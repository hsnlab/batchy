# Batchy
#
# Copyright (C) 2019 by its authors (See AUTHORS)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import logging
import pprint
import random

from . import settings
from . import tcontroller
from . import utils


class Controller:
    """Main controller for a worker object.

       Can be used as a base class.
    """

    def __init__(self, batchy):
        self.period = 0
        self.batchy = batchy
        self.denylisted_names = ['source', 'src']
        self.stat = []

    def control(self):
        """ Controller function, called by the framework. """
        self.period += 1

    def get_controlled_tasks(self):
        """ Collect controlled tasks on controlled workers.

            Returns a list of controlled tasks.

        """
        c_workers = utils.filter_obj_list(self.batchy.workers, 'name',
                                          self.denylisted_names)
        c_tasks = [task for worker in c_workers
                   for task in utils.filter_obj_list(worker.tasks, 'name',
                                                     self.denylisted_names)]
        return c_tasks


class DecompMainController(Controller):
    """Master Controller for primal decomposition.

       Works with run-to-completion ProjectedGradient task controller
       (ProjGradientRTCTaskController).

    """

    def __init__(self, batchy):
        super().__init__(batchy)
        # enable decomposition in controlled tasks
        self.control_mode = 'fixstep'
        self.gradients = []

    def set_control_mode(self, ctrl_mode):
        """ Set control mode. Raises ValueError if control mode is not supported. """
        if ctrl_mode not in self.get_control_modes():
            raise ValueError("invalid control mode")
        self.control_mode = ctrl_mode

    def get_control_modes(self):
        """ Get a list of available control modes. """
        return [m.replace('_control_', '')
                for m in dir(self) if '_control_' in m]

    def get_projgrad_ctasks(self):
        """ Collect tasks controlled by a ProjGradientRTCTaskController. """
        return [ctask for ctask in self.get_controlled_tasks()
                if isinstance(ctask.controller,
                              tcontroller.ProjGradientRTCTaskController)]

    def _collect_flows(self):
        """ Collect flows of controlled tasks. """
        return set(flow
                   for task in self.get_projgrad_ctasks()
                   for flow in task.get_flows())

    def get_gradients_flow(self, flow, store=True):
        """Collect flow gradients

            Paramaters:
            flow (Flow): collect gradients on the path of this flow
            store (bool): store gradients in self.stat; default: True

            Returns:
            gradients (list): gradients implicitely following order of
            taskflows

        """
        # collect task flows
        tflows = flow.get_tflows()
        tflows_on_leader = [tf
                            for _flow in flow.leader_task.get_flows()
                            for tf in _flow.get_tflows()
                            if tf.task != flow.leader_task]
        # calculate gradients
        ctrlr_grads = flow.leader_task.controller.sub_grad
        leader_subgrads = ctrlr_grads[len(tflows_on_leader)-1:]

        gradients = []
        for idx, tf in enumerate(tflows):
            if tf.task != flow.leader_task:
                leader_grad = leader_subgrads[max(0, idx-1)]
                __idx = min(tf.id, len(tf.task.controller.sub_grad)-1)
                follower_grad = tf.task.controller.sub_grad[__idx]
                gradients.append(leader_grad + follower_grad)

        # store stats
        if store:
            if not gradients:
                gradients = [0.0] * (len(tflows) - 1)
            self.stat[-1]['gradients'][flow.name] = gradients
        return gradients

    @staticmethod
    def _redistribute_over_delays(tflows, flow_delay_bound):
        """ Redistribure the over delay among taskflows. """
        delay_sum = sum(tflow.delay_bound for tflow in tflows)
        over = delay_sum - flow_delay_bound
        decrement = max(over / len(tflows), 0)
        if decrement:
            for tflow in tflows:
                tflow.delay_bound -= decrement
        return delay_sum, over

    def _control_fixstep(self):
        """ Adjust task flow delay bounds using a fix-value step size. """
        for flow in self._collect_flows():
            tflows = flow.get_tflows()
            followers = [tf for tf in tflows if tf.task != flow.leader_task]
            gradients = self.get_gradients_flow(flow, store=True)
            self.gradients = gradients

            if not gradients or all(g == 0.0 for g in gradients):
                self.gradients = [0] * len(followers)
                logging.log(logging.INFO,
                            'MAIN CONTROLLER: period: %s, '
                            'working on flow "%s" with D=%d\n'
                            '\t All Gradients are zero: %s',
                            self.period, flow.name, flow.D, gradients)
                continue

            # guess a step size
            step_size = flow.D * settings.DEFAULT_DECOMP_STEPSIZE_PERCENTAGE / 100

            # update tflow bounds
            for zipped in zip(gradients, followers):
                grad, tflow = zipped
                tflow.delay_bound += step_size * grad

            # ensure tflow delay bounds are below flow delay bound
            tflow_delay_sum, over_delay = self._redistribute_over_delays(tflows,
                                                                         flow.D)

            # ensure tflow delay bounds are over zero
            delay_min_tresh = 0.0
            delay_correction_sum = 0
            for tflow in tflows:
                if tflow.delay_bound < delay_min_tresh:
                    delay_correction_sum += delay_min_tresh - tflow.delay_bound
                    tflow.delay_bound = delay_min_tresh
                if tflow.delay_bound > flow.D:
                    delay_correction_sum -= tflow.delay_bound - flow.D
                    tflow.delay_bound = flow.D

            tflow_delay_sum, over_delay = self._redistribute_over_delays(tflows,
                                                                         flow.D)

            logging.log(logging.INFO,
                        'MAIN CONTROLLER: period: %s, working on flow "%s" with D=%d\n'
                        '\t sum taskflow delay bound: %.2f, over delay: %.2f, '
                        'delay_correction_sum: %.2f,\n\t '
                        'gradients: %s',
                        self.period, flow.name, flow.D,
                        tflow_delay_sum, over_delay, delay_correction_sum,
                        gradients)
            logging.log(logging.DEBUG,
                        '\t delay bounds: {%s}',
                        ', '.join([f"'{tf.get_name()}': {tf.delay_bound:.2f}"
                                   for tf in tflows]))

    def _control_varstep(self):
        """ Find proper step size, and adjust task flow delay bounds. """
        raise NotImplementedError

    def _control_dummystep(self):
        """ Dummy control function applying no control. """
        for flow in self._collect_flows():
            self.get_gradients_flow(flow, store=True)

    def _handle_infeasibility(self):
        """ Apply heuristics to recover infeasibility

            Returns:
            had_effect (bool): True if infeasibility recovery occured
        """
        had_effect = False

        for flow in self._collect_flows():
            tflows = flow.get_tflows()
            infeas_tflows = []
            feas_tflows = []
            for _tflow in tflows:
                # collect taskflows in infeasible state: delay > delay_bound
                if _tflow.get_delay_stat() >= _tflow.delay_bound:
                    infeas_tflows.append(_tflow)
                # collect tflows in feasuble state
                else:
                    feas_tflows.append(_tflow)
            logging.log(logging.DEBUG,
                        "MAIN CONTROLLER: INFEASIBILITY RECOVERY ON %s; "
                        "infeasible taskflows:  %s, feasible taskflows:  %s",
                        flow.name,
                        [tf.get_name() for tf in infeas_tflows],
                        [tf.get_name() for tf in feas_tflows])

            if not infeas_tflows or not feas_tflows:
                # no infeasible/feasible tflows on this path, jump to next flow
                logging.log(logging.DEBUG,
                            "MAIN CONTROLLER: INFEASIBILITY RECOVERY ON %s; "
                            "all taskflows are either feasible or infeasible, exiting.",
                            flow.name)
                continue

            # calculate delay budget
            delay_increment = flow.D * settings.DELAY_BOUND_PERCENTAGE / 100
            sum_delay_increment = len(infeas_tflows) * delay_increment
            logging.log(logging.DEBUG,
                        "MAIN CONTROLLER: INFEASIBILITY RECOVERY ON %s; "
                        "sum delay bound increment:  %.2f",
                        flow.name, sum_delay_increment)

            # increase delay bounds of infeasible tflows
            for _tflow in infeas_tflows:
                old_tflow_bound = _tflow.delay_bound
                _tflow.delay_bound += delay_increment
                logging.log(logging.INFO,
                            "MAIN CONTROLLER: INFEASIBILITY RECOVERY ON %s/%s; "
                            "increase delay bound from %.2f to %.2f",
                            flow.name, _tflow.get_name(),
                            old_tflow_bound, _tflow.delay_bound)

            # decrase feasible tflows budget to compensate the budget reallocation
            sum_delay_decrement = 0
            _iter = 0
            per_tflow_delay_decrement = delay_increment / len(feas_tflows)
            while sum_delay_decrement < sum_delay_increment and _iter < 10:
                _iter += 1
                for _tflow in feas_tflows:
                    if _tflow.delay_bound > per_tflow_delay_decrement:
                        old_tflow_bound = _tflow.delay_bound
                        _tflow.delay_bound -= per_tflow_delay_decrement
                        logging.log(logging.INFO,
                                    "MAIN CONTROLLER: INFEASIBILITY RECOVERY ON %s/%s; "
                                    "decrease delay bound from %.2f to %.2f",
                                    flow.name, _tflow.get_name(),
                                    old_tflow_bound, _tflow.delay_bound)
                        sum_delay_decrement += per_tflow_delay_decrement
                        if sum_delay_decrement >= sum_delay_increment:
                            break
            had_effect = True
        return had_effect

    def control(self):
        """Control function facade, runs control implementation set by
           self.control_mode.
        """
        super().control()

        # add placeholder for new stats
        self.stat.append({'gradients': {}})

        # try recover from infeasibility
        self._handle_infeasibility()

        # execute control function
        try:
            control_func = getattr(self, f'_control_{self.control_mode}')
        except AttributeError as attr_err:
            txt = f'{self.control_mode} is not valid controller mode, ' \
                f'try one of these: {self.get_control_modes()}.'
            raise Exception(txt) from attr_err
        control_func()


class NormalizedGreedyController(Controller):
    """NormalizedGreedyController takes tasks/flows and move to a new
       worker until the current worker has enough capacity to process
       its own flows.

       Input is a normalized pipeline, otherwise this will do nothing.

    """

    def __init__(self, batchy):
        super().__init__(batchy)
        self.cyc_per_second = None

    @staticmethod
    def _has_slo_viol(delay, rate, delay_slo=None, rate_slo=None):
        if delay_slo is not None:
            if delay > delay_slo * settings.CRITICAL_SLO_VIOLATION_RATIO:
                return True
        if rate_slo is not None:
            if rate > rate_slo * settings.CRITICAL_SLO_VIOLATION_RATIO:
                return True
        return False

    def detect(self):
        """ Detects if there is an slo violation on a worker.

        Returns:
        viol, w_state tuple
        viol (list): parameters of SLO violating flows
        w_state (list): worker info
        """
        viol = []
        w_state = []
        c_workers = utils.filter_obj_list(self.batchy.workers, 'name',
                                          self.denylisted_names)
        for worker in c_workers:
            worker_load = 0
            slo_viol = False
            worker_max_delay = {'max_delay': 0}
            c_tasks = utils.filter_obj_list(worker.tasks, 'name',
                                            self.denylisted_names)
            for task in c_tasks:
                flows = task.get_flows()

                if not all(utils.get_rate_slo_resource(f) == 'packet' for f in flows
                           if f.has_rate_slo()):
                    text = f'batchy.control: only "packet"-type rate-SLO ' \
                           f'supported for worker "{worker}"'
                    raise Exception(text)

                task_rate = 0
                for flow in flows:
                    delay_slo = flow.D
                    delay = flow.stat[-1][f'latency_{settings.DELAY_MAX_PERC}']
                    rate_slo = utils.get_rate_slo_value(flow) or 0
                    rate = flow.stat[-1]['pps']

                    if self._has_slo_viol(delay, rate, delay_slo, rate_slo):
                        slo_viol = True
                        viol_params = {'flow': flow,
                                       'task': task,
                                       'worker': worker,
                                       'delay': delay,
                                       'delay_slo': delay_slo,
                                       'rate': rate,
                                       'rate_slo': rate_slo,
                                       }
                        if delay_slo:
                            viol_params['delay_slo_viol'] = delay - delay_slo
                        if rate_slo:
                            viol_params['rate_slo_viol'] = rate - rate_slo
                        viol.append(viol_params)

                    if rate_slo is not None:
                        task_rate += rate_slo
                    if delay_slo is not None and \
                       worker_max_delay['max_delay'] < delay_slo:
                        worker_max_delay = {'max_delay': delay_slo,
                                            'flow': flow,
                                            'task': task,
                                            }

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG, pprint.pformat(task.stat))

                task_load = task_rate * task.stat[-1]['cyc_per_packet']
                worker_load += task_load

                if self.cyc_per_second is None:
                    self.cyc_per_second = task.stat[-1]['cyc_per_time']

            w_state.append({'worker': worker,
                            'load': worker_load,
                            'slo_viol': slo_viol,
                            'max_delay': worker_max_delay})

        return viol, w_state

    def control(self):
        super().control()
        viol, w_state = self.detect()
        if not viol:
            logging.log(logging.INFO,
                        'BATCHY CONTROL: no critical SLO violation detected')
            return

        # take the first flow that is in violation and try to handle that
        v = viol[0]
        flow = v['flow']
        task = v['task']
        worker = v['worker']

        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.log(logging.INFO,
                        f'BATCHY CONTROL: flow {flow.name}: '
                        f'critical SLO violation detected: '
                        f'worker={worker.name}, task={task.name}: '
                        f'delay={v["delay"]:.2f}/'
                        f'delay_slo={v["delay_slo"] or 0:.2f}, '
                        f'rate={v["rate"]:.2f}/rate_slo={v["rate_slo"] or 0:.2f}')

        # try to find a worker with free capacity
        target_worker = next((e['worker'] for e in w_state
                              if e['load'] < self.cyc_per_second and e['slo_viol'] is False),
                             None)

        if target_worker is None:
            logging.log(logging.INFO,
                        'BATCHY CONTROL: couldnt find worker with free capacity: '
                        'try to add new workers by calling batchy.add_worker()')
            return

        # choose the flow with the highest delay_slo on worker as a
        # candidate for being moved to the target worker
        worker_maxdelay = next((e['max_delay'] for e in w_state
                                if e['worker'].name == worker.name))
        if getattr(worker_maxdelay, 'task', None) is None:
            # this should never happen: at least the flow with
            # slo-violation must have been added
            text = 'BATCHY CONTROL: this should never happen: cannot find ' \
                   'candidate flow for being moved'
            raise Exception(text)

        logging.log(logging.INFO,
                    'BATCHY CONTROL: moving task %s from worker %s to worker %s',
                    worker_maxdelay['task'].name, worker.name, target_worker.name)

        worker.move_task(worker_maxdelay['task'], target_worker)


class GreedyController(Controller):
    """GreedyController takes tasks/flows and move to a new worker until
       the current worker has enough capacity to process its own flows.

       Input is a non-normalized pipeline.
    """

    def __init__(self, batchy):
        super().__init__(batchy)
        self.cyc_per_second = None
        self.denylisted_names.append('decomp')
        workers = [self.batchy.add_worker(f'decomp-{i}')
                   for i in range(settings.DECOMP_EXTRA_WORKERS)]
        self.extra_workers = {'index': 0, 'workers': workers}

    def _get_worker(self, mode='rr'):
        try:
            func = getattr(self, f'_get_worker_{mode}')
            return func()
        except AttributeError as attr_err:
            txt = f'BATCHY CONTROL: unknown worker get method: "{mode}"'
            raise Exception(txt) from attr_err

    def _get_worker_rr(self):
        """ Choose worker in a round-robin fashion. """
        workers = self.extra_workers['workers']
        idx = self.extra_workers['index']
        ret = workers[idx]
        new_index = (idx + 1) % len(workers)
        self.extra_workers['index'] = new_index
        return ret

    def _get_worker_random(self):
        """ Choose a worker randomly. """
        return random.choice(self.extra_workers['workers'])

    @staticmethod
    def check_delay_slo(resident_modules, resident_flows, turnaround_time, new_flow):
        """ Check delay_slo feasibility if flow is added to task.

        Returns:
        False if adding new_flow causes SLO violation, otherwise True

        """
        for flow in resident_flows + [new_flow]:
            sum_time = turnaround_time
            sum_time += sum([m.get_delay_estimate(batch_size=settings.BATCH_SIZE)
                             for tflow in flow.path for m in tflow['path']
                             if m not in resident_modules])
            sum_time += sum([m.get_delay_estimate(batch_size=1)
                             for tflow in flow.path for m in tflow['path']])
            if sum_time > (flow.D or 0):
                return False
        return True

    def migrate_flows(self, task, resident_modules, resident_flows, modules_done=None,
                      worker_select_mode='rr'):
        """ Migrate flows from overloaded task to a new task.

        Returns:
        list of modules done
        """
        modules_done = modules_done or []
        flow_diff = [f for f in task.get_flows(
        ) if f not in set(resident_flows)]
        has_modules = any((m for flow in flow_diff
                           for tflow in flow.path for m in tflow['path']
                           if m not in resident_modules and m not in modules_done))
        if has_modules:
            for flow in flow_diff:
                tflow = next(t for t in flow.path if t['task'] == task)
                tflow_idx = flow.path.index(tflow)
                flow_modules = [m for m in tflow['path']
                                if m not in resident_modules
                                and m not in modules_done]
                if not flow_modules:
                    # no modules to move from flow on task,
                    # check next flow
                    continue
                new_worker = self._get_worker(mode=worker_select_mode)
                new_task = new_worker.add_task()
                ctrlr = self.batchy.resolve_task_controller(
                    settings.DECOMP_EXTRA_TASK_CONTROLLER)
                new_task.set_controller(ctrlr)
                mtype, disconnect = ('ingress', True)
                new_path = []
                for module in flow_modules:
                    tflow['path'].remove(module)
                    new_path.append(module)
                    prev_mod = module.parent
                    if prev_mod is None:
                        prev_tflow = flow.path[tflow_idx - 1]
                        prev_mod = prev_tflow['path'][-1]
                    prev_mod_ogate = module.get_parent_name_ogate()[1]
                    task.remove_module(module, disconnect)
                    new_task.add_module(module, type=mtype)
                    if mtype == 'ingress':
                        prev_mod.connect(new_task.ingress,
                                         ogate=prev_mod_ogate)
                        mtype, disconnect = ('internal', False)
                    modules_done.append(module)
                new_tflow = new_task.add_tflow(flow, new_path)
                flow.path.insert(tflow_idx, {'task': new_task,
                                             'path': new_path,
                                             'tflow': new_tflow})
        return modules_done

    def decompose(self, task):
        """ Decompose pipeline. """
        resident_modules = []
        resident_flows = []
        resident_turntime = 0
        modules_done = []
        flows = task.get_flows().copy()
        flows.sort(key=lambda x: x.D or settings.DEFAULT_DELAY_BOUND)
        for flow in flows:
            if self.check_delay_slo(resident_modules, resident_flows,
                                    resident_turntime, flow):
                sum_delay = sum([m.get_delay_estimate(batch_size=settings.BATCH_SIZE)
                                 for tflow in flow.path for m in tflow['path']
                                 if m not in resident_modules])
                resident_turntime += sum_delay
                resident_flows.append(flow)
                resident_modules.extend([m for tflow in flow.path
                                         for m in tflow['path']])
            else:
                modules_done = self.migrate_flows(task, resident_modules,
                                                  resident_flows, modules_done)

    def control(self):
        super().control()
        for task in self.get_controlled_tasks():
            logging.log(logging.INFO,
                        'BATCHY CONTROL: working on %s/%s',
                        task.worker.name, task.name)
            self.decompose(task)
