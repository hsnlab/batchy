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
from . import utils


class Controller:
    '''Master controller for a worker object.

       Currently does not do anything.
    '''
    def __init__(self, batchy):
        self.batchy = batchy
        self.period = 0

    def control(self):
        self.period += 1


class NormalizedGreedyController(Controller):
    '''NormalizedGreedyController takes tasks/flows and move to a new
       worker until the current worker has enough capacity to process
       its own flows.

       Input is a normalized pipeline, otherwise this will do nothing.

    '''
    def __init__(self, batchy):
        super(NormalizedGreedyController, self).__init__(batchy)
        self.cyc_per_second = None
        self.blacklisted_names = ('source', 'src')

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
        ''' Detects if there is an slo violation on a worker.

        Returns:
        viol, w_state tuple
        viol (list): parameters of SLO violating flows
        w_state (list): worker info
        '''
        viol = []
        w_state = []
        c_workers = utils.filter_obj_list(self.batchy.workers, 'name',
                                          self.blacklisted_names)
        for worker in c_workers:
            worker_load = 0
            slo_viol = False
            worker_max_delay = {'max_delay': 0}
            c_tasks = utils.filter_obj_list(worker.tasks, 'name',
                                            self.blacklisted_names)
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
        target_worker = None
        for e in w_state:
            if e['load'] < self.cyc_per_second and e['slo_viol'] is False:
                target_worker = e['worker']
                break

        if target_worker is None:
            logging.log(logging.INFO,
                        'BATCHY CONTROL: couldnt find worker with free capacity: '
                        'try to add new workers by calling batchy.add_worker()')
            return

        # choose the flow with the highest delay_slo on worker as a
        # candidate for being moved to the target worker
        m = next((e['max_delay'] for e in w_state
                  if e['worker'].name == worker.name))
        if getattr(m, 'task', None) is None:
            # this should never happen: at least the flow with
            # slo-violation must have been added
            text = 'BATCHY CONTROL: this should never happen: cannot find ' \
                   'candidate flow for being moved'
            raise Exception(text)

        logging.log(logging.INFO,
                    f'BATCHY CONTROL: moving task {m["task"].name} '
                    f'from worker {worker.name} to '
                    f'worker {target_worker.name}')

        worker.move_task(m['task'], target_worker)


class GreedyController(Controller):
    '''GreedyController takes tasks/flows and move to a new worker until
       the current worker has enough capacity to process its own flows.

       Input is a non-normalized pipeline.
    '''
    def __init__(self, batchy):
        super(GreedyController, self).__init__(batchy)
        self.cyc_per_second = None
        self.blacklisted_names = ('source', 'src', 'decomp')
        workers = [self.batchy.add_worker(f'decomp-{i}')
                   for i in range(settings.DECOMP_EXTRA_WORKERS)]
        self.extra_workers = {'index': 0, 'workers': workers}

    def _get_worker(self, mode='rr'):
        try:
            func = getattr(self, f'_get_worker_{mode}')
            return func()
        except AttributeError:
            raise Exception(f'BATCHY CONTROL: unknown worker get method: "{mode}"')

    def _get_worker_rr(self):
        ''' Choose worker in a round-robin fashion. '''
        workers = self.extra_workers['workers']
        idx = self.extra_workers['index']
        ret = workers[idx]
        new_index = (idx + 1) % len(workers)
        self.extra_workers['index'] = new_index
        return ret

    def _get_worker_random(self):
        ''' Choose worker randomly. '''
        return random.choice(self.extra_workers['workers'])

    @staticmethod
    def check_delay_slo(resident_modules, resident_flows, turnaround_time, new_flow):
        ''' Check delay_slo feasibility if flow is added to task.

        Returns:
        False if adding new_flow causes SLO violation, otherwise True

        '''
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
        ''' Migrate flows from overloaded task to a new task.

        Returns:
        list of modules done
        '''
        modules_done = modules_done or []
        flow_diff = [f for f in task.get_flows() if f not in set(resident_flows)]
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
                        prev_mod.connect(new_task.ingress, ogate=prev_mod_ogate)
                        mtype, disconnect = ('internal', False)
                    modules_done.append(module)
                new_tflow = new_task.add_tflow(flow, new_path)
                flow.path.insert(tflow_idx, {'task': new_task,
                                             'path': new_path,
                                             'tflow': new_tflow})
        return modules_done

    def decompose(self, task):
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
        c_workers = utils.filter_obj_list(self.batchy.workers, 'name',
                                          self.blacklisted_names)
        for worker in c_workers:
            c_tasks = utils.filter_obj_list(worker.tasks, 'name',
                                            self.blacklisted_names)
            for task in c_tasks:
                logging.log(logging.INFO,
                            'BATCHY CONTROL: working on '
                            f'{worker.name}/{task.name}')
                self.decompose(task)
