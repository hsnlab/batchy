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

import collections
import logging
import pprint
import numpy as np

from . import settings


np.set_printoptions(precision=5)


def get_tcontroller(task, base_class):
    ''' Resolve task controller by controller type and task type

        Parameters:
        task (Task): task to set controller for
        base_class (str): name of controller class

        Returns:
        a controller instance set for the task

    '''
    name = f'{base_class}{task.type}TaskController'
    ctrlr_class = globals().get(name)
    if ctrlr_class is None:
        raise Exception(f'Unknown task controller class "{name}"')
    return ctrlr_class(task)


class TaskController:
    ''' Task controller base class '''

    def __init__(self, task):
        self.task = task
        self.period = 0
        self.flows = [f for f in task.batchy.flows if f.traverses_task(task)]
        if task.type == 'RTC' and not task.cmodules:
            logging.log(logging.DEBUG,
                        f'No controllable module found in RTC task "{task.name}"')
            task.controller = None
        else:
            task.controller = self

    def get_violating_flows(self):
        ''' Collect flows' delay-SLO violation data

            Returns:
            delay (dict): last measured delay of flows
            delay_bound (float):  delay bounds of flows
            over (list): modules violating contraint
            error (float): sum of delay violations
            dist (float): smallest distance from the delay contraint over all flows

        '''
        delay_key = f'latency_{settings.DELAY_MAX_PERC}'
        delay = {f.id: f.stat[-1][delay_key] for f in self.flows}
        delay_bound = {f.id: (f.D if f.D is not None
                              else settings.DEFAULT_DELAY_BOUND)
                       for f in self.flows}
        error = 0.0
        over = []
        dist = sum(delay_bound.values())  # large number
        for flow in self.flows:
            error_f = delay[flow.id] - delay_bound[flow.id]
            if error_f > 0:
                error += error_f
                over.append(flow)
            else:
                dist = min(dist, -error_f)
        return delay, delay_bound, over, error, dist

    def control(self, *args, **kwargs):
        self.period += 1


class RTCTaskController(TaskController):
    ''' Base class for run-to-completion task controllers '''
    @staticmethod
    def get_control_at(cmodule, t):
        return cmodule.stat[t]['q_v']


class WFQTaskController(TaskController):
    ''' Base class for weighted-fair-queueing task controllers '''
    @staticmethod
    def get_control_at(cmodule, t):
        return cmodule.stat[t]['w_v']


class NullTaskController(TaskController):
    ''' Leaves all triggers at zero '''

    def __init__(self, task):
        super(NullTaskController, self).__init__(task)
        self.real_controller = get_tcontroller(task, 'Null')

    def __repr__(self, *args, **kwargs):
        return self.real_controller.__repr__(*args, **kwargs)

    def control(self, *args, **kwargs):
        return self.real_controller.control(*args, **kwargs)


class NullRTCTaskController(RTCTaskController):
    def __init__(self, task):
        super(NullRTCTaskController, self).__init__(task)
        for m in task.cmodules:
            m.set_trigger(0)

    def __repr__(self):
        return 'NullRTCTaskController'


class NullWFQTaskController(WFQTaskController):
    def __init__(self, task):
        super(NullWFQTaskController, self).__init__(task)
        for m in task.cmodules:
            m.set_weight(1)

    def __repr__(self):
        return 'NullWFQTaskController'


class MaxTaskController(TaskController):
    ''' Leaves all triggers at settings.BATCH_SIZE (default: 32) '''

    def __init__(self, task):
        super(MaxTaskController, self).__init__(task)
        self.real_controller = get_tcontroller(task, 'Max')

    def __repr__(self, *args, **kwargs):
        return self.real_controller.__repr__(*args, **kwargs)

    def control(self, *args, **kwargs):
        return self.real_controller.control(*args, **kwargs)


class MaxRTCTaskController(RTCTaskController):
    def __init__(self, task):
        super(MaxRTCTaskController, self).__init__(task)
        for m in task.cmodules:
            m.set_trigger(settings.BATCH_SIZE)

    def __repr__(self):
        return 'MaxRTCTaskController'


class MaxWFQTaskController(WFQTaskController):
    def __init__(self, task):
        super(MaxWFQTaskController, self).__init__(task)
        for m in task.cmodules:
            m.set_weight(settings.WEIGHT_MAX)

    def __repr__(self):
        return 'MaxWFQTaskController'


# Fix{i}TaskController leaves all triggers at a preset value
# betweeen 0 and settings.BATCH_SIZE

def _fix_t_init(self, task):
    super(self.__class__, self).__init__(task)
    name = self.__class__.__name__.replace('TaskController', '')
    self.real_controller = get_tcontroller(task, name)


def _fix_t_repr(self, *args, **kwargs):
    return self.real_controller.__repr__(*args, **kwargs)


def _fix_t_control(self, *args, **kwargs):
    return self.real_controller.control(*args, **kwargs)


def _fix_init(self, task):
    super(self.__class__, self).__init__(task)
    for m in task.cmodules:
        m.set_trigger(self.size)


def _fix_repr(self):
    return f'Fix{self.size}RTCTaskController'


for fix_batch_size in range(1, settings.BATCH_SIZE):
    tname = f'Fix{fix_batch_size}TaskController'
    globals()[tname] = type(tname, (TaskController,),
                            {'__init__': _fix_t_init,
                             '__repr__': _fix_t_repr,
                             'control': _fix_t_control,
                            })

    rtctname = f'Fix{fix_batch_size}RTCTaskController'
    globals()[rtctname] = type(rtctname, (RTCTaskController,),
                               {'size': fix_batch_size,
                                '__init__': _fix_init,
                                '__repr__': _fix_repr,
                               })


class FeasDirTaskController(TaskController):
    ''' Create a proper controller for the task
         RTC-controller for an RTC task
         WFQ-controller for an WFQ task
        then delegate calls to the new controller
    '''

    def __init__(self, task):
        super(FeasDirTaskController, self).__init__(task)
        self.real_controller = get_tcontroller(task, 'FeasDir')

    def __repr__(self, *args, **kwargs):
        return self.real_controller.__repr__(*args, **kwargs)

    def control(self, *args, **kwargs):
        return self.real_controller.control(*args, **kwargs)


class FeasDirRTCTaskController(RTCTaskController):
    ''' RTC task controller based on the method of feasible directions
        takes a single module, get a feasible improving direction,
        and trigger move along that
    '''

    def __init__(self, task):
        super(FeasDirRTCTaskController, self).__init__(task)
        self.gradient = []
        self.ptr = 0

    def __repr__(self):
        return f'FeasDirRTCTaskController: ptr={self.ptr}'

    def control(self, *args, **kwargs):
        super().control()
        batchy = self.task.batchy
        cmodules = self.task.cmodules

        module = self.task.cmodules[self.ptr]
        self.ptr = (self.ptr + 1) % len(cmodules)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG, module.format_stat())

        q_v = module.q_v
        b_v = module.stat[-1]['b_in']
        min_q, max_q = self.get_min_max_trigger(module)

        # do we satisfy the delay requirements?
        delay, delay_bound, over, sum_error, delay_dist = self.get_violating_flows()

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG,
                        f'CONTROL: {self.task.name} : '
                        f'module {module.name} (id={module.id}), '
                        f'q_v = {q_v}, min_q = {min_q}, max_q = {max_q}, '
                        f'sum_error={sum_error:.3f}, dist={delay_dist:.3f}')

        # special case for q_v = 0: if all flows traversing the module
        # could tolerate an additional 1/x_v delay, set q_v = b_min
        x_v = module.stat[-1]['x_v']
        if q_v == 0 and b_v < settings.DEFAULT_PUSH_RATIO * settings.BATCH_SIZE:
            if x_v == 0:
                # we do not even get scheduled: we can do whatever we want
                module.set_trigger(min_q)
                return
            t_v = 1e9 / x_v
            delay_budget = min([delay_bound[f.id] - delay[f.id]
                                for f in batchy.flows
                                if f.traverses_module(module)])
            delay_budget = min(delay_budget, settings.DEFAULT_DELAY_BOUND)

            q = min(max_q, min_q + settings.DEFAULT_BUFFER_PULL)
            if delay_budget > (q / min_q * t_v):
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'CONTROL: {self.task.name} : '
                                f'module {module.name} (id={module.id}), '
                                f'delay_budget={delay_budget:.3f} > '
                                f'1/x_v={t_v:.3f}: setting q_v -> {q}')
                module.set_trigger(q)
                return
            if delay_budget > 0 and min_q == 1:
                # q_v=1 will not yield additional delay: if it is allowed, set it
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'CONTROL: {self.task.name} : '
                                f'module {module.name} (id={module.id}), '
                                f'delay_budget={delay_budget:.3f} and '
                                f'q_v=1 is feasible: setting q_v - > {min_q}')
                module.set_trigger(min_q)
                return

            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.log(logging.INFO,
                            f'CONTROL: {self.task.name} : '
                            f'module {module.name} (id={module.id}), '
                            f'delay_budget={delay_budget:.3f} <= '
                            f'1/x_v={t_v:.3f}: leaving q_v=0')
            return

        # module receives batches that are already too large
        if b_v >= settings.DEFAULT_PUSH_RATIO * settings.BATCH_SIZE and \
           q_v > 0 and x_v > 0:
            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.log(logging.INFO,
                            f'\tmodule {module.name}: PUSH: input batch rate '
                            f'b[v]={b_v:.3f} >= '
                            f'{settings.DEFAULT_PUSH_RATIO:.2f} * BATCH_SIZE: '
                            f'setting q[v]: {q_v:d} -> 0')
            module.set_trigger(0)
            return

        # special case for q_v -> 0: if some flows traversing the module is
        # overdelayed and module is on minimum trigger, reset q_v to zero
        # maybe we should need a more clever heuristics
        if q_v == min_q and x_v > 0:
            delay_budget = 0.0
            for f in batchy.flows:
                if f.traverses_module(module):
                    try:
                        val = (delay[f.id] - delay_bound[f.id]) / delay_bound[f.id]
                    except ZeroDivisionError:
                        val = 0.0
                    delay_budget = max(val, delay_budget)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG,
                            f'CONTROL: {self.task.name} : '
                            f'module {module.name} (id={module.id}), '
                            f'delay_budget={delay_budget:.3f} percent, '
                            f'overdelay_bound={settings.OVERDELAY_BOUND:.3f}: '
                            f'considering setting q_v={q_v:d} -> 0')
            if delay_budget > settings.OVERDELAY_BOUND:
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'CONTROL: {self.task.name} : '
                                f'module {module.name} (id={module.id}), '
                                f'delay_budget={delay_budget:.3f} percent > '
                                f'overdelay_bound={settings.OVERDELAY_BOUND:.3f}: '
                                f'setting q_v={q_v:d} -> 0')
                module.set_trigger(0)
                return

        df_dq, dDf_dq = self.get_gradient(module)
        sum_over_grad = sum([dDf_dq[f.id] for f in over])

        # no module violates delay constraints OR some flows violate the
        # delay contraints but this module would not affect those modules
        if sum_error == 0 or (sum_over_grad == 0 and df_dq < 0):
            # get the largest possible increase in q_v if possible
            delta_qv = min(max_q - q_v, settings.DELTA_TRIGGER_MAX)
            for f in self.flows:
                if dDf_dq[f.id] > 0:
                    diff = (delay_bound[f.id] - delay[f.id]) / dDf_dq[f.id]
                    delta_qv = min(delta_qv, int(diff) + settings.EXTRA_TRIGGER)

            if df_dq < 0 and logging.getLogger().isEnabledFor(logging.INFO):
                logging.log(logging.INFO,
                            f'CONTROL: {self.task.name} : '
                            f'module {module.name} (id={module.id}): '
                            f'NO OVERDELAYED FLOW WOULD GET HIGHER DELAY '
                            f'and df_dq={df_dq:.3f}<0: '
                            f'increasing q_v: {q_v:d} -> {q_v+delta_qv:d}')
            else:
                delta_qv = 0
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'CONTROL: {self.task.name} : '
                                f'module {module.name} (id={module.id}): '
                                f'NO OVERDELAYED FLOW WOULD GET HIGHER DELAY '
                                f'but df_dq={df_dq:.3f} is not negative: '
                                f'leaving q_v={q_v:d}')
        else:
            # some modules violate delay constraints
            # max_delta = int(math.ceil(abs(sum_error/sum_over_grad)))
            if sum_over_grad > 0:
                # decrease trigger
                delta_trigger_max = min(settings.DELTA_TRIGGER_MAX,
                                        q_v - min_q,
                                        int(abs(sum_error / sum_over_grad)) + 1)
                for f in self.flows:
                    if dDf_dq[f.id] > 0 and f in over:
                        # flow overdelayed and delay will decrease
                        tmp = int((delay[f.id] - delay_bound[f.id]) / abs(dDf_dq[f.id]))
                    elif dDf_dq[f.id] < 0 and not f in over:
                        # flow is not overdelayed and delay will increase
                        tmp = int((delay_bound[f.id] - delay[f.id]) / abs(dDf_dq[f.id]))
                    else:
                        continue
                    tmp += settings.EXTRA_TRIGGER
                    delta_trigger_max = min(delta_trigger_max, tmp)
                delta_qv = -delta_trigger_max
            else:
                # increase trigger
                delta_trigger_max = min(settings.DELTA_TRIGGER_MAX, max_q - q_v)
                for f in self.flows:
                    if dDf_dq[f.id] < 0 and f in over:
                        # flow overdelayed and delay will decrease
                        tmp = int((delay[f.id] - delay_bound[f.id]) / abs(dDf_dq[f.id]))
                    elif dDf_dq[f.id] > 0 and not f in over:
                        # flow is not overdelayed and delay will increase
                        tmp = int((delay_bound[f.id] - delay[f.id]) / abs(dDf_dq[f.id]))
                    else:
                        continue
                    tmp += settings.EXTRA_TRIGGER
                    delta_trigger_max = min(delta_trigger_max, tmp)
                delta_qv = delta_trigger_max

            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.log(logging.INFO,
                            f'CONTROL: {self.task.name} : '
                            f'module {module.name} (id={module.id}): '
                            f'SOME FLOWs OVERDELAYED '
                            f'(sum_over_grad={sum_over_grad:.3f}), '
                            f'setting q_v: {q_v:d} -> {q_v+delta_qv:d}')

        # adjust
        module.set_trigger(q_v + delta_qv)

    def get_gradient(self, u):
        bs = self.task.stat[-1]
        # get statistics from the INGRESS MODULE rather than the tc as they are more accurate
        b_0 = bs['b_in']
        t_0 = bs['t_in']
        w = self.task.weight  # should be 1

        # objective
        us = u.stat[-1]
        r_u = us['r_v']
        q_u = us['q_v']
        T_0u, T_1u = u.get_sum_delay(u.get_unbuffered_descs())

        df_dqu = 0
        if q_u > 0:
            df_dqu = -b_0 * r_u * T_0u / (w * pow(q_u, 2))

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG,
                        f'df_dq: {u}:\n\t b_0={b_0:.3f} r_u={r_u:.3f} '
                        f'T_0u={T_0u:.3f} w={w:.3f} '
                        f'q_u={q_u:.3f} -> {df_dqu:.3f}')

        df_dq = df_dqu

        # flows
        dDf_dq = [0] * len(self.flows)
        for f in self.flows:
            # should have a single task flow
            tf = f.path[0]['tflow']
            cbr = f.is_rate_limited()
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                rl_text = 'not rate-limited'
                if cbr:
                    rl_text = 'RATE_LIMITED'
                logging.log(logging.DEBUG, f'dDf_dq: flow {f} {rl_text}')

            qv_rv = 0    # sum_{v \in p_f} q_v/r_v
            for v in tf.cpath:
                vs = v.stat[-1]
                q_v = vs['q_v']
                r_v = vs['r_v']
                if r_v > 0:
                    qv_rv += q_v / r_v

            # common component of all derivatives
            dDf_dqu = 0
            if q_u > 0 and not cbr:
                dDf_dqu = - (r_u * T_0u * qv_rv) / (w * pow(q_u, 2))

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG,
                            f'dDf_dqu: {u} -- common component:\n\t'
                            f'cbr={cbr}, r_u={r_u:.3f} '
                            f'T_u0={T_0u:.3f} q_u={q_u:.3f} '
                            f'sum_qv_rv={qv_rv:.3f} -> {dDf_dqu:.3f}')

            if u in tf.cpath:
                # module on the path of the flow, differential must be
                # adjusted
                diff = 0.0
                if r_u > 0.0:
                    diff = t_0 / (b_0 * r_u) + T_1u
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'dDf_dqu: {u} -- adjusted:\n\t'
                                f't_0={t_0:.3f} b_0={b_0:.3f} r_u={r_u:.3f} '
                                f'T_1u={T_1u:.3f}: '
                                f'{dDf_dqu:.3f} -> {dDf_dqu + diff:.3f}')
                dDf_dqu += diff

            dDf_dq[f.id] = dDf_dqu

        return (df_dq, dDf_dq)

    @staticmethod
    def get_min_max_trigger(u):
        if not u.stat:
            raise ValueError(f'{u.name} has no stats available')
        if not u.is_controlled:
            raise ValueError(f'{u.name} is not controlled')

        r_u = u.stat[-1]['r_v']

        max_q = settings.BATCH_SIZE
        for v in u.c_desc:
            q_v = v.q_v
            if q_v == 0:
                continue  # uncontrolled descendant
            r_v = v.stat[-1]['r_v']
            r_uv = r_v / r_u
            max_q = min(max_q, int(1 / r_uv * q_v))

        min_q = int(u.stat[-1]['b_in'])

        return min_q, max_q

    def get_trigger_effect(self, module, delta_q, delay, delay_bound):
        dDf_dq = self.gradient[-1]['dDf_dq']
        error = 0
        for i, _ in enumerate(self.flows):
            dDf_dqv = dDf_dq[i][module.id]
            err = delay[i] + dDf_dqv * delta_q - delay_bound[i]
            if err > 0:
                error += err
        return error


class FeasDirWFQTaskController(WFQTaskController):
    ''' WFQ task controller based on the method of feasible directions
        takes a single module, get a feasible improving direction,
        and weight move along that
    '''

    def __init__(self, task):
        super(FeasDirWFQTaskController, self).__init__(task)
        self.ptr = 0
        self.gradient = []

    def __repr__(self):
        return f'FeasDirWFQTaskController: ptr={self.ptr}'

    def control(self, *args, **kwargs):
        super().control()
        module = self.task.cmodules[self.ptr]
        self.ptr = (self.ptr + 1) % len(self.task.cmodules)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG, module.format_stat())

        cid = module.cid
        w_v = module.w_v

        # do we satisfy the delay requirements?
        _, _, over, sum_error, delay_dist = self.get_violating_flows()

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG,
                        f'CONTROL: {self.task.name} : '
                        f'module {module.name} (id={module.id}): '
                        f'w_v = {w_v}, sum_error={sum_error:.3f}, '
                        f'dist={delay_dist:.3f}')

        self.get_gradient()
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG, self.format_grad())

        df_dw = self.gradient[-1]['df_dw']
        dtf_dw = self.gradient[-1]['dtf_dw']

        sum_over_grad = sum([dtf_dw[f.id][cid] for f in over])

        delta_wv = 0
        # no module violates delay constraints OR some flows violate the
        # delay contraints but this module would not affect those modules:
        # we may DECREASE module weight
        if sum_error == 0 or (sum_over_grad == 0 and df_dw[cid] < 0):
            delta_wv = 0 if w_v <= 1 else settings.WEIGHT_GRADIENT

            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.log(logging.INFO,
                            f'CONTROL: {self.task.name} : '
                            f'module {module.name} (id={module.id}): '
                            f'NO OVERDELAYED FLOW WOULD GET HIGHER DELAY: '
                            f'decrasing w_v: {w_v:d} -> {w_v-delta_wv:d}')

        else:
            # some modules violate delay constraints
            # we must INCREASE module weight
            delta_wv = min(settings.WEIGHT_GRADIENT,
                           settings.DELTA_WEIGHT_MAX - w_v)
            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.log(logging.INFO,
                            f'CONTROL: {self.task.name} : module {module.name} '
                            f'(id={module.id}): '
                            f'SOME FLOWs OVERDELAYED '
                            f'(sum_over_grad={sum_over_grad:.3f}), '
                            f'setting w_v: {w_v:d} -> {w_v+delta_wv:d}')
        # adjust
        module.set_weight(w_v + delta_wv)

    def get_gradient(self):
        bs = self.task.stat[-1]
        b_0 = bs['b_in']
        t_0 = bs['t_in']
        x_0 = bs['x_in']
        grad = {}

        T_0 = {}
        T_1 = {}
        for v in self.task.cmodules:
            T_0[v], T_1[v] = v.get_sum_delay(v.get_uncontrolled_descs(v))

        # objective: make this as simple as possible: we know that
        # increasing w_v will increase the objective function and
        # decreasing it will redouce it, bu we are not interested in as to
        # how much
        df_wu = []
        for u in self.task.cmodules:
            us = u.stat[-1]
            r_u = us['r_v']
            w_u = us['w_v']

            df_dwu = 1
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG,
                            f'df_dw: {u}:\n\t b_0={b_0:.3f} r_u={r_u:.3f} '
                            f'T_0u={T_0[u]:.3f} w_u={w_u:.3f} -> {df_dwu:.3f}')
            df_wu.append(df_dwu)

        grad['df_dw'] = df_wu

        # flows: delay has two components, one arising from that if we
        # increase w_v that the scheduling unit (the share received by a
        # module with w=1), and another one arisig from that w affects the
        # frequency with which a module is scheduled; below, we ignore the first component
        grad['dtf_dw'] = collections.defaultdict(list)
        for f in self.flows:
            # should have a single task flow
            tf = f.path[0]['tflow']
            cbr = f.is_rate_limited()
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                rl_text = 'not rate-limited'
                if cbr:
                    rl_text = 'RATE_LIMITED'
                logging.log(logging.DEBUG, f'dtf_dw: flow {f} {rl_text}')

            dtf_dw = []
            for u in self.task.cmodules:
                # common component of all derivatives
                dtf_dwu = 0

                if u in tf.cpath:
                    R_u = us['R_v']
                    # module on the path of the flow, differential must be
                    # adjusted
                    diff = - (1 + T_1[u] * R_u) / (x_0 * pow(w_u, 2))
                    dtf_dwu += diff

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'dtf_dwu: {u} -- adjusted:\n\tt_0={t_0:.3f} '
                                f'b_0={b_0:.3f} r_u={R_u:.3f} '
                                f'T_1u={T_1[u]:.3f}: {dtf_dwu:.3f}')
                dtf_dw.append(dtf_dwu)

            grad['dtf_dw'][f.id].extend(dtf_dw)

        self.gradient.append(grad)

    def format_grad(self):
        grad_content = pprint.pformat(self.gradient[-1], indent=4, width=1)
        return f'Gradient:\n{grad_content}\n'


class ProjGradientTaskController(TaskController):
    ''' Create a proper controller for the task
         RTC-controller for an RTC task
         WFQ-controller for an WFQ task
        then delegate calls to the new controller
    '''

    def __init__(self, task):
        super(ProjGradientTaskController, self).__init__(task)
        self.real_controller = get_tcontroller(task, 'ProjGradient')

    def __repr__(self, *args, **kwargs):
        return self.real_controller.__repr__(*args, **kwargs)

    def control(self, *args, **kwargs):
        return self.real_controller.control(*args, **kwargs)


class ProjGradientRTCTaskController(RTCTaskController):
    ''' RTC task controller based on the method of projected gradient '''

    def __repr__(self):
        return 'ProjGradientRTCTaskController'

    def control(self, *args, **kwargs):
        super().control()
        batchy = self.task.batchy
        cmodules = self.task.cmodules

        delay, delay_bound, _, sum_error, delay_dist = self.get_violating_flows()
        delay_budget = {f:delay_bound[f] - delay[f] for f in self.flows}

        # count times in [nsec]
        T_0, T_1, x, b, q, t, R, q_min, dt_dq, fxm = \
                        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for v in cmodules:
            T_0[v], T_1[v] = v.get_sum_delay(v.get_uncontrolled_descs(v))
            x[v] = v.stat[-1]['x_v'] / 1e9
            q[v] = v.stat[-1]['q_v']
            b[v] = v.stat[-1]['b_in']
            R[v] = v.stat[-1]['R_v'] / 1e9
            t[v] = 0.0
            if x[v] > 0.0:
                t[v] = 1/x[v] + T_0[v] + T_1[v] * q[v]
            dt_dq[v] = 0.0
            if R[v] > 0.0:
                dt_dq[v] = 1/R[v] + T_1[v]
            q_min[v] = int(v.stat[-1]['b_in']) + settings.DEFAULT_BUFFER_PULL
            fxm[v] = False

        by_delay = [(t[v], v) for v in cmodules]
        by_delay.sort(key=lambda x: x[0], reverse=True)
        fxf = {f: False for f in self.flows}

        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.log(logging.INFO,
                        f'CONTROL: {self.task.name}: '
                        f'error={sum_error:.3f}, dist={delay_dist:.3f}')

        # PUSH module to qv=0 if it already receives large batches
        for v in cmodules:
            if q[v] > 0 and b[v] >= settings.DEFAULT_PUSH_RATIO * settings.BATCH_SIZE:
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'\tmodule {v.name}: PUSH: input batch rate '
                                f'b[v]={b[v]:.3f} >= {settings.DEFAULT_PUSH_RATIO:.2f} '
                                f'* BATCH_SIZE: setting q[v]: {q[v]:d} -> 0')
                fxm[v] = True
                v.set_trigger(0)

        if sum_error > 0:
            # FEASIBILITY RECOVERY PHASE: push back all flows to below delay
            # bound, PUSH q_v to zero if needed
            for f in self.flows:
                if delay_budget[f] >= 0:
                    continue

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'RECOVERY MODE for flow {f.name}: '
                                f'delay={delay[f]:.3f}, bound={delay_bound[f]:.3f}')

                fxf[f] = True
                for _, v in by_delay:
                    if fxm[v] or q[v] == 0 or R[v] == 0 or not f.traverses_module(v):
                        continue

                    delta_q = int(abs(delay_budget[f]) / dt_dq[v]) + 1

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'RECOVERY MODE for flow {f.name} at '
                                    f'MODULE {v.name}: '
                                    f't[v]={t[v]:.3f}, dt_dq[v]={dt_dq[v]:.3f}')

                    q_new = 0
                    if delta_q < (q[v] - q_min[v]):
                        q_new = q[v] - delta_q

                    delay_diff = (q[v] - q_new) * dt_dq[v]

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'\tmodule {v.name}: set q_v: {q[v]:d} -> {q_new:d}, '
                                    f'delay_budget: {delay_budget[f]:.3f} -> '
                                    f'{delay_budget[f] + delay_diff:.3f}')

                    for ff in batchy.flows_via_module(v):
                        delay_budget[ff] += delay_diff

                    if logging.getLogger().isEnabledFor(logging.INFO):
                        logging.log(logging.INFO,
                                    f'\tmodule {v.name}: RECOVERY for flow {f.name}: '
                                    f'setting: q[v]: {q[v]:d} -> {q_new:d}')
                    fxm[v] = True
                    q[v] = q_new
                    v.set_trigger(q_new)

                    if delay_budget[f] >= 0:
                        break

                if delay_budget[f] > 0 and logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'\t--> tflow {f.name}: RECOVERY ready: '
                                f'delay_budget: {delay_budget[f]:.3f}')
                    continue

        # PULL PHASE: pull free modules
        for v in cmodules:
            if fxm[v] or q[v] > 0 or R[v] <= 1e-9:
                continue
            # do not pull a module if it already receives large batches
            if b[v] >= settings.DEFAULT_PUSH_RATIO * settings.BATCH_SIZE:
                continue

            flows_via_module = batchy.flows_via_module(v)
            if any(fxf[f] for f in flows_via_module):
                continue
            max_delay = min(delay_budget[f] for f in flows_via_module)

            delay_diff = q_min[v] * dt_dq[v]
            if delay_diff < max_delay:
                for ff in flows_via_module:
                    delay_budget[ff] -= delay_diff
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'\tmodule {v.name}: PULLING: q_v: {q[v]:.3f}, '
                                f'delay_diff = {delay_diff:.3f} < '
                                f'max_delay={max_delay:.3f}')
                fxm[v] = True
                q[v] = q_min[v]
                v.set_trigger(q_min[v])
            elif logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG,
                            f'\tmodule {v.name} (R_v={R[v]:.9f}, '
                            f'dt_dq[v]={dt_dq[v]:.6f}): '
                            f'NOT PULLING to q_min: {q_min[v]:.3f}, '
                            f'delay_diff = {delay_diff:.3f} > '
                            f'max_delay={max_delay:.3f}')

        # PROJECTED GRADIENT OPTIMIZATION PHASE: optimize the modules
        # that are still free (fx=true)
        unfxm = [v for v in cmodules if not fxm[v] and q[v] > 0]
        if not unfxm or all(fxf[f] for f in self.flows):
            return

        # obtain tight bounds (within DEFAULT_TOLERANCE of bound)
        # delay
        A1 = []
        for f in self.flows:
            if fxf[f] or delay[f] < delay_bound[f] * (1 - settings.DEFAULT_TOLERANCE):
                continue

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG,
                            f'GRADIENT MODE: flow {f.name} is TIGHT: '
                            f'delay={delay[f]:.3f}, bound={delay_bound[f]:.3f}')

            tpath = next((e['tflow'].path for e in f.path if e['task'] == self.task), [])

            a1 = [0] * len(unfxm)
            for i, v in enumerate(unfxm):
                if v in tpath:
                    a1[i] = dt_dq[v]

            A1.append(a1)

        # trigger
        for i, u in enumerate(unfxm):
            # optimization: as long as all flows satisfy delay bounds we'll
            # never decrase q, so the below can be ignored
            if sum_error > 0:
                if q[u] <= q_min[u] * (1 + settings.DEFAULT_TOLERANCE):
                    a1 = [0] * len(unfxm)
                    a1[i] = -1
                    A1.append(a1)

            if q[u] == settings.BATCH_SIZE:
                a1 = [0] * len(unfxm)
                a1[i] = 1
                A1.append(a1)

            cdesc = u.get_controlled_descs(u)
            ## make setting parameter to disable this stability optimization to improve performance
            for j, v in enumerate(unfxm):
                if i == j or not v in cdesc:
                    continue
                # v is a DAG node with multiple inputs, ignore condition
                if R[v] > R[u] * (1 + settings.DEFAULT_TOLERANCE):
                    continue
                q_v_bound = settings.BATCH_SIZE + 1
                if R[v] > 0:
                    q_v_bound = q[u] >= (R[u]/R[v] * q[v]) * (1 - settings.DEFAULT_TOLERANCE)
                if q[u] >= q_v_bound:
                    ru_per_rv = R[u] / R[v]
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'GRADIENT MODE: module {u.name} is TIGHT '
                                    f'on module {v.name}: '
                                    f'q[u]={q[u]:d}, q[v]={q[v]:d}, '
                                    f'(R[u]/R[v]={ru_per_rv:.3f}')
                    a1 = [0] * len(unfxm)
                    a1[i] = 1
                    a1[j] = -ru_per_rv
                    A1.append(a1)

        # note: Rv is [packet/sec] but T_0 is [nsec]
        df = np.asarray([-(x[v] * T_0[v]) / q[v] for v in unfxm])
        df.shape = (len(unfxm), 1)
        A1 = np.asarray(A1)
        I = np.identity(len(unfxm))

        d = -df

        loopcounter = 0
        while A1.shape[0] > 0:
            loopcounter += 1
            if loopcounter > settings.MAX_PROJGRAD_ITERATIONS:
                break
            # we take the pseudo-inverse here, if MxM^T is invertible this
            # will give the normal inverse, otherwise we hope for the best
            mp_A1 = np.linalg.pinv(A1 @ A1.T)
            P = I - A1.T @ mp_A1 @ A1
            d = - P @ df

            if not np.all(d <= 1e-8):
                break

            w = (- mp_A1 @ A1) @ df

            if np.all(w >= 0):
                logging.log(logging.WARNING,
                            'GRADIENT PROJECTION: obtained a KKT point, doing nothing')
                return

            for i in range(w.shape[0]):
                if w[i] < 0:
                    A1 = np.delete(A1, (i), axis=0)
                    break

        nonzero = d[np.nonzero(d)]
        if nonzero.size == 0:
            return
        minval = np.min(np.abs(nonzero))
        delta_q = {v: d[i] / minval for i, v in enumerate(unfxm)}

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG,
                        f'GRADIENT MODE: projected gradient obtained:\n{d}\n'
                        f'normalized projected gradient:\n{delta_q}')

        # line-search
        lmbd = 1e6  # large number

        # delay
        for flow in self.flows:
            tpath = next((e['tflow'].path for e in flow.path if e['task'] == self.task), [])
            delta_d = sum([dt_dq[v] * delta_q[v] for v in unfxm if v in tpath])
            if delta_d > 0:
                lmbd = min(lmbd, delay_budget[flow] / delta_d)

        # trigger
        for u in unfxm:
            max_q = 0.0
            if delta_q[u] > 0.0:
                max_q = (settings.BATCH_SIZE - q[u]) / delta_q[u]
            elif delta_q[u] < 0.0:
                max_q = (q[u] - q_min[u]) / delta_q[u]

            if max_q > 0.0:
                lmbd = min(lmbd, max_q)

        for i, u in enumerate(unfxm):
            cdesc = u.get_controlled_descs(u)
            for j, v in enumerate(unfxm):
                if i == j or not v in cdesc:
                    continue
                if delta_q[u] == 0 and delta_q[v] == 0:
                    continue
                deltaq_per_r_diff = delta_q[u] / R[u] - delta_q[v] / R[v]
                if deltaq_per_r_diff == 0.0:
                    continue
                # v is a DAG node with multiple inputs, ignore condition
                if R[v] > R[u] * (1 + settings.DEFAULT_TOLERANCE):
                    continue
                max_q = (q[v]/R[v] - q[u]/R[u]) / deltaq_per_r_diff
                # TODO: avoid numerical instability here?
                if max_q < 0:
                    continue
                lmbd = min(lmbd, max_q)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG,
                        f'GRADIENT MODE: line search ended: lambda={lmbd}')

        for v in unfxm:
            # take care of rounding errors: if all flows traversing a
            # module allow then round fractional trigger up, otherwise
            # round down
            delta_qv = int(delta_q[v] * lmbd) + 1
            for f in self.flows:
                if fxf[f] or not f.traverses_module(v):
                    continue
                if delay[f] + dt_dq[v] * delta_qv > \
                   delay_bound[f] * (1 - settings.DEFAULT_TOLERANCE):
                    delta_qv -= 1
                    break

            # keep q_new between 0 and batch size
            q_new = max(0, min(q[v] + delta_qv + settings.EXTRA_TRIGGER,
                               settings.BATCH_SIZE))
            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.log(logging.INFO,
                            f'\tmodule {v.name}: GRADIENT PROJECTION: setting: '
                            f'q[v]: {q[v]:d} -> {q_new:d}')
            v.set_trigger(q_new)

    def get_violating_flows(self):
        ''' Collect flows' delay-SLO violation data from flow-data stored in
            dicts.

            Returns:
            delay (dict): last measured delay of flows
            delay_bound (float):  delay bounds of flows
            over (list): modules violating contraint
            error (float): sum of delay violations
            dist (float): smallest distance from the delay contraint over all flows

        '''
        delay_key = f'latency_{settings.DELAY_MAX_PERC}'
        delay = {f: f.stat[-1][delay_key] for f in self.flows}
        delay_bound = {f: (f.D if f.D is not None
                           else settings.DEFAULT_DELAY_BOUND)
                       for f in self.flows}
        error = 0.0
        over = []
        dist = sum(delay_bound.values())  # large number
        for flow in self.flows:
            error_f = delay[flow] - delay_bound[flow]
            if error_f > 0:
                error += error_f
                over.append(flow)
            else:
                dist = min(dist, -error_f)
        return delay, delay_bound, over, error, dist


class OnOffTaskController(TaskController):
    ''' Create a proper controller for the task
         RTC-controller for an RTC task
         WFQ-controller for an WFQ task
        then delegate calls to the new controller
    '''

    def __init__(self, task):
        super(OnOffTaskController, self).__init__(task)
        self.real_controller = get_tcontroller(task, 'OnOff')

    def __repr__(self, *args, **kwargs):
        return self.real_controller.__repr__(*args, **kwargs)

    def control(self, *args, **kwargs):
        return self.real_controller.control(*args, **kwargs)


class OnOffRTCTaskController(RTCTaskController):
    ''' Toggle module buffering: set FractionalBuffer size to 0 or 32 '''

    def __repr__(self):
        return 'OnOffRTCTaskController'

    def control(self, *args, **kwargs):
        super().control()
        batchy = self.task.batchy
        cmodules = self.task.cmodules

        delay, delay_bound, _, sum_error, delay_dist = self.get_violating_flows()
        delay_budget = {f:delay_bound[f] - delay[f] for f in self.flows}

        # count times in [nsec]
        T_0, T_1, x, b, q, t, R, dt_dq, fxm = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for v in cmodules:
            T_0[v], T_1[v] = v.get_sum_delay(v.get_uncontrolled_descs(v))
            x[v] = v.stat[-1]['x_v'] / 1e9
            q[v] = v.stat[-1]['q_v']
            b[v] = v.stat[-1]['b_in']
            R[v] = v.stat[-1]['R_v'] / 1e9
            t[v] = 0.0
            if x[v] > 0.0:
                t[v] = 1/x[v] + T_0[v] + T_1[v] * q[v]
            dt_dq[v] = 0.0
            if R[v] > 0.0:
                dt_dq[v] = 1/R[v] + T_1[v]
            fxm[v] = False

        by_delay = [(t[v], v) for v in cmodules]
        by_delay.sort(key=lambda x: x[0], reverse=True)
        fxf = {f: False for f in self.flows}

        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.log(logging.INFO,
                        f'CONTROL: {self.task.name}: '
                        f'error={sum_error:.3f}, dist={delay_dist:.3f}')

        # PUSH module to qv=0 if it already receives large batches
        for v in cmodules:
            if q[v] > 0 and b[v] >= settings.DEFAULT_PUSH_RATIO * settings.BATCH_SIZE:
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'\tmodule {v.name}: PUSH: input batch rate '
                                f'b[v]={b[v]:.3f} >= {settings.DEFAULT_PUSH_RATIO:.2f} '
                                f'* BATCH_SIZE: setting q[v]: {q[v]:d} -> 0')
                fxm[v] = True
                v.set_trigger(0)

        if sum_error > 0:
            # FEASIBILITY RECOVERY PHASE: push back all flows to below delay
            # bound, PUSH q_v to zero if needed
            for f in self.flows:
                if delay_budget[f] >= 0:
                    continue

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'RECOVERY MODE for flow {f.name}: '
                                f'delay={delay[f]:.3f}, bound={delay_bound[f]:.3f}')

                fxf[f] = True
                for _, v in by_delay:
                    if fxm[v] or q[v] == 0 or R[v] == 0 or not f.traverses_module(v):
                        continue

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'RECOVERY MODE for flow {f.name} at '
                                    f'MODULE {v.name}: '
                                    f't[v]={t[v]:.3f}, dt_dq[v]={dt_dq[v]:.3f}')

                    delay_diff = q[v] * dt_dq[v]

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'\tmodule {v.name}: set q_v: {q[v]:d} -> 0, '
                                    f'delay_budget: {delay_budget[f]:.3f} -> '
                                    f'{delay_diff:.3f}')

                    for ff in batchy.flows_via_module(v):
                        delay_budget[ff] += delay_diff

                    if logging.getLogger().isEnabledFor(logging.INFO):
                        logging.log(logging.INFO,
                                    f'\tmodule {v.name}: RECOVERY for flow {f.name}: '
                                    f'setting: q[v]: {q[v]:d} -> 0')
                    fxm[v] = True
                    q[v] = 0
                    v.set_trigger(0)

                if delay_budget[f] > 0 and logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'\t--> tflow {f.name}: RECOVERY ready: '
                                f'delay_budget: {delay_budget[f]:.3f}')
                    continue

        # PULL PHASE: pull free modules
        for v in cmodules:
            if fxm[v] or q[v] > 0 or R[v] <= 1e-9:
                continue
            # do not pull a module if it already receives large batches
            if b[v] >= settings.DEFAULT_PUSH_RATIO * settings.BATCH_SIZE:
                continue

            flows_via_module = batchy.flows_via_module(v)
            if any(fxf[f] for f in flows_via_module):
                continue
            max_delay = min(delay_budget[f] for f in flows_via_module)

            delay_diff = settings.BATCH_SIZE * dt_dq[v]
            if delay_diff < max_delay:
                for ff in flows_via_module:
                    delay_budget[ff] -= delay_diff
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'\tmodule {v.name}: PULLING: q_v: {q[v]:.3f}, '
                                f'delay_diff = {delay_diff:.3f} < '
                                f'max_delay={max_delay:.3f}')
                fxm[v] = True
                q[v] = settings.BATCH_SIZE
                v.set_trigger(settings.BATCH_SIZE)
            elif logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG,
                            f'\tmodule {v.name} (R_v={R[v]:.9f}, '
                            f'dt_dq[v]={dt_dq[v]:.6f}): '
                            f'NOT PULLING to q_min: {settings.BATCH_SIZE:.3f}, '
                            f'delay_diff = {delay_diff:.3f} > '
                            f'max_delay={max_delay:.3f}')

    def get_violating_flows(self):
        ''' Collect flows' delay-SLO violation data from flow-data stored in
            dicts.

            Returns:
            delay (dict): last measured delay of flows
            delay_bound (float):  delay bounds of flows
            over (list): modules violating contraint
            error (float): sum of delay violations
            dist (float): smallest distance from the delay contraint over all flows

        '''
        delay_key = f'latency_{settings.DELAY_MAX_PERC}'
        delay = {f: f.stat[-1][delay_key] for f in self.flows}
        delay_bound = {f: (f.D if f.D is not None
                           else settings.DEFAULT_DELAY_BOUND)
                       for f in self.flows}
        error = 0.0
        over = []
        dist = sum(delay_bound.values())  # large number
        for flow in self.flows:
            error_f = delay[flow] - delay_bound[flow]
            if error_f > 0:
                error += error_f
                over.append(flow)
            else:
                dist = min(dist, -error_f)
        return delay, delay_bound, over, error, dist


# ###################################################
# #
# # EXPERIMENTAL: A LOG-BARRIER CONTROLLER
# #
# ###################################################
# # Create a proper controller for the task
# #   RTC-controller for an RTC task
# #   WFQ-controller for an WFQ task
# # then delegate calls to the new controller
# class LogBarrierTaskController(TaskController):
#     def __init__(self, task):
#         super(LogBarrierTaskController, self).__init__(task)
#         self.real_controller = get_tcontroller(task, 'LogBarrier')

#     def __repr__(self, *args, **kwargs):
#         return self.real_controller.__repr__(*args, **kwargs)

#     def control(self, *args, **kwargs):
#         return self.real_controller.control(*args, **kwargs)


# class LogBarrierRTCTaskController(RTCTaskController):
#     def __init__(self, task):
#         super(LogBarrierRTCTaskController, self).__init__(task)
#         self.ak = []
#         self.k1 = self.k2 = self.k3 = self.k4 = 0
#         self.maxiter = 20

#     def __repr__(self):
#         return 'LogBarrierRTCTaskController'

#     def get_x0(self, unfxm):
#         x0 = np.asarray([v.stat[-1]['q_v'] for v in unfxm])
#         x0.shape = (1, len(unfxm))
#         return x0

#     def f(self, x):
#         f = 0
#         for i in range(x.shape[0]):
#             f += self.R[i] * self.T_0[i] / (x[i] * 1e6) + \
#                  self.R[i] * self.T_1[i] / 1e6
#         return f

#     def gradient(self, x):
#         return self.A.T @ self.dx

#     def hessian(self, x):
#         return self.A.T @ np.diag(self.dx) @ self.A

#     def control(self, *args, **kwargs):
#         super().control()
#         task = self.task
#         batchy = task.batchy
#         cmodules = task.cmodules

#         delay, delay_bound, over, sum_error, \
#             delay_dist = self.get_violating_flows()
#         delay_budget = {f:delay_bound[f] - delay[f] for f in self.flows}

#         T_0, T_1, load, x, q, t, R, q_min, dt_dq, fxm = \
#                         {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#         for v in cmodules:
#             T_0[v], T_1[v] = v.get_sum_delay(v.get_uncontrolled_descs(v))
#             x[v] = v.stat[-1]['x_v']
#             q[v] = v.stat[-1]['q_v']
#             R[v] = v.stat[-1]['R_v']
#             t[v] = T_0[v] + T_1[v] * q[v]
#             load[v] = x[v] * (T_0[v] + T_1[v] * q[v])
#             q_min[v] = int(v.stat[-1]['b_in']) + settings.DEFAULT_BUFFER_PULL
#             fxm[v] = False
#             dt_dq[v] = 1/R[v] + T_1[v]
#         by_delay = [(t[v], v) for v in cmodules]
#         by_delay.sort(key=lambda x: x[0], reverse=True)
#         fxf = {f: False for f in self.flows}

#         logging.log(logging.INFO,
#                   'CONTROL: %s: error=%.3lf, dist=%.3f' %
#                   (self.task.name, sum_error, delay_dist))

#         if sum_error > 0:
#             # FEASIBILITY RECOVERY PHASE: push back all flows to below delay
#             # bound, PUSH q_v to zero if needed
#             for f in self.flows:
#                 if delay_budget[f] > 0: continue
#                 fxf[f] = True

#                 logging.log(logging.DEBUG,
#                           'RECOVERY MODE for flow %f: delay=%.3lf, bound=%.3f' %
#                           (flow.name, delay[v], delay_bound[v]))
#                 for _, v in by_delay:
#                     if q_v[v] == 0 or not f.traverses_module(v) or R[v] == 0:
#                         continue

#                     delta_q = t[v] / dt_dq[v]
#                     if delta_q < q[v]:
#                         if delta_q < q[v] - q_min[v]:
#                             q_new = q[v] - delta_q
#                         else:
#                             q_new = 0

#                     logging.log(logging.DEBUG,
#                               '\tmodule %s: set q_v: %d -> %d, '
#                               'delay_budget: %.3f -> %.3f' %
#                               (v.name, q[v], q_new,
#                               delay_budget[f],
#                               delay_budget[f] + (q[v] - q_new) * dt_dq[v]))
#                     for ff in batchy.flows_via_module(v):
#                         delay_budget[ff] += (q[v] - q_new) * dt_dq[v]
#                     q[v] = q_new
#                     fxm[v] = True

#                 if delay_budget[f] > 0:
#                     logging.log(logging.DEBUG,
#                               '\t--> tflow %s: RECOVERY ready: '
#                               'delay_budget: %.3f' % (f.name, delay_budget[f]))
#                     continue

#         # PULL PHASE: pull free modules
#         for v in cmodules:
#             if fxm[v] or q[v] > 0 or x[v] == 0: continue

#             fv = batchy.flows_via_module(v)
#             if not all(fxf[f] == False for f in fv): continue
#             max_delay = min([delay_budget[f] for f in fv])

#             if q_min[v] * dt_dq[v] < max_delay:
#                 q[v] = q_min[v]
#                 for ff in fv:
#                     delay_budget[ff] -= q_min[v] * dt_dq[v]
#                 logging.log(logging.DEBUG,
#                           '\tmodule %s: PULLING: q_v: %.3f, '
#                           'delay_diff = %.3f < max_delay=%.3f' %
#                           (v.name, q[v], q_min[v] * dt_dq[v], max_delay))

#         # LOG-BARRIER OPTIMIZATION PHASE: optimize the modules that are
#         # still free (fx=true)
#         unfxm = [v for v in cmodules if not fxm[v] and q[v] > 0]
#         unfxf = [f for f in self.flows if not fxf[f]]

#         ak = []
#         dx = []

#         # delay
#         for f in unfxf:
#             tpath = None
#             for e in f.path:
#                 if e['task'] == task:
#                     tpath = e['tflow'].path
#                     break

#             a = [0] * len(unfxm)
#             for i, v in enumerate(unfxm):
#                 if v in tpath:
#                     a[i] = dt_dq[v]

#             ak.append(a)
#             dx.append(1 / delay_budget[f])
#         self.k1 = len(ak)

#         # trigger
#         for i, u in enumerate(unfxm):
#             # q_v >= q_min
#             a = [0] * len(unfxm)
#             a[i] = -1
#             ak.append(a)
#             dx.append(1 / (q[u] - q_min[u]) if q[u] > q_min[u] else 1.0)
#         self.k2 = len(ak) - self.k1

#         for i, u in enumerate(unfxm):
#             # q_v <= B
#             a = [0] * len(unfxm)
#             a[i] = 1
#             ak.append(a)
#             dx.append(1 / settings.BATCH_SIZE - q[u] if
#                       q[u] < settings.BATCH_SIZE else 1.0)
#         self.k3 = len(ak) - self.k2

#         for i, u in enumerate(unfxm):
#             cdesc = u.get_controlled_descs(u)
#             for j, v in enumerate(unfxm):
#                 if i == j or not v in cdesc: continue
#                 # q[u] >= (R[u]/R[v] * q[v])
#                 a = [0] * len(unfxm)
#                 a[i] = 1
#                 a[j] = -R[u]/R[v]
#                 ak.append(a)
#                 budget = -q[u]/R[u] + q[v]/R[v]
#                 dx.append(1 / budget if budget != 0.0 else 1.0)

#         self.k4 = len(ak) - self.k3
#         self.k = len(ak)
#         self.A = np.asarray(ak)
#         self.dx = np.asarray(dx)
#         self.dx.shape = (self.k, 1)
#         self.R = np.asarray([R[v] for v in unfxm])
#         self.T_0 = np.asarray([T_0[v] for v in unfxm])
#         self.T_1 = np.asarray([T_1[v] for v in unfxm])
#         self.unfxm = unfxm

#         # root = minimize(f, self.get_x0(unfxm),
#         #                 fprime=lambda x:self.gradient(x),
#         #                 fprime2=lambda x:self.hessian(x),
#         #                 maxiter=self.maxiter)
#         res = minimize(lambda x: self.f(x),
#                        self.get_x0(unfxm),
#                        method='Newton-CG',
#                        jac=lambda x:self.gradient(x),
#                        hess=lambda x:self.hessian(x),
#                        options={'xtol': 1e-8,
#                                 'maxiter': self.maxiter,
#                                 'disp': True})

#         pprint.pprint(res)

#         for i, v in enumerate(unfxm):
#             q_new = int(res.x[i])
#             logging.log(logging.DEBUG,
#                       '\tmodule %s: setting: q[v]: %d -> %d' % (
#                       v.name, q[v], q_new))
#             v.set_trigger(q_new)

#         return

#     # override: saner API
#     def get_violating_flows(self):
#         delay_key = f'latency_{settings.DELAY_MAX_PERC}'
#         delay = {f: f.stat[-1][delay_key] for f in self.flows}
#         delay_bound = {f: f.D for f in self.flows}

#         error = 0
#         over = []
#         dist = sum(delay_bound.values())  # large number
#         for f in self.flows:
#             error_f = delay[f] - delay_bound[f]
#             if error_f > 0:
#                 error += error_f
#                 over.append(f)
#             else:
#                 dist = min(dist, -error_f)
#         return delay, delay_bound, over, error, dist
