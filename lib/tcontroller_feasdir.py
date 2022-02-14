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
from .tcontroller import \
    TaskController, RTCTaskController, WFQTaskController, get_tcontroller


np.set_printoptions(precision=5)


class FeasDirTaskController(TaskController):
    ''' Create a proper controller for the task
        then delegate calls to the new controller:
         - RTC-controller for RTC task
         - WFQ-controller for WFQ task
    '''

    def __init__(self, task):
        super().__init__(task)
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
        super().__init__(task)
        self.gradient = []
        self.ptr = 0

    def __repr__(self):
        return f'FeasDirRTCTaskController: ptr={self.ptr}'

    def control(self, *args, **kwargs):
        super().control()
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
            delay_budget = min([delay_bound[tf.id] - delay[tf.id]
                                for tf in self.tflows if module in tf.path])
            delay_budget = min(delay_budget, settings.DEFAULT_DELAY_BOUND)

            q = min(max_q, min_q + settings.DEFAULT_BUFFER_PULL)
            if min_q and delay_budget > (q / min_q * t_v):
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
            for tf in self.tflows:
                if module in tf.path:
                    try:
                        val = (delay[tf.id] - delay_bound[tf.id]
                               ) / delay_bound[tf.id]
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
        sum_over_grad = sum([dDf_dq[tf.id] for tf in over])

        # no module violates delay constraints OR some flows violate the
        # delay constraints but this module would not affect those modules
        if sum_error == 0 or (sum_over_grad == 0 and df_dq < 0):
            # get the largest possible increase in q_v if possible
            delta_qv = min(max_q - q_v, settings.DELTA_TRIGGER_MAX)
            for tf in self.tflows:
                if dDf_dq[tf.id] > 0:
                    diff = (delay_bound[tf.id] - delay[tf.id]) / dDf_dq[tf.id]
                    delta_qv = min(delta_qv, int(diff) +
                                   settings.EXTRA_TRIGGER)

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
                for tf in self.tflows:
                    if dDf_dq[tf.id] > 0 and tf in over:
                        # flow overdelayed and delay will decrease
                        tmp = int(
                            (delay[tf.id] - delay_bound[tf.id]) / abs(dDf_dq[tf.id]))
                    elif dDf_dq[tf.id] < 0 and not tf in over:
                        # flow is not overdelayed and delay will increase
                        tmp = int(
                            (delay_bound[tf.id] - delay[tf.id]) / abs(dDf_dq[tf.id]))
                    else:
                        continue
                    tmp += settings.EXTRA_TRIGGER
                    delta_trigger_max = min(delta_trigger_max, tmp)
                delta_qv = -delta_trigger_max
            else:
                # increase trigger
                delta_trigger_max = min(
                    settings.DELTA_TRIGGER_MAX, max_q - q_v)
                for tf in self.tflows:
                    if dDf_dq[tf.id] < 0 and tf in over:
                        # flow overdelayed and delay will decrease
                        tmp = int(
                            (delay[tf.id] - delay_bound[tf.id]) / abs(dDf_dq[tf.id]))
                    elif dDf_dq[tf.id] > 0 and not tf in over:
                        # flow is not overdelayed and delay will increase
                        tmp = int(
                            (delay_bound[tf.id] - delay[tf.id]) / abs(dDf_dq[tf.id]))
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
        dDf_dq = [0] * len(self.tflows)
        for tf in self.tflows:
            try:
                flow = next(f for f in self.task.get_flows()
                            if f.has_tflow(tf))
                cbr = flow.is_rate_limited()
            except StopIteration:
                cbr = False
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                rl_text = 'not rate-limited'
                if cbr:
                    rl_text = 'RATE_LIMITED'
                flow = next((f for f in self.task.get_flows()
                             if f.has_tflow(tf)), None)
                logging.log(logging.DEBUG, f'dDf_dq: flow {flow} {rl_text}')

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

            dDf_dq[tf.id] = dDf_dqu

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
        for i, _ in enumerate(self.tflows):
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
        super().__init__(task)
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

        self.gradient.append(self.get_gradient())
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG, self.format_grad())

        df_dw = self.gradient[-1]['df_dw']
        dtf_dw = self.gradient[-1]['dtf_dw']

        sum_over_grad = sum([dtf_dw[tf.id][cid] for tf in over])

        delta_wv = 0
        # no module violates delay constraints OR some flows violate the
        # delay constraints but this module would not affect those modules:
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
        for tf in self.tflows:
            try:
                flow = next(f for f in self.task.get_flows()
                            if f.has_tflow(tf))
                cbr = flow.is_rate_limited()
            except StopIteration:
                cbr = False
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                rl_text = 'not rate-limited'
                if cbr:
                    rl_text = 'RATE_LIMITED'
                flow = next((f for f in self.task.get_flows()
                             if f.has_tflow(tf)), None)
                logging.log(logging.DEBUG, f'dtf_dw: flow {flow} {rl_text}')

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

            grad['dtf_dw'][tf.id].extend(dtf_dw)

        return grad

    def format_grad(self):
        grad_content = pprint.pformat(self.gradient[-1], indent=4, width=1)
        return f'Gradient:\n{grad_content}\n'
