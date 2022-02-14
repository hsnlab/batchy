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

import numpy as np

from . import controller
from . import settings
from .tcontroller import \
    TaskController, RTCTaskController, get_tcontroller

np.set_printoptions(precision=5)


class ProjGradientTaskController(TaskController):
    ''' Create a proper controller for the task
         RTC-controller for an RTC task
         WFQ-controller for an WFQ task
        then delegate calls to the new controller
    '''

    def __init__(self, task):
        super().__init__(task)
        self.real_controller = get_tcontroller(
            task, 'ProjGradient')

    def __repr__(self, *args, **kwargs):
        return self.real_controller.__repr__(*args, **kwargs)

    def control(self, *args, **kwargs):
        return self.real_controller.control(*args, **kwargs)


class ProjGradientRTCTaskController(RTCTaskController):
    ''' RTC task controller based on the method of projected gradient '''

    def __init__(self, task):
        super().__init__(task)
        self.sub_grad = None
        self.set_subgrad(0)

    def __repr__(self):
        return 'ProjGradientRTCTaskController'

    def set_subgrad(self, value):
        """ Set subgradients to a given value """
        if hasattr(value, "__iter__"):
            self.sub_grad = value
        else:
            self.sub_grad = [value]
        logging.log(logging.DEBUG,
                    "Updating subgradients to %s", self.sub_grad)

    def _works_under_decomp_controller(self):
        return isinstance(self.task.batchy.controller,
                          controller.DecompMainController)

    def control(self, *args, **kwargs):
        """ Revised control function to work for primal decomposition"""
        super().control()

        cmodules = self.task.cmodules

        delay, delay_bound, _, sum_error, delay_dist = self.get_violating_flows()
        # use delay estimate instead of measured delay
        delay = [tf.stat[-1]['tf_estimate'] for tf in self.tflows]
        delay_budget = {tf.id: delay_bound[tf.id] - delay[tf.id]
                        for tf in self.tflows}

        # reset subgrads
        self.set_subgrad([0] * (len(cmodules) + len(delay)))

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
            # fxm: True if module was optimized
            fxm[v] = False

        modules_by_delay = [(t[v], v) for v in cmodules]
        modules_by_delay.sort(key=lambda x: x[0], reverse=True)
        # fxf: True if taskflow was optimized
        fxf = {tf: False for tf in self.tflows}

        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.log(logging.INFO,
                        f'CONTROL: {self.task.name}: '
                        f'error={sum_error:.3f}, dist={delay_dist:.3f}')

        ##########################################################

        # # PUSH module to qv=0 if it already receives large batches
        for v in cmodules:
            if q[v] > 0 and b[v] >= settings.DEFAULT_PUSH_RATIO * settings.BATCH_SIZE:
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'\tmodule {v.name}: PUSH: input batch rate '
                                f'b[v]={b[v]:.3f} >= {settings.DEFAULT_PUSH_RATIO:.2f} '
                                f'* BATCH_SIZE: setting q[v]: {q[v]:d} -> 0')
                fxm[v] = True
                v.set_trigger(0)

        ##########################################################

        # FEASIBILITY RECOVERY PHASE: push back all flows to below delay
        # bound, PUSH q_v to zero if needed
        if sum_error > 0:
            for tf in self.tflows:
                if delay_budget[tf.id] >= 0:
                    continue

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'RECOVERY MODE for taskflow {tf.get_name()}: '
                                f'delay={delay[tf.id]:.3f}, bound={delay_bound[tf.id]:.3f}')

                fxf[tf] = True

                # go over taskflow modules and minimize queueing
                # delays while taskflow delay is or over te SLO
                for _, v in modules_by_delay:
                    if fxm[v] or q[v] == 0 or R[v] == 0 or not v in tf.path:
                        continue

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'RECOVERY MODE for task flow {tf.get_name()} at '
                                    f'MODULE {v.name}: '
                                    f't[v]={t[v]:.3f}, dt_dq[v]={dt_dq[v]:.3f}')

                    q_delta = int(abs(delay_budget[tf.id]) / dt_dq[v]) + 1

                    q_new = 0
                    if q_delta < (q[v] - q_min[v]):
                        q_new = q[v] - q_delta

                    delay_diff = (q[v] - q_new) * dt_dq[v]

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'\tmodule {v.name}: set q_v: {q[v]:d} -> {q_new:d}, '
                                    f'delay_budget: {delay_budget[tf.id]:.3f} -> '
                                    f'{delay_budget[tf.id] + delay_diff:.3f}')

                    for tff in self.tflows:
                        if v in tff.path:
                            delay_budget[tff.id] += delay_diff

                    if logging.getLogger().isEnabledFor(logging.INFO):
                        logging.log(logging.INFO,
                                    f'\tmodule {v.name}: RECOVERY for taskflow {tf.get_name()}: '
                                    f'setting: q[v]: {q[v]:d} -> {q_new:d}')
                    fxm[v] = True
                    q[v] = q_new
                    v.set_trigger(q_new)

                    if delay_budget[tf.id] >= 0:
                        break

                if delay_budget[tf.id] > 0 and logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'\t--> tflow {tf.get_name()}: RECOVERY ready: '
                                f'delay_budget: {delay_budget[tf.id]:.3f}')

        ##########################################################

        # PULL PHASE: pull free modules (increase their trigger sizes)
        for v in cmodules:
            if fxm[v] or q[v] > 0 or R[v] <= 1e-9:
                continue
            # do not pull a module if it already receives large batches
            if b[v] >= settings.DEFAULT_PUSH_RATIO * settings.BATCH_SIZE:
                continue

            tflows_via_module = [tf for tf in self.tflows if v in tf.path]
            if any(fxf[tf] for tf in tflows_via_module):
                continue
            max_delay = min(delay_budget[tf.id] for tf in tflows_via_module)

            delay_diff = q_min[v] * dt_dq[v]
            if delay_diff < max_delay:
                # account taskflow delay changes
                for tff in tflows_via_module:
                    delay_budget[tff.id] -= delay_diff
                if logging.getLogger().isEnabledFor(logging.INFO):
                    logging.log(logging.INFO,
                                f'\tmodule {v.name}: '
                                f'PULLING: q_v: {q[v]} to {q_min[v]}, '
                                f'delay_diff = {delay_diff:.3f} < '
                                f'max_delay={max_delay:.3f}')
                # pull module to q_min
                fxm[v] = True
                q[v] = q_min[v]
                v.set_trigger(q_min[v])
            else:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'\tmodule {v.name} (R_v={R[v]:.9f}, '
                                f'dt_dq[v]={dt_dq[v]:.6f}): '
                                f'NOT PULLING to q_min: {q_min[v]:.3f}, '
                                f'delay_diff = {delay_diff:.3f} > '
                                f'max_delay={max_delay:.3f}')

        ##########################################################

        # PROJECTED GRADIENT OPTIMIZATION PHASE: optimize the modules
        # that are still free (fxm or fxf is False)
        unfxm = [v for v in cmodules if not fxm[v] and q[v] > 0]
        if not unfxm or all(fxf[tf] for tf in self.tflows):
            logging.log(logging.DEBUG,
                        'Return (%s): %d/%d modules are done; flows done? %s',
                        self.task.name,
                        len(cmodules)-len(unfxm),
                        len(cmodules),
                        all(fxf[tf] for tf in self.tflows))
            return

        # obtain tight bounds (within DEFAULT_TOLERANCE of bound)

        # delay
        A1 = []
        for tf in self.tflows:
            if fxf[tf]:
                continue

            delay_bound_tolerance = 1 - settings.DEFAULT_TOLERANCE
            delay_bound_lower = delay_bound[tf.id] * delay_bound_tolerance

            is_close_to_delay_bound = any((delay[tf.id] + dt_dq[v]) >= delay_bound_lower
                                          for v in tf.cpath)

            if not is_close_to_delay_bound and delay[tf.id] < delay_bound_lower:
                continue

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG,
                            f'GRADIENT MODE: task flow {tf.get_name()} is TIGHT: '
                            f'delay={delay[tf.id]:.3f}, bound={delay_bound[tf.id]:.3f}'
                            f'dt_dqs={[dt_dq[v] for v in tf.cpath]}')

            for i, v in enumerate(unfxm):
                # if queue is at settings.BATCH_SIZE, we add it later
                if v in tf.path and q[v] < settings.BATCH_SIZE:
                    a1 = [0] * len(unfxm)
                    a1[i] = dt_dq[v]
                    A1.append(a1)

        # trigger
        for i, u in enumerate(unfxm):
            # optimization: as long as all flows satisfy delay bounds we'll
            # never decrase q, so the below can be ignored
            if sum_error > 0 and not self._works_under_decomp_controller():
                if q[u] <= q_min[u] * (1 + settings.DEFAULT_TOLERANCE):
                    a1 = [0] * len(unfxm)
                    a1[i] = -1
                    A1.append(a1)

            if q[u] == settings.BATCH_SIZE:
                a1 = [0] * len(unfxm)
                a1[i] = 1
                A1.append(a1)

            cdesc = u.get_controlled_descs(u)
            # make setting parameter to disable this stability optimization to improve performance
            for j, v in enumerate(unfxm):
                if i == j or not v in cdesc:
                    continue
                # v is a DAG node with multiple inputs, ignore condition
                if R[v] > R[u] * (1 + settings.DEFAULT_TOLERANCE):
                    continue
                q_v_bound = settings.BATCH_SIZE + 1
                if R[v] > 0:
                    q_v_bound = q[u] >= (R[u]/R[v] * q[v]) * \
                        (1 - settings.DEFAULT_TOLERANCE)
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

        # self.is_leader is defined in the base class,
        # somehow pylint is not aware of that.
        # pylint: disable=E0203,W0201
        if (self.is_leader or self.task.is_leader()) \
           and A1.shape[0] > 0:
            self.is_leader = True

            # create additional matrixes
            Q = []
            _tflows = [tf
                       for _flow in self.task.get_flows()
                       for tf in _flow.get_tflows()
                       if tf.task != self.task]
            for _idx, _tflow in enumerate(_tflows):
                q_row = [0] * len(_tflows)
                q_row[_idx] = 1
                Q.append(q_row)
            Q = np.asarray(Q)

            B = []
            for i, _mod in enumerate(unfxm):
                b_row = [0] * len(_tflows)
                mod_flows = self.task.batchy.flows_via_module(_mod)
                for _flow in mod_flows:
                    for _tf in _flow.get_tflows():
                        try:
                            b_row[_tflows.index(_tf)] = 1
                        except ValueError:
                            pass
                B.append(b_row)
            B = np.asarray(B)

            Z = np.zeros((Q.shape[0], A1.shape[1]))

            # form matrix M
            M1 = np.concatenate((A1, B), axis=1)
            M2 = np.concatenate((Z, Q), axis=1)
            M = np.concatenate((M1, M2), axis=0)

            # add extra columns to the vector of derivatives
            # to match dimensions of M
            df = np.append(df, [0] * Q.shape[0])
            try:
                df_size = M.shape[1]
            except IndexError:
                df_size = 0
            df.shape = (df_size, 1)

        else:
            # the task is not a leader, matrix M is A1
            M = A1

        try:
            eye_size = M.shape[1]
        except IndexError:
            eye_size = 0
        I = np.identity(eye_size)
        d = -df

        # main loop of the gradient optimization algorithm
        loopcounter = 0
        while M.shape[0] > 0:
            loopcounter += 1
            if loopcounter > settings.MAX_PROJGRAD_ITERATIONS:
                break

            # we take the pseudo-inverse here, if MxM^T is invertible this
            # will give the normal inverse, otherwise we hope for the best
            mp_M = np.linalg.pinv(M @ M.T)

            P = I - M.T @ mp_M @ M

            d = - P @ df

            if not np.all(d[:len(unfxm)] <= 1e-8):
                break

            w = (- mp_M @ M) @ df

            u_grad = w[:len(unfxm)]
            # vv_grad = w[len(unfxm):]
            self.set_subgrad(w.flatten())

            if np.all(u_grad >= 0):
                logging.log(logging.WARNING,
                            'GRADIENT PROJECTION: '
                            'obtained a KKT point, doing nothing')
                return

            for i in range(u_grad.shape[0]):
                if u_grad[i] < 0:
                    M = np.delete(M, (i), axis=0)
                    break

        nonzero = d[np.nonzero(d)]
        if nonzero.size == 0:
            logging.log(logging.WARNING,
                        'GRADIENT PROJECTION: '
                        'all elements of d are zero.')
            return
        minval = np.min(np.abs(nonzero))
        delta_q = {v: d[i] / minval for i, v in enumerate(unfxm)}

        logging.log(logging.DEBUG,
                    'GRADIENT MODE: projected gradient obtained:\n%s\n'
                    'normalized projected gradient:\n%s',
                    d, delta_q[v])

        # line-search
        lmbd = 1e6  # large number

        # delay
        for tflow in self.tflows:
            delta_d = sum([dt_dq[v] * delta_q[v]
                           for v in unfxm if v in tflow.path])
            if delta_d > 0:
                lmbd = min(lmbd, delay_budget[tflow.id]/delta_d)

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
                deltaq_per_r_diff = delta_q[u]/R[u] - delta_q[v]/R[v]
                if deltaq_per_r_diff == 0.0:
                    continue
                # v is a DAG node with multiple inputs, ignore condition
                if R[v] > R[u] * (1 + settings.DEFAULT_TOLERANCE):
                    continue
                max_q = (q[v]/R[v] - q[u]/R[u]) / deltaq_per_r_diff
                # NOTE: expect numerical instability here?
                if max_q < 0:
                    continue
                lmbd = min(lmbd, max_q)

        logging.log(logging.DEBUG,
                    'GRADIENT MODE: line search ended: lambda=%s', lmbd)

        for v in unfxm:
            # take care of rounding errors: if all flows traversing a
            # module allow then round fractional trigger up, otherwise
            # round down
            delta_qv = int(delta_q[v] * lmbd) + 1
            for tf in self.tflows:
                if fxf[tf] or not v in tf.path:
                    continue
                _delay_bound = delay_bound[tf.id] * \
                    (1 - settings.DEFAULT_TOLERANCE)
                _new_delay = delay[tf.id] + dt_dq[v] * delta_qv
                if _new_delay > _delay_bound:
                    delta_qv -= 1
                    break

            # keep q_new between 0 and batch size
            _q_inc = q[v] + delta_qv + settings.EXTRA_TRIGGER
            q_new = max(0, min(_q_inc, settings.BATCH_SIZE))
            logging.log(logging.INFO,
                        '\tmodule %s: GRADIENT PROJECTION: setting: '
                        'q[v]: %d -> %d',
                        v.name, q[v], q_new)
            # apply new trigger
            v.set_trigger(q_new)
