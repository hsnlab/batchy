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

from . import settings
from .tcontroller import \
    TaskController, RTCTaskController, get_tcontroller

np.set_printoptions(precision=5)


class OnOffTaskController(TaskController):
    ''' Create a proper controller for the task
         RTC-controller for an RTC task
         WFQ-controller for an WFQ task
        then delegate calls to the new controller
    '''

    def __init__(self, task):
        super().__init__(task)
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
        cmodules = self.task.cmodules

        delay, delay_bound, _, sum_error, delay_dist = self.get_violating_flows()
        delay_budget = {
            tf.id: delay_bound[tf.id] - delay[tf.id] for tf in self.tflows
        }

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
        fxf = {tf: False for tf in self.tflows}

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
            for tf in self.tflows:
                if delay_budget[tf.id] >= 0:
                    continue

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'RECOVERY MODE for task flow {tf.get_name()}: '
                                f'delay={delay[tf.id]:.3f}, bound={delay_bound[tf.id]:.3f}')

                fxf[tf] = True
                for _, v in by_delay:
                    if fxm[v] or q[v] == 0 or R[v] == 0 or not v in tf.path:
                        continue

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'RECOVERY MODE for task flow {tf.get_name()} at '
                                    f'MODULE {v.name}: '
                                    f't[v]={t[v]:.3f}, dt_dq[v]={dt_dq[v]:.3f}')

                    delay_diff = q[v] * dt_dq[v]

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.log(logging.DEBUG,
                                    f'\tmodule {v.name}: set q_v: {q[v]:d} -> 0, '
                                    f'delay_budget: {delay_budget[tf.id]:.3f} -> '
                                    f'{delay_diff:.3f}')

                    for tff in self.tflows:
                        if v in tff.path:
                            delay_budget[tff.id] += delay_diff

                    if logging.getLogger().isEnabledFor(logging.INFO):
                        logging.log(logging.INFO,
                                    f'\tmodule {v.name}: RECOVERY for task flow {tf.get_name()}: '
                                    f'setting: q[v]: {q[v]:d} -> 0')
                    fxm[v] = True
                    q[v] = 0
                    v.set_trigger(0)

                if delay_budget[tf.id] > 0 and logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.log(logging.DEBUG,
                                f'\t--> tflow {tf.get_name()}: RECOVERY ready: '
                                f'delay_budget: {delay_budget[tf.id]:.3f}')
                    continue

        # PULL PHASE: pull free modules
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

            delay_diff = settings.BATCH_SIZE * dt_dq[v]
            if delay_diff < max_delay:
                for tff in tflows_via_module:
                    delay_budget[tff.id] -= delay_diff
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
