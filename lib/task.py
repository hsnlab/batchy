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

from collections import defaultdict
import pprint
import time
import logging

from . import tcontroller
from . import module
from . import tflow
from . import settings
from . import utils


class Task:
    """ Represents a Batchy task """

    def __init__(self, batchy, worker, name='', queue=None, weight=1,
                 queue_size=None, backpressure=None):
        self.batchy = batchy
        self.bess = self.batchy.bess
        self.worker = worker
        self.name = name
        self.type = None
        self.weight = weight
        self.ingress = None
        self.egress = []
        self.modules = []
        self.cmodules = []  # controlled modules
        self.tflows = []
        self.sink = None
        self.tc_name = None
        self.stat = []
        self.controller = None
        self.last_info = {'timestamp': time.time(),
                          'count': 0, 'pkts': 0, 'cycles': 0}
        self.loglevel = batchy.loglevel
        if backpressure is None:
            backpressure = settings.ENABLE_BACKPRESSURE
        self.backpressure = backpressure
        if queue_size is None:
            queue_size = settings.DEFAULT_QUEUE_SIZE
        if queue is None:
            queue_args = {'name': f'queue_{self.name}',
                          'size': queue_size,
                          'backpressure': self.backpressure}
            queue = utils.create_bess_module(self.bess, 'Queue', queue_args)
        self.queue = queue

    def __repr__(self):
        content_repr = pprint.pformat({'TC': self.tc_name,
                                       'Weight': self.weight,
                                       'Queue': self.queue.name,
                                       'Modules': self.modules,
                                       'Controller': self.controller},
                                      indent=12, width=1)
        return (f'Task {self.name}: mode={self.type}, '
                f'backpressure={self.backpressure}\n'
                f'{content_repr}')

    def create_module(self, mod, task, T_0=0, T_1=0, id=-1,
                      type='internal', controlled=None):
        """ Create a new task module. Implemented by derived classes """
        raise NotImplementedError

    def add_module(self, mod, T_0=0, T_1=0, type='internal', controlled=None,
                   prev_mod=None):
        """ Add a module to task """
        if isinstance(mod, module.Module):
            m = mod
        else:
            if not T_0 and not T_1:
                T_0, T_1 = utils.get_t0t1_values(mod.mclass)
            m = self.create_module(mod, self, T_0, T_1,
                                   id=len(self.modules),
                                   type=type, controlled=controlled)
        self.modules.append(m)

        if m.is_controlled:
            m.cid = len(self.cmodules)
            self.cmodules.append(m)

        if type == 'ingress':
            if self.ingress is not None:
                raise Exception(f'add_module: task "{self.name}" '
                                f'already has an ingress: "{self.ingress.name}"')
            self.ingress = m
            ingress = m.ingress  # possibly a buffer
            m.ingress = self.queue
            m.egress = mod
            self.bess.connect_modules(self.queue.name, ingress.name, 0, 0)

            self.bess.track_gate(True, '', mod.name, False, 'in', 0)
            self.bess.track_gate(settings.ENABLE_OGATE_TRACKING or
                                 self.loglevel == 'DEBUG', '',
                                 self.queue.name, False, 'out', 0)

        elif prev_mod is not None:
            prev_mod.connect(mod)

        if type == 'egress':
            self.egress.append(m)

        m.task = self
        m.type = type
        return m

    def remove_module(self, mod, disconnect=True):
        """ Remove a task module.

            Arguments:
            disconnect (bool): If true, disconnect module input
        """
        try:
            self.modules.remove(mod)
        except ValueError as val_err:
            txt = f'Task "{self.name}" has no module "{mod.name}"'
            raise Exception(txt) from val_err
        if mod.is_controlled:
            self.cmodules.remove(mod)

        if mod.type == 'ingress':
            # NB: this might lead to packet loss if the Queue is not empty
            mod.ingress = mod.buff
            try:
                prev_mod_info = self.bess.get_module_info(self.queue.name)
                prev_mod = prev_mod_info.igates[0].ogates[0]
                self.bess.disconnect_modules(prev_mod.name, prev_mod.ogate)
            except IndexError:
                pass  # task Queue has no upstream module
        elif mod.type == 'egress':
            self.egress.remove(mod)

        if disconnect:
            mod.disconnect_in()

    def get_module(self, module_id):
        """ Get task module by its id """
        return next((m for m in self.modules if m.id == module_id), None)

    def get_cmodule(self, module_id):
        """ Get controlled task module by its id """
        return next((m for m in self.cmodules if m.id == module_id), None)

    def add_sink(self):
        """ Create BESS Sink for the task """
        if self.sink is None:
            sink = utils.create_bess_module(self.bess, 'Sink')
            self.sink = self.add_module(sink, type='bess')

    def add_tflow(self, parent, path):
        """ Add and assign taskflow to the task """
        new_tflow = tflow.TaskFlow(path, parent.D, id=len(self.tflows))
        self.tflows.append(new_tflow)
        return new_tflow

    def is_leader(self):
        """ Check if task is a leader for any flow """
        return any(flow.leader_task == self for flow in self.get_flows())

    def is_rate_limited(self):
        """ Check whether all flows traversing task are rate limited

        Returns:
        True if all traversing flows are rate-limited

        """
        if any(f.traverses_task(self) and not f.is_rate_limited()
               for f in self.batchy.flows):
            return False
        return True

    def get_flows(self):
        """ Collect flows traversing the task """
        return [f for f in self.batchy.flows if f.traverses_task(self)]

    def get_slo_flows(self):
        """ Collect SLO flows traversing the task """
        return [f for f in self.batchy.flows
                if f.traverses_task(self) and f.has_slo()]

    def has_slo(self):
        """ Check if any flow traversing the task has an SLO """
        return any(self.get_slo_flows())

    def get_ratelimit(self):
        slo_flows = self.get_slo_flows()
        all_flows = [f for f in self.batchy.flows if
                     f.traverses_task(self)]
        if slo_flows and len(slo_flows) == len(all_flows):
            # all flows via the task as rate-limited
            return utils.get_ratelimit_for_flows(slo_flows)
        # fall back to default limit if there is a flow w/ delay_slo and
        # w/0 rate_slo
        return utils.default_rate_slo()

    def set_controller(self, cclass, *args, **kwargs):
        """ Set task controller """
        logging.log(logging.DEBUG,
                    'set_controller: setting controller for task "%s" to "%s"',
                    self.name, cclass)
        ctrlr = tcontroller.get_tcontroller(self, cclass)
        return ctrlr

    def reset(self, meas=None):
        """ Reset task statistics """
        if meas is None:
            for mod in self.modules:
                mod.reset()
            meas = self.bess.get_tc_stats(self.tc_name)
        self.last_info = {'timestamp': meas.timestamp,
                          'count': meas.nonidle_count,
                          'pkts': meas.packets,
                          'cycles': meas.nonidle_cycles}

    def erase_stat(self):
        """ Clear task and module statistics """
        for mod in self.modules:
            mod.erase_stat()
        self.stat = []

    def get_stat(self):
        """Collect task statistics from the corresponding BESS traffic class
           and task modules.

           Appends the collected statistics (dict) to self.stat.

        """
        m = self.bess.get_tc_stats(self.tc_name)
        diff_ts = m.timestamp - self.last_info['timestamp']
        s = defaultdict(lambda: 0.0)

        s['batch'] = m.nonidle_count - self.last_info['count']
        s['pkts'] = m.packets - self.last_info['pkts']
        s['cycles'] = m.nonidle_cycles - self.last_info['cycles']

        s['x_0'] = s['batch'] / diff_ts
        if s['batch'] > 0:
            s['t_0'] = 1e9 * diff_ts / s['batch']

        s['pps'] = s['pkts'] / diff_ts
        # AVERAGE BATCH SIZE!
        if s['batch'] > 0:
            s['b_0'] = s['pkts'] / s['batch']

        if s['pkts'] > 0:
            s['cyc_per_packet'] = s['cycles'] / s['pkts']
        if s['batch'] > 0:
            s['cyc_per_batch'] = s['cycles'] / s['batch']
        s['cyc_per_time'] = s['cycles'] / diff_ts

        # then the module stat
        self.get_module_stat(s)

        # use average stats for estimation
        s['t_0_estimate'] = m.nonidle_count / m.nonidle_cycles

        # NB: The task statistics (task.cnt) contain the times when the
        # task was scheduled but no packets were found on the input so the
        # task did not indeed run. This causes the task statistics to be
        # somewhat wrong, especially with small queue size: e.g., if
        # ingress_queue_size=32 and source.cnt=task.cnt, then the task may
        # find packets only for every second schedule rounds on its input,
        # which causes b_0 to be only half of what's the reality. Of
        # course, we may let larger queue sizes (e.g.,
        # ingress_queue_size=64), this will also improve speed, but the
        # cost will be twice the delay. Below, we try to correct for the
        # task statistics, especially for b_0: we take the cnt and the b_0
        # parameters from the ingress module, which will be the _real_
        # number of times/batch size for the cases when the task was indeed
        # executed.
        ms = self.ingress.stat[-1]
        s['b_in'] = ms['b_v']
        s['x_in'] = ms['x_v']
        if ms['batch'] > 0:
            s['t_in'] = 1e9 * diff_ts / ms['batch']
        s['pps_in'] = ms['R_v']
        if s['b_in'] > 0.0:
            s['cyc_per_b_in'] = s['cyc_per_time'] / s['b_in']

        self.stat.append(s)
        self.reset(m)

    def format_stat(self):
        """ Format statistics riport """
        last_stat = pprint.pformat(dict(self.stat[-1]), indent=4, width=1)
        return f'Task {self.name}:\n{last_stat}'

    def get_module_stat(self, prev_stats):
        for mod in self.modules:
            mod.get_stat(prev_stats)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG, mod.format_stat())


class RTCTask(Task):
    """ Task (self-)scheduled in run-to-completion mode """

    def __init__(self, batchy, worker, name='', queue=None, weight=1,
                 queue_size=None, backpressure=None):
        super().__init__(batchy, worker, name, queue, weight,
                         queue_size, backpressure)
        self.type = 'RTC'

    def create_module(self, mod, task, T_0=0, T_1=0, id=-1,
                      type='internal', controlled=None):
        """ Create an RTC module """
        return module.RTCModule(mod, task, T_0, T_1, id,
                                type=type, controlled=controlled)


class WFQTask(Task):
    """ Task scheduled in weighted-fair-queuing mode """

    def __init__(self, batchy, worker, name='', queue=None, weight=1,
                 queue_size=None, backpressure=None):
        super().__init__(batchy, worker, name, queue, weight,
                         queue_size, backpressure)
        self.type = 'WFQ'

    def create_module(self, mod, task, T_0=0, T_1=0, id=-1,
                      type='internal', controlled=None):
        """ Create a WFQ module """
        return module.WFQModule(mod, task, T_0, T_1, id,
                                type=type, controlled=controlled)
