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
import time

from . import settings
from . import utils


class Module:
    """ Represents a Batchy module.

        Type can be:
         - 'bess' (native BESS module, do not prepend anything)
         - 'ingress' (prepend queue)
         - 'egress' (add fractional buffer if controlled)
         - 'internal'  (add fractional buffer if controlled)

    """

    def __init__(self, module, task, per_batch=0, per_packet=0, id=-1,
                 type='internal', controlled=None):
        self.task = task
        self.bess = task.bess
        self.id = id
        self.cid = -1   # id on the 'controlled modules' list
        self.module = module
        self.type = type
        self.buff = None
        self.ingress = module
        self.egress = module
        self.name = module.name
        self.q_v = 0
        self.T_0 = per_batch
        self.T_1 = per_packet
        self.parent = None
        self.cparent = None     # controlled parents
        self.children = []
        self.c_desc = []        # controlled descendants
        self.uncont_desc = []   # uncontrolled descendants
        self.stat = []
        if controlled is None:
            self.is_controlled = self.batchyness() <= settings.CONTROLLABLE_BOUND
        else:
            # convert controlled to bool
            self.is_controlled = controlled is True

        now = time.time()
        self.last_mod_info = {'timestamp': now, 'count': 0, 'pkts': 0}
        self.last_buf_info = {'timestamp': now, 'count': 0, 'pkts': 0}

    def __repr__(self):
        return f'Module: {self.name}:id={self.id:d}/cid={self.cid:d} ' \
               f'(T={self.T_0}/{self.T_1}, q_v={self.q_v:d}, ' \
               f'controlled={self.is_controlled})'

    def batchyness(self):
        """ Calculate module's batchyness metric """
        try:
            return self.T_1 / (self.T_0 + self.T_1)
        except ZeroDivisionError:
            return 1.0

    def reset(self, meas=None, bmeas=None):
        """ Reset module: update self.last_*_info """
        if meas is None:
            meas = self.bess.get_module_info(self.module.name).igates[0]
        if bmeas is None:
            bmeas = self.bess.get_module_info(self.ingress.name).igates[0]

        self.last_mod_info = {'timestamp': meas.timestamp,
                              'count': meas.cnt,
                              'pkts': meas.pkts}
        self.last_buf_info = {'timestamp': bmeas.timestamp,
                              'count': bmeas.cnt,
                              'pkts': bmeas.pkts}

    def erase_stat(self):
        """ Delete collected module statistics """
        self.stat = []

    def connect(self, nextm, ogate=0, igate=0):
        """ Connect module on output gate to next module's input gate

            Parameters:
            nextm (module): Next module
            ogate (int): Output gate id
            igate (int): Input gate id of next module

            Returns:
            nextm (module): Next module

        """
        # connect on BESS level
        self.bess.connect_modules(self.egress.name, nextm.ingress.name,
                                  ogate, igate)
        itask = self.task
        otask = nextm.task

        if itask.name == otask.name:
            # link internal to task
            nextm.parent = self
            self.children.append(nextm)
            # request tracking on the input gate, set tracking on the output
            self.set_tracking(nextm.ingress.name, self.egress.name, igate, ogate)
        else:
            # link between tasks, do nothing for now
            pass
        return nextm

    def disconnect_in(self):
        """ Disconnect module on input gate """
        self.bess.disconnect_modules(*self.get_parent_name_ogate())
        self.parent = None

    def disconnect_out(self, nextm, ogate=0):
        """ Disconnect module on given output gate """
        self.bess.disconnect_modules(self.egress.name, ogate)
        self.children.remove(nextm)

    def set_tracking(self, in_name, out_name, igate=0, ogate=0):
        """ Request tracking on the input gate """
        do_track = (settings.ENABLE_OGATE_TRACKING or
                    self.task.loglevel == 'DEBUG')
        self.bess.track_gate(True, '', in_name, False, 'in', igate)
        self.bess.track_gate(do_track, '', out_name, False, 'out', ogate)

    def get_parent_name_ogate(self):
        """Get parent BESS module's name and connecting ogate."""
        try:
            info = self.bess.get_module_info(self.ingress.name).igates[0].ogates[0]
            return info.name, info.ogate
        except IndexError:
            raise Exception(f'{self.name} has no parent')

    def get_spec_stat(self, stats):
        """ Add module specific field to statistics

            Parameters:
            stats (dict): Statistics

            Returns:
            stats (dict): Statistics

        """
        raise NotImplementedError

    def get_stat(self, prev_stats):
        """ Get statistics proxy function """
        method = settings.MODULE_GET_STAT_METHOD
        try:
            get_stat_func = getattr(self, f'_get_stat_{method}')
            get_stat_func(prev_stats)
        except AttributeError:
            pfix = '_get_stat_'
            opts = [x.replace(pfix, '') for x in dir(self) if pfix in x]
            text = f'module.get_stat: invalid method: "{method}", ' \
                   f'available options: {opts}'
            logging.log(logging.ERROR, text)
            raise Exception(text) from None

    def _get_stat_full(self, prev_stats):
        """ Get module statistics and buffer statistics

           NOTE: must be called in preorder

        """
        mod_info = self.bess.get_module_info(self.module.name).igates[0]
        diff_ts = mod_info.timestamp - self.last_mod_info['timestamp']

        stats = {}
        # module stat
        stats['batch'] = mod_info.cnt - self.last_mod_info['count']
        stats['pkts'] = mod_info.pkts - self.last_mod_info['pkts']

        try:
            stats['x_v'] = stats['batch'] / diff_ts
        except ZeroDivisionError:
            stats['x_v'] = 0.0

        try:
            stats['R_v'] = stats['pkts'] / diff_ts
        except ZeroDivisionError:
            stats['R_v'] = 0.0

        try:
            # AVERAGE BATCH SIZE!
            stats['b_v'] = stats['pkts'] / stats['batch']
        except ZeroDivisionError:
            stats['b_v'] = 0.0

        try:
            stats['r_v'] = stats['pkts'] / prev_stats['pkts']
        except ZeroDivisionError:
            stats['r_v'] = 0.0

        stats = self.get_spec_stat(stats)

        # buffer/queue stat
        buf_info = self.bess.get_module_info(self.ingress.name).igates[0]
        batch_in = buf_info.cnt - self.last_buf_info['count']
        pkts_in = buf_info.pkts - self.last_buf_info['pkts']
        try:
            stats['b_in'] = pkts_in / batch_in
        except ZeroDivisionError:
            stats['b_in'] = 0.0

        # estimated delay profile
        stats['T_0v'], stats['T_1v'] = self.T_0, self.T_1
        b_v = stats['b_in']
        if self.is_controlled:
            b_v = stats['b_v']
        stats['t_m_estimate'] = self.get_delay_estimate(b_v)

        self.stat.append(stats)
        self.reset(mod_info, buf_info)

    def _get_stat_partial(self, prev_stats):
        """ Get buffer statistics, calculate module statistics

           NOTE: must be called in preorder

        """
        buf_info = self.bess.get_module_info(self.ingress.name).igates[0]
        diff_ts = buf_info.timestamp - self.last_buf_info['timestamp']

        stats = {}
        # buffer/queue stat
        stats['batch'] = buf_info.cnt - self.last_buf_info['count']
        stats['pkts'] = buf_info.pkts - self.last_buf_info['pkts']
        try:
            stats['b_in'] = stats['pkts'] / stats['batch']
        except ZeroDivisionError:
            stats['b_in'] = 0.0

        # module stat
        try:
            stats['x_v'] = stats['batch'] / diff_ts
        except ZeroDivisionError:
            stats['x_v'] = 0.0

        try:
            stats['R_v'] = stats['pkts'] / diff_ts
        except ZeroDivisionError:
            stats['R_v'] = 0.0

        # ESTIMATED BATCH SIZE
        stats['b_v'] = self.q_v

        try:
            stats['r_v'] = stats['pkts'] / prev_stats['pkts']
        except ZeroDivisionError:
            stats['r_v'] = 0.0

        stats = self.get_spec_stat(stats)

        # estimated delay profile
        stats['T_0v'], stats['T_1v'] = self.T_0, self.T_1
        b_v = stats['b_in']
        if self.is_controlled:
            b_v = stats['b_v']
        stats['t_m_estimate'] = self.get_delay_estimate(b_v)

        self.stat.append(stats)
        self.reset(buf_info, buf_info)

    def format_stat(self):
        """Format module stats.

           Returns: a string representation of module including
           statistics.

        """
        stat = self.stat[-1]
        return f'Module {self.name}({stat["T_0v"]}/{stat["T_1v"]}): ' \
               f'x_v={stat["x_v"]:.3f}, b_v={stat["b_v"]:0.3f}, ' \
               f'pps={stat["R_v"]:0.3f}, b_in={stat["b_in"]:.3f}, ' \
               f'T_est={stat["t_m_estimate"]:.3f}'

    def get_controlled_descs(self, cur):
        """ Collect controlled descendant modules """
        for mod in cur.children:
            if mod.is_controlled:
                self.c_desc.append(mod)
            else:
                self.get_controlled_descs(mod)
        return self.c_desc

    def get_uncontrolled_descs(self, cur):
        """ Collect uncontrolled descendant modules """
        self.uncont_desc.append(cur)
        for mod in cur.children:
            if not mod.is_controlled:
                self.get_uncontrolled_descs(mod)
        return self.uncont_desc

    def get_cparent(self):
        """ Find first controlled parent module """
        cur = self.parent
        while cur is not None:
            if not cur.is_controlled:
                cur = cur.parent
            else:
                break
        self.cparent = cur

    def get_sum_delay(self, desc):
        """ Calculate delay parameters of desc modules

            Parameters:
            desc (list): list of modules

            Returns:
            T_0, T_1 (float): delay parameters

        """
        if not self.is_controlled:
            raise ValueError(f'{self.name} is not controlled')
        r_0 = self.stat[-1]['r_v']
        T_0 = 0.0
        T_1 = 0.0
        for mod in desc:
            T_0 += mod.T_0
            r_v = mod.stat[-1]['r_v']
            if r_0 > 0.0:
                T_1 += (r_v / r_0) * mod.T_1
        return T_0, T_1

    def get_delay_estimate(self, batch_size):
        """ Calculate module's estimated delay """
        return self.T_0 + batch_size * self.T_1


class RTCModule(Module):
    """ Module in a run-to-completion task """

    def __init__(self, module, task, per_batch=0, per_packet=0, id=-1,
                 type='internal', controlled=None):
        super().__init__(module, task, per_batch, per_packet,
                         id, type, controlled)
        self.buff = None
        self.q_v = 0

        if type in ('egress', 'internal', 'ingress'):
            # add a fractional buffer if controlled
            if self.is_controlled:
                frac_buf_params = {'size': self.q_v,
                                   'name': f'{module.name}_buffer'}
                self.buff = utils.create_bess_module(self.bess, 'FractionalBuffer',
                                                     frac_buf_params)
                self.ingress = self.buff
                self.egress = module
                self.bess.connect_modules(self.buff.name, module.name, 0, 0)

                # request tracking on the input gate, set tracking off on the
                # output (can only do this once we have a connection set up)
                self.set_tracking(module.name, self.buff.name)
        elif type in ('bess', 'native'):
            pass
        else:
            raise Exception(f'add_module: unknown type "{type}"')

    def __repr__(self):
        return f'Module: {self.name}:id={self.id:d}/cid={self.cid:d} ' \
               f'(T={self.T_0}/{self.T_1}, q_v={self.q_v:d}, ' \
               f'controlled={self.is_controlled})'

    def set_trigger(self, q_v):
        """ Set module buffer's trigger size """
        if not self.is_controlled:
            raise Exception('module: attempt to call set_trigger '
                            'on an uncontrolled module')
        if not 0 <= q_v <= settings.BATCH_SIZE:
            raise Exception('module: set_trigger: q_v out of bound')

        if self.q_v == q_v:
            return

        self.buff.set_size(size=q_v)
        self.q_v = q_v

    def get_spec_stat(self, stats):
        """ Add module buffer's trigger size to stats """
        stats['q_v'] = self.q_v
        return stats

    def is_buffered(self):
        """ Check whether the module is buffered or not.
            Returns True if the module is buffered.
        """
        return self.is_controlled and self.q_v > 0

    def get_buffered_descs(self):
        """ Collect buffered descendant modules """
        descs = []
        self._get_buffered_descs(self, descs)
        return descs

    def _get_buffered_descs(self, cur, descs):
        """ Helper function to collect buffered descendant modules """
        for mod in cur.children:
            if mod.is_buffered():
                descs.append(mod)
            else:
                self._get_buffered_descs(mod, descs)

    def get_unbuffered_descs(self):
        """ Collect unbuffered descendant modules """
        descs = []
        self._get_unbuffered_descs(self, descs)
        return descs

    def _get_unbuffered_descs(self, cur, descs):
        """ Helper function to collect unbuffered descendant modules """
        descs.append(cur)
        for mod in cur.children:
            if not mod.is_buffered():
                self._get_unbuffered_descs(mod, descs)


class WFQModule(Module):
    """ Module in a weighted-fair-queuing task """

    def __init__(self, module, task, per_batch=0, per_packet=0, id=-1,
                 type='internal', controlled=None):
        super().__init__(module, task, per_batch, per_packet,
                         id, type, controlled=controlled)
        self.queue = None
        self.w_v = 1

        if type in('egress', 'internal'):
            # add a queue and a tc if controlled
            if self.is_controlled:
                self.queue = utils.create_bess_module(self.bess, 'Queue',
                                                      {'size':
                                                       settings.DEFAULT_QUEUE_SIZE,
                                                       'name':
                                                       f'{module.name}_queue'})
                self.ingress = self.queue
                self.egress = module
                self.bess.connect_modules(self.queue.name, module.name, 0, 0)

                # attach our queue to task's tc using initial share
                self.queue.attach_task(parent=self.task.tc_name, share=self.w_v)

                # request tracking on the input gate, set tracking on the
                # output (can only do this once we have a connection set up)
                self.set_tracking(module.name, self.queue.name)

                # add this to the worker as a new task
                # avoid circular imports
                from . import task
                worker = self.task.worker
                new_task = task.WFQTask(self.task.batchy, worker,
                                        name=f'task_{self.module.name}',
                                        queue=self.queue)
                new_task.tc_name = self.task.tc_name
                new_task.ingress = self
                worker.tasks.append(new_task)

        elif type in ('ingress', 'bess', 'native'):
            pass

        else:
            raise Exception(f'add_module: unknown type "{type}"')

    def __repr__(self):
        return f'Module: {self.name}:id={self.id:d}/cid={self.cid:d} ' \
               f'(T={self.T_0}/{self.T_1}, q_v={self.w_v:d}, ' \
               f'controlled={self.is_controlled})'

    def set_weight(self, w_v):
        """ Set module's weight by adjusting the share of its Queue """
        if not self.is_controlled:
            raise Exception('module: '
                            'attempt to call set_weight '
                            'on an uncontrolled module')
        if w_v < 1:
            raise Exception('module: set_weight: w_v out of bound')

        if self.w_v != w_v:
            # update share
            self.queue.attach_task(parent=self.task.tc_name, share=w_v)
            self.w_v = w_v

    def get_spec_stat(self, stats):
        """ Add module weight to stats """
        stats['w_v'] = self.w_v
        return stats
