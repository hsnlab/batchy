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

import collections.abc
import pprint
import time

from . import settings
from . import utils


class Flow:
    """ Represents a Batchy flow. """

    def __init__(self, bess, path, name=None, delay_slo=None, rate_slo=None,
                 source_params=None, id=-1, leader_task=None):
        self.bess = bess
        self.path = path
        self.name = name or f'flow{id:d}'
        self.tpath = [e['task'] for e in path]
        self.leader_task = leader_task
        self.id = id
        self.ingress = path[0]['task']   # task!
        self.egress = path[-1]['task']   # task!
        self.ingress_module = self.ingress.ingress
        self.egress_module = path[-1]['path'][-1]
        self.egress_m = None             # measure
        self.stat = []
        self.src_task = []
        self.last_time = time.time()
        self.D = delay_slo or utils.default_delay_slo()
        self.R = self.set_rate_slo(rate_slo)

        self.source_params = {
            'burst_size': settings.BATCH_SIZE,
            'limit': rate_slo or utils.default_rate_slo(),
            'templates': None,
            'weight': 1,
            'burst': 1,  # works only if templates are set
            'ts_offset': settings.FLOW_TIMESTAMP_OFFSET,
            'ts_offset_diff': 0}
        # 'ts_offset_diff': diff between timestamp offset position and
        # measure offset position measured in bytes.

        if isinstance(source_params, collections.abc.Mapping):
            self.source_params.update(source_params)

        num_tflows = len(self.path)
        for path_segment in self.path:
            task = path_segment['task']
            new_tflow = task.add_tflow(self, path_segment['path'])
            new_tflow.task = task
            if path_segment.get('delay_bound', False):
                tf_delay_bound = path_segment['delay_bound']
            else:
                tf_delay_bound = self.D / num_tflows
            new_tflow.delay_bound = tf_delay_bound
            path_segment['tflow'] = new_tflow

    def __repr__(self):
        resource = 'packet'
        delay = self.get_delay() or 'NA'
        rate = 'NA'
        rate_slo = 'None'

        if self.R:
            (resource, limit), = self.R.items()
            rate_slo = f'{resource}->{limit}'

        if self.stat:
            rate_dim = 'pps' if resource == 'packet' else 'bps'
            rate = f'{self.stat[-1][rate_dim]:.1f} {rate_dim}'

        return (f'Flow {self.name}:\t'
                f'{self.ingress.name}/{self.ingress_module.name} -> '
                f'{self.egress.name}/{self.egress_module.name}:\t'
                f'delay-slo={self.D}/delay={delay}\t'
                f'rate-slo={rate_slo}/rate={rate}')

    def reset(self, meas=None):
        """ Reset timer and conditionally clear egress Measure module stats """
        if meas is None:
            # clear stats
            meas = self.egress_m.get_summary(
                latency_percentiles=[int(settings.DELAY_MAX_PERC)],
                clear=True)
        self.last_time = time.time()

    def erase_stat(self):
        """ Clear statistics """
        self.stat = []

    def set_limit(self, rate_limit):
        """ Set rate limit to a given value """
        self.source_params['limit'] = utils.check_ratelimit(rate_limit)
        return self.source_params['limit']

    def set_rate_slo(self, rate):
        """ Set rate SLO to the given value """
        self.R = utils.check_ratelimit(rate)
        return self.R

    # this will need its own task so that we can schedule each flow
    # individually
    def add_source(self, worker, ts_offset=None):
        """ Setup traffic generator """
        if ts_offset is None:
            ts_offset = settings.FLOW_TIMESTAMP_OFFSET
        name = f'src:{worker.name}:{self.name}'
        src = utils.create_bess_module(self.bess, 'Source',
                                       {'name': f'src_{name}'})
        src.set_burst(burst=self.source_params['burst_size'])

        # create a new QoS task for the source
        limit = self.source_params['limit']
        qos_hint = limit is not None
        task = worker.add_task(name, limit=limit,
                               weight=self.source_params['weight'],
                               queue=src, qos=qos_hint)
        timestamp = utils.create_bess_module(self.bess, 'Timestamp',
                                             {'name': f'ts_{name}',
                                              'offset':
                                              self.source_params['ts_offset']})
        self.src_task.append(task)

        if self.source_params['templates'] is not None:
            rewrite = utils.create_bess_module(self.bess, 'Rewrite',
                                               {'name': f'rw_{name}',
                                                'templates':
                                                self.source_params['templates']})
            task.add_module(rewrite, type='ingress')
            rewrite.connect(timestamp)
            timstamp_module = task.add_module(timestamp)
        else:
            timstamp_module = task.add_module(timestamp, type='ingress')
        task.egress.append(timstamp_module)

        # connect to the task ingress
        timstamp_module.connect(self.ingress.ingress)

    def add_sink(self):
        """ Adds the measurement module for the flow, goes INSIDE the task.

            ASSUMPTION: flow egresses are unique

        """
        offset = self.source_params['ts_offset'] + \
            self.source_params['ts_offset_diff']
        lat_max = settings.DEFAULT_MEASURE_LATENCY_NS_MAX
        lat_res = settings.DEFAULT_MEASURE_LATENCY_NS_RESOLUTION
        self.egress_m = utils.create_bess_module(self.bess, 'Measure',
                                                 {'name': f'meas_{self.name}',
                                                  'latency_ns_resolution': lat_res,
                                                  'latency_ns_max': lat_max,
                                                  'offset': offset})
        measure = self.egress.add_module(self.egress_m, type='bess')
        self.egress_module.connect(measure)
        self.egress_module = measure
        self.path[-1]['path'].append(measure)

        self.egress.add_sink()  # will add only one sink/task
        self.egress_module.connect(self.egress.sink)

    def traverses_worker(self, worker):
        """ Check if a worker is traversed by the flow. Returns True if so. """
        return any(map(self.traverses_task, worker.tasks))

    def traverses_task(self, task):
        """ Check if a task is traversed by the flow. Returns True if so. """
        return task in self.tpath

    def traverses_module(self, module):
        """ Check if a module is traversed by the flow. Returns True if so. """
        return any(e for e in self.path if e['tflow'].traverses_module(module))

    def has_tflow(self, tflow):
        """ Checks if a task flow is part of the flow. Returns True if so. """
        return any(tflow == segment['tflow'] for segment in self.path)

    def get_tflows(self):
        """ Returns list of task flows """
        return [segment['tflow'] for segment in self.path]

    def get_delay(self):
        """ Reads last delay measurement from statistics.
            If no measurement is available, return 0.

        """
        delay_key = f'latency_{settings.DELAY_MAX_PERC}'
        try:
            return self.stat[-1][delay_key]
        except (IndexError, KeyError):
            return 0.0

    def get_stat(self):
        """ Get flow statistics """
        meas = self.egress_m.get_summary(
            latency_percentiles=[int(settings.DELAY_MAX_PERC)], clear=True)
        delay_key = f'latency_{settings.DELAY_MAX_PERC}'
        diff_ts = meas.timestamp - self.last_time
        stat = {}
        stat['pkts'] = meas.packets
        stat['pps'] = meas.packets / diff_ts
        stat['bps'] = meas.bits / diff_ts
        stat[delay_key] = meas.latency.percentile_values_ns[0]

        t_f_estimate = 0  # sum of task flow estimates
        for path_segment in self.path:
            tf_estimate = 0  # task flow estimate
            task = path_segment['task']
            tflow = path_segment['tflow']
            if task.stat:
                t_f_estimate += task.stat[-1]['t_0_estimate']
                tf_estimate += task.stat[-1]['t_0_estimate']
            for module in tflow.path:
                if module.stat:
                    t_f_estimate += module.stat[-1]['t_m_estimate']
                    tf_estimate += module.stat[-1]['t_m_estimate']
                if module.is_controlled and module.q_v > 0:
                    # add queuing delay
                    x_v = float(module.stat[-1]['x_v'])
                    if x_v != 0.0:
                        t_f_estimate += 1e9 / x_v
                        tf_estimate += 1e9 / x_v
            stat['delay_bound'] = tflow.delay_bound
            stat['tf_estimate'] = tf_estimate * settings.TF_ESTIMATE_MULTIPLIER
            path_segment['tflow'].stat.append(stat.copy())

        stat['delay_bound'] = self.D
        stat['t_f_estimate'] = t_f_estimate * settings.TF_ESTIMATE_MULTIPLIER

        self.stat.append(stat)
        self.reset(meas)

    def format_stat(self):
        """ Format statistics """
        content = pprint.pformat(self.stat[-1], indent=4, width=1)
        return f'Stats of Flow {self.name}:\n{content}'

    def is_rate_limited(self):
        """ Returns true if flow has rate limit """
        if self.source_params['limit'] is None:
            return False
        if not self.stat:
            raise Exception(f'{self.name} has no stats available')
        stat = self.stat[-1]
        (resource, limit), = self.source_params['limit'].items()
        if resource == 'packet' and stat['pps'] >= settings.CBR_RATIO * limit:
            return True
        if resource == 'bit' and stat['bps'] >= settings.CBR_RATIO * limit:
            return True
        return False

    def has_rate_slo(self):
        """ Returns true if flow has rate SLO """
        try:
            (resource, limit), = self.R.items()
            if resource in ('packet', 'bit') and limit > 0:
                return True
        except (ValueError, AttributeError):
            pass
        return False

    def has_delay_slo(self):
        """ Returns true if flow has delay SLO """
        if self.D is None or self.D <= 0:
            return False
        return True

    def has_slo(self):
        """ Returns true if any delay or rate SLO is set for the flow """
        return self.has_rate_slo() or self.has_delay_slo()
