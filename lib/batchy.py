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

import csv
import itertools
import logging
import os
import random
import subprocess
import time

from . import controller
from . import flow
from . import pipeline
from . import settings
from . import utils
from . import worker


class Batchy(metaclass=utils.Singleton):
    def __init__(self, bess=None):
        self.bess = bess
        self.round = 1
        self.stat = []  # a list of dicts
        self.flows = []
        self.workers = []
        self.controller = None
        self.proc = None  # tcpreplay process of self.add_pcap_source()

    def __repr__(self):
        workers = '\n'.join([f'{w}' for w in self.workers])
        flows = '\n'.join([f'{f}' for f in self.flows])
        return f'Batchy object dump:\n{workers}\n{flows}'

    def cleanup(self):
        if self.proc:
            pgrp = os.getpgid(self.proc.pid)
            cmd = f'sudo kill -s 2 -- -{pgrp}'
            try:
                subprocess.run(cmd.split(' '), check=True)
            except KeyboardInterrupt:
                pass

    def reset_all(self):
        self.bess.reset_all()
        self.flows.clear()
        self.workers.clear()
        self.__init__(self.bess)

    def add_worker(self, name=None, wid=None, core=None):
        if wid is None:
            wid = len(self.workers)
        if name is None:
            name = f"w{wid}"
        if self.get_worker(name) is not None:
            raise Exception(f'add_worker: worker "{name}" already exists')

        new_worker = worker.Worker(self, name, wid, core)
        self.workers.append(new_worker)

        return new_worker

    def get_worker(self, name):
        return next((w for w in self.workers if w.name == name), None)

    @staticmethod
    def add_task(name, worker, type=settings.DEFAULT_TASK_TYPE,
                 queue_size=None, backpressure=None):
        return worker.add_task(name, type, queue_size, backpressure)

    def get_task(self, name):
        for w in self.workers:
            for t in w.tasks:
                if t.name == name:
                    return t
        return None

    @staticmethod
    def resolve_task_controller(name):
        name = name.lower().replace('taskcontroller', '')
        controllers = {'feasdir': 'FeasDir',
                       'projgrad': 'ProjGradient',
                       'projgradient': 'ProjGradient',
                       'onoff': 'OnOff',
                       'null': 'Null',
                       'max': 'Max'}
        for i in range(1, settings.BATCH_SIZE):
            controllers[f'fix{i}'] = f'Fix{i}'
        try:
            return controllers[name]
        except KeyError:
            raise Exception(f'Unknown task controller: "{name}"') from None

    @staticmethod
    def set_task_controller(task, *args, **kwargs):
        return task.set_controller(*args, **kwargs)

    @staticmethod
    def resolve_controller(name):
        name = name.lower().replace('controller', '')
        controllers = {'null': 'Null',
                       'greedy': 'Greedy',
                       'normgreedy': 'NormalizedGreedy',
                       'normalizedgreedy': 'NormalizedGreedy'}
        try:
            _controller = f'{controllers[name]}Controller'
            return _controller
        except KeyError:
            raise Exception(f'Unknown controller: "{name}"') from None

    def set_controller(self, cclass, *args, **kwargs):
        logging.log(logging.DEBUG,
                    f'set_controller: setting Batchy controller to "{cclass}"')
        ctrl_class = getattr(controller, cclass)
        ctrlr = ctrl_class(self, *args, **kwargs)
        self.controller = ctrlr
        return ctrlr

    def add_flow(self, path, name=None, delay_slo=None, rate_slo=None,
                 source_params=None, id=None, leader_task=None):
        """ Add a new flow """
        if id is None:
            id = len(self.flows)
        name = name or f'flow{id:d}'
        if self.get_flow(name) is not None:
            raise Exception(f'add_flow: flow "{name}" already exists')

        new_flow = flow.Flow(self.bess, path=path, name=name,
                             delay_slo=delay_slo, rate_slo=rate_slo,
                             source_params=source_params, id=id,
                             leader_task=leader_task)
        self.flows.append(new_flow)
        # if new_flow.has_slo():
        for path_segment in new_flow.path:
            task = path_segment['task']
            if task.type == 'RTC':
                task.worker.update_task_tc(task.name)
        return new_flow

    def get_flow(self, name):
        return next((f for f in self.flows if f.name == name), None)

    def flows_via_module(self, module):
        return [f for f in self.flows if f.traverses_module(module)]

    def create_bess_module(self, mclass, args=None):
        return utils.create_bess_module(self.bess, mclass, args or {})

    def create_bess_port(self, pclass, args=None):
        return utils.create_bess_port(self.bess, pclass, args or {})

    def create_pipeline(self, pl_name, *args, **kwargs):
        pl_class = getattr(pipeline, f'{pl_name}Pipeline')
        return pl_class(self, *args, **kwargs)

    def add_pcap_source(self, pcap, worker, task, mode='replicate', ts_offset=None):
        if ts_offset is None:
            ts_offset = settings.FLOW_TIMESTAMP_OFFSET
        try:
            add_src_func = getattr(self, f'_add_pcap_source_{mode}')
            add_src_func(pcap, worker, task, ts_offset)
        except AttributeError:
            pfix = '_add_pcap_source_'
            opts = [x.replace(pfix, '') for x in dir(self) if pfix in x]
            text = f'add_pcap_source: invalid mode: "{mode}", ' \
                   f'available options: {opts}'
            logging.log(logging.ERROR, text)
            raise Exception(text) from None

    def _add_pcap_source_looper(self, pcap, worker, task, ts_offset):
        name = 'pmd1'
        looper = self.create_bess_module('Looper')
        src = self._create_pcap_port(name, pcap)
        src.connect(looper)
        self._add_source(name, worker, [looper], task.ingress,
                         ts_offset, utils.default_rate_slo(), 1, False)

    def _add_pcap_source_replicate(self, pcap, worker, task, ts_offset):
        name = 'pmd1'
        # NB: BESS must be recompiled with large enough Queue size
        queue = self.create_bess_module('Queue', {'size': 2**26})
        repl = self.create_bess_module('Replicate', {'gates': [0, 1]})
        src = self._create_pcap_port(name, pcap)
        src.connect(queue)
        repl.connect(queue, ogate=1)
        self._add_source(name, worker, [queue, repl], task.ingress,
                         ts_offset, utils.default_rate_slo(), 1, False)

    def _add_pcap_source_tcpreplay(self, pcap, worker, task, ts_offset):
        name = 'pmd1'
        iface = 'batchy0'
        in_port_args = {'name': name, 'vdev': f'net_tap0,iface={iface}'}
        in_port = self.create_bess_port('PMDPort', in_port_args)
        src = self.create_bess_module('PortInc', {'port': in_port})
        self._add_source(name, worker, [src], task.ingress,
                         ts_offset, utils.default_rate_slo(), 1, False)

        cmd = f'sudo ip link set dev {iface} mtu 5000'.split(' ')
        subprocess.run(cmd, check=True)
        cmd = f'sudo tcpreplay '\
              f'-q --preload-pcap --topspeed -l 0 --pktlen -i {iface} {pcap}'
        self.proc = subprocess.Popen(cmd.split(' '))

    def _create_pcap_port(self, name, pcap):
        pcap_files = f'rx_pcap={pcap},tx_pcap=/dev/null'
        in_port_args = {'name': name, 'vdev': f'eth_pcap0,{pcap_files}'}
        try:
            in_port = self.create_bess_port('PMDPort', in_port_args)
        except:
            in_port_args['vdev'] = f'net_pcap0,{pcap_files}'
            in_port = self.create_bess_port('PMDPort', in_port_args)
        return self.create_bess_module('PortInc', {'port': in_port})

    def add_source(self, worker=None, ts_offset=None):
        """ Add a multi-source for QoS flows and another multi-source for bulk
            flows
        """
        if worker is None:
            worker = self.add_worker()
        qos_flows, bulk_flows = [], []
        for f in self.flows:
            if f.has_rate_slo():
                qos_flows.append(f)
            else:
                bulk_flows.append(f)
        self.add_bulk_source(bulk_flows, worker, ts_offset=ts_offset)
        self.add_qos_source(qos_flows, worker, ts_offset=ts_offset)

    def add_bulk_source(self, flows, worker=None, burst_size=None,
                        postfix=None, ts_offset=None):
        for f in flows:
            if f.has_rate_slo():
                logging.log(logging.WARNING,
                            f'Adding flow "{f.name}" '
                            f'to bulk multi-source will drop rate-SLO')
        return self.add_multi_source(flows, worker, burst_size, postfix,
                                     ts_offset, qos=None)

    def add_qos_source(self, flows, worker=None, burst_size=None,
                       postfix=None, ts_offset=None):
        if not flows:
            return None
        for f in flows:
            if not f.has_rate_slo():
                logging.log(logging.WARNING,
                            f'Cannot add non-rate-limited '
                            f'flow "{f.name}" as a QoS multi-source')
        return self.add_multi_source(flows, worker, burst_size, postfix,
                                     ts_offset, qos=True)

    def _add_infra_source(self, name, worker, ingress, burst_size,
                          ts_offset, templates, limit, weight, is_qos):
        q = 'qos' if is_qos else 'bulk'
        src = self.create_bess_module('Source',
                                      {'name': f'{q}_multi_src_{name}'})
        src.set_burst(burst=burst_size)
        mods = [src]
        if templates:
            rewrite = self.create_bess_module('Rewrite',
                                              {'name': f'bulk_rw_{name}',
                                               'templates': templates})
            mods.append(rewrite)

        self._add_source(name, worker, mods, ingress,
                         ts_offset, limit, weight, is_qos)

    def _add_source(self, name, worker, modules, ingress,
                    ts_offset, limit, weight, is_qos):
        # create modules
        ts = self.create_bess_module('Timestamp', {'name': f'ts_{name}',
                                                   'offset': ts_offset})
        # create a new task for the source
        task = worker.add_task(name, weight=weight, limit=limit,
                               queue=modules[0], qos=is_qos)
        modules.append(ts)
        mtype = 'ingress'
        for i, module in enumerate(modules[1:], start=1):
            last = task.add_module(module, type=mtype, prev_mod=modules[i-1])
            mtype = 'internal'

        task.egress.append(last)

        # connect to the task ingress
        last.connect(ingress)

    def add_multi_source(self, flows, worker=None, burst_size=None,
                         postfix=None, ts_offset=None, qos=False):
        """ Add multi source by the following invariants:

            INVARIANT 1: either all flows have templates or none of them, we do
                         not support mixed setups

            INVARIANT 2: all flows with a common multi-source share ingress

        """

        postfix = f'_{postfix}' if postfix else ''
        worker = worker or self.add_worker()
        burst_size = burst_size or settings.DEFAULT_BULK_SOURCE_BURST_SIZE
        if ts_offset is None:
            ts_offset = settings.FLOW_TIMESTAMP_OFFSET

        if not flows:
            text = 'add_multi_source: zero flows specified'
            logging.log(logging.ERROR, text)
            raise Exception(text)

        # check INVARIANT 1
        have_template = len([f for f in flows
                             if f.source_params['templates'] is not None])
        if not (have_template == 0 or have_template == len(flows)):
            text = 'add_multi_source: invariant failed, number of ' \
                   f'flows with template ({have_template}) != number of ' \
                   f'all flows ({len(flows)}), ' \
                   'nor number of bulk flows with template == 0'
            logging.log(logging.ERROR, text)
            raise Exception(text)

        # check INVARIANT 2
        ingress = flows[0].ingress.ingress
        if any(f.ingress.ingress != ingress for f in flows):
            text = 'add_multi_source: invariant failed, all flows with ' \
                   'common multi-source must share the ingress'
            logging.log(logging.ERROR, text)
            raise Exception(text)

        # bail out if we get weird weight in one of our flows
        max_weight = max([f.source_params['weight'] for f in flows])
        if max_weight > 10:
            text = f'add_multi_source: weight={max_weight} is too large for a flow'
            logging.log(logging.ERROR, text)
            raise Exception(text)

        src_type = 'qos' if qos else 'bulk'
        name_prefix = f'{src_type}_multi_source{postfix}_{worker.name}'
        if have_template:
            # add templates in proportion to weight and burstiness in
            # a random order
            templates = [
                [(f, t)
                 for t in f.source_params['templates'] * f.source_params['weight']
                 for _ in range(f.source_params['burst'])]
                for f in flows
            ]
            if max([f.source_params['burst'] for f in flows]) < 2:
                random.shuffle(templates)
            templates = list(itertools.chain.from_iterable(templates))
            # start creating sources for the bulk sources, at most 20 templates
            # at a time
            i, j = 0, 0
            while i < len(templates):
                tnum = min(len(templates) - i, 20)
                if qos:
                    burst_size = tnum + 1
                # templates
                fts = templates[i:(i + tnum)]
                limit = utils.default_rate_slo()
                if qos:
                    _flows = list(set(ft[0] for ft in fts))
                    limit = utils.get_ratelimit_for_flows(_flows)
                self._add_infra_source(f'{name_prefix}-{j}', worker, ingress,
                                       burst_size, ts_offset,
                                       templates=[ft[1] for ft in fts],
                                       limit=limit, weight=tnum, is_qos=qos)
                j += 1
                i += tnum
        else:
            limit = utils.default_rate_slo()
            if qos:
                limit = utils.get_ratelimit_for_flows(flows)
            self._add_infra_source(name_prefix, worker, ingress,
                                   burst_size, ts_offset,
                                   templates=None, limit=limit,
                                   weight=len(flows), is_qos=qos)

    def add_sink(self):
        """ Connects all flow egresses into a common sink through a measure """
        for f in self.flows:
            f.add_sink()

    def run_one_period(self, control_period=settings.DEFAULT_CONTROL_PERIOD):
        if settings.ENABLE_STOPTHEWORLD:
            self.bess.resume_all()
        self.reset()
        time.sleep(control_period)
        if settings.ENABLE_STOPTHEWORLD:
            self.bess.pause_all()

    def __run_workers(self, rounds, control_period):
        for _ in range(rounds):
            sum_rate = int(self.get_cumulative_flow_rate())
            logging.log(logging.INFO,
                        f'*** CONTROL ROUND: {self.round}..., '
                        f'cumulative flow rate: '
                        f'{utils.format_sum_rate(sum_rate)}')
            self.run_one_period(control_period)
            ts_start = time.time()
            # statistics
            for _worker in self.workers:
                for task in _worker.tasks:
                    if task.controller is not None:
                        self.get_stat_task(task)
            self.get_flow_stat()
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                for _flow in self.flows:
                    logging.log(logging.DEBUG, _flow)
            # add extra controller dead time
            if settings.EXTRA_CONTROLLER_DEADTIME > 0:
                time.sleep(settings.EXTRA_CONTROLLER_DEADTIME)
            # call task controllers in all workers
            for _worker in self.workers:
                for task in _worker.tasks:
                    if task.controller is not None:
                        task.controller.control()
            # call the master controller (if needed)
            if self.controller is not None and \
               self.round % settings.DEFAULT_TASK_CONTROL_ROUNDS == 0:
                self.controller.control()
            # collect stats
            ts_diff = time.time() - ts_start
            stats = {'controller_deadtime': ts_diff,
                     'sum_fpps': sum_rate}
            try:
                self.stat[self.round].update(stats)
            except IndexError:
                self.stat.append(stats)
            # increment round counter
            self.round += 1

    def run(self, rounds=3, control_period=None, warmup=True):
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG, self)
        if control_period is None:
            control_period = settings.DEFAULT_CONTROL_PERIOD

        self.bess.resume_all()

        if warmup:
            logging.log(logging.INFO, '*** WARMUP... ***')
            self.run_one_period(control_period=settings.DEFAULT_WARMUP_PERIOD)
            self.reset()

        if rounds > 0:
            self.__run_workers(rounds, control_period)

    def reset(self):
        for w in self.workers:
            w.reset()
        for f in self.flows:
            f.reset()

    def get_stat(self):
        for w in self.workers:
            w.get_stat()
        for f in self.flows:
            f.get_stat()

    @staticmethod
    def get_stat_task(t):
        """ Query task statistics """
        t.get_stat()
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.log(logging.DEBUG, t.format_stat())

    def get_flow_stat(self):
        """ Query flow statistics """
        for f in self.flows:
            f.get_stat()
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.log(logging.DEBUG, f.format_stat())

    def erase_stat(self):
        """ Clear statistics """
        self.round = 0
        self.stat = []
        for w in self.workers:
            w.erase_stat()
        for f in self.flows:
            f.erase_stat()
        self.reset()

    def get_cumulative_flow_rate(self, t=-1):
        """ Calculate cumulative flow rates """
        return sum([f.stat[t]['pps'] for f in self.flows if f.stat])

    @staticmethod
    def _import_matplotlib():
        """ Helper function to import matplotlib package """
        plt = None
        try:
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            import matplotlib.pyplot as plt
        except ImportError:
            txt = 'plot: matplotlib can not be imported, plotting skipped'
            logging.log(logging.ERROR, txt)
        return plt

    def collect_gradients(self, store=True):
        """ Collect gradients, and optionally store in self.stat.

            Paramaters:
            store (bool): store gradients in self.stat; default: True

            Returns:
            gradients (dict): list of gradients indexed by task name
            and flow name
        """
        # collect all gradients
        gradients = {}
        if self.controller \
           and self.controller.stat \
           and 'gradients' in self.controller.stat[-1]:
            _stats = self.controller.stat
            for f in self.flows:
                followers = [tf for tf in f.get_tflows()
                             if tf.task != f.leader_task]
                for tf in followers:
                    # rounds = [n*settings.DEFAULT_TASK_CONTROL_ROUNDS
                    #           for n in range(len(_stats)+1)]
                    values = [0.0] + [s['gradients'][f.name][0]
                                      for s in _stats]
                    gradients[f'gradient_{tf.task.name}_{f.name}'] = values
        if store:
            # register gradients to self.stat
            for ts, stat in enumerate(self.stat):
                idx = ts // settings.DEFAULT_TASK_CONTROL_ROUNDS
                _grads = {name: tf_grads[idx]
                          for name, tf_grads in gradients.items()}
                stat.update(_grads)
        return gradients

    def plot(self, filename, modules=None, flows=None):
        """ Plot statistics.

            Parameters:
            filename (str): Output PNG filename
            modules (list): List of modules to plot. If not set: plot all.
            flows (list): List of flows to plot. If not set: plot all.

        """
        plt = self._import_matplotlib()
        if plt is None:
            return
        plt.style.use('seaborn-colorblind')
        size = 12.5
        params = {'legend.fontsize': size,
                  'axes.labelsize': size,
                  'axes.titlesize': size,
                  'figure.titlesize': size*1.5,
                  'xtick.labelsize': size,
                  'ytick.labelsize': size,
                  'axes.titlepad': size*0.75}
        plt.rcParams.update(params)
        plt.switch_backend('Agg')
        logging.log(logging.INFO,
                    f'plot: plotting statistics to file: "{filename}"')

        modules = modules or [m for w in self.workers
                              for t in w.tasks for m in t.cmodules]
        flows = flows or self.flows
        plt.figure(1)
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True,
                                               figsize=(15, 14))
        t = None

        # modules: control
        for i, m in enumerate(modules, start=5):
            if not m.is_controlled or m.task.controller is None:
                continue
            t = t or range(len(m.stat))
            if len(m.stat) != len(t):
                raise Exception('plot: stats size mismatch:'
                                f'{len(t)} <> {len(m.stat)}')
            ax1.plot(t,
                     [m.task.controller.get_control_at(m, x)
                      for x in range(len(m.stat))],
                     marker='d',
                     label=f'{m.module.name}')

        ax1.set_title('Module Controls')
        ax1.set_ylabel('Control')

        # delay 1/2: tflows
        for i, f in enumerate(flows):
            for tf in f.get_tflows():
                color = next(ax2._get_lines.prop_cycler)['color']
                t = t or range(len(tf.stat))
                if len(tf.stat) != len(t):
                    raise Exception('plot: stats size mismatch:'
                                    f'{len(t)} <> {len(tf.stat)}')
                ax2.plot(t,
                         [s['delay_bound'] for s in tf.stat],
                         marker='*',
                         color=color,
                         label=f'{tf.get_name()} (bound)')
                ax2.plot(t,
                         [s['tf_estimate'] for s in tf.stat],
                         marker='H',
                         color=color,
                         label=f'{tf.get_name()} (estimate)')

        # delay 2/2: flows
        for i, f in enumerate(flows):
            color = next(ax2._get_lines.prop_cycler)['color']
            t = t or range(len(f.stat))
            # if len(f.stat) != len(t):
            #     raise Exception('plot: stats size mismatch:'
            #                     f'{len(t)} <> {len(f.stat)}')
            ax2.plot(t,
                     [s[f'latency_{settings.DELAY_MAX_PERC}'] for s in f.stat],
                     marker=i % 11,
                     color=color,
                     label=f'{f.name}')
            ax2.plot(t,
                     [s['t_f_estimate'] for s in f.stat],
                     marker='s',
                     color=color,
                     label=f'{f.name} (estimate)')

        ax2.set_title('Flows: Delay')
        ax2.set_ylabel('Delay [nsec]')

        # gradients
        if self.controller \
           and self.controller.stat \
           and 'gradients' in self.controller.stat[-1]:
            _stats = self.controller.stat
            for f in flows:
                color = next(ax2._get_lines.prop_cycler)['color']
                followers = [tf for tf in f.get_tflows()
                             if tf.task != f.leader_task]
                for i, tf in enumerate(followers):
                    ax3.plot([n*settings.DEFAULT_TASK_CONTROL_ROUNDS
                              for n in range(len(_stats)+1)],
                             [0.0] + [s['gradients'][f.name][0]
                                      for s in _stats],
                             marker='^',
                             color=color,
                             label=f'({tf.task.name}, {f.name})')
        ax3.set_title('(Task,Flow) Gradients')
        ax3.set_ylabel('')

        # rate
        for i, f in enumerate(flows):
            t = t or range(len(f.stat))
            if len(f.stat) != len(t):
                raise Exception('plot: stats size mismatch:'
                                f'{len(t)} <> {len(f.stat)}')
            ax4.plot(t, [s['pps'] for s in f.stat],
                     marker=i % 11, label=f'{f.name}')

        if t is not None:
            ax4.plot(t, [s['sum_fpps'] for s in self.stat],
                     marker='>', label='Total Rate')

        ax4.set_title('Flows: Packet Rate')
        ax4.set_ylabel('Rate [pps]')

        for a in (ax1, ax2, ax3, ax4):
            num_legend_cols = min(max(len(a.lines) // 8, 1), 3)
            a.legend(ncol=num_legend_cols,
                     loc="center left",
                     bbox_to_anchor=(1, .5))

        plt.subplots_adjust(left=.07, right=.57, top=.95, bottom=.05)
        plt.savefig(filename, dpi=150)

    def dump(self, filename, modules=None, flows=None, tasks=None):
        """ Dump statistics to tsv

            Parameters:
            filename (str): Output filename
            modules (list): List of modules to dump. If not set: dump all.
            flows (list): List of flows to dump. If not set: dump all.
            tasks (list): List of tasks to dump. If not set: dump all.

        """

        logging.log(logging.INFO,
                    f'dump: dumping statistics to CSV file: "{filename}"')

        modules = modules or [m for w in self.workers
                              for t in w.tasks for m in t.cmodules]
        flows = flows or self.flows
        tasks = tasks or [t for w in self.workers for t in w.tasks]

        self.collect_gradients(store=True)
        gradient_keys = [key for key in self.stat[0] if "gradient_" in key]

        with open(filename, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file,
                                    delimiter='\t',
                                    lineterminator='\n',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            for t in range(self.round - 1):
                row = {'t': t}

                for m in modules:
                    if not m.is_controlled or m.task.controller is None:
                        continue
                    val = m.task.controller.get_control_at(m, t)
                    row[f'{m.name}:control'] = val
                    for s in ('x_v', 'R_v', 'b_v', 'b_in'):
                        row[f'{m.name}:{s}'] = m.stat[t][s]

                delay_key = f'latency_{settings.DELAY_MAX_PERC}'
                for f in flows:
                    row[f'{f.name}:delay'] = f.stat[t][delay_key]
                    row[f'{f.name}:delay_estimate'] = f.stat[t]['t_f_estimate']
                for f in flows:
                    row[f'{f.name}:pps'] = f.stat[t]['pps']
                for f in flows:
                    for tf in f.get_tflows():
                        row[f'{tf.get_name()}:delay_bound'] = tf.stat[t]['delay_bound']
                        row[f'{tf.get_name()}:estimate'] = tf.stat[t]['tf_estimate']


                for task in tasks:
                    if task.controller is None:
                        continue
                    row[f'{task.name}:task_pps'] = task.stat[t]['pps_in']
                    row[f'{task.name}:t_0'] = task.stat[t]['t_in']
                    row[f'{task.name}:t_0_estimate'] = task.stat[t]['t_0_estimate']
                    row[f'{task.name}:b_0'] = task.stat[t]['b_in']
                    row[f'{task.name}:cyc_per_packet'] = task.stat[t]['cyc_per_packet']

                # gradients
                row.update({key: self.stat[t][key] for key in gradient_keys})

                row['sum_fpps'] = self.stat[t]['sum_fpps']
                row['ctrlr_deadtime'] = self.stat[t]['controller_deadtime']

                if t == 0:
                    # print header
                    csv_writer.writerow(k for k in row)

                csv_writer.writerow([v for k, v in row.items()])
