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

from . import task
from . import settings
from . import utils


class Worker(object):
    def __init__(self, batchy, name, wid, core=None):
        self.wid = wid
        self.core = core or wid
        self.name = f'w{self.wid}' if name == '' else name
        self.batchy = batchy
        self.bess = batchy.bess
        self.tasks = []
        self.bess.add_worker(wid=self.wid, core=self.core)

        self.tc_root = f'tc_root:{self.name}'
        self.tc_wfq = f'tc_wfq:{self.name}'
        self.tc_qos = f'tc_qos:{self.name}'
        self.tc_bulk_root = f'tc_bulk_root:{self.name}'
        self.tc_bulk = f'tc_bulk:{self.name}'

        self.bess.add_tc(self.tc_root,
                         wid=self.wid,
                         policy='weighted_fair',
                         resource=settings.SCHEDULER_DOMAIN)
        self.bess.add_tc(self.tc_wfq,
                         parent=self.tc_root,
                         share=1,
                         policy='weighted_fair',
                         resource=settings.WFQ_SCHEDULER_DOMAIN)
        self.bess.add_tc(self.tc_qos,
                         parent=self.tc_root,
                         share=settings.DEFAULT_TC_QOS_SHARE,
                         policy='weighted_fair',
                         resource=settings.SCHEDULER_DOMAIN)
        self.bess.add_tc(self.tc_bulk_root,
                         parent=self.tc_root,
                         share=1,
                         policy='rate_limit',
                         resource=settings.DEFAULT_RATE_LIMIT_RESOURCE,
                         limit={settings.DEFAULT_RATE_LIMIT_RESOURCE:
                                settings.DEFAULT_RATE_LIMIT})
        self.bess.add_tc(self.tc_bulk,
                         parent=self.tc_bulk_root,
                         policy='weighted_fair',
                         resource=settings.SCHEDULER_DOMAIN)

    def __repr__(self):
        content_repr = pprint.pformat({'Wid': self.wid,
                                       'Core': self.core,
                                       'Tasks': self.tasks},
                                      indent=8, width=1)
        return f'Worker {self.name}:\n{content_repr}'

    def _add_RTC_task(self, name, limit, weight, queue, queue_size, backpressure, qos):
        new_task = task.RTCTask(self.batchy, self, name, queue=queue,
                                queue_size=queue_size, backpressure=backpressure)
        new_task.weight = weight
        # set task scheduler
        if qos:
            if limit is None:
                limit = {settings.DEFAULT_RATE_LIMIT_RESOURCE:
                         settings.DEFAULT_RATE_LIMIT}
            resource = next(iter(limit.keys()))
            tc_name = f'tc_qos:{new_task.name}'
            if tc_name in (new_task.name for t in self.list_tcs()):
                logging.log(logging.WARNING,
                            f'Adding QoS tc for task "{new_task.name}": '
                            f'QoS tc with the same name already exists, updating')
                self.bess.update_tc_params(tc_name,
                                           resource=resource,
                                           limit=limit)
            else:
                self.bess.add_tc(tc_name,
                                 policy='rate_limit',
                                 resource=resource,
                                 parent=self.tc_qos,
                                 share=new_task.weight,
                                 limit=limit)
        else:
            tc_name = f'tc_bulk:{name}'
            self.bess.add_tc(tc_name, policy='round_robin',
                             parent=self.tc_bulk,
                             share=new_task.weight)
        new_task.tc_name = tc_name
        new_task.queue.attach_task(tc_name)
        return new_task

    def _add_WFQ_task(self, name, limit, weight, queue, queue_size, backpressure, qos):
        new_task = task.WFQTask(self.batchy, self, name, queue=queue,
                                queue_size=queue_size, backpressure=backpressure)
        new_task.weight = weight
        resource = next(iter(limit.keys()))

        # set task scheduler
        # first add a rate limiter
        rate_limit_tc_name = f'tc_task:rate_limit:{name}'
        self.bess.add_tc(rate_limit_tc_name,
                         parent=self.tc_wfq,
                         policy='rate_limit',
                         share=1,
                         resource=resource,
                         limit=limit)
        new_task.rate_limit_tc_name = rate_limit_tc_name

        tc_name = f'tc_wfq:{name}'
        self.bess.add_tc(tc_name,
                         parent=rate_limit_tc_name,
                         policy='weighted_fair',
                         resource=settings.WFQ_SCHEDULER_DOMAIN)
        new_task.tc_name = tc_name
        # finally, add the input queue as a child of the wrr tc
        new_task.queue.attach_task(tc_name, share=1)
        return new_task

    def add_task(self, name=None, type=settings.DEFAULT_TASK_TYPE, limit=None,
                 weight=1, queue=None, queue_size=None, backpressure=None,
                 qos=False):
        if name is None:
            name = f'w{self.wid}:task{len(self.tasks)}'
        if limit is None:
            limit = utils.default_rate_slo()
        if self.batchy.get_task(name) is not None:
            text = f'add_task: task with name "{name}" already exists'
            raise Exception(text)
        if queue_size is None:
            queue_size = settings.DEFAULT_QUEUE_SIZE
        if backpressure is None:
            backpressure = settings.ENABLE_BACKPRESSURE

        # add task
        try:
            add_func = getattr(self, f'_add_{type}_task')
            new_task = add_func(name, limit, weight, queue, queue_size,
                                backpressure, qos)
        except AttributeError:
            raise Exception(f'add_task: unknown task type "{type}"')

        self.tasks.append(new_task)
        return new_task

    def set_bulk_ratelimit(self, limit):
        '''Sets the maximum share of total resources bulk flows may get useful
           for rate-limiting bulk sources
        '''
        limit = utils.check_ratelimit(limit)
        if limit is not None:
            (resource, _), = limit.items()
            self.bess.update_tc_params(self.tc_bulk_root,
                                       resource=resource,
                                       limit=limit)

    def update_task_tc(self, task_name):
        ''' Updates a task TC.
            As long as there is a flow in the task that is
            not rate-limited, we cannot set hard rate-limit on the task.
        '''
        t = self.get_task(task_name)
        if t.has_slo():
            tc_name = f'tc_qos:{t.name}'
            limit = t.get_ratelimit()
            resource = next(iter(limit.keys()))
            if tc_name in (t.name for t in self.list_tcs()):
                self.bess.update_tc_params(tc_name,
                                           resource=resource,
                                           limit=limit)
            else:
                self.bess.add_tc(tc_name,
                                 policy='rate_limit',
                                 resource=resource,
                                 parent=self.tc_qos,
                                 share=t.weight,
                                 limit=limit)
                t.tc_name = tc_name
                t.queue.attach_task(tc_name)
        else:  # put tc back to bulk
            tc_name = f'tc_bulk:{t.name}'
            t.queue.attach_task(tc_name)
            t.tc_name = tc_name

    def set_task_controller(self, cclass, *args, **kwargs):
        for t in self.tasks:
            t.set_controller(cclass, *args, **kwargs)

    def move_task(self, task, dest):
        ''' Move a task to other worker.

        Arguments:
        task: task to move
        dest: target worker

        '''
        # set target traffic class
        if task.type == 'WFQ':
            target_tc = dest.tc_wfq
        else:
            if any((f for f in task.get_flows() if f.has_slo())):
                # if a flow in the task to be moved has slo,
                # move to qos tc
                target_tc = dest.tc_qos
            else:
                target_tc = dest.tc_bulk

        # update references
        self.tasks.remove(task)
        dest.tasks.append(task)
        task.worker = dest

        # attach task queue to new tc
        task.queue.attach_task(target_tc, share=1)

        # TODO: set rate-limit

    def num_controlled_tasks(self):
        return len([t for t in self.tasks if t.controller is not None])

    def list_tcs(self):
        tcs = self.bess.list_tcs(wid=self.wid).classes_status
        return [getattr(tc, 'class') for tc in tcs]

    def get_flows(self):
        return [f for f in self.batchy.flows if f.traverses_worker(self)]

    def get_task(self, name):
        return next((t for t in self.tasks if t.name == name), None)

    def get_stat(self):
        for t in self.tasks:
            if t.controller is not None:
                t.get_stat()

    def get_gradient(self):
        for t in self.tasks:
            t.get_gradient()

    def reset(self):
        for t in self.tasks:
            if t.controller is not None:
                t.reset()

    def erase_stat(self):
        for t in self.tasks:
            t.erase_stat()
