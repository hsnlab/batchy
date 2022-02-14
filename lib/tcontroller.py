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

from . import settings


def __import_controllers(base_class):
    ''' Import the implementation of a controller class

        Parameters:
        base_class (str): name of controller class
    '''
    # pylint: disable=C0103,C0415,W0602,W0611
    if 'feasdir' in base_class.lower():
        global FeasDirTaskController
        global FeasDirRTCTaskController
        global FeasDirWFQTaskController
        from .tcontroller_feasdir import \
            FeasDirTaskController, FeasDirRTCTaskController, FeasDirWFQTaskController
        return
    if 'onoff' in base_class.lower():
        global OnOffTaskController
        global OnOffRTCTaskController
        from .tcontroller_onoff import \
            OnOffTaskController, OnOffRTCTaskController
    if 'projgrad' in base_class.lower():
        global ProjGradientTaskController
        global ProjGradientRTCTaskController
        from .tcontroller_projgradient import \
            ProjGradientTaskController, ProjGradientRTCTaskController
        return
    raise Exception(f'Unknown task controller base class "{base_class}"')


def get_tcontroller(task, base_class):
    ''' Resolve task controller by controller type and task type

        Parameters:
        task (Task): task to set controller for
        base_class (str): name of controller class

        Returns:
        a controller instance set for the task

    '''
    name = f'{base_class}{task.type}TaskController'
    try:
        ctrlr_class = globals()[name]
    except KeyError as class_err:
        __import_controllers(base_class)
        ctrlr_class = globals().get(name)
        if ctrlr_class is None:
            txt = f'Unknown task controller class "{name}"'
            raise Exception(txt) from class_err
    return ctrlr_class(task)


class TaskController:
    ''' Task controller base class '''

    def __init__(self, task):
        self.task = task
        self.period = 0
        self.tflows = task.tflows
        self.is_leader = self.task.is_leader()
        if task.type == 'RTC' and not task.cmodules:
            logging.log(logging.DEBUG,
                        'No controllable module found in RTC task "%s"',
                        task.name)
            task.controller = None
        else:
            task.controller = self

    def control(self, *args, **kwargs):
        ''' Execute the control algoritm '''
        # pylint: disable=W0613
        self.period += 1

    def get_violating_flows(self):
        ''' Collect flows' delay-SLO violation data

            Returns:
            delay (dict): last measured delay of flows
            delay_bound (float):  delay bounds of flows
            over (list): modules violating constraint
            error (float): sum of delay violations
            dist (float): smallest distance from the delay constraint over all flows

        '''
        delay_key = f'latency_{settings.DELAY_MAX_PERC}'
        delay = {tf.id: tf.stat[-1][delay_key] for tf in self.tflows}
        delay_bound = {tf.id: (tf.delay_bound if tf.delay_bound is not None
                               else settings.DEFAULT_DELAY_BOUND)
                       for tf in self.tflows}
        error = 0.0
        over = []
        dist = sum(delay_bound.values())  # large number
        for tflow in self.tflows:
            error_tf = delay[tflow.id] - delay_bound[tflow.id]
            if error_tf > 0:
                error += error_tf
                over.append(tflow)
            else:
                dist = min(dist, -error_tf)
        return delay, delay_bound, over, error, dist


class RTCTaskController(TaskController):
    ''' Base class for run-to-completion task controllers '''
    @staticmethod
    def get_control_at(cmodule, time):
        ''' Get controled value of a module '''
        return cmodule.stat[time]['q_v']


class WFQTaskController(TaskController):
    ''' Base class for weighted-fair-queueing task controllers '''
    @staticmethod
    def get_control_at(cmodule, time):
        ''' Get controled value of a module '''
        return cmodule.stat[time]['w_v']


class NullTaskController(TaskController):
    ''' Leaves all triggers at zero '''

    def __init__(self, task):
        super().__init__(task)
        self.real_controller = get_tcontroller(task, 'Null')

    def __repr__(self, *args, **kwargs):
        return self.real_controller.__repr__(*args, **kwargs)

    def control(self, *args, **kwargs):
        return self.real_controller.control(*args, **kwargs)


class NullRTCTaskController(RTCTaskController):
    ''' Leaves all triggers at 0 '''

    def __init__(self, task):
        super().__init__(task)
        for module in task.cmodules:
            module.set_trigger(0)

    def __repr__(self):
        return 'NullRTCTaskController'


class NullWFQTaskController(WFQTaskController):
    ''' Leaves all weights at 1 '''

    def __init__(self, task):
        super().__init__(task)
        for module in task.cmodules:
            module.set_weight(1)

    def __repr__(self):
        return 'NullWFQTaskController'


class MaxTaskController(TaskController):
    ''' Leaves all triggers at settings.BATCH_SIZE '''

    def __init__(self, task):
        super().__init__(task)
        self.real_controller = get_tcontroller(task, 'Max')

    def __repr__(self, *args, **kwargs):
        return self.real_controller.__repr__(*args, **kwargs)

    def control(self, *args, **kwargs):
        return self.real_controller.control(*args, **kwargs)


class MaxRTCTaskController(RTCTaskController):
    ''' Leaves all triggers at settings.BATCH_SIZE '''

    def __init__(self, task):
        super().__init__(task)
        for module in task.cmodules:
            module.set_trigger(settings.BATCH_SIZE)

    def __repr__(self):
        return 'MaxRTCTaskController'


class MaxWFQTaskController(WFQTaskController):
    ''' Leaves all weights at settings.WEIGHT_MAX '''

    def __init__(self, task):
        super().__init__(task)
        for module in task.cmodules:
            module.set_weight(settings.WEIGHT_MAX)

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
    for module in task.cmodules:
        module.set_trigger(self.size)


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
