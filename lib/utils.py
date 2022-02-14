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

import json
import logging
import sys

from . import settings


creators = None


class Singleton(type):
    """ Singleton class based on https://stackoverflow.com/q/6760685 """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_local_bess_handle(bessdir):
    """ Import pybess and instantiate a pybess.BESS module """
    try:
        sys.path.insert(1, bessdir)
        global BESS
        global BESSModule
        global BESSPort
        from pybess.bess import BESS
        from pybess.module import Module as BESSModule
        from pybess.port import Port as BESSPort
    except ImportError:
        print(('Cannot import the API module (pybess) from %s' % bessdir))
        raise
    return BESS()


def get_bess_creators(bess):
    global creators
    if creators is None:
        if not hasattr(globals(), 'BESS'):
            get_local_bess_handle(settings.DEFAULT_BESSDIR)
        creators = get_bess_module_and_port_creators(bess)
    return creators


def create_bess_module(bess, mclass, args=None):
    args = args or {}
    creators = get_bess_creators(bess)
    args['_do_not_create'] = False
    module = creators[mclass](**args)
    return module


def create_bess_port(bess, pclass, args=None):
    args = args or {}
    creators = get_bess_creators(bess)
    port = creators[pclass](**args)
    return port


@staticmethod
def _choose_arg(arg, kwargs):
    if kwargs:
        if arg:
            raise TypeError('You cannot specify both arg and keyword args')
        for key in kwargs:
            if isinstance(kwargs[key], (BESSModule, BESSPort)):
                kwargs[key] = kwargs[key].name
        return kwargs
    if isinstance(arg, (BESSModule, BESSPort)):
        return arg.name
    return arg


def get_bess_module_and_port_creators(bess):
    global creators
    creators = {}
    class_names = [str(i) for i in bess.list_mclasses().names]
    driver_names = [str(i) for i in bess.list_drivers().driver_names]
    for name in class_names:
        creators[name] = type(str(name), (BESSModule,),
                              {'bess': bess, 'choose_arg': _choose_arg})
    for name in driver_names:
        creators[name] = type(str(name), (BESSPort,),
                              {'bess': bess, 'choose_arg': _choose_arg})
    return creators


def get_t0t1_values(module_class, profiler_file=None):
    """ Read T_0 and T_1 values from module profiling data """
    profiler_file = profiler_file or settings.PROFILER_FILE
    filtered_modules = ('Source', 'Sink', 'Measure', 'Timestamp', 'Rewrite')
    if module_class in filtered_modules:
        return 0, 0
    return _get_t0t1_values_from_json(module_class, profiler_file)


def _get_t0t1_values_from_json(module_class, profiler_file):
    """ Helper function to parse module profiling data """
    t_0 = settings.DEFAULT_T0
    t_1 = settings.DEFAULT_T1
    try:
        with open(profiler_file, 'r') as json_file:
            data = json.load(json_file)
            t_0 = data[module_class]["T_0"]
            t_1 = data[module_class]["T_1"]
            cause = None
    except FileNotFoundError:
        cause = f'Results file "{profiler_file}" not found'
    except (ValueError, KeyError):
        cause = f'No profiling data available'
    if cause:
        effect = f'using default T_0 and T_1 values for {module_class}'
        logging.log(logging.INFO, f'profiler: {cause}, {effect}.')
    return t_0, t_1


def check_ratelimit(rate_limit):
    """ Validate rate limit """
    if rate_limit is None:
        return rate_limit
    if not isinstance(rate_limit, dict):
        logging.log(logging.WARNING,
                    (f'Invalid rate-limit "{rate_limit}": must be a '
                     '{resource: limit} pair, ignoring'))
        return None
    (resource, limit), = rate_limit.items()
    if resource in ('packet', 'bit'):
        return rate_limit
    if resource == 'count':
        # we cannot rate-limit to batch rate: even though we could set the
        # limit at the source but we cannot measure whether the flow is
        # saturated as batch size/count changes inside the pipeline
        logging.log(logging.WARNING,
                    (f'Invalid rate-limit: cannot rate-limit '
                     f'to batch count: "{resource}"={limit:.3f} ignored'))
        return None
    logging.log(logging.WARNING,
                (f'Invalid rate-limit:  unknown '
                 f'resource limit "{resource}"={limit:.3f} ignored'))
    return None


def get_ratelimit_for_flows(flows):
    """ Calc sum rate of flows. Mixed limits are not supported: all flows are
        handled either as 'packet' or ar 'bit' rate-limited depending on
        the rate dimension of the first flow's rate-limit.

        Parameters:
        flows (list): flows to sum rate limit

        Returns:
        a {rate dimension (str): sum rate (int)} dict

    """
    rate_dim = None
    sum_rate = 0
    for flow in flows:
        rate_resource = get_rate_slo_resource(flow)
        rate_value = get_rate_slo_value(flow)
        rate_dim = rate_dim or rate_resource
        if rate_resource is None or rate_value is None:
            logging.log(logging.WARNING,
                        f'Invalid rate-limit: flows w/ and w/o rate-limit in flow '
                        f'set, falling back to default rate-limit')
            return default_rate_slo()
        if rate_resource != rate_dim:
            logging.log(logging.WARNING,
                        f'Invalid rate-limit on flow set: some flows rate-limit '
                        f'dimensions differ, falling back to default rate-limit')
            return default_rate_slo()
        sum_rate += rate_value
    return {rate_dim: sum_rate}


def get_rate_slo_resource(flow):
    if flow.R is not None:
        return next(iter(flow.R.keys()))
    return None


def get_rate_slo_value(flow):
    if flow.R is not None:
        return next(iter(flow.R.values()))
    return None


def default_delay_slo():
    """ Get default delay SLO """
    return settings.DEFAULT_DELAY_BOUND


def default_rate_slo():
    """ Get default rate SLO """
    return {settings.DEFAULT_RATE_LIMIT_RESOURCE:
            settings.DEFAULT_RATE_LIMIT}


def filter_obj_list(object_list, obj_attrib, filtered_words):
    return [e for e in object_list
            if not any(n in getattr(e, str(obj_attrib))
                       for n in filtered_words)]


def format_sum_rate(sum_rate, unit='pps'):
    """ Format rates with SI prefixes: mega, kilo, or none"""
    if sum_rate > 1_000_000:
        return f'{sum_rate / 1e6} M{unit}'
    if sum_rate > 1_000:
        return f'{sum_rate / 1e3} k{unit}'
    return f'{sum_rate} {unit}'
