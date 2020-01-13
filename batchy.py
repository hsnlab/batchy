#! /usr/bin/env python3

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

import argparse
import functools
import logging
import os
import sys

from lib import batchy
from lib import settings
from lib import utils


def get_arg(name, defval, type=str, globals_dict=None):
    try:
        return type(globals_dict.get(name, defval))
    except AttributeError:
        return defval


def run_batchy_config(batchy, bess, conf_textiowrapper, arg_map=None):
    new_globals = {
        '__builtins__': __builtins__,
        '__file__': os.path.abspath(conf_textiowrapper.name),
        'batchy': batchy,
        'bess': bess,
        'settings': settings,
    }
    new_globals.update(utils.get_bess_module_and_port_creators(bess))
    if arg_map:
        new_globals.update(arg_map)
    partial_get_arg = functools.partial(get_arg, globals_dict=new_globals)
    new_globals['get_arg'] = partial_get_arg

    content = str(conf_textiowrapper.read())
    code = compile(content, conf_textiowrapper.name, 'exec')
    exec(code, new_globals)


def convert_config_args_to_argmap(config_args):
    try:
        if config_args.strip() == '':
            return {}
    except AttributeError:
        return {}

    arg_map = {}
    for s in config_args.split(','):
        s.strip()
        try:
            k = s.split('=')[0].strip()
            v = s.split('=')[1].strip()
            if all((k, v)):
                arg_map[k] = v
            else:
                raise ValueError
        except (IndexError, ValueError):
            print('Invalid config_args format, please use: key1=value1,key2=value2')
            sys.exit()
    return arg_map


def get_logging_level_names():
    try:
        level_names = logging._levelToName
    except AttributeError:
        level_names = logging._levelNames
    return [logging.getLevelName(x) for x in level_names]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bessdir', '-b', type=str,
                        help='BESS root directory')
    parser.add_argument('config', type=argparse.FileType('r'),
                        help='batchy script to run')
    parser.add_argument('config_args', type=str, nargs='?', default='',
                        help='arguments of the batchy script (format: c=2,r=4)')
    parser.add_argument('--loglevel', '-l', type=str,
                        default='DEBUG', choices=get_logging_level_names(),
                        help='loglevel')
    parser.add_argument('--reset', '-r', action='store_true',
                        help='Reset BESS daemon before running batchy')
    args = parser.parse_args()
    args_map = convert_config_args_to_argmap(args.config_args)

    logging.basicConfig(stream=sys.stdout,
                        level=getattr(logging, str(args.loglevel)),
                        format='%(message)s')

    bessdir = args.bessdir or settings.DEFAULT_BESSDIR
    bessdir = os.path.expanduser(bessdir)
    bess = utils.get_local_bess_handle(bessdir)
    try:
        bess.connect()
    except (utils.BESS.APIError, utils.BESS.RPCError) as e:
        text = f'{e}.\n' \
               'Perhaps BESS daemon is not running locally?\n' \
               'Before running Batchy, start BESS daemon: bessctl daemon start'
        raise Exception(text)

    if args.reset:
        bess.reset_all()

    batchy = batchy.Batchy(bess)
    batchy.loglevel = args.loglevel
    run_batchy_config(batchy, bess, args.config, args_map)
    batchy.cleanup()

    print('Done.')
