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


class TaskFlow:
    def __init__(self, path, delay_bound, id=-1):
        self.path = path
        self.delay_bound = delay_bound
        self.id = id
        self.task = None
        self.cpath = []        # path on controlled modules
        self.uncont_desc = {}  # uncontrolled descendants, indexed by cmodule
        self.egress_m = None
        self.stat = []

        for i, module in enumerate(self.path):
            if module.is_controlled:
                self.cpath.append(module)
                self.uncont_desc[module] = []
                j = i
                while True:
                    self.uncont_desc[module].append(self.path[j])
                    j += 1
                    if j == len(self.path) or self.path[j].is_controlled:
                        break

    def __repr__(self):
        return f'TaskFlow: {self.path[0]} -> {self.path[-1]}: ' \
               f'delay_bound={self.delay_bound}'

    def get_name(self):
        task_name = "NoTask"
        if hasattr(self.task,'name'):
            task_name = self.task.name
        return f'{task_name}_{self.id}'

    def erase_stat(self):
        ''' Erase collected statistics '''
        self.stat = []

    def get_last_cmodule(self):
        ''' Get last controlled module on task flow path '''
        try:
            return self.cpath[-1]
        except IndexError:
            return None

    def traverses_module(self, module):
        ''' Check if module is traversed by this task flow '''
        return any(m for m in self.cpath if m == module)

    def get_delay_estimate(self, batch_size):
        ''' Calculate delay estimate for modules along the task flow path '''
        return sum([m.get_delay_estimate(batch_size) for m in self.path])
