#!/bin/bash
#
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


source `dirname $0`/settings.sh
L2L3_CONF="$CONF_DIR/l2l3_vbr.batchy"
MGW_CONF="$CONF_DIR/mgw_vbr.batchy"


# NOTE
# results are in /tmp


ctrlrs=(null max onoff projgrad)

# L2L3
for ctrlr in ${ctrlrs[@]}; do
    $BATCHYPY -r -l $LOGLEVEL $L2L3_CONF controller=$ctrlr
done

# MGW
for ctrlr in ${ctrlrs[@]}; do
    $BATCHYPY -r -l $LOGLEVEL $MGW_CONF controller=$ctrlr
done
