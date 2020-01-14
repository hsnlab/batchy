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
RC_CONF="$CONF_DIR/rc-static.batchy"
OUT_DIR="$RESULTS_DIR/RC_RESULTS"
ROUNDS=200


function run_rc_scenario() {
    local controller=$1
    local mode=${2:-"RTC"}
    local meas_id=${3:-0}
    local delay_slo=${4:-"1_000_000"}
    local leaves=16
    local args="controller=$controller,mode=$mode,rounds=$ROUNDS,delay_bound1=$delay_slo"
    $BATCHYPY -b $BESSDIR -r -l $LOGLEVEL $RC_CONF $args
    mkdir -p $OUT_DIR
    local ofile="${OUT_DIR}/${controller}_${mode}_${meas_id}.txt"
    mv /tmp/rcstatic${leaves}_${mode}_${controller}.txt ${ofile}
}


####
# WFQ

for i in `seq 0 2`; do
    run_rc_scenario null WFQ $i
done


####
# RTC

ctrlrs=(null max onoff projgrad)
for i in `seq 0 2`; do
    for ctrlr in ${ctrlrs[@]}; do
	run_rc_scenario $ctrlr RTC $i 35000
    done
done
