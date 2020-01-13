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
CONF="$CONF_DIR/simpletree.batchy"
OUT_DIR="$RESULTS_DIR/BATCHINESS_RESULTS"
ROUNDS=80
NUM_MEAS=2


function run_simpletree_scenario() {
    local controlled=$1
    local numleaves=${2:-2}
    local t0="10_000"
    local t1=${3:-10_000}
    local meas_id=${4:-0}
    local args="controller=projgrad,controlled=${controlled},rounds=$ROUNDS,"
    args+="leaves=${numleaves},cycles_per_batch=${t0},cycles_per_packet=${t1}"
    $BATCHYPY -r -l $LOGLEVEL $CONF $args
    mkdir -p $OUT_DIR
    local ofile="${OUT_DIR}/${numleaves}_${controlled}_${t0}_${t1}_${meas_id}.txt"
    mv /tmp/simple_tree_RTC.txt ${ofile}
}


controlled=(false true)
t1s=(526 1111 1765 2500 3333 4286 5385 6667 8182 10_000 12_222 15_000 18_571 23_333 30_000 40_000 56_667 90_000 190_000)
for ctrld in ${controlled[@]}; do
    for t in ${t1s[@]}; do
	for l in `seq 1 32`; do
	    for i in `seq 1 $NUM_MEAS`; do
		run_simpletree_scenario $ctrld $l $t $i
	    done
	done
    done
done
