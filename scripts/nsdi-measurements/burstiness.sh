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
CONF="$CONF_DIR/simpletree-burstiness.batchy"
OUT_DIR="$RESULTS_DIR/BURSTINESS_RESULTS"
ROUNDS=80
NUM_MEAS=2


function run_simpletree_scenario() {
    local controller=$1
    local numleaves=${2:-2}
    local burst=${3:-1}
    local meas_id=${4:-0}
    local args="controller=${controller},rounds=$ROUNDS,"
    args+="leaves=${numleaves},burstiness=${burst}"
    $BATCHYPY -b $BESSDIR -r -l $LOGLEVEL $CONF $args
    mkdir -p $OUT_DIR
    local ofile="${OUT_DIR}/${numleaves}_${controller}_bu${burst}_${meas_id}.txt"
    mv /tmp/simple_tree2_RTC.txt ${ofile}
}


controller=(null projgrad)
bursts=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
leaves=(1 2 4 8 16 32)
for ctrlr in ${controller[@]}; do
    for b in ${bursts[@]}; do
        for l in ${leaves[@]}; do
            for i in `seq 1 $NUM_MEAS`; do
                run_simpletree_scenario $ctrlr $l $b $i
            done
        done
    done
done
