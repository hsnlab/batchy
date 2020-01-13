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
MGW_CONF="$CONF_DIR/mgwcomplex.batchy"
OUT_DIR="$RESULTS_DIR/MGW_RESULTS"
ROUNDS=200


function run_mgw_scenario {
    local bearer_num=$1
    local user_num=$2
    local bearer0_user=${3:-1}
    local controller=$4
    local mode=${5:-"RTC"}
    local meas_id=${6:-0}
    local normalized=${7:-0}
    local delay_slo=${8:-"1_000_000"}
    local an=""
    local args="rounds=$ROUNDS,controller=$controller,mode=$mode,"
    args+="bearer_num=$bearer_num,user_num=$user_num,bearer0_user=$bearer0_user,"
    args+="delay_slo=$delay_slo,normalized=$normalized"
    $BATCHYPY -r -l $LOGLEVEL $MGW_CONF $args
    local odir="$OUT_DIR/b${bearer_num}_norm_${normalized}"
    mkdir -p $odir
    local ofile="${odir}/${controller}_${mode}_norm_${normalized}_${meas_id}.txt"
    mv /tmp/mgw_${mode}_${controller}_norm_${normalized}.txt ${ofile}
}


####
# WFQ

for i in `seq 0 2`; do
    run_mgw_scenario 1 4 4 null WFQ $i 0
    run_mgw_scenario 2 4 4 null WFQ $i 0
    run_mgw_scenario 4 4 4 null WFQ $i 0
    run_mgw_scenario 8 4 4 null WFQ $i 0
    run_mgw_scenario 16 4 4 null WFQ $i 0
done


####
# RTC

ctrlrs=(null max onoff projgrad)
for i in `seq 0 2`; do
    n=0  # normalized parameter
    #for n in `seq 0 2`; do
	for ctrlr in ${ctrlrs[@]}; do
	    run_mgw_scenario 1 4 4 $ctrlr RTC $i $n 80_260
	    run_mgw_scenario 2 4 4 $ctrlr RTC $i $n 112_437
	    run_mgw_scenario 4 4 4 $ctrlr RTC $i $n 193_342
	    run_mgw_scenario 8 4 4 $ctrlr RTC $i $n 415_462
	    run_mgw_scenario 16 4 4 $ctrlr RTC $i $n 788_899
	done
    #done
done
