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
L3VRF_CONF="$CONF_DIR/l3vrf.batchy"
OUT_DIR="$RESULTS_DIR/L3VRF_RESULTS"
ROUNDS=200


function run_l3vrf_scenario {
    local vlan_num=$1
    local nhop_num=$2
    local controller=$3
    local mode=${4:-"RTC"}
    local meas_id=${5:-0}
    local delay_slo=${6:-"1_000_000"}
    local args="rounds=$ROUNDS,controller=$controller,mode=$mode,"
    args+="vlan_num=$vlan_num,nhop_num=$nhop_num,max_delay=$delay_slo"

    $BATCHYPY -r -l $LOGLEVEL $L3VRF_CONF $args

    local odir="$OUT_DIR/v${vlan_num}_n${nhop_num}"
    mkdir -p $odir
    local ofile="${odir}/${controller}_${mode}_${meas_id}.txt"
    mv /tmp/l3vrf_${controller}_${mode}_stat.txt ${ofile}
}


####
# WFQ

for i in `seq 0 2`; do
    # run_l3vrf_scenario 1 1 null WFQ $i
    # run_l3vrf_scenario 2 4 null WFQ $i
    # run_l3vrf_scenario 4 8 null WFQ $i
    # run_l3vrf_scenario 8 8 null WFQ $i
    # run_l3vrf_scenario 16 4 null WFQ $i
    # run_l3vrf_scenario 32 4 null WFQ $i

    run_l3vrf_scenario 1 1 null WFQ $i
    run_l3vrf_scenario 2 4 null WFQ $i
    run_l3vrf_scenario 16 4 null WFQ $i
    run_l3vrf_scenario 12 8 null WFQ $i
done


####
# RTC

ctrlrs=(null max onoff projgrad)
for i in `seq 0 2`; do
    for ctrlr in ${ctrlrs[@]}; do
	# run_l3vrf_scenario 1 1 $ctrlr RTC $i 18_018
	# run_l3vrf_scenario 2 4 $ctrlr RTC $i 30_110
        # run_l3vrf_scenario 4 8 $ctrlr RTC $i 80_027
        # run_l3vrf_scenario 8 8 $ctrlr RTC $i 145_913
	# run_l3vrf_scenario 16 4 $ctrlr RTC $i 195_240
	# run_l3vrf_scenario 32 4 $ctrlr RTC $i 411_261

	run_l3vrf_scenario 1 1 $ctrlr RTC $i 22_108
	run_l3vrf_scenario 2 4 $ctrlr RTC $i 72_840
	run_l3vrf_scenario 16 4 $ctrlr RTC $i 752_590
	run_l3vrf_scenario 12 8 $ctrlr RTC $i 1_130_310
    done
done
