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
L2L3_CONF="$CONF_DIR/l2l3.batchy"
OUT_DIR="$RESULTS_DIR/L2L3_RESULTS"
ROUNDS=200


function run_l2l3_pcap_scenario {
    local nhop_num=$1
    local controller=$2
    local mode=${3:-"RTC"}
    local meas_id=${4:-0}
    local pcap_src=$5
    local delay_slo=${6:-"1_000_000"}
    local args="rounds=$ROUNDS,controller=$controller,mode=$mode,"
    args+="branch_num=1,nhop_num=$nhop_num,max_delay=$delay_slo,"
    args+="src_pcap=$pcap_src,acl=false,nat=false"

    $BATCHYPY -r -l $LOGLEVEL $L2L3_CONF $args

    local odir="$OUT_DIR/b1_n${nhop_num}_pcap"
    mkdir -p $odir
    local ofile="${odir}/${controller}_${mode}_${meas_id}.txt"
    mv /tmp/l2l3_${mode}_${controller}_b1_n${nhop_num}_pcap.txt ${ofile}
}


####
# WFQ

for i in `seq 0 2`; do
    run_l2l3_pcap_scenario 16 null WFQ $i $SRC_PCAP
    run_l2l3_pcap_scenario 32 null WFQ $i $SRC_PCAP
    run_l2l3_pcap_scenario 64 null WFQ $i $SRC_PCAP
done


####
# RTC

ctrlrs=(null max onoff projgrad)
for i in `seq 0 2`; do
    for ctrlr in ${ctrlrs[@]}; do
	run_l2l3_pcap_scenario 16 $ctrlr RTC $i $SRC_PCAP 98_494
	run_l2l3_pcap_scenario 32 $ctrlr RTC $i $SRC_PCAP 206_226
	run_l2l3_pcap_scenario 64 $ctrlr RTC $i $SRC_PCAP 465_823
    done
done
