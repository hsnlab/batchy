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


function run_l2l3_scenario {
    local branch_num=$1
    local nhop_num=$2
    local aclnat=${3:-"acl=false,nat=false"}
    local controller=$4
    local mode=${5:-"RTC"}
    local meas_id=${6:-0}
    local delay_slo=${7:-"1_000_000"}
    local an=""
    local args="rounds=$ROUNDS,controller=$controller,mode=$mode,"
    args+="branch_num=$branch_num,nhop_num=$nhop_num,$aclnat,max_delay=$delay_slo"

    $BATCHYPY -b $BESSDIR -r -l $LOGLEVEL $L2L3_CONF $args

    case "$aclnat" in
	*acl=true*)
	    an+="_a"
	    ;;
    esac
    case "$aclnat" in
	*nat=true*)
	    an+="n"
	    ;;
    esac
    local odir="$OUT_DIR/b${branch_num}_n${nhop_num}${an}"
    mkdir -p $odir
    local ofile="${odir}/${controller}_${mode}_${meas_id}.txt"
    mv /tmp/l2l3_${mode}_${controller}_b${branch_num}_n${nhop_num}*.txt ${ofile}

}


####
# WFQ

for i in `seq 0 2`; do
    run_l2l3_scenario 1 1 "acl=false,nat=false" null WFQ $i
    # run_l2l3_scenario 1 2 "acl=false,nat=false" null WFQ $i
    run_l2l3_scenario 1 4 "acl=false,nat=false" null WFQ $i
    run_l2l3_scenario 1 8 "acl=false,nat=false" null WFQ $i
    run_l2l3_scenario 1 16 "acl=false,nat=false" null WFQ $i
    # run_l2l3_scenario 1 24 "acl=false,nat=false" null WFQ $i
    # run_l2l3_scenario 1 32 "acl=false,nat=false" null WFQ $i
    # run_l2l3_scenario 1 48 "acl=false,nat=false" null WFQ $i
    # run_l2l3_scenario 1 64 "acl=false,nat=false" null WFQ $i

    run_l2l3_scenario 1 1 "acl=true,nat=true" null WFQ $i
    # run_l2l3_scenario 1 2 "acl=true,nat=true" null WFQ $i
    # run_l2l3_scenario 1 4 "acl=true,nat=true" null WFQ $i
    run_l2l3_scenario 1 8 "acl=true,nat=true" null WFQ $i
    run_l2l3_scenario 1 16 "acl=true,nat=true" null WFQ $i
    # run_l2l3_scenario 1 24 "acl=true,nat=true" null WFQ $i
    # run_l2l3_scenario 1 32 "acl=true,nat=true" null WFQ $i
    # run_l2l3_scenario 1 48 "acl=true,nat=true" null WFQ $i
    run_l2l3_scenario 1 64 "acl=true,nat=true" null WFQ $i
done


####
# RTC

ctrlrs=(null max onoff projgrad)
for i in `seq 0 2`; do
    for ctrlr in ${ctrlrs[@]}; do
	run_l2l3_scenario 1 1 "acl=false,nat=false" $ctrlr RTC $i 11_164
	# run_l2l3_scenario 1 2 "acl=false,nat=false" $ctrlr RTC $i 16_679
        run_l2l3_scenario 1 4 "acl=false,nat=false" $ctrlr RTC $i 22_990
        run_l2l3_scenario 1 8 "acl=false,nat=false" $ctrlr RTC $i 41_180
	run_l2l3_scenario 1 16 "acl=false,nat=false" $ctrlr RTC $i 79_121
	# run_l2l3_scenario 1 24 "acl=false,nat=false" $ctrlr RTC $i 115_891
	# run_l2l3_scenario 1 32 "acl=false,nat=false" $ctrlr RTC $i 146_875
	# run_l2l3_scenario 1 48 "acl=false,nat=false" $ctrlr RTC $i 240_591
	# run_l2l3_scenario 1 64 "acl=false,nat=false" $ctrlr RTC $i 359_281

        run_l2l3_scenario 1 1 "acl=true,nat=true" $ctrlr RTC $i 16_243
	# run_l2l3_scenario 1 2 "acl=true,nat=true" $ctrlr RTC $i 22_728
        # run_l2l3_scenario 1 4 "acl=true,nat=true" $ctrlr RTC $i 34_428
        run_l2l3_scenario 1 8 "acl=true,nat=true" $ctrlr RTC $i 62_250
	run_l2l3_scenario 1 16 "acl=true,nat=true" $ctrlr RTC $i 118_926
	# run_l2l3_scenario 1 24 "acl=true,nat=true" $ctrlr RTC $i 161_700
	# run_l2l3_scenario 1 32 "acl=true,nat=true" $ctrlr RTC $i 204_278
	# run_l2l3_scenario 1 48 "acl=true,nat=true" $ctrlr RTC $i 330_169
	run_l2l3_scenario 1 64 "acl=true,nat=true" $ctrlr RTC $i 477_653
    done
done
