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
L2L3_CONF="conf/l2l3.batchy"
OUT_DIR="BATCHGAIN_RESULTS"
LOGLEVEL="CRITICAL"
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

    $BATCHYPY -r -l $LOGLEVEL $L2L3_CONF $args

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
    mv /tmp/${controller}_stat.txt ${ofile}
}

ctrlrs=(null max onoff projgrad)
branch_num=1
nhop_nums=(1 2 4 8 16 24 32 48 64)
delay_slos=(11_440 17_722 24_886 45_728 88_561 125_397 161_485 276_840 432_508)

for i in `seq 0 2`; do
    for ctrlr in ${ctrlrs[@]}; do
	for ((j=0; j<${#nhop_nums[@]}; ++j)); do
	    run_l2l3_scenario $branch_num ${nhop_nums[j]} "acl=false,nat=false" $ctrlr RTC $i ${delay_slos[j]}
	done
    done
done
