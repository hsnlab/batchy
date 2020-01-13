#!/usr/bin/env bash

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


BESSDIR=$HOME/bess
MEASUREMENTS=10
HUGEPAGES=12

MEMORY=$(expr 1024 \* ${HUGEPAGES})
MODULES=(ExactMatch WildcardMatch RandomSplit HashLB IPLookup StaticNAT MACSwap Bypass Replicate ACL DPDKACL)
#EXTRA_ARGS="rate_limit=1000000000"

if [ -z $EXTRA_ARGS ]; then
    EXTRA_ARGS="max_batch=32"
fi
for MOD in ${MODULES[@]}; do
    OFILE="${MOD}.tsv"
    OLOGFILE="${MOD}.log"
    python3 ./profiler.py -b ~/bess -a " -m ${MEMORY}" -e $EXTRA_ARGS -r $MEASUREMENTS -f tsv -m $MOD -o ${OFILE,,} > ${OLOGFILE,,}
done
