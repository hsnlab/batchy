#!/usr/bin/env python3

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

import logging
import sys
import scapy.all as scapy

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

try:
    pcap_in = sys.argv[1]
except IndexError as e:
    print('Usage: create_pcap_stats.py filename.pcap')
    sys.exit(-1)

if not pcap_in.endswith('.pcap'):
    print('input filename should end with ".pcap"')
    sys.exit(-1)

stats_out = pcap_in.replace('.pcap', '-stat.txt')

def write_stats(stats):
    with open(stats_out, 'w') as f:
        for item in stats.items():
            f.write("%s %s\n" % item)

dst = {}
for i, pkt in enumerate(scapy.PcapReader(pcap_in)):
    try:
        ip = pkt[scapy.IP]
    except IndexError:
        print("pkt #%d has no IP header" % i)
    dst[ip.dst] = dst.get(ip.dst, 0) + 1
    if i % 50000 == 0:
        print(i)

write_stats(dst)
print('bye')
