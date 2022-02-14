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



# A greedy algorithm to divide packets into roughly equaly sized LPM
# groups.  Input file should consists of lines with the format of
# <ip address> <number of packet with that destination address>

import sys

DEBUG = False

class Btree:
    def __init__(self, bit='', depth=0):
        self.z = None
        self.o = None
        self.pkts = 0
        self.depth = depth
        self.bit = bit
        self.branches = 0

    def add(self, bits="", pkts=0):
        self.pkts += pkts
        if not bits:
            self.branches = 1
            return self.branches
        if bits[0] == '0':
            if self.z is None:
                self.z = Btree('0', depth=self.depth+1)
                b_orig = 0
            else:
                b_orig = self.z.branches
            b_new = self.z.add(bits[1:], pkts)
        if bits[0] == '1':
            if self.o is None:
                self.o = Btree('1', depth=self.depth+1)
                b_orig = 0
            else:
                b_orig = self.o.branches
            b_new = self.o.add(bits[1:], pkts)
        self.branches += b_new - b_orig
        return self.branches

    def find(self, pkts, s=''):
        ss = s + self.bit
        #print("a:%s %s %s" % (ss, self.branches, self.pkts))
        if self.pkts <= pkts or self.branches <= 1:
            return [(self.depth, ss, self.pkts)]
        else:
            r = []
            if self.z:
                r += (self.z.find(pkts, ss))
            if self.o:
                r += (self.o.find(pkts, ss))
            return r

    def __repr__(self):
        self.print()

    def print(self, bits=''):
        bits += self.bit
        leaf = ''
        if self.z is None and self.o is None:
            leaf = '*'
        print(self.pkts, self.branches, bits, leaf)
        if self.z:
            self.z.print(bits)
        if self.o:
            self.o.print(bits)

    def delete_subtree(self, s, pkts):
        "Delete subtree with is pkt count."
        self.pkts -= pkts
        if len(s) == 1:
            if s[0] == '0':
                self.z = None
                self.branches = self.o.branches if self.o else 1
            if s[0] == '1':
                self.o = None
                self.branches = self.z.branches if self.z else 1
        else:
            if s[0] == '0':
                b_orig = self.z.branches
                b_new = self.z.delete_subtree(s[1:], pkts)
                self.branches += b_new - b_orig
                if b_new == 1:
                    self.z = None
            if s[0] == '1':
                b_orig = self.o.branches
                b_new = self.o.delete_subtree(s[1:], pkts)
                self.branches += b_new - b_orig
                if b_new == 1:
                    self.o = None
        return self.branches

    def cut_subtree(self, s):
        "Delete subtree, but add its pkts to its parent."
        if len(s) == 1:
            if s[0] == '0' or s[0] == '1':
                self.z = None
                self.o = None
                self.branches = 1
        else:
            if s[0] == '0':
                b_orig = self.z.branches
                b_new = self.z.cut_subtree(s[1:])
                self.branches += b_new - b_orig
            if s[0] == '1':
                b_orig = self.o.branches
                b_new = self.o.cut_subtree(s[1:])
                self.branches += b_new - b_orig
        return self.branches

def bin2ip(bin_addr, depth):
    bin_addr = (bin_addr + '0' * (32-depth))
    ip = []
    for i in range(4):
        ip.append(str(int(bin_addr[i*8:(i+1)*8], 2)))
    return '.'.join(ip)

def format_percent(pkts, all_pkts):
    return "%2.2f" % (100.0 * (float(pkts)/all_pkts))

def print_cgroups(prefix, cgroups, all_pkts):
    for c in cgroups:
        dprint(prefix, c, format_percent(c[2], all_pkts))


def test():
    bt = Btree()
    print("bt.add('001', 2)",
          bt.add('001', 2))
    bt.print()
    print("bt.add('001', 2)",
          bt.add('001', 2))
    bt.print()
    print("bt.add('010', 3)",
          bt.add('010', 3), bt.pkts)
    bt.print()
    print("bt.add('011', 3)",
          bt.add('011', 3), bt.pkts)
    bt.print()
    # print("bt.delete_subtree('001', 4)",
    #        bt.delete_subtree('001', 4), bt.pkts)
    # bt.print()
    print('bt.find(0)', bt.find(0))


def dprint(*args, **kv):
    if DEBUG:
        print(*args, **kv)

def get_groups(pcap_in, groups):
    if not pcap_in.endswith('-stat.txt'):
        raise ValueError('input filename should end with "-stat.txt"')

    npkts = 0
    ips = []
    with open(pcap_in) as f:
        for line in f:
            ip, pkts = line.rstrip().split(' ')
            b = ''.join([bin(int(x)+256)[3:] for x in ip.split('.')])
            #dprint("%s ip:%s pkts:%s" % (b, ip, pkts))
            npkts += int(pkts)
            ips.append((b, int(pkts), ip))
    npkts_orig = npkts
    dprint(npkts, npkts/groups)
    ips = sorted(ips)

    btree = Btree()
    for _, v in enumerate(ips):
        btree.add(v[0], v[1])

    final = []
    while True:
        dprint(' ')
        cgroups = btree.find(npkts/groups)
        cgroups = sorted(cgroups, key=lambda x: (-x[0], -x[2]))
        print_cgroups("d", cgroups, npkts_orig)
        if len(cgroups) <= groups:
            print_cgroups("final", cgroups, npkts_orig)
            final += cgroups
            break
        head = cgroups[0]
        if head[2] > npkts/groups:
            dprint('final', head, format_percent(head[2], npkts_orig))
            final.append(head)
            btree.delete_subtree(head[1], head[2])
            npkts -= head[2]
            groups -= 1
            dprint("npkts: %s, groups: %s, n/g: %s" % (npkts, groups, npkts/groups))
        else:
            btree.cut_subtree(head[1])
            dprint('cut_subtree', head[1])

    dprint(" ")
    print_cgroups("Final", final, npkts_orig)
    return ["%s/%s" % (bin2ip(g[1], g[0]), g[0]) for g in final]


if __name__ == "__main__":
    #test(); exit()
    DEBUG = True
    try:
        pcap_in = sys.argv[1]
        groups = int(sys.argv[2])
    except IndexError as e:
        print('Usage: %s filename-stat.txt groups' % sys.argv[0])
        exit(-1)

    groups = get_groups(pcap_in, groups)

    outfile = pcap_in.replace('-stat.txt', '-lpm-rules.txt')
    with open(outfile, 'w') as f:
        for cidr in groups:
            print(cidr)
            f.write("%s\n" % cidr)
