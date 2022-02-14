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

import binascii
import scapy.all as scapy
import socket
import struct


class PipelineBase:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        globals().update(vars(self))
        self.make_config()
        globals().update(vars(self))
        self.build_pipeline()
        globals().update(vars(self))
        self.set_controllers()
        globals().update(vars(self))
        self.add_test_traffic()

    def make_config(self):
        pass

    def gen_packet(self):
        pass

    def build_pipeline(self):
        pass

    def set_controllers(self):
        pass

    def add_test_traffic(self):
        pass

    def get_vars(self):
        d = vars(self).copy()
        d.pop('self')
        return d


def aton(ip):
    return socket.inet_aton(ip)


def mac_from_str(s, mode='bytes'):
    ret = binascii.unhexlify(s.replace(':', ''))
    if mode == 'int':
        ret = int.from_bytes(ret, byteorder='big')
    return ret


def get_octets(seq, offset=11, last=0):
    return (int(seq / 64516) + offset,
            int(seq % 64516 / 254),
            (seq % 254) + 1,
            last)


def get_ip(seq, offset=11, last=0):
    return '.'.join(map(str, get_octets(seq, offset, last)))


def gen_acl_rules(num_entries=128):
    acl_rules = []
    for i in range(num_entries):
        acl_rules.append(
            {'src_ip': '%s/24' % get_ip(i, offset=2),
             'dst_ip': '%s/24' % get_ip(i, offset=2),
             'drop': True})
    return acl_rules


class SimpleTreePipeline(PipelineBase):
    """Implements conf/multistagetree.batchy

    Arguments:
        leaves: Number of leaves
        weight1: Weight of first branch split
        cycles_per_batch: Bypass param
        cycles_per_batch1: 1st branch Bypass param
        cycles_per_packet: Bypass param
        cycles_per_packet1: 1st branch Bypass param
        delay_bound: Delay SLO of flows
        delay_bound1: Delay SLO of the 1st flow
        packet_bound: Rate SLO of flows
        packet_bound1: Rate SLO of the 1st flow
        mode: RTC or WFQ
        rounds: Number of control rounds
        control_period: Interval of control periods
        controller: Controller type

    Attributes:
        gates_weights: WeightedRandomSplit config
    """

    def __init__(self, *args, **kwargs):
        super(SimpleTreePipeline, self).__init__(*args, **kwargs)

    def make_config(self):
        locals().update(vars(self))
        gates_weights = {i: 1.0 for i in range(1, leaves)}
        gates_weights[0] = weight1
        vars(self).update(locals())

    def build_pipeline(self):
        locals().update(vars(self))
        w0 = batchy.add_worker('w0')
        t0 = w0.add_task('task0', type=mode)
        r0 = t0.add_module(WeightedRandomSplit(gates_weights=gates_weights,
                                               drop_rate=0.0),
                           type='ingress')
        bp0 = t0.add_module(Bypass(cycles_per_batch=cycles_per_batch1,
                                   cycles_per_packet=cycles_per_packet1),
                            cycles_per_batch1 / settings.BYPASS_RATIO,
                            cycles_per_packet1 / settings.BYPASS_RATIO)

        r0.connect(bp0, ogate=0)

        batchy.add_flow(name='flow0', path=[{'task': t0, 'path': [r0, bp0]}],
                        delay_slo=delay_bound1,
                        source_params={'rate_limit':
                                       {'packet': packet_bound1}})

        for i in range(1, leaves):
            bp = t0.add_module(Bypass(cycles_per_batch=cycles_per_batch,
                                      cycles_per_packet=cycles_per_packet),
                               cycles_per_batch / settings.BYPASS_RATIO,
                               cycles_per_packet / settings.BYPASS_RATIO)
            r0.connect(bp, ogate=i)
            batchy.add_flow(name=f'flow{i:d}',
                            path=[{'task': t0, 'path': [r0, bp]}],
                            delay_slo=delay_bound,
                            source_params={'rate_limit':
                                           {'packet': packet_bound}})

        vars(self).update(locals())

    def set_controllers(self):
        locals().update(vars(self))
        t0.set_controller(batchy.resolve_task_controller(controller))

    def add_test_traffic(self):
        locals().update(vars(self))
        batchy.add_source()
        batchy.add_sink()



class MultiStageTreePipeline(PipelineBase):
    """Implements conf/multistagetree.batchy

    Arguments:
        leaves: Number of leaves
        weight1: Weight of first branch split
        cycles_per_batch: Bypass param
        cycles_per_batch1: 1st branch Bypass param
        cycles_per_packet: Bypass param
        cycles_per_packet1: 1st branch Bypass param
        delay_bound: Delay SLO of flows
        delay_bound1: Delay SLO of the 1st flow
        packet_bound: Rate SLO of flows
        packet_bound1: Rate SLO of the 1st flow
        mode: RTC or WFQ
        rounds: Number of control rounds
        control_period: Interval of control periods
        controller: Controller type

    Attributes:
        gates_weights: WeightedRandomSplit config
    """

    def __init__(self, *args, **kwargs):
        super(MultiStageTreePipeline, self).__init__(*args, **kwargs)

    def make_config(self):
        locals().update(vars(self))
        gates_weights = {i: 1.0 for i in range(1, leaves)}
        gates_weights[0] = weight1
        vars(self).update(locals())

    def build_pipeline(self):
        locals().update(vars(self))
        w0 = batchy.add_worker('w0')
        t0 = w0.add_task('task0', type=mode)
        bp_in = t0.add_module(Bypass(name='bp_ingress'),
                              type='ingress')
        bp1 = t0.add_module(Bypass(cycles_per_batch=cycles_per_batch1,
                                   cycles_per_packet=cycles_per_packet1),
                            cycles_per_batch1 / settings.BYPASS_RATIO,
                            cycles_per_packet1 / settings.BYPASS_RATIO)
        r1 = t0.add_module(WeightedRandomSplit(gates_weights={0: 1.0, 1: 1.0},
                                               drop_rate=0.0))

        bp11 = t0.add_module(Bypass(cycles_per_batch=cycles_per_batch1,
                                    cycles_per_packet=cycles_per_packet1),
                             cycles_per_batch1 / settings.BYPASS_RATIO,
                             cycles_per_packet1 / settings.BYPASS_RATIO)

        bp_in.connect(bp1)
        bp1.connect(r1)
        r1.connect(bp11, ogate=0)

        r2 = t0.add_module(WeightedRandomSplit(gates_weights=gates_weights,
                                               drop_rate=0.0))

        bp21 = t0.add_module(Bypass(cycles_per_batch=cycles_per_batch1,
                                    cycles_per_packet=cycles_per_packet1),
                             cycles_per_batch1 / settings.BYPASS_RATIO,
                             cycles_per_packet1 / settings.BYPASS_RATIO)
        bp11.connect(r2)
        r2.connect(bp21, ogate=0)

        batchy.add_flow(name='flow0',
                        path=[{'task': t0,
                               'path': [bp_in, bp1, r1, bp11, r2, bp21]}],
                        delay_slo=delay_bound1,
                        source_params={'rate_limit':
                                       {'packet': packet_bound1}})

        for i in range(1, leaves):
            bp = t0.add_module(Bypass(cycles_per_batch=cycles_per_batch,
                                      cycles_per_packet=cycles_per_packet),
                               cycles_per_batch / settings.BYPASS_RATIO,
                               cycles_per_packet / settings.BYPASS_RATIO)
            r2.connect(bp, ogate=i)
            batchy.add_flow(name=f'flow{i:d}',
                            path=[{'task': t0,
                                   'path': [bp_in, bp1, r1, bp11, r2, bp]}],
                            delay_slo=delay_bound,
                            source_params={'rate_limit':
                                           {'packet': packet_bound}})

        bp12 = t0.add_module(Bypass(cycles_per_batch=cycles_per_batch,
                                    cycles_per_packet=cycles_per_packet),
                             cycles_per_batch / settings.BYPASS_RATIO,
                             cycles_per_packet / settings.BYPASS_RATIO)
        r1.connect(bp12, ogate=1)
        batchy.add_flow(name=f'flow{leaves:d}',
                        path=[{'task': t0, 'path': [bp_in, bp1, r1, bp12]}],
                        delay_slo=delay_bound,
                        source_params={'rate_limit':
                                       {'packet': packet_bound}})

        vars(self).update(locals())

    def set_controllers(self):
        locals().update(vars(self))
        t0.set_controller(batchy.resolve_task_controller(controller))

    def add_test_traffic(self):
        batchy.add_source()
        batchy.add_sink()


class L2L3Pipeline(PipelineBase):
    """Implements conf/l2l3.batchy

    Arguments:
        acl: Add ACL modules to pipeline
        acl_module: ACL module type (ACL/DPDKACL)
        nat: Add NAT modules to pipeline
        burst_size: Bulk source burst size
        src_worker_num: Number of source workers
        fib_size: Size of IPLookup table
        acl_size: Number of ACL rules
        branch_num: Number of VLANs
        nhop_num: Number of next-hops per VLAN
        max_delay: Delay SLO of flows
        max_delay1: Delay SLO of the 1st flow
        mode: RTC or WFQ
        rounds: Number of control rounds
        control_period: Interval of control periods
        controller: Controller type

    Attributes:
        flow_params: parameters of flows
        fib: IPLookup rules
    """

    def __init__(self, *args, **kwargs):
        super(L2L3Pipeline, self).__init__(*args, **kwargs)

    @staticmethod
    def gen_packet(proto, src_ip, dst_ip, src_port=10001, dst_port=10002,
                   src_mac='02:1e:67:9f:4d:ae', dst_mac='ae:34:2e:16:b8:cd',
                   vlan_tci=None):
        eth = scapy.Ether(src=src_mac, dst=dst_mac)
        ip = scapy.IP(src=src_ip, dst=dst_ip)
        udp = proto(sport=src_port, dport=dst_port)
        payload = 'helloworld'
        if vlan_tci is None:
            pkt = eth/ip/udp/payload
        else:
            dotq = scapy.Dot1Q(vlan=vlan_tci)
            pkt = eth/dotq/ip/udp/payload
        return bytes(pkt)

    def make_config(self):
        locals().update(vars(self))
        flow_params = []
        for b in range(branch_num):
            for n in range(nhop_num):
                i = b * nhop_num + n
                dmac = f'11:77:{":".join(["%02x" % x for x in get_octets(b, last=1)])}'
                templates = [self.gen_packet(scapy.UDP,
                                             '10.0.0.1', get_ip(i, offset=20),
                                             dst_mac=dmac)]
                new_flow = {
                    'name': 'flow_%d' % i,
                    'id': i,
                    'delay_slo': max_delay,
                    'rate_slo': {'packet': int(1e9)},
                    'path': [],
                    'flow': None,
                    'source_params': {'limit': {'packet': int(1e9)},
                                      'templates': templates,
                                      'weight': 1}}
                flow_params.append(new_flow)
        fib = []
        for i in range(fib_size - len(flow_params)):
            entry = {
                'prefix': get_ip(i, offset=100),
                'prefixlen': 24,
                'ogate': i % nhop_num}
            fib.append(entry)
            for flow in flow_params:
                i = flow['id']
                entry = {
                    'prefix': get_ip(i, offset=20),
                    'prefixlen': 24,
                    'ogate': i % nhop_num}
                fib.append(entry)
        vars(self).update(locals())

    def build_pipeline(self):
        locals().update(vars(self))
        w0 = batchy.add_worker('w0')
        t0 = w0.add_task('task0', type=mode)
        type_check = t0.add_module(ExactMatch(name='type_check',
                                              fields=[{'offset': 12,
                                                      'num_bytes': 2}]),
                                   type='ingress')
        type_check.module.add(fields=[{'value_bin': struct.pack("!H", 0x0800)}],
                              gate=0)

        split = t0.add_module(ExactMatch(name='split',  # dst mac
                                         fields=[{'offset': 0, 'num_bytes': 6}]))
        for b in range(branch_num):
            mac_str = f'11:77:{":".join(["%02x" % x for x in get_octets(b, last=1)])}'
            split.module.add(fields=[{'value_bin': mac_from_str(mac_str)}],
                             gate=b)
        type_check.connect(split)

        if enable_acl:
            # acl_rules = [{'src_ip': '0.0.0.0/0',
            #               'dst_ip': '0.0.0.0/0',
            #               'ipproto': 17,
            #               'drop': False}] + gen_acl_rules(acl_size - 1)
            acl_rules = gen_acl_rules(acl_size - 2) + [{'src_ip': '0.0.0.0/0',
                                                        'dst_ip': '0.0.0.0/0',
                                                        'ipproto': 17,
                                                        'drop': False},
                                                       {'src_ip': '0.0.0.0/0',
                                                        'dst_ip': '0.0.0.0/0',
                                                        'ipproto': 6,
                                                        'drop': False}]
        if enable_nat:
            addr_pairs = [{'int_range':
                           {'start': '10.0.0.0', 'end': '10.0.0.255'},
                           'ext_range':
                           {'start': '20.0.0.0', 'end': '20.0.0.255'}}]

        for vn in range(branch_num):
            l3 = t0.add_module(IPLookup(name='L3_%s' % vn, max_rules=fib_size),
                               T_0=ip_t0, T_1=ip_t1)
            for entry in fib:
                l3.module.add(prefix=entry['prefix'],
                              prefix_len=entry['prefixlen'],
                              gate=entry['ogate'])
            split.connect(l3, ogate=vn)
            for n in range(nhop_num):
                fid = vn * nhop_num + n
                ttl = t0.add_module(UpdateTTL(name='update_ttl_%s' % fid),
                                    T_0=ttl_t0, T_1=ttl_t1)
                src_mac = mac_from_str('aa:bb:bb:aa:%02x:dc' % (fid % 256),
                                       mode='int')
                dst_mac = mac_from_str('ee:dd:dd:aa:01:%02x' % (fid % 256),
                                       mode='int')
                mac = t0.add_module(Update(name='update_mac_%s' % fid,
                                           fields=[{'offset': 6, 'size': 6,
                                                    'value': src_mac},
                                                   {'offset': 0, 'size': 6,
                                                    'value': dst_mac}]))
                chk = t0.add_module(IPChecksum(name='ip_checksum_%s' % fid))
                flow_params[fid]['path'] = [type_check, split, l3]

                last_mod = l3
                last_gate = n
                if enable_acl:
                    acl = t0.add_module(acl_module(name='ACL_%d' % fid,
                                                   rules=acl_rules),
                                        T_0=acl_t0, T_1=acl_t1)
                    l3.connect(acl, ogate=n)
                    flow_params[fid]['path'].append(acl)
                    last_mod = acl
                    last_gate = 0

                if enable_nat:
                    nat = t0.add_module(StaticNAT(name='NAT_%d' % fid,
                                                  pairs=addr_pairs),
                                        T_0=nat_t0, T_1=nat_t1)
                    last_mod.connect(nat, last_gate)
                    flow_params[fid]['path'].append(nat)
                    last_mod = nat
                    last_gate = 1

                last_mod.connect(ttl, last_gate)
                ttl.connect(mac)
                mac.connect(chk)
                flow_params[fid]['path'].extend([ttl, mac, chk])

        for flow_param in flow_params:
            new_flow = batchy.add_flow(name=flow_param['name'],
                                       path=[{'task': t0,
                                              'path': flow_param['path']}],
                                       delay_slo=flow_param['delay_slo'],
                                       rate_slo=flow_param['rate_slo'],
                                       source_params=flow_param['source_params'])
            flow_param['flow'] = new_flow
        flow_params[0]['flow'].D = max_delay1

        vars(self).update(locals())

    def set_controllers(self):
        locals().update(vars(self))
        t0.set_controller(batchy.resolve_task_controller(controller))

    def add_test_traffic(self):
        for i in range(src_worker_num):
            src_worker_name = f'src_{i:d}'
            src_worker = batchy.add_worker(src_worker_name)

            if next(iter(rate_slo.values())) < 0:
                # not rate limitation, add everything as bulk
                batchy.add_bulk_source(batchy.flows, src_worker, burst_size=burst_size)
            else:
                # add QoS flows for everything except the last bearer (bulk)
                bulk_flows = []
                qos_flows = []
                for fp in flow_params:
                    f = fp['flow']
                    if f.has_rate_slo():
                        qos_flows.append(f)
                    else:
                        bulk_flows.append(f)

                if qos_flows:
                    batchy.add_qos_source(qos_flows, src_worker)
                if bulk_flows:
                    batchy.add_bulk_source(bulk_flows, src_worker)

        batchy.add_sink()

    def update_qos_rate(self, resource='packet', limit=100_000):
        shares = {getattr(c,'class').name: getattr(c,'class').share
                  for c in bess.list_tcs().classes_status}
        for i in range(src_worker_num):
            src_worker = batchy.get_worker(f'src_{i}')
            for t in src_worker.tasks:
                if 'qos_multi' in t.name:
                    tc = f'tc_qos:{t.name}'
                    l = limit / len(flow_params) * shares[tc]
                    bess.update_tc_params(tc,
                                          resource=resource,
                                          limit={resource: int(l)})



class MGWPipeline(PipelineBase):
    """Implements conf/mgwcomplex.batchy

    Arguments:
        mode: RTC or WFQ
        rounds: Number of control rounds
        control_period: Interval of control periods
        controller: Controller type
        src_worker_num: Number of source workers
        fib_size: Size of IPLookup table
        user_num: Number of users
        bearer_num: Number of bearers
        bearer0_user: Number of users at bearer 0
        rate_slo: Rate limit in bps
        delay_slo: Delay SLO of bearer 0
        max_weight: Max weight of a flow
        bulk_limit: Cumulative rate limit on bulk traffic
        normalized: Whether the pipeline is decomposed into tasks:
                    0: single task,
                    1: ingress + /flow tasks,
                    2: separate QoS and bulk tasks
        t0_dl: Downstream Bypass parameter
        t1_dl: Downstream Bypass parameter
        t0_ul: Upstream Bypass parameter
        t1_ul: Upstream Bypass parameter

    Attributes:
        gw: Gateway config (IP addr and MAC addr)
        users: Paramteres of users
        fib: IPLookup rules
        controller_name: Controller class name
    """

    def __init__(self, *args, **kwargs):
        super(MGWPipeline, self).__init__(*args, **kwargs)

    @staticmethod
    def gen_packet(proto, src_ip, dst_ip, src_port=10001, dst_port=10002,
                   vxlan_id=None):
        gw = {'ip': '1.10.100.1', 'mac': 'ac:dc:ac:dc:02:9a'}  # FIXME
        eth = scapy.Ether(src='02:1e:67:9f:4d:ae', dst='ae:34:2e:16:b8:cd')
        ip = scapy.IP(src=src_ip, dst=dst_ip)
        udp = proto(sport=src_port, dport=dst_port)
        payload = 'helloworld'
        if vxlan_id is not None:
            vxeth = scapy.Ether(src='02:1d:61:3f:4d:ae', dst=gw['mac'], type=0x0800)
            vx_ip = scapy.IP(src=src_ip, dst=gw['ip'])
            vxudp = scapy.UDP(sport=4789, dport=4789)
            vxlan = scapy.VXLAN(vni=vxlan_id, flags=0x08)
            pkt = vxeth/vx_ip/vxudp/vxlan/eth/ip/udp/payload
        else:
            pkt = eth/ip/udp/payload
        return bytes(pkt)

    @staticmethod
    def get_delay_slo_for_bearer(br, dir):
        if br < bearer_num - 1:
            return delay_slo * (br + 1)
        return None

    @staticmethod
    def get_rate_slo_for_bearer(br, dir):
        if not rate_slo < 0 and br < bearer_num - 1:
            return {'packet': rate_slo * (br + 1)}
        return None

    @staticmethod
    def get_weight_for_bearer(br, dir):
        if rate_slo < 0:
            if br < bearer_num - 1:
                return min(br + 1, max_weight)
            return max_weight
        return 1

    @staticmethod
    def get_br_uid(uid, br):
        return uid * bearer_num + br


    def make_config(self):
        vxlan_size = len(scapy.Ether()/scapy.VXLAN()/scapy.IP()/scapy.UDP())

        gw = {'ip': '1.10.100.1',
              'mac': 'ac:dc:ac:dc:02:9a'}

        em_t0 = 2
        em_t1 = 1

        ip_t0 = 2
        ip_t1 = 1

        users = []
        for i in range(user_num):
            new_user = {
                'name': f'user_{i}',
                'id': i,
                'path': {'ul': {}, 'dl': {}},
                'task': {'ul': {}, 'dl': {}},
                'flow': {'ul': {}, 'dl': {}},
                'ip': {'ul': get_ip(i, offset=10), 'dl': get_ip(i, offset=10)},
                'source_params': {
                    'ul': {
                        'ts_offset': settings.FLOW_TIMESTAMP_OFFSET + vxlan_size,
                        'ts_offset_diff': -vxlan_size,
                        'weight': 1,
                        'templates': [self.gen_packet(scapy.UDP,            # per bearer
                                                      get_ip(i, offset=10),
                                                      get_ip(i, offset=100),
                                                      dst_port=1000 + i,
                                                      vxlan_id=br)
                                      for br in range(bearer_num)]},
                    'dl': {
                        'ts_offset': settings.FLOW_TIMESTAMP_OFFSET,
                        'ts_offset_diff': +vxlan_size,
                        'weight': 1,
                        'templates': [self.gen_packet(scapy.UDP,            # per bearer
                                                      get_ip(i, offset=100),
                                                      get_ip(i, offset=10),
                                                      src_port=10 + br,
                                                      dst_port=1000 + i)
                                      for br in range(bearer_num)]},
                }
            }
            users.append(new_user)

        fib = []
        for i in range(user_num):
            dl_entry = {
                'prefix': get_ip(i, offset=10, last=0),
                'prefixlen': 24,
                'ogate': 1,
            }
            fib.append(dl_entry)
            ul_entry = {
                'prefix': get_ip(i, offset=100, last=0),
                'prefixlen': 24,
                'ogate': 0,
            }
            fib.append(ul_entry)
        for i in range(fib_size - 2 * user_num):
            dummy_entry = {
                'prefix': get_ip(i, offset=150, last=0),
                'prefixlen': 24,
                'ogate': i % 2,
            }
            fib.append(dummy_entry)

        vars(self).update(locals())

    def build_pipeline(self):
        w0 = batchy.add_worker('w0')
        ti = w0.add_task('task_ingress', type=mode)
        mtype = 'ingress' if normalized > 0 else 'internal'

        for bearer in range(bearer_num):
            for i, user in enumerate(users):
                if bearer == 0 and i >= bearer0_user:
                        break
                for dir in ('dl', 'ul'):
                    if normalized == 0:
                        user['task'][dir][bearer] = ti
                    elif normalized == 1:
                        t = w0.add_task(f'task_{user["name"]}_{bearer}_{dir}',
                                        type=mode)
                        user['task'][dir][bearer] = t
                    else:
                        # qos bearers: ingress task, bulk: own task
                        if bearer < bearer_num - 1:
                            user['task'][dir][bearer] = ti
                        else:
                            t = w0.add_task(f'task_{user["name"]}_{bearer}_{dir}',
                                            type=mode)
                            user['task'][dir][bearer] = t

        mac_table = ti.add_module(ExactMatch(name='mac_table',
                                             fields=[{'offset': 0,
                                                      'num_bytes': 6}]),
                          type='ingress')
        mac_table.module.add(fields=[{'value_bin':
                                      mac_from_str('ae:34:2e:16:b8:cd')}],
                             gate=0)
        mac_table.module.set_default_gate(gate=0)

        type_check = ti.add_module(ExactMatch(name='type_check',
                                              fields=[{'offset': 12, 'num_bytes': 2}]))
        type_check.module.add(fields=[{'value_bin':
                                       struct.pack('!H', 0x0800)}],
                              gate=0)
        type_check.module.set_default_gate(gate=0)
        mac_table.connect(type_check)

        dir_selector = ti.add_module(ExactMatch(name='dir_selector',
                                                fields=[{'offset': 30,
                                                         'num_bytes': 4},    # dst IP
                                                        {'offset': 23,
                                                         'num_bytes': 1},    # IP proto
                                                        {'offset': 36,
                                                         'num_bytes': 2}]),  # dst port
                                     T_0=em_t0, T_1=em_t1, controlled=False)
        dir_selector.module.add(fields=[{'value_bin': aton(gw['ip'])},
                                        {'value_bin': struct.pack('B', 17)},
                                        {'value_bin': struct.pack('!H', 4789)}],
                                gate=1)  # uplink
        dir_selector.module.set_default_gate(gate=0) # downlink
        type_check.connect(dir_selector)

        # in the downlink direction, bearers are selected by TFT/SDF templates, ie,
        # fully-fledged 5-tuple packet filters, we substitute these for source port
        # based mappings
        dl_br_selector = ti.add_module(ExactMatch(name='dl_br_selector',
                                                  fields=[{'offset': 34,
                                                           'num_bytes': 2}]),  # src port
                                       T_0=em_t0, T_1=em_t1, controlled=False)
        dl_br_selector.module.set_default_gate(gate=0)
        dir_selector.connect(dl_br_selector, ogate=0)

        vxlan_decap = ti.add_module(VXLANDecap(name='vxlan_decap'))
        # in the uplink direction, bearers are selected by VXLAN/GTP teid
        ul_br_selector = ti.add_module(Split(name='ul_br_selector',
                                     attribute='tun_id', size=4))
        dir_selector.connect(vxlan_decap, ogate=1)
        vxlan_decap.connect(ul_br_selector)

        common_path_dl = [mac_table, type_check, dir_selector, dl_br_selector]
        common_path_ul = [mac_table, type_check, dir_selector,
                          vxlan_decap, ul_br_selector]

        ttl = ti.add_module(UpdateTTL(name=f'update_ttl'))

        for bearer in range(bearer_num):
            # dl: src_port
            dl_br_selector.module.add(fields=[{'value_bin':
                                               struct.pack("!H", 10 + bearer)}],
                                      gate=bearer)
            # ul: tun_id, handled by Split

            # user selectors
            dl_ue_selector = ti.add_module(ExactMatch(name=f'dl_ue_selector_{bearer}',
                                                      fields=[{'offset': 30,
                                                               'num_bytes': 4}]), # dst IP
                                           T_0=em_t0, T_1=em_t1, controlled=False)
            dl_br_selector.connect(dl_ue_selector, ogate=bearer)

            ul_ue_selector = ti.add_module(ExactMatch(name=f'ul_ue_selector_{bearer}',
                                                      fields=[{'offset': 26,
                                                               'num_bytes': 4}]), # src IP
                                           T_0=em_t0, T_1=em_t1, controlled=False)
            ul_br_selector.connect(ul_ue_selector, ogate=bearer)


            for i, user in enumerate(users):
                if bearer == 0 and i >= bearer0_user:
                    break
                for dir in ('dl', 'ul'):
                    selector = dl_ue_selector if dir == 'dl' else ul_ue_selector
                    selector.module.add(fields=[{'value_bin': aton(user['ip'][dir])}],
                                        gate=user['id'])
                    common_path = locals().get(f'common_path_{dir}').copy()
                    common_path.append(selector)
                    t = user['task'][dir][bearer]
                    bp_per_batch = globals().get(f't0_{dir}')
                    bp_per_packet = globals().get(f't1_{dir}')

                    if normalized == 0:
                        type = 'internal'
                    elif normalized == 1:
                        type = 'ingress'
                    else:
                        if bearer < bearer_num - 1:
                            type = 'internal'
                        else:
                            type = 'ingress'
                    bp = t.add_module(Bypass(name=f'{dir}_{user["name"]}_{bearer}_bp',
                                             cycles_per_batch=bp_per_batch,
                                             cycles_per_packet=bp_per_packet),
                                      T_0=bp_per_batch/settings.BYPASS_RATIO,
                                      T_1=bp_per_packet/settings.BYPASS_RATIO,
                                      controlled=True, type=type)
                    selector.connect(bp, ogate=user['id'])

                    _uid_br = self.get_br_uid(user['id'], bearer).to_bytes(4, byteorder='big')
                    if dir == 'dl':
                        md_attrs = [{'name': 'tun_id', 'size': 4,
                                     'value_int': user['id']},
                                    {'name': 'tun_ip_src', 'size': 4,
                                     'value_bin': aton(gw['ip'])},
                                    {'name': 'tun_ip_dst', 'size': 4,
                                     'value_bin': aton(user['ip']['dl'])},
                                    {'name': 'ether_src', 'size': 6,
                                     'value_bin': b'\x02\x01\x02\x03\x04\x05'},
                                    {'name': 'ether_dst', 'size': 6,
                                     'value_bin': b'\x02\x0a\x0b\x0c\x0d\x0e'},
                                    {'name': 'uid_br', 'size': 4,
                                     'value_bin': _uid_br}]

                        set_md = t.add_module(SetMetadata(name=f'setmd_dl_{user["id"]}_{bearer}',
                                                          attrs=md_attrs))

                        vxlan_encap = t.add_module(VXLANEncap(name=f'vxlan_encap_{user["id"]}_{bearer}',
                                                              dstport=1000 + user['id']))
                        ip_encap = t.add_module(IPEncap(name=f'ip_encap_{user["id"]}_{bearer}'))
                        eth_encap = t.add_module(EtherEncap(name=f'ether_encap_{user["id"]}_{bearer}'))

                        bp.connect(set_md)
                        set_md.connect(vxlan_encap)
                        vxlan_encap.connect(ip_encap)
                        ip_encap.connect(eth_encap)
                        eth_encap.connect(ttl)

                        path = [bp, set_md, vxlan_encap, ip_encap, eth_encap]
                        common_path.append(dl_ue_selector)

                    else:
                        md_attrs = [{'name': 'uid_br', 'size': 4,
                                     'value_bin': _uid_br}]
                        set_md = t.add_module(SetMetadata(name=f'setmd_ul_{user["id"]}_{bearer}',
                                                          attrs=md_attrs))

                        bp.connect(set_md)
                        set_md.connect(ttl)
                        path = [ul_ue_selector, bp, set_md]
                        common_path.append(ul_ue_selector)

                    if normalized == 0:
                        user['path'][dir][bearer] = [{'task': ti,
                                                      'path': common_path + path}]
                    elif normalized == 1:
                        user['path'][dir][bearer] = [{'task': ti, 'path': common_path},
                                                     {'task': t, 'path': path}]
                    else:
                        if bearer < bearer_num - 1:
                            user['path'][dir][bearer] = [{'task': ti,
                                                          'path': common_path + path}]
                        else:
                            user['path'][dir][bearer] = [{'task': ti, 'path': common_path},
                                                         {'task': t, 'path': path}]

        # these all go to the ingress task
        l3 = ti.add_module(IPLookup(name='L3', max_rules=fib_size),
                           T_0=ip_t0, T_1=ip_t1)
        ttl.connect(l3)
        for entry in fib:
            l3.module.add(prefix=entry['prefix'],
                          prefix_len=entry['prefixlen'],
                          gate=entry['ogate'])

        nhops = {'ul': {'smac': 'aa:bb:bb:aa:ac:dc',
                        'dmac': 'ee:dd:dd:aa:01:01',
                        'ogate': 0,
                        'path': None},
                 'dl': {'smac': 'aa:bb:bb:aa:ac:dc',
                        'dmac': 'ee:dd:dd:aa:02:02',
                        'ogate': 1,
                        'path': None}}
        for dir, cfg in nhops.items():
            src_mac = mac_from_str(cfg['smac'], mode='int')
            dst_mac = mac_from_str(cfg['dmac'], mode='int')
            mac = ti.add_module(Update(name=f'update_mac_{dir}',
                                       fields=[{'offset': 6, 'size': 6,
                                                'value': src_mac},
                                               {'offset': 0, 'size': 6,
                                                'value': dst_mac}]))
            chk = ti.add_module(IPChecksum(name=f'ip_checksum_{dir}'))
            flow_demuxer = ti.add_module(Split(name=f'flow_demuxer_{dir}',
                                               attribute='uid_br', size=4))
            l3.connect(mac, ogate=cfg['ogate'])
            mac.connect(chk)
            chk.connect(flow_demuxer)
            cfg['path'] = [ttl, l3, mac, chk, flow_demuxer]

            for bearer in range(bearer_num):
                for i, user in enumerate(users):
                    if bearer == 0 and i >= bearer0_user:
                        break
                    dmux_bp = ti.add_module(Bypass(name=f'demux_bp_{dir}_{user["name"]}_{bearer}'))
                    flow_demuxer.connect(dmux_bp,
                                         ogate=self.get_br_uid(user['id'], bearer))
                    # append the egress pipeline to each flow's last taskflow (ugly)
                    user['path'][dir][bearer][-1]['path'].extend(cfg['path'] + [dmux_bp])

        for bearer in range(bearer_num):
            for i, user in enumerate(users):
                if bearer == 0 and i >= bearer0_user:
                    break
                for dir in ('dl', 'ul'):
                    if not normalized:
                        path = list(itertools.chain(*((v['path'] for v in user['path'][dir][bearer]))))
                        user['path'][dir][bearer] = [{'task': ti, 'path': path}]
                    src_br = user['source_params'][dir].copy()
                    src_br['templates'] = [src_br['templates'][bearer]]
                    src_br['weight'] = self.get_weight_for_bearer(bearer, dir)
                    src_br['burst_size'] = 1 # relevant only for non-bulk
                    src_br['ts_offset'] = user['source_params'][dir]['ts_offset']
                    new_flow = batchy.add_flow(name=f'{user["name"]}_{bearer}_{dir}',
                                               path=user['path'][dir][bearer],
                                               delay_slo=self.get_delay_slo_for_bearer(bearer, dir),
                                               rate_slo=self.get_rate_slo_for_bearer(bearer, dir),
                                               source_params=src_br)
                    user['flow'][dir][bearer] = new_flow

        vars(self).update(locals())

    def set_controllers(self):
        locals().update(vars(self))
        controller_name = batchy.resolve_task_controller(controller)
        ti.set_controller(controller_name)
        if normalized > 0:
            for bearer in range(bearer_num):
                for i, user in enumerate(users):
                    if bearer == 0 and i >= bearer0_user:
                        break
                    for dir in ('dl', 'ul'):
                        t = user['task'][dir][bearer]
                        t.set_controller(controller_name)
        vars(self).update(locals())


    def add_test_traffic(self):
        for i in range(src_worker_num):
            src_worker_name = f'src_{i}'
            src_worker = batchy.add_worker(src_worker_name)
            src_worker.set_bulk_ratelimit(bulk_limit)
            for dir in ('dl', 'ul'):
                ts_offset = users[0]['source_params'][dir]['ts_offset']

                if rate_slo < 0:
                    # not rate limitation, add everything as bulk
                    batchy.add_bulk_source([f for f in batchy.flows if dir in f.name],
                                           src_worker, postfix=dir, ts_offset=ts_offset)
                else:
                    # add QoS flows for everything except the last bearer (bulk)
                    bulk_flows = []
                    qos_flows = []
                    for user in users:
                        for _, uflow in user['flow'][dir].items():
                            if uflow.has_rate_slo():
                                qos_flows.append(uflow)
                            else:
                                bulk_flows.append(uflow)

                    if qos_flows:
                        batchy.add_qos_source(qos_flows, src_worker, postfix=dir,
                                              ts_offset=ts_offset)
                    if bulk_flows:
                        batchy.add_bulk_source(bulk_flows, src_worker, postfix=dir,
                                               ts_offset=ts_offset)

        batchy.add_sink()

    def update_qos_rate(self, resource='packet', limit=100_000):
        for i in range(src_worker_num):
            src_worker_name = f'src_{i}'
            src_worker = batchy.get_worker(src_worker_name)
            for t in src_worker.tasks:
                if 'qos_multi' in t.name:
                    bess.update_tc_params(f'tc_qos:{t.name}',
                                          resource=resource,
                                          limit={resource: limit})
