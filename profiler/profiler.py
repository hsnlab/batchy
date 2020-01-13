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

import argparse
import math
import re
import subprocess
from pathlib import Path
from scipy import stats


def build_cmd(bessdir, module, bessd_args=None, extra_args=None, mode='str'):
    if bessd_args is None:
        bessd_args = ''
    if extra_args is None:
        extra_args = ''
    bessctl = Path(bessdir).resolve() / 'bessctl' / 'bessctl'
    cur_dir = Path(__file__).parent.resolve()
    bess_file = cur_dir / 'profile.bess'
    acl_file = cur_dir / 'acl1_seed_1000.rules'
    profiler_file = cur_dir.parent / 'profiler_results.json'
    args = f'runs=1,module=\'{module}\',acl_file=\'{acl_file}\',' \
           f'out_file=\'{profiler_file}\',{extra_args}'
    cmd = f'{bessctl} daemon start {bessd_args} -- run file {bess_file} "{args}"'
    if mode == 'list':
        cmd = cmd.split(' ')
    return cmd


def run_cmd(cmd, shell=True, _try_counter=0):
    try:
        comp_proc = subprocess.run(cmd, shell=shell, check=True,
                                   stdout=subprocess.PIPE)
        retval = comp_proc.stdout
    except subprocess.CalledProcessError:
        c = _try_counter + 1
        if c > 32:
            raise Exception(f'Too many errors occured during running {cmd}')
        retval = run_cmd(cmd, shell=shell, _try_counter=c)
    return retval


def process_output(output, table_title='Averaged results:'):
    pattern = re.compile(f'{table_title}\\n(.*)\\n\\n',
                         re.MULTILINE | re.DOTALL)
    table_content = pattern.search(str(output), re.MULTILINE | re.DOTALL)
    table = []
    try:
        for line in table_content.group(0).split('\n'):
            if '|' in line:
                try:
                    row = tuple(float(item) for item
                                in line.split('|') if item)
                    table.append(row)
                except ValueError:
                    continue
    except AttributeError:
        raise Exception('No averaged table was found on output')
    return table


def process_log(logfile):
    results = []
    with open(logfile, 'r') as f:
        data = ''.join(f.readlines())
    chunks = data.split('Results:')
    for chunk in chunks:
        try:
            table = process_output(chunk, '')
            results.append(table)
        except:
            continue
    results_avg = calc_avg(results)
    results_err = calc_error(results, results_avg)
    return {'results_avg': results_avg,
            'results_error': results_err}


def calc_avg(results, max_batch=32):
    results_avg = []
    for b in range(max_batch):
        tmp = [int(results[0][b][0])]  # batch size
        for i in range(1, len(results[0][0])):
            res_sum = sum([results[r][b][i] for r in range(len(results))])
            tmp.append(res_sum / len(results))
        results_avg.append(tmp)
    return results_avg


def calc_error(results, results_avg=None, max_batch=32):
    if results_avg is None:
        results_avg = calc_avg(results, max_batch)
    results_error = []
    for b in range(max_batch):
        tmp = [int(results[0][b][0])]  # batch size
        for i in range(1, len(results[0][0])):
            res_sd_sum = sum([pow(results[r][b][i] - results_avg[b][i], 2)
                              for r in range(len(results))])
            tmp.append(math.sqrt(res_sd_sum / (len(results) - 1)) /
                       math.sqrt(len(results)))
        results_error.append(tmp)
    return results_error


def format_table(results, mode='org'):
    spec_chars = {}
    spec_chars['org'] = {'sep': ' | ', 'eol': ' |\n', 'sol': '| '}
    spec_chars['csv'] = {'sep': ',', 'eol': '\n'}
    spec_chars['tsv'] = {'sep': '\t', 'eol': '\n'}
    spec_chars['tex'] = {'sep': ' & ', 'eol': '\\\\n'}
    sep = spec_chars[mode].get('sep', ' ')  # separator
    eol = spec_chars[mode].get('eol', '\n')  # end of line
    sol = spec_chars[mode].get('sol', '')  # start of line
    columns = ('batch_size', 'batches_bps', 'pkts_Mpps', 'bits_Mbps',
               'avg_ns', '1st_ns', 'med_ns', '75th_ns', '95th_ns', '99th_ns')
    header = '%s%s%s' % (sol, sep.join(columns), eol)
    format_list = [s for subl in [['%d'], ['%.3f'] * 9] for s in subl]
    result_format = f'{sol}{sep.join(format_list)}{eol}'
    rows = (result_format % tuple(result) for result in results)
    table = header
    table += ''.join(rows)
    return table


def get_maxbatch_from_arg(arg):
    try:
        pattern = re.compile(r'max_batch=(\d+)')
        return int(pattern.search(arg).group(1))
    except (AttributeError, TypeError):
        return 32


def get_avg_results(bessdir, module, runs, bessd_args=None, extra_args=None,
                    verbose=False):
    cmd = build_cmd(bessdir, module,
                    bessd_args=bessd_args, extra_args=extra_args)
    results = []
    for i in range(1, runs + 1):
        cmd_stdout = run_cmd(cmd)
        table = process_output(cmd_stdout.decode())
        results.append(table)
        if verbose:
            print(f'{module} :: {i}/{runs} done.')
            print(f'Results:\n{format_table(table)}')
    return calc_avg(results, get_maxbatch_from_arg(extra_args))


def get_result_column(column, table):
    columns = ('batch_size', 'batches_bps', 'pkts_Mpps', 'bits_Mbps',
               'avg_ns', '1st_ns', 'med_ns', '75th_ns', '95th_ns', '99th_ns')
    idx = columns.index(column)
    return [row[idx] for row in table]


def calc_lin_reg(result_column, name='', max_batch=32):
    linreg_params = stats.linregress(range(1, max_batch + 1), result_column)
    slope, intercept, r_value, p_value, std_err = linreg_params
    return {'name': name, 'intercept': intercept, 'slope': slope,
            'r_value': r_value, 'p_value': p_value, 'std_err': std_err}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bessdir', '-b',
                        type=str, required=True,
                        help='BESS root directory')
    parser.add_argument('--bessd_args', '-a',
                        type=str,
                        help='bessd cli args, eg. -m 2048')
    parser.add_argument('--module', '-m',
                        type=str, default='ExactMatch',
                        help='BESS module to test')
    parser.add_argument('--runs', '-r',
                        type=int, default=10,
                        help='Number of measurements')
    parser.add_argument('--extra_args', '-e',
                        type=str,
                        help=('Args for profile.bess, '
                              'eg. rate_limit=1000000000'))
    parser.add_argument('--format', '-f',
                        type=str, default='org',
                        choices=['org', 'csv', 'tsv', 'tex'],
                        help='Format to print table')
    parser.add_argument('--outfile', '-o',
                        type=argparse.FileType('w'),
                        help='File to write results')
    parser.add_argument('--logfile', '-l',
                        type=argparse.FileType('r'),
                        help='Previous run\'s logfile to use reading results')
    args = parser.parse_args()

    if args.logfile:
        res = process_log(args.logfile.name)
        avg_table = format_table(res['results_avg'], args.format)
        err_table = format_table(res['results_error'], args.format)
        out_str = f'Results AVERAGE:\n{avg_table}\n' \
                  f'Results ERROR:\n{err_table}'
        if args.outfile:
            args.outfile.write(out_str)
        else:
            print(out_str)
        exit()

    avg_results = get_avg_results(args.bessdir, args.module, args.runs,
                                  args.bessd_args, args.extra_args,
                                  verbose=True)

    table_str = format_table(avg_results, args.format)
    if args.outfile:
        args.outfile.write(table_str)
    else:
        print(table_str)

    for col in ('avg_ns', '95th_ns', '99th_ns'):
        avg_res_col = get_result_column(col, avg_results)
        print(calc_lin_reg(avg_res_col, col,
                           max_batch=get_maxbatch_from_arg(args.extra_args)))
