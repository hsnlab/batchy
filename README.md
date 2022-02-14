# Batchy

[Overview](#overview) | [Installation](#installation) | [Usage](#usage) | [Examples](#examples) | [Caveats and Limitations](#caveats-and-limitations) | [License](#license)


## Overview

Batchy is a a scheduler for run-to-completion packet processing engines, which uses controlled queuing to efficiently reconstruct fragmented batches in accordance with strict service-level objectives (SLOs).

Batchy comprises:

* a runtime profiler to quantify batch-processing gains on different processing functions,
* an analytical model to fine-tune queue backlogs,
* a new queuing abstraction that allows to realize the model in the run-to-completion execution model,
* a one-step receding horizon controller that adjusts backlogs across the pipeline.

Extensive experiments on networking use cases taken from an official 5G benchmark suite show that Batchy provides 2-3x the performance of prior work while accurately satisfying delay SLOs.

For more information, please refer to the [NSDI20 paper](https://www.usenix.org/conference/nsdi20/presentation/levai).

## Installation

### Dependencies

* [Python](https://www.python.org/downloads/) (>=3.6)
* [numpy](https://www.numpy.org/)
* [BESS](https://github.com/NetSys/bess)
* [our-plugins](https://github.com/levaitamas/bess/tree/our-plugins) for BESS

### Install Dependencies

BESS and our-plugins can be installed with the following steps ([Docker](https://docs.docker.com/install/) is required):

```sh
git clone https://github.com/levaitamas/bess.git -b our-plugins
cd bess
sudo ./start-container-our_plugins.sh

$ ./build.py --plugin our-plugins
```

### Clone the Repository

```sh
git clone https://github.com/levaitamas/batchy.git
```

### Setup Batchy

Before diving deep in Batchy, customize your configuration file at [lib/settings.py](lib/settings.py):

* (mandatory) set BESS install location (`DEFAULT_BESSDIR`)
* (optional) tweak Batchy parameters

## Usage

### Profiling

Batchy contains a built-in profiler that runs a standard benchmark on the system under test at the time of initialization, collects per-batch and per-packet service-time components for common NFs (BESS modules), and stores the results for later use.

The profiling data is automatically read when a new module is added to the packet-processing pipeline and thus provides a handful information to the controller.

Profiling can be initiated manually only due to its long time-span. A complete profiling can even take up to 60+ minutes.

To start profiling:
```sh
./batchy.py -profile [MODULES]
```
or
```sh
./profiler/profile_all.sh
```

The profiler results are written to `PROFILER_FILE` set in [lib/settings.py](lib/settings.py) (default: `profiler_results.json` in Batchy root).


### Writing Batchy Scripts

Batchy provides a scripting interface to build and interact with the Batchy internals as a main interface.

A Batchy scripts are Python3 scripts with extra global variables. They are also very similar to [BESS config scripts](https://github.com/NetSys/bess/wiki/Writing-a-BESS-Configuration-Script).

Relevant parts of a Batchy script are the following:

#### 1. Read run-time args
The following pattern reads the CLI arg `arg_name` (or `default_value`) to `variable` converted to `type`.
```python
variable = get_arg(arg_name, default_value, type)
```

#### 2. Create worker
```python
worker0 = batchy.add_worker(worker_name)
```

#### 3. Add tasks to a worker
```python
task0 = worker0.add_task(task_name, type=task_type)
```
Supported types are: `'RTC'` and `'WFQ'`. Type-mixing is not supported.

#### 4. Add modules to task, set internal pipeline
```python
module0 = task0.add_module(BessModule(bess_module_kw_args), type=mtype)
```

#### 5. Add flows
```python
new_flow = batchy.add_flow(name=flow_name,
                           path=[{'task': task0, 'path': flow_path}],
                           delay_slo=flow_delayslo,
                           rate_slo=flow_rateslo,
                           source_params=flow_source_params)
```

#### 6. Add test traffic

* Built-in traffic generator
```python
batchy.add_source()
batchy.add_sink()
```

* PCAP-replay
Supported modes are `'replicate'` and `'tcpreplay'`. Tcpreplay relies on the external tool `tcpreplay`and has a very limited throughput. Replicate uses a large Queue module to store packets and a Replicate module to re-add a freash-copy of the packet leaving the traffic generator. This method requires large amount of memory, but it provides a good throughput.
```python
batchy.add_pcap_source(source_pcap, worker, task, mode='replicate', ts_offset=None)
batchy.add_sink()
```

A [helper script](scripts/create_pcap_stats.py) is available to populate L3 lookup module tables. See the [l2l3 config](conf/l2l3.batchy) as an example.

#### 7. Set controllers
* Set Task controller
```python
worker0.set_task_controller(batchy.resolve_task_controller(controller_name))
```

* Set Worker controller
```python
batchy.set_controller(batchy.resolve_controller(controller_name))
```

#### 8. Run pipeline
```python
batchy.run(rounds, control_period)
```

#### 9. Get results
```python
batchy.plot(outfile_png)
batchy.dump(outfile_csv)
```

A complete example is presented in [Examples](#examples).

### Running Batchy Scripts

#### 1. Start BESS daemon if it is not running
TIP: [This BESS Wiki page](https://github.com/NetSys/bess/wiki/Build-and-Install-BESS#start-up-bess-and-run-a-sample-configuration) shows how to start the BESS daemon.

#### 2. Run Batchy config script
```sh
./batchy.py -r CONF_FILE CONF_ARGS
```


## Examples

### A Simple Pipeline

```
                  +---------+
                  |         |
               -->|   NF1   |
              /   |         |
+----------+ /    +---------+
|          |/
| Splitter |
|          |\
+----------+ \    +---------+
              \   |         |
               -->|   NF2   |
                  |         |
                  +---------+

       Simple Pipeline
```

A simple pipeline consists of 2 NFs (implemented as Bypass modules) and a splitter (Splitter module). Two flows will be defined: flow1: [splitter, nf1] and flow2: [splitter, nf2]. The configuration script is located at [conf/simple_pipeline.batchy](conf/simple_pipeline.batchy).

Details of the config script:

To run the configuration start BESS daemon and then issue the following command:
```sh
./batchy.py --reset --loglevel INFO conf/simple_pipeline.batchy rounds=100,delay_slo1=55_000
```
The `--reset` arg resets the BESS daemon before run, `--loglevel` sets Batchy loglevel, `rounds=100,delay_slo1=55_000` run-time arguments of `simple_pipeline.batchy`  specifying the number of rounds and delay SLO of flow1.

During the run Batchy dumps information about the process according to the loglevel set:
```
*** WARMUP... ***
*** CONTROL ROUND: 1..., cumulative flow rate: 0 pps
CONTROL: task0: error=0.000, dist=3600.000
        module bypass1: PULLING: q_v: 19.000, delay_diff = 5723.824 < max_delay=88650.000
*** CONTROL ROUND: 2..., cumulative flow rate: 6.23899 Mpps
CONTROL: task0: error=0.000, dist=3400.000
        module bypass1: GRADIENT PROJECTION: setting: q[v]: 19 -> 32
*** CONTROL ROUND: 3..., cumulative flow rate: 6.529956 Mpps
CONTROL: task0: error=0.000, dist=3350.000
GRADIENT PROJECTION: obtained a KKT point, doing nothing
*** CONTROL ROUND: 4..., cumulative flow rate: 6.821728 Mpps
CONTROL: task0: error=0.000, dist=3350.000
GRADIENT PROJECTION: obtained a KKT point, doing nothing
[...]
*** CONTROL ROUND: 39..., cumulative flow rate: 6.855618 Mpps
CONTROL: task0: error=0.000, dist=3350.000
GRADIENT PROJECTION: obtained a KKT point, doing nothing
*** CONTROL ROUND: 40..., cumulative flow rate: 6.862739 Mpps
CONTROL: task0: error=0.000, dist=1700.000
GRADIENT PROJECTION: obtained a KKT point, doing nothing
plot: plotting statistics to file: "/tmp/simple_pipeline_stats.png"
dump: dumping statistics to CSV file: "/tmp/simple_pipeline_stats.txt"
Done.
```

After the run Batchy dumps results to `/tmp/simple_pipeline_stats.txt` and plots a figure of control values, flow delays, and flow packet rates to `/tmp/simple_pipeline_stats.png`.

### Run our measurements
To easily recreate some of our measurements, we packed our NSDI measurements with ready-to-run shell scripts. For details, see [scripts/nsdi-measurements](scripts/nsdi-measurements).

## Caveats and Limitations

Batchy is an experimental software with limitations. Some of these are:

* tasks are not handled (in most cases 1 task/worker is used)
* by default implicit decision of configurable module, i.e. batchyness metric
* the resolution of Measure module determines the real-time controller's effective delay range

## License

Batchy is a free software and licensed under [GPLv3+](./LICENSE).
