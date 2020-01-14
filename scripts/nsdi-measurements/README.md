# Batchy NSDI measurements

[Preparations](#preparations) | [Usage](#usage) | [Known Issues](#known-issues)

## Preparations

Please install Batchy and its dependencies properly. For details, check the main [README](/../../README.md). Start BESS before running measurements.

It is advised to run the measurements on bare-metal. Virtual CPUs tend to cause significant performance fluctuation that leads to distorted results.

To prepare a PCAP file for the `l2l3pcap` scenario, generate the next hop statistics file first:
```sh
python3 ../create_pcap_stats.py PCAP_FILE
```

We used the 2019 CAIDA trace `equinix-nyc.dirA.20190117-130600.UTC.anon` for our measurements. To get access to that trace, contact CAIDA [here](https://www.caida.org/data/passive/passive_dataset_request.xml).

## Usage

Edit the configuration file [settings.sh](settings.sh) to set BESS install directory, PCAP source file and output directory.

A shell script covers a scenario. The script will fire up Batchy and run measurements. As an example, to run L2L3 measurements: `bash l2l3.sh`

## Known Issues

* BESS daemon tends to silently stop processing packets. In that case you will see 0-s after a given control period in Batchy results.

* Virtual CPUs cause serious performance fluctuations that leads to distorted results. It is advised to run measurements on bare-metal.
