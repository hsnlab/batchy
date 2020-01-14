CUR_DIR=`dirname $0`
BATCHYROOT=$CUR_DIR/../..
BATCHYPY="$BATCHYROOT/batchy.py"

# BESS install directory
BESSDIR="/opt/bess"

# batchy loglevel
LOGLEVEL="CRITICAL"

# directory to put measurement results
RESULTS_DIR="$CUR_DIR/results"

# directory containing measurement files
CONF_DIR="$CUR_DIR/configs"

# pcap to use with l2l3pcap.sh
SRC_PCAP="$CUR_DIR/pcaps/ny-20190117-130600-caida.pcap"
