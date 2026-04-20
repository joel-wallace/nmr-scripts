#!/dev/null

# PROCESSING PARAMETERS
# first and last experiments to sum
export FIRST=3
export LAST=19
# phasing, probably don't use P1
export P0=-24
export P1=0
# line-broadening to apply in Hz
export LB=10
# number of peaks to fit
export NUM_PEAKS=3
# exclude the peaks for baselining
export EXCLUDE_LEFT=-59
export EXCLUDE_RIGHT=-65


# Directory where the NMR experiments 1, 2, 3 etc are stored
export WD=$PWD/..

# Directory containing the scripts (where this script is stored)
export MASTER=$PWD
# Conversion script
export CONVERSION_SCRIPT="bruk2pipe.com"


# Environment variables for NMRPipe:
# installation location
export NMR_HOME="/opt/nmrpipe"
# executable binary location
export PATH="$PATH:$NMR_HOME/nmrbin.linux239_64"
# LD library path
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NMR_HOME/nmrbin.linux239_64/lib"
export NMR_CONT="CORRECT_ALL"

# Python executable
# make sure this can call numpy and scipy
export PYTHON="$MAMBA_ROOT_PREFIX/envs/nmrpipe/bin/python"

# Processing script
export PROC_SCRIPT="nmrproc.com"

# Processing file names
export FID_NAME="test.fid"
export FT1_NAME="test.ft1"
export OUT_NAME="data.txt"
export BASELINED_NAME="baselined.txt"
export FIT_NAME="fit.txt"
