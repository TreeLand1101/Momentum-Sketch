#!/bin/sh

make

DATASET_DIR="../../"

datasets="202111011400.dat Campus.dat"

memory="256"
alpha="0.0001"

for dataset in $datasets; do
    ./MomentumCDF ${memory} ${alpha} "${DATASET_DIR}${dataset}"
    echo "Finished run: memory=${memory}KB, alpha=${alpha}, dataset=${dataset}"
done

python3 PlotCDF.py $datasets
