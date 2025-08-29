#!/bin/sh

make

DATASET_DIR="../../"

dataset="equinix-chicago.dirA.20160121-140000.UTC.anon.dat equinix-nyc.dirA.20180315-125910.UTC.anon.dat equinix-nyc.dirA.20190117-125910.UTC.anon.dat Campus.dat 202111011400.dat 202211011400.dat"

memory=128
alpha=0.0001

> log.txt

for dataset in $dataset
do
    dataset_path="${DATASET_DIR}${dataset}"
    
    ./PacketCountDistribution "$memory" "$alpha" "$dataset_path" >> log.txt
    echo "Finished run: memory=${memory}KB, alpha=${alpha} for dataset=${dataset}"

    python3 PlotPacketCountDistribution.py --memory "$memory" --alpha "$alpha" --dataset "$dataset"
done

echo ">>> All done."