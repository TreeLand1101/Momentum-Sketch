#!/bin/sh

make

DATASET_DIR="../../"

dataset="equinix-chicago.dirA.20160121-140000.UTC.anon.dat"

memory_values="32 48 64 80 96"

for memory in $memory_values
do
    ./FilterPerformance ${memory} "${DATASET_DIR}${dataset}"
    echo "Finished run: memory=${memory}KB"
done

echo "Finished all run."

python3 metric.py $memory_values