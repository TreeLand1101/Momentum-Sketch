#!/bin/sh

# Compile the program
make

datasets="equinix-chicago.dirA.20160121-140000.UTC.anon.dat equinix-nyc.dirA.20180315-125910.UTC.anon.dat equinix-nyc.dirA.20190117-125910.UTC.anon.dat 202111011400.dat 202211011400.dat Campus.dat"

memory_values="16 32 64 128 256 512 1024"
alphas="0.0001"

for dataset in $datasets 
do
    mkdir -p "Performance/${dataset}"
    
    for memory in $memory_values
    do
        for alpha in $alphas
        do
            log_path="Performance/${dataset}/memory_${memory}KB_alpha_${alpha}.txt"
            ./CPU ${memory} ${alpha} ${dataset} >> ${log_path}
            echo "Finished run: memory=${memory}KB, alpha=${alpha} for dataset=${dataset}"
        done
    done
    python3 metric.py "$dataset" "$alphas" $memory_values
done

echo "Finished all runs and generated plots."