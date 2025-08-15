#!/bin/sh

make

DATASET_DIR="../../"

datasets="equinix-chicago.dirA.20160121-140000.UTC.anon.dat equinix-nyc.dirA.20180315-125910.UTC.anon.dat equinix-nyc.dirA.20190117-125910.UTC.anon.dat 202111011400.dat 202211011400.dat Campus.dat"

memory_values="128"
threshold_values="0.0001"

for dataset in $datasets
do
    dataset_name=$(basename "${dataset}")
    output_dir="${dataset_name}"
    mkdir -p "${output_dir}"
    
    for memory in $memory_values
    do
        for threshold in $threshold_values
        do
            ./TwoStageParameter ${memory} ${threshold} "${DATASET_DIR}${dataset}"
            echo "Finished run: dataset=${dataset}, memory=${memory}KB, threshold=${threshold}"
            python3 PlotHeatmap.py "${output_dir}/performance.csv" "${output_dir}"
        done
    done
done

echo "Finished all runs."