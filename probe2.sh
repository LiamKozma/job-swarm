#!/bin/bash
# Round 2 of the driver census: full version string from the 4 nodes whose
# driver didn't match round 1's grep (likely newer "Open Kernel Module").
set -u
OUT=/scratch/$USER/quant_swarm/job_swarm/data/driver_census
for n in ra5-2 ra7-2 ra8-3 ra8-4; do
    sbatch -p gpu_p -w "$n" -n1 -c1 --mem=1G --time=00:02:00 \
        -J "drv2_$n" -o "$OUT/${n}_full.txt" -e "$OUT/${n}_full.err" \
        --wrap "cat /proc/driver/nvidia/version"
done
