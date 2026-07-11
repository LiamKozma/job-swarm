#!/bin/bash
# One-shot NVIDIA-driver census of every usable H100 node in gpu_p.
# Run from the login node:  bash probe_drivers.sh
# Results land in job_swarm/data/driver_census/<node>.txt on scratch (one tiny
# job per node) - also readable over the sshfs mount from the workstation.
# Read them with:    grep . /scratch/$USER/quant_swarm/job_swarm/data/driver_census/*.txt
#
# Why: the H100 nodes run different driver versions; vllm_inference.sif needs
# driver >= 575.x (CUDA 12.9+). 570.x nodes kill vLLM with "driver too old".

set -u

OUT=/scratch/$USER/quant_swarm/job_swarm/data/driver_census
mkdir -p "$OUT"

# H100 nodes only, skipping ones sinfo says are unusable right now
NODES=$(sinfo -p gpu_p -N -h -o "%N %G %T" | sort -u \
        | awk '/H100/ && $NF !~ /drain|inval|maint|down/ {print $1}')

echo "Probing: $NODES"
for n in $NODES; do
    # try a CPU-only probe first (schedules instantly); if the partition
    # insists on a GPU request, fall back to borrowing one GPU for a minute
    jid=$(sbatch --parsable -p gpu_p -w "$n" -n1 -c1 --mem=1G --time=00:02:00 \
          -J "drv_$n" -o "$OUT/$n.txt" -e "$OUT/$n.err" \
          --wrap "grep -oE 'Kernel Module +[0-9.]+' /proc/driver/nvidia/version" \
          2>/dev/null) \
    || jid=$(sbatch --parsable -p gpu_p -w "$n" --gres=gpu:H100:1 -n1 -c1 \
          --mem=1G --time=00:02:00 \
          -J "drv_$n" -o "$OUT/$n.txt" -e "$OUT/$n.err" \
          --wrap "grep -oE 'Kernel Module +[0-9.]+' /proc/driver/nvidia/version" \
          2>/dev/null)
    if [ -n "${jid:-}" ]; then
        echo "  $n -> job $jid"
    else
        echo "  $n -> SUBMISSION FAILED"
    fi
done

echo ""
echo "Give them a minute, then:"
echo "  grep . $OUT/*.txt"
echo "Any probe still queued shows in:  squeue --me -n \$(echo drv_{ra5,ra7,ra8}-{1,2,3,4} | tr ' ' ',')"
echo "Cancel leftovers with:            scancel --me --name=drv_<node>"
