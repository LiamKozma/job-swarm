#!/bin/bash
# NVIDIA-driver + GPU-memory census of the A100 and L4 nodes in gpu_p -
# deciding whether the nightly analyze stage can migrate off the contended
# 4-node H100 subset. Run from the login node:  bash probe_a100_l4.sh
# Results: job_swarm/data/driver_census/<node>.txt (readable over sshfs).
#
# Lessons baked in from the H100 census (2026-07-02):
#  - capture the FULL /proc/driver/nvidia/version line, not a grep - new
#    "Open Kernel Module" drivers don't match the old 'Kernel Module' pattern
#  - vllm_inference.sif needs driver >= ~575.x; 570.x fails "driver too old"
#  - nvidia-smi also reports GPU model + VRAM (A100 40GB vs 80GB decides
#    whether TP=2 works or we need TP=4)

set -u

OUT=/scratch/$USER/quant_swarm/job_swarm/data/driver_census
mkdir -p "$OUT"

NODES=$(sinfo -p gpu_p -N -h -o "%N %G %T" | sort -u \
        | awk '/A100|L4:/ && $NF !~ /drain|inval|maint|down/ {print $1}' | sort -u)

PROBE='head -1 /proc/driver/nvidia/version; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable without GPU grant"'

echo "Probing: $NODES"
for n in $NODES; do
    case "$n" in
        b*|ra4-*) GRES="gpu:A100:1" ;;
        *)        GRES="gpu:L4:1"   ;;
    esac
    # GPU-granting probe first (gets nvidia-smi memory info); CPU-only fallback
    jid=$(sbatch --parsable -p gpu_p -w "$n" --gres="$GRES" -n1 -c1 \
          --mem=1G --time=00:02:00 \
          -J "drv_$n" -o "$OUT/$n.txt" -e "$OUT/$n.err" \
          --wrap "$PROBE" 2>/dev/null) \
    || jid=$(sbatch --parsable -p gpu_p -w "$n" -n1 -c1 --mem=1G --time=00:02:00 \
          -J "drv_$n" -o "$OUT/$n.txt" -e "$OUT/$n.err" \
          --wrap "$PROBE" 2>/dev/null)
    if [ -n "${jid:-}" ]; then
        echo "  $n ($GRES) -> job $jid"
    else
        echo "  $n -> SUBMISSION FAILED"
    fi
done

echo ""
echo "Give them a few minutes (GPU probes queue behind real jobs), then:"
echo "  grep -H . $OUT/{b,ra4,ra5-5,ra5-6,ra5-7,ra5-8,ra7-5,ra7-6,ra7-7,ra7-8,ra8-5,ra8-6,ra8-7,ra8-8}*.txt 2>/dev/null"
echo "Leftovers:  squeue --me -o '%j %T %R' | grep drv_   /   scancel --me -n drv_<node>"
