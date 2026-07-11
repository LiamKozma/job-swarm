#!/bin/bash
#SBATCH --job-name=js_qwen_dl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/job_swarm/logs/qwen_dl_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/qwen_dl_%j.err

# Downloads Qwen3-Next-80B-A3B-Instruct-FP8 (~80 GB) - the MoE upgrade
# candidate from the 2026 deep-research briefing: 80B total / 3.9B active
# params, IFEval 87.6, Apache 2.0, ~10x long-context throughput vs dense
# 70B. FP8 requires H100-class GPUs (sm89+): the nightly's H100 pool and
# the L4s qualify; the A100s do NOT (they keep the Llama-AWQ fallback).
# Safe to re-run: snapshot_download resumes.

set -uo pipefail
MODELS_DIR=/scratch/$USER/data/models
mkdir -p "$MODELS_DIR" /home/$USER/job_swarm/logs

apptainer exec \
    --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --bind $HOME:$HOME --bind /scratch/$USER:/scratch/$USER \
    /scratch/$USER/quant_swarm/agent_env.sif \
    python3 - "$MODELS_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

models_dir = sys.argv[1]
repo = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
dest = f"{models_dir}/qwen3-next-80b-a3b-instruct-fp8"
print(f"=== {repo} -> {dest} (~80 GB, the long one) ===", flush=True)
try:
    kwargs = {"local_dir": dest}
    try:
        p = snapshot_download(repo, local_dir_use_symlinks=False, **kwargs)
    except TypeError:
        p = snapshot_download(repo, **kwargs)
    print(f"done -> {p}", flush=True)
except Exception as e:
    print(f"FAILED: {e}\n"
          "If 404: the exact FP8 repo name may differ - check "
          "huggingface.co/Qwen for the Qwen3-Next-80B-A3B-Instruct FP8 "
          "variant and resubmit with the corrected repo id.", flush=True)
    sys.exit(1)
PY
RC=$?
[ $RC -ne 0 ] && exit $RC
du -sh /scratch/$USER/data/models/qwen3-next-80b-a3b-instruct-fp8
echo "PASS: now run test_vllm2.sbatch (needs build_vllm2.sbatch done too)."
