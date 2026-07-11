#!/bin/bash
#SBATCH --job-name=js_embed8b_dl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/job_swarm/logs/embed8b_dl_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/embed8b_dl_%j.err

# One-time download of Qwen3-Embedding-8B (~16 GB) into the HF cache. Once the
# cache dir exists, nightly_analyze.sbatch auto-selects it at the TOP of the
# embedder ladder and the trajectory engine's embed-model consistency guard
# wipes + re-embeds the whole corpus into 8B space on the FIRST run after this
# lands - one re-embed, no weekly/nightly space-thrash (a single corpus DB can
# hold exactly one embedding space; see trajectory_engine section 1.5).
#
# Runs fine on the A100 tier at inference time - 8B fp16 with the hard 512 seq
# cap and OOM-halving batch co-tenants the vLLM GPUs; it does NOT need the B200.
# Safe to re-run: snapshot_download resumes and skips complete files.

set -uo pipefail
mkdir -p /home/$USER/job_swarm/logs

apptainer exec \
    --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --bind $HOME:$HOME \
    --bind /scratch/$USER:/scratch/$USER \
    /scratch/$USER/quant_swarm/agent_env.sif \
    python3 - <<'PY'
from huggingface_hub import snapshot_download
print("=== Qwen/Qwen3-Embedding-8B (~16 GB, into the HF cache) ===", flush=True)
try:
    p = snapshot_download("Qwen/Qwen3-Embedding-8B")
    print(f"done -> {p}", flush=True)
except Exception as e:
    print(f"FAILED: {e}\n"
          "If this is a 401/403 the repo may be gated - resubmit with:\n"
          "  sbatch --export=ALL,HF_TOKEN=hf_xxx download_qwen_embed_8b.sbatch",
          flush=True)
    raise
PY
