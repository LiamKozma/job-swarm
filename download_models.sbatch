#!/bin/bash
#SBATCH --job-name=js_model_dl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/job_swarm/logs/model_dl_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/model_dl_%j.err

# One-time model downloads for the job swarm v2 upgrade (CPU only, batch
# partition - the same internet egress the nightly ingest already uses).
# Uses huggingface_hub from inside agent_env.sif: no pip, no modules.
# Safe to re-run - snapshot_download resumes partial downloads and skips
# files that already exist.

set -uo pipefail
MODELS_DIR=/scratch/$USER/data/models
mkdir -p "$MODELS_DIR" /home/$USER/job_swarm/logs

apptainer exec \
    --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --bind $HOME:$HOME \
    --bind /scratch/$USER:/scratch/$USER \
    /scratch/$USER/quant_swarm/agent_env.sif \
    python3 - "$MODELS_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

models_dir = sys.argv[1]

def grab(repo, local_dir=None):
    # local_dir must contain REAL files, not symlinks into the HF cache -
    # the vLLM container only binds /scratch/$USER/data, so symlinks into
    # $HOME/.cache would dangle. Newer hub versions copy real files by
    # default and dropped the flag (TypeError -> retry without it).
    kwargs = {"local_dir": local_dir} if local_dir else {}
    try:
        return snapshot_download(repo, local_dir_use_symlinks=False, **kwargs)
    except TypeError:
        return snapshot_download(repo, **kwargs)

print("=== 1/2 Llama-3.3-70B-Instruct-AWQ (~40 GB - the long one) ===", flush=True)
try:
    p = grab("casperhansen/llama-3.3-70b-instruct-awq",
             f"{models_dir}/llama-3.3-70b-instruct-awq")
    print(f"done -> {p}", flush=True)
except Exception as e:
    print(f"FAILED: {e}\n"
          "If this is a 401/403: the repo may have gone gated. Make a free "
          "HF account token and resubmit with:  sbatch --export=ALL,"
          "HF_TOKEN=hf_xxx download_models.sbatch", flush=True)
    sys.exit(1)

print("=== 2/3 BAAI/bge-large-en-v1.5 (~1.3 GB, into the HF cache) ===", flush=True)
p = grab("BAAI/bge-large-en-v1.5")
print(f"done -> {p}", flush=True)

print("=== 3/3 Qwen/Qwen3-Embedding-0.6B (~1.2 GB, into the HF cache) ===", flush=True)
# 2026 MTEB retrieval leader ≤2B params (61.83 vs stella 52.42 on Eng v2),
# Apache 2.0, 32k context. nightly_analyze auto-prefers it once cached; if
# the container's transformers is too old for the Qwen3 architecture, the
# trajectory engine's fallback ladder keeps the nightly on bge-large.
p = grab("Qwen/Qwen3-Embedding-0.6B")
print(f"done -> {p}", flush=True)
PY
RC=$?
[ $RC -ne 0 ] && exit $RC

echo "--- verification ---"
du -sh "$MODELS_DIR/llama-3.3-70b-instruct-awq"
ls "$MODELS_DIR/llama-3.3-70b-instruct-awq"
du -sh "$HOME"/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5
echo "PASS: nightly_analyze.sbatch will auto-detect both on its next run."
