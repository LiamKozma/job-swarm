#!/bin/bash
#SBATCH --job-name=js_vllm_ver
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/job_swarm/logs/vllm_ver_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/vllm_ver_%j.err

# 5-minute probe: what vLLM version lives in the CURRENT container, and is
# the xgrammar guided-decoding backend available? Decides task #21 (vLLM
# >= 0.6.x defaults to xgrammar already - the flag may be unnecessary) and
# whether the current container could even serve newer model architectures.

apptainer exec \
    --env PYTHONNOUSERSITE=1 \
    /scratch/$USER/quant_swarm/vllm_inference.sif \
    python3 - <<'PY'
import importlib
try:
    import vllm
    print(f"vllm version: {vllm.__version__}")
except Exception as e:
    print(f"vllm import failed (CPU import quirk, not fatal): {e}")
for mod in ("xgrammar", "outlines"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: available ({getattr(m, '__version__', '?')})")
    except ImportError:
        print(f"{mod}: NOT available")
try:
    import transformers, torch
    print(f"transformers: {transformers.__version__}  torch: {torch.__version__}")
except Exception as e:
    print(f"transformers/torch: {e}")
PY
