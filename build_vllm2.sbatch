#!/bin/bash
#SBATCH --job-name=js_vllm2_build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/%u/job_swarm/logs/vllm2_build_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/vllm2_build_%j.err

# Builds the NEXT-GEN vLLM container (for Qwen3-Next / Mistral-Small-4 MoE
# serving, task #25) as a SEPARATE file - the proven vllm_inference.sif is
# never touched, and the nightly only switches after test_vllm2.sbatch
# writes its PASS marker.
#
# Pin a specific tag instead of latest if the build misbehaves:
#   sbatch --export=ALL,VLLM2_TAG=v0.20.1 build_vllm2.sbatch

set -uo pipefail
TAG="${VLLM2_TAG:-latest}"
OUT=/scratch/$USER/quant_swarm/vllm2_inference.sif
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer_cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" /home/$USER/job_swarm/logs

echo "Building $OUT from docker://vllm/vllm-openai:$TAG (10-20 GB pull)..."
apptainer build --force "$OUT" "docker://vllm/vllm-openai:$TAG"
RC=$?
if [ $RC -eq 0 ]; then
    echo "BUILD OK -> $OUT"
    apptainer exec "$OUT" python3 -c \
        "import vllm; print('container vllm version:', vllm.__version__)" || true
else
    echo "BUILD FAILED (rc=$RC) - try pinning a tag, see header comment"
fi
exit $RC
