#!/bin/bash
#SBATCH --job-name=js_analyze
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:H100:2
#SBATCH --exclude=ra5-1,ra5-3,ra5-4,ra7-1,ra7-3,ra7-4,ra8-1,ra8-2
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/job_swarm/logs/analyze_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/analyze_%j.err

# Nightly stage 2 - vLLM Llama-3-70B + LangGraph analysis DAG.
# Mirrors quant_swarm_nightly.sh (the proven GPU job) on the vLLM launch,
# tunnel, and binds - but sized for THIS workload: Llama-3-70B-AWQ is ~40GB
# quantized, so 2×H100 (tensor-parallel-size 2) holds it with room for KV
# cache, and 2-GPU jobs clear the gpu_p queue much faster than 4-GPU ones.
# The DAG itself is ~80 LLM calls + tiny embeddings - nowhere near quant-scale.
# Submitted by job_swarm_nightly.sh with --dependency=afterany:<ingest jobid>.
#
# The --exclude list is LOAD-BEARING: the gpu_p H100 nodes run two NVIDIA
# driver generations (census 2026-07-02, see probe_drivers.sh):
#   good (590.48.01): ra5-2, ra7-2, ra8-3, ra8-4
#   bad  (570.124.06 = CUDA 12.8 only): ra5-1, ra5-4, ra7-3, ra7-4, ra8-1, ra8-2
#   unknown (drained at census time, excluded to be safe): ra5-3, ra7-1
# The torch inside vllm_inference.sif needs the newer driver - job 46563229
# died on a 570 node with "NVIDIA driver too old, found 12080". This is also
# why quant_swarm_nightly.sh pins ra8-4. If GACRC upgrades drivers, re-run
# probe_drivers.sh and trim the list.
# If your allocation ever moves to A100s: change the --gres line above and
# TORCH_CUDA_ARCH_LIST below to "8.0".

set -uo pipefail

SWARM_DIR=/scratch/$USER/quant_swarm/job_swarm

# ---- Runtime GPU detection ---------------------------------------------------
# The submitter may have overridden --gres to the A100 pool (more good-driver
# nodes, less contention). Arch flag must match, and the FP8 Qwen tier must
# never engage on A100s (sm80 has no FP8 tensor cores).
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *A100*) TORCH_ARCH="8.0";  GPU_IS_H100=0 ;;
    # B200 (sm_100, 192GB) is FP8-capable: counts as H100-tier so the Qwen
    # ladder engages. Only reachable via the weekly heavy branch in
    # job_swarm_nightly.sh, which is gated on the data/B200_OK marker.
    *B200*) TORCH_ARCH="10.0"; GPU_IS_H100=1 ;;
    *)      TORCH_ARCH="9.0";  GPU_IS_H100=1 ;;
esac
echo "GPU: ${GPU_NAME:-unknown} (arch $TORCH_ARCH)"

# ---- Model + container auto-detect -------------------------------------------
# Three-tier ladder, newest first, each tier gated on its artifacts existing:
#   1. Qwen3-Next-80B FP8 on the NEW container - ONLY if test_vllm2.sbatch
#      wrote its PASS marker (data/VLLM2_OK). A broken upgrade can never
#      take down a nightly; delete the marker to force-revert.
#   2. Llama-3.3-70B-AWQ on the proven container (drop-in over Llama-3).
#   3. Llama-3-70B-AWQ (original).
VLLM_SIF=/scratch/$USER/quant_swarm/vllm_inference.sif
# 0.70 util (was 0.75/0.85/0.90): the DAG's embedding model co-tenants the
# PRIMARY server's GPU 0 (the critic on GPUs 2,3 keeps 0.85). At 0.90 the
# two OOM'd each other on consecutive nights (2026-07-03). 0.85 (~12GB
# headroom) worked for bge-large/0.6B, but the standing Qwen3-Embedding-8B
# (2026-07-07 matching-core) failed to load there and its bge fallback still
# starved the vLLM worker (run 46804624). 0.75 (~19GB) fit the embedder at
# rest but not its batch activations: run 46807108 measured 18.9GB resident
# (16GB bf16 weights + batches) and the vLLM worker OOM'd with 76MB free.
# 0.70 leaves ~24GB - resident embedder + batch spikes + vLLM transient
# workspace; AWQ 70B at TP=2 (~18GB weights/GPU) still has ~37GB KV cache.
VLLM_EXTRA_ARGS="--quantization awq --dtype float16 --gpu-memory-utilization 0.70"
VLLM_V1_ENV="--env VLLM_USE_V1=0"
if [ -f "$SWARM_DIR/data/VLLM2_OK" ] && [ "$GPU_IS_H100" = "1" ] \
   && [ -d /scratch/$USER/data/models/qwen3-next-80b-a3b-instruct-fp8 ]; then
    # Existing container is vLLM 0.22.1 (probed 2026-07-03) - new enough for
    # Qwen3-Next; a fresh vllm2 build is preferred when present but optional.
    if [ -f /scratch/$USER/quant_swarm/vllm2_inference.sif ]; then
        VLLM_SIF=/scratch/$USER/quant_swarm/vllm2_inference.sif
    fi
    MODEL_PATH=/data/models/qwen3-next-80b-a3b-instruct-fp8
    # triton GDN backend: FlashInfer JIT dies in the read-only container
    VLLM_EXTRA_ARGS="--max-model-len 32768 --gdn-prefill-backend triton --gpu-memory-utilization 0.70"
    VLLM_V1_ENV=""                            # modern vLLM: v1 engine is correct
elif [ -d /scratch/$USER/data/models/llama-3.3-70b-instruct-awq ]; then
    MODEL_PATH=/data/models/llama-3.3-70b-instruct-awq
else
    MODEL_PATH=/data/models/llama-3-70b-instruct-awq
fi
export VLLM_MODEL=$MODEL_PATH
echo "vLLM container: $VLLM_SIF"
echo "vLLM model: $MODEL_PATH ($VLLM_EXTRA_ARGS)"

# ---- Embedder auto-detect ----------------------------------------------------
# Preference order: Qwen3-Embedding-0.6B (2026 MTEB retrieval leader ≤2B) >
# bge-large-en-v1.5 > MiniLM. Download via download_models.sbatch (or
# huggingface-cli on the login node). When the HF cache has a better model,
# the swarm switches automatically; the corpus is wiped and re-embedded in
# one run (embed-model guard in trajectory_engine). If the container's
# transformers can't load the Qwen3 architecture, the trajectory engine's
# fallback ladder lands on bge-large, NOT MiniLM.
if [ -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B" ]; then
    # Top tier: once download_qwen_embed_8b.sbatch has landed, 8B becomes the
    # STANDING embedder for every run (nightly + weekly). The embed-model guard
    # re-embeds the whole corpus into 8B space once, on the first run it sees
    # the switch; thereafter it is just the embedder. A single corpus DB holds
    # one embedding space, so there is no separate weekly 8B pass to thrash
    # against - promoting it here IS the "re-embed once" the plan called for.
    export JOB_SWARM_EMBED_MODEL=Qwen/Qwen3-Embedding-8B
    echo "Embedder: Qwen/Qwen3-Embedding-8B (HF cache hit)"
elif [ -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B" ]; then
    export JOB_SWARM_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
    echo "Embedder: Qwen/Qwen3-Embedding-0.6B (HF cache hit)"
elif [ -d "$HOME/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5" ]; then
    export JOB_SWARM_EMBED_MODEL=BAAI/bge-large-en-v1.5
    echo "Embedder: BAAI/bge-large-en-v1.5 (HF cache hit)"
fi

# ---- SSH egress tunnel (compute node -> internet via login node) ------------
EGRESS_PORT=$(shuf -i 40000-50000 -n 1)
ssh -N -D 127.0.0.1:$EGRESS_PORT -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
    $USER@$SLURM_SUBMIT_HOST &
SSH_TUNNEL_PID=$!
echo "Egress tunnel established (PID $SSH_TUNNEL_PID)"

# ---- Dual-model night? -------------------------------------------------------
# Cross-model critic (2026-07-06): when the allocation is 4 GPUs AND the
# primary is the Qwen tier, a SECOND model family (Llama-AWQ) serves on port
# 8001 from GPUs 2,3 as an adversarial critic - same-model self-review
# rewards its own blind spots. Strictly best-effort: any failure here leaves
# the critic env unset and the DAG runs single-model exactly as before.
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
CRITIC_MODEL_PATH=""
if [ "$GPU_COUNT" -ge 4 ] && [[ "$MODEL_PATH" == *qwen3-next* ]]; then
    if [ -d /scratch/$USER/data/models/llama-3.3-70b-instruct-awq ]; then
        CRITIC_MODEL_PATH=/data/models/llama-3.3-70b-instruct-awq
    elif [ -d /scratch/$USER/data/models/llama-3-70b-instruct-awq ]; then
        CRITIC_MODEL_PATH=/data/models/llama-3-70b-instruct-awq
    fi
fi
PRIMARY_GPU_ENV=""
if [ -n "$CRITIC_MODEL_PATH" ]; then
    PRIMARY_GPU_ENV="--env CUDA_VISIBLE_DEVICES=0,1"
    echo "Dual-model night: critic $CRITIC_MODEL_PATH on GPUs 2,3 (port 8001)"
fi

# ---- Launch tensor-parallel vLLM --------------------------------------------
export SLURM_STEP_RESV_PORTS=1
export SLURM_OVERLAP=1

echo "Launching vLLM inference engine..."
apptainer exec --nv --ipc \
    $VLLM_V1_ENV $PRIMARY_GPU_ENV \
    --env http_proxy="" --env https_proxy="" --env all_proxy="" \
    --env TORCH_CUDA_ARCH_LIST="$TORCH_ARCH" \
    --env NVIDIA_REQUIRE_CUDA="cuda>=12.0" \
    --bind /scratch/$USER/data:/data:ro \
    "$VLLM_SIF" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
    --port 8000 \
    $VLLM_EXTRA_ARGS &
VLLM_PID=$!

CRITIC_PID=""
if [ -n "$CRITIC_MODEL_PATH" ]; then
    # The proven container + AWQ args - the exact Llama config that ran the
    # pipeline before the Qwen upgrade. v0 engine env matches that era.
    apptainer exec --nv --ipc \
        --env VLLM_USE_V1=0 \
        --env CUDA_VISIBLE_DEVICES=2,3 \
        --env http_proxy="" --env https_proxy="" --env all_proxy="" \
        --env TORCH_CUDA_ARCH_LIST="$TORCH_ARCH" \
        --env NVIDIA_REQUIRE_CUDA="cuda>=12.0" \
        --bind /scratch/$USER/data:/data:ro \
        /scratch/$USER/quant_swarm/vllm_inference.sif \
        python3 -m vllm.entrypoints.openai.api_server \
        --model $CRITIC_MODEL_PATH \
        --tensor-parallel-size 2 \
        --port 8001 \
        --quantization awq --dtype float16 --gpu-memory-utilization 0.85 &
    CRITIC_PID=$!
fi

echo "Waiting for vLLM to bind on port 8000..."
while ! timeout 1 bash -c "echo > /dev/tcp/localhost/8000" 2>/dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "CRITICAL: vLLM crashed during startup. Aborting."
        kill $CRITIC_PID $SSH_TUNNEL_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done
echo "vLLM is listening."

if [ -n "$CRITIC_PID" ]; then
    # Best-effort wait (max ~15 min): a dead critic just means single-model.
    CRITIC_DEADLINE=$((SECONDS + 900))
    while ! timeout 1 bash -c "echo > /dev/tcp/localhost/8001" 2>/dev/null; do
        if ! kill -0 $CRITIC_PID 2>/dev/null || [ $SECONDS -ge $CRITIC_DEADLINE ]; then
            echo "WARN: critic vLLM failed to start - running single-model."
            kill $CRITIC_PID 2>/dev/null || true
            CRITIC_PID=""
            break
        fi
        sleep 5
    done
    if [ -n "$CRITIC_PID" ]; then
        export VLLM_CRITIC_URL=http://localhost:8001/v1/chat/completions
        export VLLM_CRITIC_MODEL=$CRITIC_MODEL_PATH
        echo "Critic vLLM is listening on 8001."
    fi
fi

# ---- Run the analysis DAG (embeddings ride the same H100s via --nv) ---------
export JOB_SWARM_DATA=$SWARM_DIR/data
export JOB_SWARM_DB=/home/$USER/job_swarm/state/job_swarm.db
export JOB_SWARM_CONFIG=$SWARM_DIR/job_swarm_config.json
export JOB_SWARM_REPORTS=/home/$USER/job_swarm_reports
export JOB_SWARM_PROFILE_DIR=$SWARM_DIR/profile
mkdir -p "$JOB_SWARM_REPORTS" "$JOB_SWARM_DATA/telemetry" "$JOB_SWARM_DATA/profile_cache"

apptainer exec --nv \
    --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --bind $HOME:$HOME \
    --bind /scratch/$USER:/scratch/$USER \
    /scratch/$USER/quant_swarm/agent_env.sif \
    python3 $SWARM_DIR/js_main.py --stage analyze
RC=$?

kill $VLLM_PID 2>/dev/null || true
[ -n "$CRITIC_PID" ] && kill $CRITIC_PID 2>/dev/null || true
kill $SSH_TUNNEL_PID 2>/dev/null || true
exit $RC
