#!/bin/bash
#SBATCH --job-name=js_b200_test
#SBATCH --partition=iai_B200_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:B200:1
#SBATCH --time=01:30:00
#SBATCH --output=/home/%u/job_swarm/logs/b200_test_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/b200_test_%j.err

# GATE for the B200 weekly heavy tier (2026-07-07): the 4 good-driver H100
# nodes wedge for days behind 7-day-wall jobs, while e4-1 (8x B200, 192GB,
# FP8-capable, 2 GPUs/user, 48h wall) has a 2-deep queue. Blackwell is
# sm_100 - the proven container (vLLM 0.22.1) has never run on it and may
# lack the kernels. This boots Qwen3-Next FP8 on ONE B200 and runs the same
# memo-shaped guided-JSON checks as test_vllm2.sbatch. Writes data/B200_OK
# ONLY on full success - job_swarm_nightly.sh submits the weekly heavy pass
# to iai_B200_p iff that marker exists, so a bad architecture can never
# take down a nightly. If this FAILS on missing sm100 kernels, rebuild:
#   sbatch build_vllm2.sbatch          (latest vllm/vllm-openai docker tag)
# then rerun this gate (it prefers vllm2_inference.sif when present).

set -uo pipefail
SWARM_DIR=/scratch/$USER/quant_swarm/job_swarm
if [ -f /scratch/$USER/quant_swarm/vllm2_inference.sif ]; then
    SIF=/scratch/$USER/quant_swarm/vllm2_inference.sif
else
    SIF=/scratch/$USER/quant_swarm/vllm_inference.sif
fi
MODEL=/data/models/qwen3-next-80b-a3b-instruct-fp8
MARKER=$SWARM_DIR/data/B200_OK
rm -f "$MARKER"
echo "container: $SIF"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

[ -d "/scratch/$USER/data/models/qwen3-next-80b-a3b-instruct-fp8" ] || {
    echo "FAIL: model missing (run download_qwen_next.sbatch)"; exit 1; }

# TP=1: one 192GB B200 holds the 80B FP8 MoE with room for KV cache.
# TORCH_CUDA_ARCH_LIST=10.0 only steers JIT extensions; the real question
# is whether the wheel ships precompiled sm100 kernels - that's what this
# gate exists to answer.
echo "Launching vLLM with Qwen3-Next FP8, TP=1 on B200..."
apptainer exec --nv --ipc \
    --env http_proxy="" --env https_proxy="" --env all_proxy="" \
    --env NVIDIA_REQUIRE_CUDA="cuda>=12.0" \
    --env TORCH_CUDA_ARCH_LIST="10.0" \
    --bind /scratch/$USER/data:/data:ro \
    "$SIF" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --gdn-prefill-backend triton \
    --port 8000 &
VLLM_PID=$!

for i in $(seq 1 240); do   # MoE + FP8 load can take a while; 20 min cap
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FAIL: vLLM crashed during startup (check stderr log for"
        echo "      'no kernel image' / sm_100 errors = container rebuild needed)"
        exit 1
    fi
    timeout 1 bash -c "echo > /dev/tcp/localhost/8000" 2>/dev/null && break
    sleep 5
done
echo "vLLM port is open - waiting for HTTP health before the gate tests."

apptainer exec --bind /scratch/$USER:/scratch/$USER \
    /scratch/$USER/quant_swarm/agent_env.sif \
    python3 - "$MODEL" "$MARKER" <<'PY'
import json, sys, time, urllib.request, urllib.error

model, marker = sys.argv[1], sys.argv[2]
BASE = "http://127.0.0.1:8000"
URL = f"{BASE}/v1/chat/completions"

def call(payload):
    req = urllib.request.Request(URL, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

# An open TCP port is not a healthy server (engine warmup can still kill
# it). Poll /v1/models for HTTP 200, up to 10 minutes.
deadline = time.time() + 600
while True:
    try:
        with urllib.request.urlopen(f"{BASE}/v1/models", timeout=10) as r:
            if r.status == 200:
                print("server healthy (/v1/models 200)")
                break
    except Exception as e:
        if time.time() > deadline:
            print(f"FAIL: server never became healthy: {e}")
            sys.exit(1)
        time.sleep(10)

schema = {"type": "object",
          "properties": {"subject": {"type": "string"},
                         "body": {"type": "string"}},
          "required": ["subject", "body"]}
prompt = ("Draft a 4-part technical memo (hook, hedged diagnosis, proof, "
          "15-minute-call CTA), 60-125 words, to a PI whose recent paper "
          "reports GMM estimator degradation under covariate shift. "
          "Candidate: MS statistics, CUDA/HPC, regime-switching models. "
          'Output ONLY JSON: {"subject": "...", "body": "..."}')

t0 = time.time()
out = call({"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600, "temperature": 0.2,
            "guided_json": schema})
dt = time.time() - t0
text = out["choices"][0]["message"]["content"]
memo = json.loads(text)          # guided decoding => must be valid JSON
words = len(memo["body"].split())
print(f"guided-JSON OK in {dt:.1f}s; subject={memo['subject']!r}; "
      f"body={words} words")
assert memo.get("subject") and memo.get("body"), "empty fields"
assert 20 <= words <= 220, f"implausible body length {words}"

# throughput sanity: single-stream speed is the nightly's bottleneck
t0 = time.time()
for _ in range(3):
    call({"model": model, "messages": [{"role": "user",
          "content": "One sentence on Hawkes processes."}],
          "max_tokens": 60, "temperature": 0.2})
print(f"3 short completions in {time.time()-t0:.1f}s")

with open(marker, "w") as f:
    f.write(f"passed {time.strftime('%Y-%m-%d %H:%M:%S')} model={model} tp=1 gpu=B200\n")
print("PASS - marker written; weekly heavy pass will target iai_B200_p.")
PY
RC=$?
kill $VLLM_PID 2>/dev/null || true
if [ $RC -eq 0 ] && [ -f "$MARKER" ]; then
    echo "GATE PASSED - $MARKER"
else
    echo "GATE FAILED - heavy tier stays off; nightly unaffected"
fi
exit $RC
