#!/bin/bash
#SBATCH --job-name=js_vllm2_test
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:H100:2
#SBATCH --exclude=ra5-1,ra5-3,ra5-4,ra7-1,ra7-3,ra7-4,ra8-1,ra8-2
#SBATCH --time=01:30:00
#SBATCH --output=/home/%u/job_swarm/logs/vllm2_test_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/vllm2_test_%j.err

# GATE for the LLM upgrade (task #25): boots the NEW container + Qwen3-Next
# on 2 good-driver H100s and runs a real memo-shaped guided-JSON request.
# Writes data/VLLM2_OK ONLY on full success - nightly_analyze switches to
# the new stack iff that marker exists, so a broken build can never take
# down a nightly. Requires build_vllm2.sbatch + download_qwen_next.sbatch.

set -uo pipefail
SWARM_DIR=/scratch/$USER/quant_swarm/job_swarm
# Probe 2026-07-03: the EXISTING container already has vLLM 0.22.1 - new
# enough for Qwen3-Next. Prefer the fresh build if it exists, else use it.
if [ -f /scratch/$USER/quant_swarm/vllm2_inference.sif ]; then
    SIF=/scratch/$USER/quant_swarm/vllm2_inference.sif
else
    SIF=/scratch/$USER/quant_swarm/vllm_inference.sif
    echo "NOTE: vllm2_inference.sif absent - testing on the existing container "
    echo "(vLLM 0.22.1, verified new enough)."
fi
MODEL=/data/models/qwen3-next-80b-a3b-instruct-fp8
MARKER=$SWARM_DIR/data/VLLM2_OK
rm -f "$MARKER"
echo "container: $SIF"

[ -d "/scratch/$USER/data/models/qwen3-next-80b-a3b-instruct-fp8" ] || {
    echo "FAIL: model missing (run download_qwen_next.sbatch)"; exit 1; }

echo "Launching vLLM (new container) with Qwen3-Next FP8, TP=2..."
apptainer exec --nv --ipc \
    --env http_proxy="" --env https_proxy="" --env all_proxy="" \
    --env NVIDIA_REQUIRE_CUDA="cuda>=12.0" \
    --bind /scratch/$USER/data:/data:ro \
    "$SIF" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --gdn-prefill-backend triton \
    --port 8000 &
VLLM_PID=$!
# --gdn-prefill-backend triton: the FlashInfer GDN kernel JIT-compiles on
# first use, which failed inside the read-only container and killed the
# engine right after it bound the port (gate run 2026-07-03: "listening"
# then connection refused). Triton kernels ship precompiled.

for i in $(seq 1 240); do   # MoE + FP8 load can take a while; 20 min cap
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FAIL: vLLM crashed during startup (check stderr log)"; exit 1
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
# it). Poll /v1/models for HTTP 200, up to 10 minutes, riding out refused
# connections from restarts/warmup.
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

# throughput sanity: 3 concurrent-ish calls shouldn't take forever on MoE
t0 = time.time()
for _ in range(3):
    call({"model": model, "messages": [{"role": "user",
          "content": "One sentence on Hawkes processes."}],
          "max_tokens": 60, "temperature": 0.2})
print(f"3 short completions in {time.time()-t0:.1f}s")

with open(marker, "w") as f:
    f.write(f"passed {time.strftime('%Y-%m-%d %H:%M:%S')} model={model}\n")
print("PASS - marker written; nightly_analyze will use the new stack.")
PY
RC=$?
kill $VLLM_PID 2>/dev/null || true
if [ $RC -eq 0 ] && [ -f "$MARKER" ]; then
    echo "GATE PASSED - $MARKER"
else
    echo "GATE FAILED - nightly stays on the proven Llama stack"
fi
exit $RC
