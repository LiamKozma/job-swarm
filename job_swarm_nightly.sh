#!/bin/bash
# =============================================================================
# job_swarm_nightly.sh - Cron-triggered SLURM submission wrapper (Job Swarm)
#
# Thin submitter: the three stages live in real .sbatch files next to this
# script (nightly_ingest / nightly_analyze / nightly_cleanup), chained with
# SLURM dependencies so GPU hours are only burned after the CPU ingest lands.
#
# Why files and not heredocs: an earlier version generated the stage scripts
# inline via heredocs piped to sbatch stdin - that corrupted the submitted
# scripts (apptainer received the code directory as its "image"). Plain
# sbatch files are the pattern proven by quant_swarm_nightly.sh,
# smoke_ingest.sbatch, and backfill_ingest.sbatch. Do not go back.
#
# CRON SETUP (login node, `crontab -e`) - nightly at 04:00 UTC (23:00 EST):
#   0 4 * * * /scratch/YOUR_USER/quant_swarm/job_swarm/job_swarm_nightly.sh \
#       >> $HOME/job_swarm/logs/cron_submit.log 2>&1
#
# Deploy layout (see README.md):
#   /scratch/$USER/quant_swarm/job_swarm - code + data (this directory)
#   $HOME/job_swarm                          - state DB, logs (backed up)
#   $HOME/job_swarm_reports                  - the human review queue + TRACKER.md
# =============================================================================

set -euo pipefail

USER_ID="${USER:-$(id -un)}"
SWARM_DIR="/scratch/${USER_ID}/quant_swarm/job_swarm"
LOG_DIR="/home/${USER_ID}/job_swarm/logs"
REPORTS_DIR="/home/${USER_ID}/job_swarm_reports"

mkdir -p "$LOG_DIR" "$REPORTS_DIR" "$SWARM_DIR/data/raw" \
         "$SWARM_DIR/data/telemetry" "$SWARM_DIR/data/profile_cache"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Submitting job swarm pipeline..."

STAGE1_JOBID=$(sbatch --parsable "$SWARM_DIR/nightly_ingest.sbatch")
echo "  Stage 1 (ingestion)  JOBID=$STAGE1_JOBID"

# ---- GPU pool selection (census 2026-07-03) ---------------------------------
# Qwen3-Next is FP8 -> H100-only, so if the upgrade gate has passed we must
# queue for H100s (the sbatch header's default). Otherwise Llama-AWQ runs on
# the A100 pool, which has MORE good-driver nodes (6+: b6-1/2/3, b7-4, b8-4,
# ra4-2) and far less contention than the 4 usable H100 nodes - near-instant
# starts instead of multi-hour waits. sbatch CLI flags override the script's
# #SBATCH headers. Bad/unprobed A100s excluded; trim as probes land.
GPU_ARGS=()
if [ ! -f "$SWARM_DIR/data/VLLM2_OK" ]; then
    # Census complete 2026-07-03: 10 good A100 nodes (b6-1..4, b7-1..4,
    # b8-4, ra4-2 all on 590.48.01); only b8-1/2/3 run the bad 570 driver.
    GPU_ARGS=(--gres=gpu:A100:2 --exclude=b8-1,b8-2,b8-3,ra4-1)
    echo "  GPU pool: A100 (Llama-AWQ; Qwen gate not passed yet)"
elif [ -f "$SWARM_DIR/data/B200_OK" ] && [ "$(date -u +%u)" = "6" ]; then
    # Weekly HEAVY pass (2026-07-07, gated on test_b200.sbatch's marker):
    # Saturday 04:00 UTC = Friday night ET. 2x B200 (the per-user cap) =
    # 384GB FP8-capable - more memory than the whole 4xH100 node, on a
    # 2-deep queue instead of a 48-deep one. Runs Qwen3-Next FP8 TP=2
    # (analyze sbatch auto-detects B200); no critic at 2 GPUs. This branch
    # deliberately beats FORCE_A100 - rm data/B200_OK to kill the tier.
    GPU_ARGS=(--partition=iai_B200_p --gres=gpu:B200:2)
    echo "  GPU pool: B200 x2 (weekly heavy pass, Qwen3-Next FP8)"
elif [ -f "$HOME/job_swarm/state/FORCE_A100" ]; then
    # Manual override (2026-07-07): all 4 good-driver H100 nodes were held by
    # 7-day-wall jobs, wedging the nightly for days. touch/rm
    # ~/job_swarm/state/FORCE_A100 to flip; analyze sbatch auto-detects A100
    # -> Llama-3.3-AWQ + bge embeddings (corpus re-embeds on switch). Remove
    # the marker when the H100 queue clears to restore Qwen + dual-model.
    GPU_ARGS=(--gres=gpu:A100:2 --exclude=b8-1,b8-2,b8-3,ra4-1)
    echo "  GPU pool: A100 (Llama-AWQ; FORCE_A100 override active)"
elif [ "${JS_DUAL_MODEL:-1}" = "1" ]; then
    # Dual-model night (2026-07-06): 4 H100s = a whole node, so Qwen3-Next
    # (GPUs 0,1) drafts and Llama-3.3-AWQ (GPUs 2,3) criticizes with a kill
    # mandate - two model FAMILIES, the cheap defense against same-model
    # sycophancy. Costs queue time (full node on the 4 good-driver H100
    # hosts); export JS_DUAL_MODEL=0 in the crontab line to fall back to
    # the 2-GPU single-model job if waits get bad.
    GPU_ARGS=(--gres=gpu:H100:4)
    echo "  GPU pool: H100 x4 (Qwen3-Next primary + Llama-3.3 critic)"
else
    echo "  GPU pool: H100 (Qwen3-Next FP8 requires it; JS_DUAL_MODEL=0)"
fi

# afterany (not afterok): a partially-failed ingest still leaves the
# accumulated corpus in SQLite, so analysis remains worthwhile.
STAGE2_JOBID=$(sbatch --parsable --dependency=afterany:$STAGE1_JOBID \
    "${GPU_ARGS[@]}" "$SWARM_DIR/nightly_analyze.sbatch")
echo "  Stage 2 (analysis)   JOBID=$STAGE2_JOBID  [afterany:$STAGE1_JOBID]"

STAGE3_JOBID=$(sbatch --parsable --dependency=afterany:$STAGE2_JOBID \
    "$SWARM_DIR/nightly_cleanup.sbatch")
echo "  Stage 3 (verdict)    JOBID=$STAGE3_JOBID  [afterany:$STAGE2_JOBID]"

echo ""
echo "Dependency chain: $STAGE1_JOBID (ingest) -> $STAGE2_JOBID (analyze) -> $STAGE3_JOBID (verdict)"
echo "Monitor: squeue -u \$USER --format='%.10i %.20j %.8T %.10M %.9l %R'"
echo "Review queue lands in: $REPORTS_DIR/\$(date +%F)/REVIEW_QUEUE.md"
