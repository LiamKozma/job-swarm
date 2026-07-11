#!/bin/bash
#SBATCH --job-name=js_backfill
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/job_swarm/logs/backfill_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/backfill_%j.err

# ONE-TIME deep-history sweep so the δ-shift/GMM machinery has dense
# trajectories on day one: ~3 years of NSF/NIH grants (paginated), arXiv to
# depth 300 per query, the past 12 HN Who-is-Hiring threads, 45 days of
# Form D raises (still inside their 30-90 day hiring window).
#
# Safe to re-run: doc_ids are stable, so duplicates are skipped on upsert.
# The payload is absorbed automatically by the next nightly analyze stage
# (it consumes every pending payload, not just the newest).
#
# Submit from the login node:   sbatch backfill_ingest.sbatch

set -euo pipefail

SWARM_DIR=/scratch/$USER/quant_swarm/job_swarm

export JOB_SWARM_DATA=$SWARM_DIR/data
export JOB_SWARM_DB=/home/$USER/job_swarm/state/job_swarm.db
export JOB_SWARM_CONFIG=$SWARM_DIR/job_swarm_config.json
mkdir -p "$JOB_SWARM_DATA/raw" "$(dirname "$JOB_SWARM_DB")"

apptainer exec \
    --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --bind $HOME:$HOME \
    --bind /scratch/$USER:/scratch/$USER \
    /scratch/$USER/quant_swarm/agent_env.sif \
    python3 $SWARM_DIR/js_main.py --stage backfill

echo "Backfill complete. Payload in $JOB_SWARM_DATA/raw/ - next analyze absorbs it."
