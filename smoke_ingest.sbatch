#!/bin/bash
#SBATCH --job-name=js_smoke_ingest
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/job_swarm/logs/smoke_ingest_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/smoke_ingest_%j.err

# Smoke test for the ingestion stage - CPU only, no GPU, no vLLM.
# Submit from the login node:   mkdir -p ~/job_swarm/logs && sbatch smoke_ingest.sbatch
# Watch it:                     tail -f ~/job_swarm/logs/smoke_ingest_<jobid>.out

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
    python3 $SWARM_DIR/js_main.py --stage ingest

echo "Smoke test complete. Raw payload is in $JOB_SWARM_DATA/raw/"
