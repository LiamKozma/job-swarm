#!/bin/bash
#SBATCH --job-name=js_ingest
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/job_swarm/logs/ingest_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/ingest_%j.err

# Nightly stage 1 - alpha-source ingestion (CPU only).
# Same proven pattern as smoke_ingest.sbatch. Submitted by job_swarm_nightly.sh;
# can also be run standalone: sbatch nightly_ingest.sbatch

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

echo "Ingestion stage complete."
