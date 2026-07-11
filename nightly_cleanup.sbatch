#!/bin/bash
#SBATCH --job-name=js_cleanup
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/job_swarm/logs/cleanup_%j.out
#SBATCH --error=/home/%u/job_swarm/logs/cleanup_%j.err

# Nightly stage 3 - end-of-chain verdict. Runs afterany:<analyze jobid>, so it
# always fires and its log tells you at a glance whether the night produced a
# report. (vLLM/tunnel reaping happens inside nightly_analyze.sbatch itself -
# and SLURM kills any stragglers when that job ends.)

REPORT_DIR=/home/$USER/job_swarm_reports/$(date +%F)

if [ ! -f "$REPORT_DIR/REVIEW_QUEUE.md" ]; then
    echo "FAIL - no REVIEW_QUEUE.md in $REPORT_DIR"
    echo "Check the newest analyze log: ls -t /home/$USER/job_swarm/logs/analyze_*.out | head -1"
    exit 1
fi

# A report full of dead-LLM artifacts is a failure wearing a report's
# clothes (2026-07-03: vLLM worker crashed mid-run; every audit failed
# fast, zero memos drafted, and the old check still said PASS).
N_DOSSIERS=$(ls "$REPORT_DIR" 2>/dev/null | grep -c '^[0-9][0-9]_.*\.md')
if grep -q "Error connecting to vLLM" "$REPORT_DIR/REVIEW_QUEUE.md"; then
    echo "FAIL - report rendered but the LLM was unreachable during the run"
    echo "(REVIEW_QUEUE contains 'Error connecting to vLLM' artifacts)."
    echo "Check: ls -t /home/$USER/job_swarm/logs/analyze_*.err | head -1"
    exit 1
elif [ "$N_DOSSIERS" -eq 0 ]; then
    echo "WARN - report landed but ZERO memo dossiers were drafted."
    echo "Legitimate only if every audit scored below threshold; check the"
    echo "analyze log for vLLM crashes before trusting this."
    ls -la "$REPORT_DIR"
else
    echo "PASS - tonight's report landed ($N_DOSSIERS dossiers):"
    ls -la "$REPORT_DIR"
    echo ""
    echo "Tracker: /home/$USER/job_swarm_reports/TRACKER.md"
fi
