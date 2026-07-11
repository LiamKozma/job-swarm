"""
Job Swarm - entrypoint.

Stages map to SLURM partitions:
  --stage ingest    CPU only (batch partition). Hits the public alpha-source
                    APIs and writes the raw payload to Lustre. No GPU, no vLLM.
  --stage backfill  CPU only, ONE-TIME deep-history sweep (~3 yrs of grants,
                    arXiv to depth 300, 12 months of HN threads, 45 days of
                    Form D). The next analyze absorbs it automatically -
                    every pending payload is consumed, not just the newest.
  --stage analyze   GPU (gpu_p partition, vLLM already listening on :8000).
                    Runs the LangGraph DAG: profile -> filter -> audit ->
                    synthesis -> review queue.
  --stage all       Both, sequentially (single-job / interactive testing).

Smoke test (login node or laptop, no GPU needed):
  python3 js_main.py --stage ingest
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def run_ingest(backfill: bool = False) -> str:
    from ingest_engines import run_all_ingestion
    from js_graph import RAW_DIR
    return await run_all_ingestion(RAW_DIR, backfill=backfill)


async def run_analyze(ingest_path=None):
    from js_graph import app, RAW_DIR
    from ingest_engines import latest_ingest_payload
    from js_state import JobSwarmState

    initial_state: JobSwarmState = {
        "messages": [{"role": "user", "content": "Nightly strategic placement run."}],
        "profile_path": None,
        "ingest_payload_path": ingest_path or latest_ingest_payload(RAW_DIR),
        "shortlist_path": None,
        "audit_path": None,
        "memo_path": None,
        "application_path": None,
        "report_dir": None,
        "run_stats": {},
    }
    final_state = await app.ainvoke(initial_state)

    print("\nJob Swarm run complete.")
    print(f"  Ingest payload -> {final_state.get('ingest_payload_path')}")
    print(f"  Shortlist      -> {final_state.get('shortlist_path')}")
    print(f"  Audits         -> {final_state.get('audit_path')}")
    print(f"  Memo drafts    -> {final_state.get('memo_path')}")
    print(f"  REVIEW QUEUE   -> {final_state.get('report_dir')}/REVIEW_QUEUE.md")
    print(f"  Stats          -> {final_state.get('run_stats')}")
    return final_state


async def main():
    parser = argparse.ArgumentParser(description="Sapelo2 Job Swarm")
    parser.add_argument("--stage", choices=["ingest", "backfill", "analyze", "all"],
                        default="all")
    args = parser.parse_args()

    ingest_path = None
    if args.stage == "backfill":
        ingest_path = await run_ingest(backfill=True)
        print(f"Backfill payload written: {ingest_path}\n"
              "It will be absorbed into the corpus by the next analyze stage.")
        return
    if args.stage in ("ingest", "all"):
        ingest_path = await run_ingest()
    if args.stage in ("analyze", "all"):
        await run_analyze(ingest_path)


if __name__ == "__main__":
    asyncio.run(main())
