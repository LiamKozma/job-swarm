"""
Job Swarm - LangGraph state definition.

Follows the quant swarm's disk-pointer convention: heavy payloads (raw ingest
documents, embeddings, audit telemetry) live on Lustre scratch; only file paths
travel through the DAG so the state remains serialisation-safe.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator


class JobSwarmState(TypedDict):
    # LangGraph requires the operator.add annotation to append messages rather than overwrite
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # ---------------------------------------------------------------------
    # Disk pointers - written by one node, consumed by the next.
    # ---------------------------------------------------------------------
    # Written by profile_loader -> structured candidate profile JSON (+ .npy embedding)
    profile_path: Optional[str]
    # Newest raw ingest payload produced by the (separate) ingestion stage
    ingest_payload_path: Optional[str]
    # Written by trajectory_filter -> per-org δ-shift/GMM/alignment shortlist
    shortlist_path: Optional[str]
    # Written by llm_audit -> structured bottleneck audits for the shortlist
    audit_path: Optional[str]
    # Written by strategy_synthesis -> drafted technical memos (DRAFTS ONLY)
    memo_path: Optional[str]
    # Written by application_forge -> tailored notes/bullets for direct openings
    application_path: Optional[str]
    # Written by artifact_nominator -> proof-of-work artifact briefs
    artifact_path: Optional[str]
    # Written by compile_review -> directory containing the markdown review queue
    report_dir: Optional[str]

    # Lightweight run counters (orgs ingested / shortlisted / audited / drafted)
    run_stats: Dict[str, Any]
