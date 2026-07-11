"""
Job Swarm - persistent SQLite state store.

Lives in the ZFS home directory (small, backed up) per the storage topology:
heavy artefacts go to /scratch, but the accumulated document corpus, org
registry, audits, and memo drafts must survive the 30-day scratch purge.

The docs table is the swarm's long-term memory: every grant abstract, paper,
and startup description ever ingested is retained with its embedding, so the
per-organization δ-shift trajectory gets richer every night the swarm runs.
"""

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np

DB_PATH = os.environ.get(
    "JOB_SWARM_DB",
    os.path.expanduser("~/job_swarm/state/job_swarm.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
    org_key          TEXT PRIMARY KEY,
    display_name     TEXT NOT NULL,
    sources          TEXT NOT NULL DEFAULT '[]',
    website          TEXT,
    first_seen       TEXT NOT NULL,
    last_seen        TEXT NOT NULL,
    latest_prescore  REAL,
    latest_alignment REAL,
    latest_hurdle    REAL,
    regime           TEXT,
    -- new -> shortlisted -> audited -> memo_drafted -> contacted -> followed_up
    --   -> replied / rejected
    -- ('contacted', 'followed_up', 'replied', 'rejected' are set manually by
    --  the human operator via js_review.py; all four stop re-targeting)
    status           TEXT NOT NULL DEFAULT 'new',
    contacted_at     TEXT,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS docs (
    doc_id     TEXT PRIMARY KEY,
    org_key    TEXT NOT NULL,
    source     TEXT NOT NULL,
    date       TEXT,
    title      TEXT,
    url        TEXT,
    text       TEXT,
    contacts   TEXT,
    extra      TEXT,
    embedding  BLOB,
    added_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_org ON docs(org_key);
CREATE INDEX IF NOT EXISTS idx_docs_source ON docs(source);

-- ATS board resolution cache: which career-board provider/slug serves each
-- org, so slug probing costs one 404 per company *ever*, not per night.
CREATE TABLE IF NOT EXISTS ats_boards (
    org_key      TEXT NOT NULL,
    provider     TEXT NOT NULL,           -- greenhouse | lever | ashby
    slug         TEXT NOT NULL,
    status       TEXT NOT NULL,           -- resolved | miss
    last_checked TEXT NOT NULL,
    PRIMARY KEY (org_key, provider)
);

-- Nightly headcount snapshots per board. The time series of posting counts
-- is the raw signal for the hiring-acceleration estimator: a rising slope in
-- n_total with zero n_relevant means a team scaling around a gap the swarm
-- can predict (see ghost_engine).
CREATE TABLE IF NOT EXISTS board_snapshots (
    org_key    TEXT NOT NULL,
    snap_date  TEXT NOT NULL,             -- YYYY-MM-DD
    n_total    INTEGER NOT NULL,
    n_relevant INTEGER NOT NULL,
    PRIMARY KEY (org_key, snap_date)
);

CREATE TABLE IF NOT EXISTS audits (
    org_key                 TEXT NOT NULL,
    run_date                TEXT NOT NULL,
    bottleneck_diagnosis    TEXT,
    distribution_shift_risk INTEGER,
    alignment_score         REAL,
    intervention_vector     TEXT,
    raw_json                TEXT,
    PRIMARY KEY (org_key, run_date)
);

-- Raw ingest payloads already absorbed into the corpus. Lets the analyze
-- stage consume every pending payload (nightly + one-time backfills) exactly
-- once, instead of only the newest file.
CREATE TABLE IF NOT EXISTS payloads (
    fname        TEXT PRIMARY KEY,
    n_docs       INTEGER,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memos (
    org_key   TEXT NOT NULL,
    run_date  TEXT NOT NULL,
    subject   TEXT,
    body      TEXT,
    contacts  TEXT,
    -- Memos are ALWAYS drafts. Sending is a manual, human action.
    status    TEXT NOT NULL DEFAULT 'draft',
    PRIMARY KEY (org_key, run_date)
);

-- Small key-value store: active embedding model, one-shot migration flags,
-- OpenAlex affiliation cache (oa_aff:<arxiv_id> -> institution name).
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- One row per memo actually drafted: the features the swarm saw at draft
-- time plus (later) the human-observed outcome synced from TRACKER.md.
-- This is the supervised dataset that eventually replaces the hand-set
-- prescore weights with a fitted reply-probability model.
CREATE TABLE IF NOT EXISTS outreach_log (
    org_key     TEXT NOT NULL,
    drafted_at  TEXT NOT NULL,
    audit_align REAL,
    prescore    REAL,
    hurdle      REAL,
    recency     REAL,
    regime      TEXT,
    chosen_by   TEXT NOT NULL DEFAULT 'score',   -- score | explore (ε-greedy slot)
    outcome     TEXT,                            -- replied | dropped (from tracker)
    PRIMARY KEY (org_key, drafted_at)
);

-- Employer salary priors (DOL H-1B LCA disclosure medians via lca_engine.py).
-- Used to impute comp for postings with no disclosed salary.
CREATE TABLE IF NOT EXISTS salary_priors (
    org_key    TEXT NOT NULL,
    title_norm TEXT NOT NULL DEFAULT '*',
    p50        REAL NOT NULL,
    n          INTEGER NOT NULL,
    source     TEXT NOT NULL DEFAULT 'lca',
    PRIMARY KEY (org_key, title_norm)
);

-- Proof-of-work artifact briefs (artifact_nominator node). One nomination
-- per org per run; the recency guard keeps the same org from being
-- re-nominated night after night while the human decides whether to build.
CREATE TABLE IF NOT EXISTS artifact_briefs (
    org_key    TEXT NOT NULL,
    run_date   TEXT NOT NULL,
    title      TEXT,
    brief_json TEXT,
    PRIMARY KEY (org_key, run_date)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def org_key_from_name(name: str) -> str:
    """Normalizes an organization name into a stable dedup key."""
    key = re.sub(r"[^a-z0-9]+", "", (name or "").lower())
    # Strip common corporate suffixes so 'Acme Inc.' == 'Acme'
    for suffix in ("incorporated", "corporation", "inc", "llc", "ltd", "corp", "co"):
        if key.endswith(suffix) and len(key) > len(suffix) + 3:
            key = key[: -len(suffix)]
            break
    return key or hashlib.sha1((name or "unknown").encode()).hexdigest()[:12]


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for databases created by earlier versions."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(docs)").fetchall()}
    if "last_seen_at" not in cols:
        # last_seen_at: refreshed every time a source re-serves the doc, so a
        # posting that is STILL LIVE on a board stays in the review queue
        # instead of aging out 10 days after first discovery.
        conn.execute("ALTER TABLE docs ADD COLUMN last_seen_at TEXT")
        conn.execute("UPDATE docs SET last_seen_at = added_at")
        conn.commit()


# =========================================================================
# DOCS - long-term corpus with embeddings
# =========================================================================

def upsert_doc(conn: sqlite3.Connection, doc: dict) -> bool:
    """Inserts a document if unseen. Returns True when the doc is new.
    A re-seen doc refreshes last_seen_at - the liveness signal for postings."""
    org_key = org_key_from_name(doc.get("org", ""))
    now = _now()
    try:
        conn.execute(
            "INSERT INTO docs (doc_id, org_key, source, date, title, url, text, contacts, extra, added_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc["doc_id"], org_key, doc.get("source", ""), doc.get("date"),
                doc.get("title"), doc.get("url"),
                (doc.get("text") or "")[:40_000],
                json.dumps(doc.get("contacts") or {}),
                json.dumps(doc.get("extra") or {}),
                now, now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        conn.execute("UPDATE docs SET last_seen_at = ? WHERE doc_id = ?",
                     (now, doc["doc_id"]))
        return False


def touch_org(conn: sqlite3.Connection, name: str, source: str, website: Optional[str] = None) -> str:
    """Registers/refreshes an org row; returns its org_key."""
    org_key = org_key_from_name(name)
    now = _now()
    row = conn.execute("SELECT sources, website FROM orgs WHERE org_key = ?", (org_key,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO orgs (org_key, display_name, sources, website, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (org_key, name, json.dumps([source]), website, now, now),
        )
    else:
        sources = set(json.loads(row["sources"] or "[]")) | {source}
        conn.execute(
            "UPDATE orgs SET last_seen = ?, sources = ?, website = COALESCE(?, website) "
            "WHERE org_key = ?",
            (now, json.dumps(sorted(sources)), website or row["website"], org_key),
        )
    return org_key


def docs_missing_embeddings(conn: sqlite3.Connection) -> list:
    return conn.execute(
        "SELECT doc_id, text FROM docs WHERE embedding IS NULL AND text IS NOT NULL AND length(text) > 40"
    ).fetchall()


def store_embedding(conn: sqlite3.Connection, doc_id: str, vector: np.ndarray) -> None:
    conn.execute(
        "UPDATE docs SET embedding = ? WHERE doc_id = ?",
        (vector.astype(np.float32).tobytes(), doc_id),
    )


def org_doc_series(conn: sqlite3.Connection, org_key: str) -> list:
    """Time-ordered embedded documents for one org (the δ-shift trajectory input)."""
    rows = conn.execute(
        "SELECT doc_id, date, title, url, source, contacts, embedding, "
        "substr(text, 1, 900) AS excerpt FROM docs "
        "WHERE org_key = ? AND embedding IS NOT NULL "
        "ORDER BY COALESCE(date, added_at) ASC",
        (org_key,),
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "doc_id": r["doc_id"], "date": r["date"], "title": r["title"],
            "url": r["url"], "source": r["source"],
            "excerpt": r["excerpt"] or "",
            "contacts": json.loads(r["contacts"] or "{}"),
            "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
        })
    return out


# Statuses set by the human operator; the swarm must never re-target these.
HANDS_OFF_STATUSES = ("contacted", "followed_up", "replied", "rejected")


def active_org_keys(conn: sqlite3.Connection) -> list:
    """Orgs not manually closed out by the operator."""
    rows = conn.execute(
        "SELECT org_key, display_name, website, status FROM orgs "
        f"WHERE status NOT IN {HANDS_OFF_STATUSES!r}"
    ).fetchall()
    return [dict(r) for r in rows]


def followups_due(conn, min_days: int = 4) -> list:
    """Orgs you contacted ≥min_days ago with no reply and no follow-up yet."""
    rows = conn.execute(
        "SELECT org_key, display_name, contacted_at FROM orgs "
        "WHERE status = 'contacted' AND contacted_at IS NOT NULL "
        "AND substr(contacted_at, 1, 10) <= date('now', ?) "
        "ORDER BY contacted_at",
        (f"-{min_days} days",),
    ).fetchall()
    return [dict(r) for r in rows]


# =========================================================================
# SCORES / AUDITS / MEMOS
# =========================================================================

def update_org_scores(conn, org_key: str, prescore: float, alignment: float,
                      hurdle: float, regime: str, status: str) -> None:
    # Scores always refresh; status only advances from the pre-human states.
    # 'memo_drafted' is protected so an unsent draft stays visible in
    # TRACKER.md's "Ready to send" instead of silently reverting overnight.
    conn.execute(
        "UPDATE orgs SET latest_prescore = ?, latest_alignment = ?, latest_hurdle = ?, "
        "regime = ?, last_seen = ? WHERE org_key = ?",
        (prescore, alignment, hurdle, regime, _now(), org_key),
    )
    conn.execute(
        "UPDATE orgs SET status = ? WHERE org_key = ? AND status NOT IN "
        "('memo_drafted', 'contacted', 'followed_up', 'replied', 'rejected')",
        (status, org_key),
    )


def store_audit(conn, org_key: str, run_date: str, audit: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO audits "
        "(org_key, run_date, bottleneck_diagnosis, distribution_shift_risk, "
        " alignment_score, intervention_vector, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            org_key, run_date,
            audit.get("bottleneck_diagnosis"),
            1 if audit.get("distribution_shift_risk") else 0,
            float(audit.get("alignment_score") or 0.0),
            audit.get("intervention_vector"),
            json.dumps(audit),
        ),
    )


# =========================================================================
# ATS BOARD CACHE + JOB POSTINGS
# =========================================================================

def ats_resolution(conn, org_key: str):
    """Cached board lookups for an org: {provider: (slug, status)}."""
    rows = conn.execute(
        "SELECT provider, slug, status FROM ats_boards WHERE org_key = ?", (org_key,)
    ).fetchall()
    return {r["provider"]: (r["slug"], r["status"]) for r in rows}


def store_ats_resolution(conn, org_key: str, provider: str, slug: str, status: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ats_boards (org_key, provider, slug, status, last_checked) "
        "VALUES (?, ?, ?, ?, ?)",
        (org_key, provider, slug, status, _now()),
    )


def resolved_boards(conn) -> list:
    return [dict(r) for r in conn.execute(
        "SELECT org_key, provider, slug FROM ats_boards WHERE status = 'resolved'"
    ).fetchall()]


def unprobed_orgs(conn, limit: int) -> list:
    """Orgs never probed for an ATS board, oldest-seen first.
    PI-level orgs ('Name · Institution') and arXiv author pseudo-orgs
    ('Name group') never have careers boards - don't waste probe budget."""
    rows = conn.execute(
        "SELECT o.org_key, o.display_name, o.website FROM orgs o "
        "WHERE o.status NOT IN ('contacted', 'rejected') "
        "AND o.display_name NOT LIKE '% · %' "
        "AND o.display_name NOT LIKE '% group' "
        "AND NOT EXISTS (SELECT 1 FROM ats_boards b WHERE b.org_key = o.org_key) "
        "ORDER BY o.first_seen ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def record_board_snapshot(conn, org_key: str, n_total: int, n_relevant: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO board_snapshots (org_key, snap_date, n_total, n_relevant) "
        "VALUES (?, date('now'), ?, ?)",
        (org_key, n_total, n_relevant),
    )


def board_snapshot_series(conn, org_key: str, days: int = 28) -> list:
    """[(days_ago_float, n_total, n_relevant), ...] oldest first."""
    rows = conn.execute(
        "SELECT julianday('now') - julianday(snap_date) AS age, n_total, n_relevant "
        "FROM board_snapshots WHERE org_key = ? AND snap_date >= date('now', ?) "
        "ORDER BY snap_date ASC",
        (org_key, f"-{days} days"),
    ).fetchall()
    return [(float(r["age"]), int(r["n_total"]), int(r["n_relevant"])) for r in rows]


def recent_docs_by_source(conn, sources: tuple, days: int = 10) -> list:
    """Recently-SEEN embedded docs from the given sources (review-queue
    sections). Filters on last_seen_at, so a posting still live on its board
    stays included however long ago it was first discovered; days_open carries
    how long it has been in the corpus (the hard-to-fill signal)."""
    placeholders = ",".join("?" for _ in sources)
    rows = conn.execute(
        f"SELECT doc_id, org_key, source, date, title, url, contacts, extra, embedding, "
        f"added_at, round(julianday('now') - julianday(added_at), 1) AS days_open "
        f"FROM docs WHERE source IN ({placeholders}) "
        f"AND COALESCE(last_seen_at, added_at) >= datetime('now', ?) "
        f"AND embedding IS NOT NULL",
        (*sources, f"-{days} days"),
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "doc_id": r["doc_id"], "org_key": r["org_key"], "source": r["source"],
            "date": r["date"], "title": r["title"], "url": r["url"],
            "days_open": float(r["days_open"] or 0.0),
            "contacts": json.loads(r["contacts"] or "{}"),
            "extra": json.loads(r["extra"] or "{}"),
            "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
        })
    return out


def payload_processed(conn, fname: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM payloads WHERE fname = ?", (fname,)
    ).fetchone() is not None


def mark_payload_processed(conn, fname: str, n_docs: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO payloads (fname, n_docs, processed_at) VALUES (?, ?, ?)",
        (fname, n_docs, _now()),
    )


def store_memo(conn, org_key: str, run_date: str, subject: str, body: str, contacts: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO memos (org_key, run_date, subject, body, contacts, status) "
        "VALUES (?, ?, ?, ?, ?, 'draft')",
        (org_key, run_date, subject, body, json.dumps(contacts)),
    )


# =========================================================================
# META KV - embedding-model tracking, migration flags, small caches
# =========================================================================

def get_meta(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else default


def set_meta(conn, key: str, value) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                 (key, str(value)))


# =========================================================================
# OUTREACH LOG - the supervised dataset for calibrating prescore weights
# =========================================================================

def log_outreach(conn, org_key: str, audit_align, prescore, hurdle, recency,
                 regime, chosen_by: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO outreach_log "
        "(org_key, drafted_at, audit_align, prescore, hurdle, recency, regime, chosen_by) "
        "VALUES (?, date('now'), ?, ?, ?, ?, ?, ?)",
        (org_key, audit_align, prescore, hurdle, recency, regime, chosen_by),
    )


def record_outreach_outcome(conn, org_key: str, outcome: str) -> None:
    """Stamps the human-observed outcome (replied/dropped) from tracker marks."""
    conn.execute(
        "UPDATE outreach_log SET outcome = ? WHERE org_key = ? AND outcome IS NULL",
        (outcome, org_key),
    )


# =========================================================================
# SALARY PRIORS - H-1B LCA employer medians (populated by lca_engine.py)
# =========================================================================

def set_salary_prior(conn, org_key: str, p50: float, n: int,
                     title_norm: str = "*", source: str = "lca") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO salary_priors (org_key, title_norm, p50, n, source) "
        "VALUES (?, ?, ?, ?, ?)",
        (org_key, title_norm, p50, n, source),
    )


def salary_prior(conn, org_key: str):
    """(median_annual_salary, n_filings) for an employer, or None."""
    row = conn.execute(
        "SELECT p50, n FROM salary_priors WHERE org_key = ? AND title_norm = '*'",
        (org_key,),
    ).fetchone()
    return (float(row["p50"]), int(row["n"])) if row else None


# =========================================================================
# POSTING ARRIVALS - event times for the Hawkes hiring-burst estimator
# =========================================================================

def posting_arrival_ages(conn, org_key: str, window_days: int = 120) -> list:
    """Ages (days before now, float, ascending age) at which this org's
    postings FIRST appeared in the corpus. First-seen times are the arrival
    process the Hawkes estimator models - last_seen refreshes are not events."""
    rows = conn.execute(
        "SELECT julianday('now') - julianday(added_at) AS age FROM docs "
        "WHERE org_key = ? AND source IN ('ats_jobs', 'hn_hiring', 'usajobs') "
        "AND added_at >= datetime('now', ?) ORDER BY added_at ASC",
        (org_key, f"-{window_days} days"),
    ).fetchall()
    return [float(r["age"]) for r in rows]


def org_recent_embedding(conn, org_key: str, n_docs: int = 3):
    """Mean of the org's latest ≤n embedded docs - the same 'recent semantic
    state' org_metrics aligns against, cheap enough to call per-finalist."""
    rows = conn.execute(
        "SELECT embedding FROM docs WHERE org_key = ? AND embedding IS NOT NULL "
        "ORDER BY COALESCE(date, added_at) DESC LIMIT ?",
        (org_key, n_docs),
    ).fetchall()
    if not rows:
        return None
    vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return vecs.mean(axis=0)


# =========================================================================
# ARTIFACT BRIEFS - proof-of-work nominations (artifact_nominator node)
# =========================================================================

def store_artifact_brief(conn, org_key: str, run_date: str, title: str,
                         brief: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO artifact_briefs (org_key, run_date, title, brief_json) "
        "VALUES (?, ?, ?, ?)",
        (org_key, run_date, title, json.dumps(brief)),
    )


def recently_nominated_orgs(conn, days: int = 14) -> set:
    """Orgs already carrying a fresh artifact brief - don't re-nominate while
    the human is still deciding whether to build."""
    rows = conn.execute(
        "SELECT DISTINCT org_key FROM artifact_briefs "
        "WHERE run_date >= date('now', ?)", (f"-{days} days",),
    ).fetchall()
    return {r["org_key"] for r in rows}


# =========================================================================
# PI-LEVEL REKEY - universities are not a single research voice
# =========================================================================

def rekey_grant_docs_by_pi(conn) -> int:
    """
    Splits institution-level NSF/NIH docs into PI-level orgs ('PI · Institution').
    A university's δ-trajectory mixes unrelated departments, so the Hurdle
    State regime was measuring cross-department heterogeneity, not a team
    pivoting. Idempotent: docs already attached to a ' · ' org are skipped
    (new ingests emit PI-level names directly).
    """
    rows = conn.execute(
        "SELECT d.doc_id, d.contacts, d.source, o.display_name FROM docs d "
        "JOIN orgs o ON o.org_key = d.org_key "
        "WHERE d.source IN ('nsf', 'nih') AND o.display_name NOT LIKE '% · %'"
    ).fetchall()
    moved = 0
    for r in rows:
        contacts = json.loads(r["contacts"] or "{}")
        pi = (contacts.get("pi_name") or "").strip()
        if not pi:
            continue
        new_key = touch_org(conn, f"{pi} · {r['display_name']}", r["source"])
        conn.execute("UPDATE docs SET org_key = ? WHERE doc_id = ?",
                     (new_key, r["doc_id"]))
        moved += 1
    if moved:
        conn.commit()
        print(f"[DB] {moved} grant docs re-keyed to PI-level orgs")
    return moved


def rekey_url_mangled_orgs(conn) -> int:
    """
    One-shot repair for HN orgs ingested before the parser stripped URLs:
    'Kog ( https://kog.ai )' -> org_key 'koghttpsx2fx2fkogai'. Recomputes the
    clean name, moves the docs, carries the more-advanced status over to the
    clean org, and deletes the mangled row (unless the human already acted on
    it). Idempotent: cleaned names no longer match the LIKE filter.
    """
    import html as _html
    rows = conn.execute(
        "SELECT org_key, display_name, status, contacted_at FROM orgs "
        "WHERE display_name LIKE '%http%' OR display_name LIKE '%&#x%'"
    ).fetchall()
    moved = 0
    for r in rows:
        name = _html.unescape(r["display_name"])
        name = re.sub(r"\(\s*(?:https?://|www\.)[^)]*\)", " ", name)
        name = re.sub(r"(?:https?://|www\.)\S+", " ", name)
        name = re.sub(r"\s{2,}", " ", name).strip(" \t---|,;:")[:80]
        if not name or org_key_from_name(name) == r["org_key"]:
            continue
        if r["status"] in HANDS_OFF_STATUSES:
            continue  # human already engaged under the old key - leave it
        new_key = touch_org(conn, name, "rekey")
        conn.execute("UPDATE docs SET org_key = ? WHERE org_key = ?",
                     (new_key, r["org_key"]))
        # Preserve draft visibility: memo_drafted must survive the move
        if r["status"] == "memo_drafted":
            conn.execute(
                "UPDATE orgs SET status = 'memo_drafted' WHERE org_key = ? "
                f"AND status NOT IN {HANDS_OFF_STATUSES!r}", (new_key,))
        conn.execute("DELETE FROM orgs WHERE org_key = ?", (r["org_key"],))
        moved += 1
    if moved:
        conn.commit()
        print(f"[DB] {moved} URL-mangled orgs re-keyed to clean names")
    return moved


# =========================================================================
# REMOVAL CHANNEL - the negative-space signals (reposts, freezes)
# =========================================================================

def detect_reposts(conn, sim_threshold: float = 0.92, dead_after_days: int = 4,
                   window_days: int = 60) -> int:
    """
    A posting that vanishes from a board and reappears days later with a new
    native id is a REPOST - but reposts tell three different stories, and the
    2026 ghost-jobs evidence says treating them all as positive was wrong:

      revised   - text/salary changed, short gap -> genuinely failed to fill,
                  THE apply signal (extra.salary_up marks a raised offer:
                  a hiring manager getting desperate).
      evergreen - near-identical text (sim ≥ .995), same salary -> pipeline/
                  resume-harvesting requisition; downweight.
      churn     - long gap (>45d) since the old posting died -> the role
                  FILLED and the hire left; research the employer first.

    Stamps extra: repost_count, repost_kind, repost_gap_days,
    repost_similarity, salary_up; plus repost_mill=True on every flagged
    posting of an org where >30% of tonight's fresh postings are reposts
    (an evergreen mill - no repost there means anything).
    """
    from ats_engine import salary_values   # function-level: avoids import cycle
    fresh = conn.execute(
        "SELECT doc_id, org_key, extra, embedding FROM docs "
        "WHERE source = 'ats_jobs' AND embedding IS NOT NULL "
        "AND added_at >= datetime('now', '-2 days')"
    ).fetchall()
    if not fresh:
        return 0
    org_keys = {r["org_key"] for r in fresh}
    n_flagged = 0
    for org_key in org_keys:
        dead = conn.execute(
            "SELECT doc_id, extra, embedding, "
            "julianday('now') - julianday(COALESCE(last_seen_at, added_at)) AS gap "
            "FROM docs "
            "WHERE source = 'ats_jobs' AND org_key = ? AND embedding IS NOT NULL "
            "AND COALESCE(last_seen_at, added_at) < datetime('now', ?) "
            "AND COALESCE(last_seen_at, added_at) >= datetime('now', ?)",
            (org_key, f"-{dead_after_days} days", f"-{window_days} days"),
        ).fetchall()
        if not dead:
            continue
        dead_vecs = np.stack([np.frombuffer(d["embedding"], dtype=np.float32)
                              for d in dead])
        norms = np.maximum(np.linalg.norm(dead_vecs, axis=1), 1e-12)
        org_fresh = [r for r in fresh if r["org_key"] == org_key]
        org_flagged = []
        for f in org_fresh:
            v = np.frombuffer(f["embedding"], dtype=np.float32)
            nv = float(np.linalg.norm(v))
            if nv == 0.0:
                continue
            sims = (dead_vecs @ v) / (norms * nv)
            best = int(np.argmax(sims))
            sim = float(sims[best])
            if sim < sim_threshold:
                continue
            dead_extra = json.loads(dead[best]["extra"] or "{}")
            extra = json.loads(f["extra"] or "{}")
            gap_days = float(dead[best]["gap"] or 0.0)

            old_sal = salary_values(dead_extra.get("salary"))
            new_sal = salary_values(extra.get("salary"))
            salary_same = (old_sal == new_sal)
            salary_up = bool(old_sal and new_sal and new_sal[0] > old_sal[0])

            if gap_days > 45:
                kind = "churn"
            elif sim >= 0.995 and salary_same:
                kind = "evergreen"
            else:
                kind = "revised"

            extra.update({
                "repost_count": int(dead_extra.get("repost_count", 0)) + 1,
                "repost_kind": kind,
                "repost_gap_days": round(gap_days, 1),
                "repost_similarity": round(sim, 4),
                "salary_up": salary_up,
            })
            org_flagged.append((f["doc_id"], extra))
            n_flagged += 1
        # Evergreen mill: when a third of an org's fresh postings are reposts,
        # reposting is that org's normal posture, not a failed-fill signal.
        mill = len(org_fresh) >= 4 and len(org_flagged) / len(org_fresh) > 0.30
        for doc_id, extra in org_flagged:
            if mill:
                extra["repost_mill"] = True
            conn.execute("UPDATE docs SET extra = ? WHERE doc_id = ?",
                         (json.dumps(extra), doc_id))
    if n_flagged:
        conn.commit()
        print(f"[DB] {n_flagged} postings flagged as reposts "
              f"(kinds stamped: revised/evergreen/churn)")
    return n_flagged


# =========================================================================
# ENTITY RESOLUTION - one company, one org row
# =========================================================================
# Name normalization can't see that 'Meta' and 'META PLATFORMS INC' are the
# same company, but a shared website domain can. Suggestions only - merging
# is a human decision via `js_review.py merge <keep> <absorb>` because a
# shared domain is near-perfect but not perfect (agencies, portfolio sites).

_SHARED_HOSTS = {
    "github.com", "github.io", "linkedin.com", "twitter.com", "x.com",
    "greenhouse.io", "lever.co", "ashbyhq.com", "myworkdayjobs.com",
    "notion.site", "google.com", "docs.google.com", "youtube.com",
    "substack.com", "medium.com", "angel.co", "wellfound.com", "ycombinator.com",
}
_MULTI_TLDS = {"co.uk", "ac.uk", "com.au", "co.jp", "co.in", "com.br"}


def registrable_domain(url: str):
    """Naive eTLD+1: 'https://careers.acme.com/x' -> 'acme.com'. None for
    shared hosts, .edu/.gov (departments of one university are NOT one org),
    and unparseable values."""
    host = re.sub(r"^[a-z]+://", "", (url or "").lower()).split("/")[0].split(":")[0]
    if not host or "." not in host:
        return None
    labels = host.split(".")
    domain = ".".join(labels[-3:]) if ".".join(labels[-2:]) in _MULTI_TLDS \
        else ".".join(labels[-2:])
    if domain in _SHARED_HOSTS or host in _SHARED_HOSTS:
        return None
    if labels[-1] in ("edu", "gov", "mil") or ".".join(labels[-2:]).endswith(".edu"):
        return None
    return domain


def suggest_domain_merges(conn) -> list:
    """[(domain, [(org_key, display_name), ...]), ...] where ≥2 org rows share
    one registrable website domain - near-certain duplicates for human review."""
    groups: dict = {}
    for r in conn.execute(
            "SELECT org_key, display_name, website FROM orgs "
            "WHERE website IS NOT NULL AND website != ''").fetchall():
        dom = registrable_domain(r["website"])
        if dom:
            groups.setdefault(dom, []).append((r["org_key"], r["display_name"]))
    return sorted(((d, orgs) for d, orgs in groups.items() if len(orgs) >= 2),
                  key=lambda t: -len(t[1]))


# Tables where absorbed-org rows move to the surviving org. PK collisions
# (same run_date etc. on both sides) keep the SURVIVOR's row.
_MERGE_TABLES = ("ats_boards", "board_snapshots", "audits", "memos",
                 "outreach_log", "artifact_briefs")


def merge_orgs(conn, keep_key: str, absorb_key: str) -> None:
    """Folds org `absorb_key` into `keep_key`: docs and per-org rows move,
    sources union, the more-advanced human status survives, absorbed row is
    deleted. Irreversible - the CLI asks for confirmation."""
    keep = conn.execute("SELECT * FROM orgs WHERE org_key = ?", (keep_key,)).fetchone()
    absorb = conn.execute("SELECT * FROM orgs WHERE org_key = ?", (absorb_key,)).fetchone()
    if keep is None or absorb is None:
        raise ValueError(f"unknown org_key: {keep_key if keep is None else absorb_key}")
    conn.execute("UPDATE docs SET org_key = ? WHERE org_key = ?",
                 (keep_key, absorb_key))
    for table in _MERGE_TABLES:
        conn.execute(f"UPDATE OR IGNORE {table} SET org_key = ? WHERE org_key = ?",
                     (keep_key, absorb_key))
        conn.execute(f"DELETE FROM {table} WHERE org_key = ?", (absorb_key,))
    sources = sorted(set(json.loads(keep["sources"] or "[]"))
                     | set(json.loads(absorb["sources"] or "[]")))
    conn.execute("UPDATE orgs SET sources = ?, website = COALESCE(website, ?) "
                 "WHERE org_key = ?",
                 (json.dumps(sources), absorb["website"], keep_key))
    # The human's engagement state must survive the merge, whichever side held it
    if absorb["status"] in HANDS_OFF_STATUSES and keep["status"] not in HANDS_OFF_STATUSES:
        conn.execute(
            "UPDATE orgs SET status = ?, contacted_at = COALESCE(?, contacted_at) "
            "WHERE org_key = ?",
            (absorb["status"], absorb["contacted_at"], keep_key))
    conn.execute("DELETE FROM orgs WHERE org_key = ?", (absorb_key,))
    conn.commit()


def warn_org_keys(conn, days: int = 90) -> set:
    """Orgs that filed a WARN layoff notice recently - outreach there is
    wasted (freezing/shrinking), same treatment as a dark careers board."""
    rows = conn.execute(
        "SELECT DISTINCT org_key FROM docs WHERE source = 'warn' "
        "AND COALESCE(date, added_at) >= date('now', ?)",
        (f"-{days} days",),
    ).fetchall()
    return {r["org_key"] for r in rows}


def frozen_org_keys(conn, quiet_days: int = 14, prior_min_postings: int = 3) -> set:
    """
    Orgs whose board went dark: latest snapshot shows 0 postings after having
    ≥prior_min_postings within the last 45 days, with nothing new since.
    A hiring freeze - memo money spent here is wasted for a month or two.
    """
    rows = conn.execute(
        "SELECT org_key, "
        "       (SELECT n_total FROM board_snapshots b2 WHERE b2.org_key = b.org_key "
        "        ORDER BY snap_date DESC LIMIT 1) AS latest_total, "
        "       MAX(n_total) AS peak_total, "
        "       MAX(snap_date) AS latest_snap "
        "FROM board_snapshots b "
        "WHERE snap_date >= date('now', '-45 days') "
        "GROUP BY org_key"
    ).fetchall()
    frozen = set()
    for r in rows:
        if (r["latest_total"] == 0 and (r["peak_total"] or 0) >= prior_min_postings
                and r["latest_snap"] >= _now()[:10]):
            frozen.add(r["org_key"])
    return frozen
