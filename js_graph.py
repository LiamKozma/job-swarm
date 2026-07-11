"""
Job Swarm - LangGraph DAG (analysis stage, runs alongside vLLM on gpu_p).

    profile_loader -> trajectory_filter -> llm_audit -> strategy_synthesis
                             |
    application_forge -> artifact_nominator -> compile_review

Same conventions as the quant swarm: async nodes, disk-pointer state, the
70B model reached over the local vLLM OpenAI endpoint. The terminal node
produces a HUMAN review queue - the swarm drafts, the candidate sends.
"""

import ast
import asyncio
import glob
import hashlib
import html
import json
import os
import random
import re
from datetime import datetime

import aiohttp
import numpy as np
from langgraph.graph import END, START, StateGraph

import profile_engine
import swarm_db
import trajectory_engine
from ingest_engines import latest_ingest_payload
from js_state import JobSwarmState

# ---------------------------------------------------------------------------
# Paths - override on the cluster via environment (see job_swarm_nightly.sh)
# ---------------------------------------------------------------------------
DATA_ROOT     = os.environ.get("JOB_SWARM_DATA", "./data")
RAW_DIR       = os.path.join(DATA_ROOT, "raw")
TELEMETRY_DIR = os.path.join(DATA_ROOT, "telemetry")
PROFILE_CACHE = os.path.join(DATA_ROOT, "profile_cache")
REPORTS_DIR   = os.environ.get(
    "JOB_SWARM_REPORTS", os.path.expanduser("~/job_swarm_reports")
)

VLLM_URL   = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "/data/models/llama-3-70b-instruct-awq")
# Cross-model critic (2026-07-06): a SECOND base model, different lab/family,
# served on its own port when the night got 4 GPUs (nightly_analyze.sbatch
# sets both envs, or neither). Same-model self-review rewards its own
# blind spots; a cross-family critic with a kill mandate is the cheapest
# defense. Unset -> critic calls transparently run on the primary model.
VLLM_CRITIC_URL   = os.environ.get("VLLM_CRITIC_URL", "")
VLLM_CRITIC_MODEL = os.environ.get("VLLM_CRITIC_MODEL", "")

AUDIT_TOP_N = int(os.environ.get("JOB_SWARM_AUDIT_N", "60"))
# 8 (was 12): the 2026-07-05 audit found the marginal slots went to
# undeliverable or dead-end targets; depth beats volume at realistic
# 8-12% reply rates, and the human sends 2-5/week anyway.
MEMO_TOP_M  = int(os.environ.get("JOB_SWARM_MEMO_M", "8"))
# 0.45 (was 0.55): the auditor is instructed to be calibrated ("most orgs
# below 0.5"), so a 0.55 hard floor left 11 of 12 memo slots empty nightly.
MEMO_MIN_ALIGNMENT = float(os.environ.get("JOB_SWARM_MEMO_MIN_ALIGN", "0.45"))
# ε-greedy: reserve up to this many memo slots for randomly-drawn orgs from
# the 0.30-floor band. Without exploration the prescore weights can never be
# calibrated against reply outcomes - there's no counterfactual data.
EXPLORE_SLOTS = int(os.environ.get("JOB_SWARM_EXPLORE_SLOTS", "2"))
APP_TOP_K   = int(os.environ.get("JOB_SWARM_APP_K", "8"))

# Shared writing constraints for everything a human might send. The goal is
# text that reads like a busy engineer wrote it, because a human will edit
# and send it under their own name.
_STYLE_RULES = (
    "STYLE RULES (hard constraints):\n"
    "- Banned words/phrases: 'leverage', 'spearheaded', 'passionate', 'delve', "
    "'seamless(ly)', 'cutting-edge', 'showcase', 'utilize', 'robust', 'excited to', "
    "'I am writing to', 'aligns perfectly', 'proven track record', 'hit the ground "
    "running'.\n"
    "- No exclamation points. No rhetorical questions. No bullet-point emoji.\n"
    "- Prefer short declarative sentences; vary length; concrete nouns and numbers "
    "over adjectives.\n"
    "- NEVER invent numbers, metrics, or accomplishments. Where a quantity would "
    "strengthen a claim but is unknown, insert a bracketed prompt for the human, "
    "e.g. '[ADD REAL NUMBER: dataset row count]'.\n"
    "- If it would sound natural on LinkedIn, rewrite it."
)

_LLM_SEMAPHORE = asyncio.Semaphore(4)   # vLLM batches internally; 4 in flight is plenty

# ---------------------------------------------------------------------------
# Pre-send lint (2026-07-05 audit M5): mechanical send-safety net under the
# LLM verify pass. A draft that trips any pattern must not be sent as-is;
# the reasons ride on the dossier and the inbox card so the human sees WHY.
# ---------------------------------------------------------------------------
_LINT_PATTERNS = (
    (re.compile(r"\[ADD [^\]]*\]?"), "unfilled [ADD ...] placeholder"),
    (re.compile(r"alexmorgan-smoketest|alex\s+morgan", re.I),
     "cites the synthetic smoke-test profile"),
    (re.compile(r"https?://|\bwww\."), "URL in body (first-touch spam trigger)"),
)


def _lint_draft(*texts) -> list:
    """Reasons this draft is not sendable as written (empty list = clean)."""
    reasons = []
    for t in texts:
        if not t:
            continue
        for rx, why in _LINT_PATTERNS:
            if rx.search(str(t)) and why not in reasons:
                reasons.append(why)
    return reasons


# =====================================================================
# ASYNC vLLM HELPER (same pattern as graph.py, longer generation budget)
# =====================================================================

async def query_vllm(system_prompt: str, user_prompt: str, max_tokens: int = 1200,
                     schema: dict = None, critic: bool = False) -> str:
    """
    schema: optional JSON schema for vLLM guided decoding (guided_json) -
    the model CANNOT emit malformed JSON when it's honored. If the server
    rejects the parameter (older vLLM), the call transparently retries
    unguided and the _extract_json parse ladder still applies downstream.
    critic=True routes to the second-family model when one is serving
    (VLLM_CRITIC_URL/_MODEL); otherwise it falls back to the primary, so
    callers never need to know whether tonight is a dual-model night.
    """
    url = VLLM_CRITIC_URL if (critic and VLLM_CRITIC_URL) else VLLM_URL
    model = VLLM_CRITIC_MODEL if (critic and VLLM_CRITIC_URL) else VLLM_MODEL

    async def _once(extra: dict) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            **extra,
        }
        timeout = aiohttp.ClientTimeout(total=300)
        async with _LLM_SEMAPHORE:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]

    if schema is not None:
        try:
            return await _once({"guided_json": schema})
        except Exception:
            pass  # older vLLM without guided decoding - fall through
    try:
        return await _once({})
    except Exception as e:
        return f"Error connecting to vLLM: {e}"


# JSON schemas for guided decoding (mirror the prompt-specified shapes)
_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "organization_name": {"type": "string"},
        "bottleneck_diagnosis": {"type": "string"},
        "distribution_shift_risk": {"type": "boolean"},
        "alignment_score": {"type": "number"},
        "intervention_vector": {"type": "string"},
    },
    "required": ["organization_name", "bottleneck_diagnosis",
                 "distribution_shift_risk", "alignment_score",
                 "intervention_vector"],
}
_MEMO_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
    "required": ["subject", "body"],
}
_APP_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "string"},
        "cover_letter": {"type": "string"},
        "resume_bullets": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"theme": {"type": "string"},
                                     "bullet": {"type": "string"}},
                      "required": ["theme", "bullet"]},
        },
        "keywords_matched": {"type": "array", "items": {"type": "string"}},
        "keywords_missing": {"type": "array", "items": {"type": "string"}},
        "likely_interview_topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["note", "cover_letter", "resume_bullets", "keywords_matched",
                 "keywords_missing", "likely_interview_topics"],
}


def _evidence_block(org: dict, n: int = 6, excerpt_chars: int = 700) -> str:
    """Evidence WITH text excerpts - the auditor and memo writer must ground
    claims in actual document content, not titles."""
    lines = []
    for e in org.get("evidence", [])[-n:]:
        lines.append(f"- [{e['source']}] ({e['date']}) {e['title']} - {e['url']}")
        excerpt = (e.get("excerpt") or "").strip()
        if excerpt:
            lines.append(f"  EXCERPT: {excerpt[:excerpt_chars]}")
    return "\n".join(lines)


# Seniority heuristics for posting ranking: an MS new grad won't clear a
# Staff/Manager screen, and 'graduate program' style roles are the archetype
# to hunt (e.g. IMC Graduate Quant Researcher at $250-300k).
_SENIOR_RE = re.compile(
    r"\b(senior|staff|principal|lead|manager|director|head|vp|vice president|"
    r"distinguished|supervisory|intern(?:ship)?)\b", re.I)
_EARLY_RE = re.compile(
    r"\b(graduate|new grad|entry[- ]level|junior|early[- ]career|campus|"
    r"university grad)\b", re.I)


def _score_posting(p: dict) -> float:
    """fit × repost-forensics × ghost-flags × clearance × seniority.

    NO staleness boost - 2026 red-team REFUTED it: posting age predicts
    ghost jobs, not hard-to-fill desperation, so age is display-only and
    only the unmaintained penalty survives. Reposts: 'revised' keeps a
    TRIMMED boost (our native-ID boards have no LinkedIn auto-renewal
    noise, but indecisive-committee risk is real); a RAISED salary remains
    the strongest tell. 'evergreen'/mill downweighted; 'churn' tagged for
    the human. PERM ads are visa-compliance (~zero external hire).
    Clearance-sponsoring postings get a small boost: US-citizen candidate,
    a fraction of the applicant pool (red-team section 2)."""
    ex = p.get("extra") or {}
    flags = set(ex.get("ghost_flags") or [])
    score = p.get("align", 0.0)
    kind = ex.get("repost_kind") or ("revised" if ex.get("repost_count") else None)
    if ex.get("repost_mill") or kind == "evergreen":
        score *= 0.60
    elif kind == "revised":
        score *= 1.0 + 0.05 * min(int(ex.get("repost_count") or 0), 2)
        if ex.get("salary_up"):
            score *= 1.10
    if ex.get("clearance_sponsor"):
        score *= 1.10
    # churn: no boost, no penalty - the tag warns the human instead
    if "evergreen_title" in flags:
        score *= 0.50
    if "perm_language" in flags:
        score *= 0.50
    if "unmaintained_45d" in flags:
        score *= 0.80
    if "wide_salary_band" in flags:
        score *= 0.90
    title = p.get("title") or ""
    if _EARLY_RE.search(title):
        score *= 1.15
    elif _SENIOR_RE.search(title):
        score *= 0.75
    if not _is_us_tier(ex.get("location")):
        score *= _NON_US_SCORE_MULT
    return score


# Hard-constraint gates (2026-07-07 matching-report edit). Cosine treats a
# posting's requirements as topic overlap; a "requires active TS/SCI" or "PhD
# required, 8+ years" line is a BINARY wall for a US-citizen MS-level early-
# career candidate, not a similarity signal. Regex-first (deterministic, no LLM
# spend): a confident hard miss multiplies the posting score down so it falls
# out of the queue, with the reason surfaced on the card. Deliberately
# conservative - only unambiguous walls fire, and "preferred"/"or equivalent"/
# "ability to obtain" phrasings are explicitly spared.
_YOE_RE = re.compile(
    r"(\d{1,2})\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)[\s\w]{0,24}?"
    r"(?:experience|exp\b)", re.I)
_PHD_REQ_RE = re.compile(
    r"\b(ph\.?\s?d|doctorate|doctoral)\b(?![^.]{0,40}\b(?:preferred|"
    r"a plus|nice to have|or equivalent|or ms|or master)\b)", re.I)
_CLEAR_HELD_RE = re.compile(
    r"\b(?:active|current|existing|must\s+(?:have|possess)|maintain\s+an?)\b"
    r"[^.]{0,30}\b(clearance|ts/sci|top\s*secret|security\s+clearance)\b", re.I)
_CLEAR_OBTAIN_RE = re.compile(
    r"\b(?:ability|able|eligible|eligibility)\b[^.]{0,30}\bobtain\b[^.]{0,20}"
    r"\bclearance\b", re.I)


def _hard_constraints(text: str, title: str) -> tuple:
    """(multiplier, reasons). multiplier<1 means a hard requirement the
    candidate cannot meet was found; reasons are shown on the card. Only
    unambiguous walls clamp - everything else passes at 1.0."""
    blob = f"{title or ''}. {text or ''}"
    reasons = []
    mult = 1.0
    # Years of experience: take the MAX stated requirement. 8+ is a hard wall
    # for an early-career candidate; 6-7 is a strong penalty, not a zero (some
    # "6+ years" reqs still interview a strong new grad).
    yoe = [int(m) for m in _YOE_RE.findall(blob) if m.isdigit()]
    max_yoe = max(yoe) if yoe else 0
    if max_yoe >= 8:
        mult *= 0.05
        reasons.append(f"requires {max_yoe}+ years experience")
    elif max_yoe >= 6:
        mult *= 0.35
        reasons.append(f"requires {max_yoe}+ years experience")
    # Active clearance ALREADY held (distinct from a sponsor who clears you,
    # and from "ability to obtain" which a US citizen can): a true wall.
    if _CLEAR_HELD_RE.search(blob) and not _CLEAR_OBTAIN_RE.search(blob):
        mult *= 0.05
        reasons.append("requires an active security clearance (not sponsored)")
    # PhD strictly required (spared when preferred / or-equivalent / or-MS).
    if _PHD_REQ_RE.search(blob):
        mult *= 0.20
        reasons.append("PhD required")
    return mult, reasons


def _lca_demand_boost(conn, org_key: str) -> float:
    """H-1B LCA filings as a DEMAND signal, not just a salary prior: an
    employer repeatedly attesting wages for stat/ML titles demonstrably
    hires this role and can't fill it domestically. Small multiplicative
    boost, log-damped (n=10 -> ~1.06×, n=100 -> ~1.12×, capped ~1.12×)."""
    prior = swarm_db.salary_prior(conn, org_key)
    if not prior:
        return 1.0
    return 1.0 + 0.06 * min(float(np.log10(1 + prior[1])), 2.0)


def _extract_json(text: str, opener: str = "{", closer: str = "}"):
    """Parse-ladder for LLM output: strict JSON, then Python-dict style
    (models sometimes mimic single quotes), then light repairs."""
    start, end = text.find(opener), text.rfind(closer)
    if start == -1 or end == -1:
        return None
    frag = text[start: end + 1]
    try:
        return json.loads(frag)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(frag)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    repaired = (frag.replace("“", '"').replace("”", '"')
                    .replace("‘", "'").replace("’", "'"))
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except Exception:
        return None


# =====================================================================
# NODE 1 - PROFILE LOADER
# =====================================================================

async def profile_loader(state: JobSwarmState):
    print("[Node] profile_loader")
    # Deep-research people paste ingest: bind named humans from last night's
    # Gemini paste to their orgs (opens outreach gates before the memo node
    # runs). Best-effort - a bad/absent paste never blocks the pipeline.
    try:
        import research_engine
        research_engine.ingest_research()
    except Exception as e:
        print(f"[Research] people ingest skipped (non-fatal): {e}")
    result = await profile_engine.build_candidate_profile(query_vllm, PROFILE_CACHE)
    profile = result["profile"]
    tag = "rebuilt from changed documents" if result["rebuilt"] else "loaded from cache"
    if result["rebuilt"]:
        # Profile changed -> every unsent draft cites the OLD profile and can
        # no longer be trusted (2026-07-05 audit M5: the synthetic-persona
        # drafts must die the day the real resume lands, mechanically, not by
        # the human remembering). Supersede the drafts and put their orgs
        # back into rotation so the next run re-drafts against the new
        # profile. Sent/replied/rejected orgs are untouched.
        conn = swarm_db.connect()
        n_memo = conn.execute(
            "UPDATE memos SET status = 'superseded' WHERE status = 'draft'"
        ).rowcount
        n_org = conn.execute(
            "UPDATE orgs SET status = 'audited' WHERE status = 'memo_drafted'"
        ).rowcount
        conn.commit()
        conn.close()
        if n_memo or n_org:
            print(f"[Profile] rebuilt -> {n_memo} stale draft(s) superseded, "
                  f"{n_org} org(s) returned to rotation")
    summary = (
        f"Candidate profile {tag}.\n"
        f"Headline: {profile.get('headline')}\n"
        f"Expertise matrix: {', '.join(profile.get('expertise_matrix') or [])[:600]}"
    )
    return {
        "messages": [{"role": "system", "name": "ProfileLoader", "content": summary}],
        "profile_path": result["profile_path"],
        "run_stats": {"profile_rebuilt": result["rebuilt"]},
    }


# =====================================================================
# NODE 2 - TRAJECTORY FILTER (embeddings + δ-shift GMM + Euler-Maruyama)
# =====================================================================

DECISIONS_PATH = os.path.expanduser("~/job_swarm/state/decisions.jsonl")


def _absorb_decisions(conn) -> None:
    """Apply dashboard inbox decisions (synced from the repo by the
    publisher). 'never' -> org permanently rejected; 'skip'/'applied' on a
    posting -> its URL never resurfaces in openings/forge. Append-only and
    idempotent - the whole file re-applies every night."""
    if not os.path.exists(DECISIONS_PATH):
        return
    # Funnel instrumentation (2026-07-06, pre-registered in the data repo's
    # jobs/PREREGISTRATION.md): every decision line is also an event in
    # funnel_events, keyed uniquely so the append-only file re-applies
    # idempotently. This is the table the weekly review and the 30/60/90-day
    # decision points read - it is the only evidence this system will have.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS funnel_events ("
        " ts TEXT NOT NULL, item_id TEXT, action TEXT NOT NULL,"
        " kind TEXT, lane TEXT, org_key TEXT, url TEXT,"
        " freshness_days REAL, artifact_depth TEXT,"
        " UNIQUE(ts, item_id, action))")
    never, urls, found_contacts, liked = set(), set(), {}, {}
    n_events = 0
    with open(DECISIONS_PATH, errors="replace") as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            act = d.get("action")
            if act and d.get("ts"):
                n_events += conn.execute(
                    "INSERT OR IGNORE INTO funnel_events (ts, item_id, action,"
                    " kind, lane, org_key, url, freshness_days, artifact_depth)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (d["ts"], d.get("id"), act, d.get("kind"), d.get("lane"),
                     d.get("org_key"), d.get("url"), d.get("freshness_days"),
                     d.get("artifact_depth"))).rowcount
            if act == "never" and d.get("org_key"):
                never.add(d["org_key"])
            elif act in ("skip", "applied") and d.get("url"):
                urls.add(d["url"])
            elif act == "contact" and d.get("org_key") and d.get("value"):
                # Human found an address (find-contact chore / check-in):
                # {"action":"contact","org_key":"x","value":"a@b.com"}.
                # Stored in meta and merged into the org's contacts at memo
                # selection, re-opening the deliverable-contact gate.
                found_contacts[d["org_key"]] = {"email": str(d["value"]).strip()}
            elif act == "liked":
                # Taste signal: "this role is my style (and here is why)".
                # Keyed by url (falling back to card id) so a re-like just
                # refreshes the note; the whole file re-applies nightly, so
                # meta ends up reflecting the latest like per item.
                key = d.get("url") or d.get("id")
                if key:
                    liked[key] = {"ts": d["ts"], "url": d.get("url"),
                                  "org_key": d.get("org_key"),
                                  "note": (d.get("note") or "").strip()}
    if found_contacts:
        manual = json.loads(swarm_db.get_meta(conn, "manual_contacts", "{}"))
        merged = {**manual, **{k: {**manual.get(k, {}), **v}
                               for k, v in found_contacts.items()}}
        if merged != manual:
            swarm_db.set_meta(conn, "manual_contacts", json.dumps(merged))
            print(f"[Decisions] {len(found_contacts)} manual contact(s) stored")
    n_rej = 0
    for k in never:
        n_rej += conn.execute(
            "UPDATE orgs SET status = 'rejected' WHERE org_key = ? "
            "AND status != 'rejected'", (k,)).rowcount
    old = set(json.loads(swarm_db.get_meta(conn, "suppressed_urls", "[]")))
    if urls - old:
        swarm_db.set_meta(conn, "suppressed_urls",
                          json.dumps(sorted(old | urls)))
    if liked != json.loads(swarm_db.get_meta(conn, "liked_items", "{}")):
        swarm_db.set_meta(conn, "liked_items", json.dumps(liked))
    conn.commit()
    if never or urls or n_events or liked:
        print(f"[Decisions] {n_rej} org(s) newly rejected, "
              f"{len(urls)} posting URL(s) suppressed, "
              f"{len(liked)} liked item(s) on record, "
              f"{n_events} new funnel event(s) logged")


_EXEC_ACTIONS = ("sent", "applied", "followed_up", "done")
_OUTCOME_ACTIONS = ("replied", "screen", "interview", "offer")


def _funnel_lines() -> list:
    """Per-lane tally of executed actions and outcomes from funnel_events,
    rendered for the queue header. Deliberately a TALLY, not a verdict:
    the pre-registered sample sizes (jobs/PREREGISTRATION.md in the data
    repo) decide when comparison against base rates means anything."""
    conn = swarm_db.connect()
    try:
        rows = conn.execute(
            "SELECT COALESCE(lane, 'other') AS lane, action,"
            " COUNT(*) AS n,"
            " SUM(CASE WHEN ts >= datetime('now', '-7 days') THEN 1 ELSE 0"
            " END) AS n7"
            " FROM funnel_events GROUP BY lane, action").fetchall()
        first_ts = conn.execute(
            "SELECT MIN(ts) AS t FROM funnel_events WHERE action IN "
            "('sent','applied','followed_up','done')").fetchone()["t"]
    except Exception:
        return []
    finally:
        conn.close()
    lanes: dict = {}
    for r in rows:
        d = lanes.setdefault(r["lane"], {"exec": 0, "exec7": 0})
        if r["action"] in _EXEC_ACTIONS:
            d["exec"] += r["n"]
            d["exec7"] += r["n7"]
        elif r["action"] in _OUTCOME_ACTIONS:
            d[r["action"]] = d.get(r["action"], 0) + r["n"]
    lanes = {k: v for k, v in lanes.items() if v["exec"] or
             any(v.get(o) for o in _OUTCOME_ACTIONS)}
    if not lanes:
        return []
    out = [
        "## Funnel to date - your logged actions, the only evidence that counts",
        "",
        "| Lane | 7d | Total | Replies | Screens | Interviews | Offers |",
        "|---|---|---|---|---|---|---|",
    ]
    for lane in ("breadth", "warm", "deep", "ops", "other"):
        v = lanes.get(lane)
        if not v:
            continue
        out.append(
            f"| {lane} | {v['exec7']} | {v['exec']} "
            f"| {v.get('replied', 0)} | {v.get('screen', 0)} "
            f"| {v.get('interview', 0)} | {v.get('offer', 0)} |")
    out += [
        "",
        "*Tally, not verdict - the pre-registered thresholds and 30/60/90-day "
        "decision rules live in jobs/PREREGISTRATION.md. Log outcomes by "
        "tapping the executed card in the app (replied / screen booked / "
        "interview / offer) - that tap is what writes the event.*",
    ]
    # Pre-registration clock: day counts run from the FIRST executed action.
    if first_ts:
        try:
            day = (datetime.now() -
                   datetime.strptime(first_ts[:10], "%Y-%m-%d")).days
        except ValueError:
            day = None
        if day is not None:
            deep_done = lanes.get("deep", {}).get("exec", 0)
            nxt = ("day-30 execution check" if day < 30 else
                   "day-60 artifact kill switch" if day < 60 else
                   "day-90 lane reallocation" if day < 90 else
                   "past day 90 - reallocation rules apply every review")
            out += [
                f"*Pre-registration clock: day {day} since the first logged "
                f"send. Next checkpoint: {nxt}. Artifact kill-switch "
                f"progress: {deep_done}/15 deep-lane ships.*",
            ]
    out.append("")
    return out


def _posting_excerpt(text: str, title: str, n: int = 320) -> str:
    """Readable role blurb for a card: unescape (ATS text arrives
    double-escaped), strip tags, drop the leading title repetition."""
    t = html.unescape(html.unescape(text or ""))
    t = " ".join(re.sub(r"<[^>]+>", " ", t).split())
    lead = (title or "").split(" — ")[0].strip()
    if lead and t.lower().startswith(lead.lower()):
        t = t[len(lead):].lstrip(" .:-")
    return (t[: n - 1] + "…") if len(t) > n else t


def _facet_matches(emb, facets, names, k: int = 2) -> str:
    """Top-k expertise_matrix items by cosine vs a posting embedding -
    the 'why you fit' line the cards were missing (2026-07-06 review)."""
    if emb is None or facets is None or not names:
        return ""
    v = np.asarray(emb, dtype=np.float32)
    v = v / (np.linalg.norm(v) or 1.0)
    F = facets / (np.linalg.norm(facets, axis=1, keepdims=True) + 1e-9)
    idx = np.argsort(F @ v)[::-1][:k]
    return "; ".join(str(names[i])[:70] for i in idx if i < len(names))


_CJK_RE = re.compile(
    "[　-ヿ㐀-䶿一-鿿豈-﫿＀-￯]")


def _cjk_heavy(title: str) -> bool:
    """True when a title is substantially CJK/fullwidth - a multinational's
    Japan/China listing the human can't even read; noise in his queue."""
    t = (title or "").strip()
    return bool(t) and len(_CJK_RE.findall(t)) > 0.2 * len(t)


def _posting_filter(conn):
    """-> callable(posting) that drops rejected orgs, suppressed URLs, and
    CJK-script titles."""
    rejected = {r["org_key"] for r in conn.execute(
        "SELECT org_key FROM orgs WHERE status = 'rejected'")}
    suppressed = set(json.loads(swarm_db.get_meta(conn, "suppressed_urls", "[]")))
    return lambda p: (p["org_key"] not in rejected
                      and p["url"] not in suppressed
                      and not _cjk_heavy(p.get("title")))


async def trajectory_filter(state: JobSwarmState):
    print("[Node] trajectory_filter")
    # Absorb the human's checkbox marks from TRACKER.md first, so orgs marked
    # sent/replied/dropped in vim are hands-off for tonight's targeting -
    # then the dashboard inbox decisions (never-this-company, skipped posts).
    import tracker_engine
    conn = swarm_db.connect()
    tracker_engine.sync_marks(conn)
    _absorb_decisions(conn)
    conn.close()

    ingest_path = state.get("ingest_payload_path") or latest_ingest_payload(RAW_DIR)
    loaded = profile_engine.load_profile(PROFILE_CACHE)

    shortlist_path = await asyncio.to_thread(
        trajectory_engine.run_filter_engine, ingest_path,
        {"embedding": loaded["embedding"], "facets": loaded.get("facets")},
        TELEMETRY_DIR
    )
    with open(shortlist_path) as f:
        telemetry = json.load(f)

    top = telemetry["shortlist"][:10]
    lines = [
        f"Filter engine: {telemetry['n_orgs_evaluated']} orgs evaluated, "
        f"{telemetry['n_new_docs']} new docs. Top 10 by prescore:"
    ]
    for o in top:
        lines.append(
            f"  {o['display_name'][:48]:48s} prescore={o['prescore']:.3f} "
            f"align={o['alignment']:.3f} regime={o['regime']} "
            f"hurdle={o['hurdle_prob']:.2f}"
        )
    return {
        "messages": [{"role": "system", "name": "TrajectoryFilter", "content": "\n".join(lines)}],
        "ingest_payload_path": ingest_path,
        "shortlist_path": shortlist_path,
        "run_stats": {**state.get("run_stats", {}),
                      "orgs_evaluated": telemetry["n_orgs_evaluated"],
                      "new_docs": telemetry["n_new_docs"]},
    }


# =====================================================================
# NODE 3 - LLM AUDIT (the elite computational auditor)
# =====================================================================

_AUDIT_SYSTEM_PROMPT = (
    "You are an elite computational auditor evaluating whether an organization is in a "
    "technical Hurdle State - facing statistical or computational bottlenecks it cannot "
    "resolve with its current expertise. You will receive the organization's recent public "
    "output (grant abstracts, papers, descriptions) plus a candidate expertise matrix.\n\n"
    "Evaluate three strict criteria: (1) extract the core technical bottleneck implied by "
    "the texts; (2) assess whether their models/pipelines are sensitive to high-dimensional "
    "distribution shift; (3) score how precisely the candidate's expertise matrix addresses "
    "the bottleneck.\n\n"
    "Output ONLY one strictly valid JSON object - double quotes on every key and "
    "string, no preamble, no markdown fences. Exact shape:\n"
    '{"organization_name": "...", "bottleneck_diagnosis": "...", '
    '"distribution_shift_risk": true, "alignment_score": 0.0, '
    '"intervention_vector": "..."}\n'
    "  organization_name: string\n"
    "  bottleneck_diagnosis: highly technical 2-3 sentence summary\n"
    "  distribution_shift_risk: boolean\n"
    "  alignment_score: float 0.0-1.0 - candidate skill applicability\n"
    "  intervention_vector: the ONE specific candidate skill that solves the bottleneck\n"
    "Be objective and calibrated: most organizations should score below 0.5. Ground every "
    "claim in the provided text only.\n"
    "EVIDENCE DISCIPLINE: you also receive the evidence volume (document count and "
    "sources). A single document - one abstract, one filing, one page - is NOT "
    "sufficient evidence of a real, addressable bottleneck, however well its topic "
    "matches the candidate. With one document, score alignment_score at most 0.5 and "
    "say in bottleneck_diagnosis that the diagnosis rests on thin evidence."
)


async def _audit_one(org: dict, expertise: str, run_date: str) -> dict:
    # NO quant signals in the prompt: the audit is the INDEPENDENT second
    # opinion on the quant filter; feeding it prescore/regime anchors it.
    # Evidence carries real text excerpts, not titles - the auditor cannot
    # diagnose a bottleneck from headlines.
    src_mix = ", ".join(sorted({e["source"] for e in org.get("evidence", [])})) or "unknown"
    user_prompt = (
        f"ORGANIZATION: {org['display_name']}\n"
        f"EVIDENCE VOLUME: {org.get('n_docs') or len(org.get('evidence', []))} "
        f"document(s) in corpus, sources: {src_mix}\n"
        f"RECENT PUBLIC OUTPUT (with document excerpts):\n"
        f"{_evidence_block(org)}\n\n"
        f"CANDIDATE EXPERTISE MATRIX:\n{expertise}"
    )
    response = await query_vllm(_AUDIT_SYSTEM_PROMPT, user_prompt,
                                max_tokens=600, schema=_AUDIT_SCHEMA)
    audit = _extract_json(response) or {
        "organization_name": org["display_name"],
        "bottleneck_diagnosis": f"AUDIT PARSE FAILURE: {response[:300]}",
        "distribution_shift_risk": False,
        "alignment_score": 0.0,
        "intervention_vector": None,
    }
    # Hard calibration fence (2026-07-05 audit M6): 1-doc orgs averaged HIGHER
    # LLM alignment (0.605, max 1.0) than 2-5-doc orgs (0.26-0.43) - a single
    # well-matched abstract reads as a perfect bottleneck story because
    # nothing contradicts it. The prompt asks for restraint; this clamp
    # guarantees it whatever the model does.
    try:
        a_s = float(audit.get("alignment_score") or 0.0)
    except (TypeError, ValueError):
        a_s = 0.0
    if int(org.get("n_docs") or 0) <= 1 and a_s > 0.6:
        a_s = 0.6
        audit["alignment_capped"] = "single-document org: audit score clamped to 0.6"
    audit["alignment_score"] = a_s
    audit["org_key"] = org["org_key"]
    audit["prescore"] = org["prescore"]
    return audit


async def llm_audit(state: JobSwarmState):
    print("[Node] llm_audit")
    with open(state["shortlist_path"]) as f:
        telemetry = json.load(f)
    shortlist = telemetry["shortlist"][:AUDIT_TOP_N]

    loaded = profile_engine.load_profile(PROFILE_CACHE)
    expertise = "\n".join(f"- {s}" for s in loaded["profile"].get("expertise_matrix") or [])
    run_date = datetime.now().strftime("%Y-%m-%d")

    audits = await asyncio.gather(*[_audit_one(o, expertise, run_date) for o in shortlist])

    conn = swarm_db.connect()
    for a in audits:
        swarm_db.store_audit(conn, a["org_key"], run_date, a)
        # 'memo_drafted' protected: an unsent draft must stay visible in the
        # tracker's Ready-to-send section, not silently revert to 'audited'.
        conn.execute(
            "UPDATE orgs SET status = 'audited' WHERE org_key = ? "
            "AND status NOT IN ('memo_drafted', 'contacted', 'followed_up', "
            "'replied', 'rejected')",
            (a["org_key"],),
        )
    conn.commit()
    conn.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = os.path.join(TELEMETRY_DIR, f"audits_{timestamp}.json")
    with open(audit_path, "w") as f:
        json.dump({"run_date": run_date, "audits": audits}, f, indent=2)

    strong = sorted(
        (a for a in audits if a.get("alignment_score", 0) >= MEMO_MIN_ALIGNMENT),
        key=lambda a: a["alignment_score"], reverse=True,
    )
    lines = [f"LLM audit: {len(audits)} orgs audited; {len(strong)} above "
             f"{MEMO_MIN_ALIGNMENT} alignment threshold."]
    for a in strong[:8]:
        lines.append(f"  {a['organization_name'][:44]:44s} align={a['alignment_score']:.2f} "
                     f"-> {str(a.get('intervention_vector'))[:60]}")
    return {
        "messages": [{"role": "assistant", "name": "LLMAudit", "content": "\n".join(lines)}],
        "audit_path": audit_path,
        "run_stats": {**state.get("run_stats", {}), "audited": len(audits),
                      "above_threshold": len(strong)},
    }


# =====================================================================
# NODE 4 - STRATEGY SYNTHESIS (founder-directed technical memos, DRAFTS)
# =====================================================================

_MEMO_SYSTEM_PROMPT = (
    "You draft peer-to-peer technical memos from one computational researcher to a technical "
    "founder / PI. The memo follows a rigid four-part rhetorical structure - ONE to TWO "
    "sentences per part, no more:\n"
    "  1. HOOK - cite one specific formulation, architecture, or pipeline detail from THEIR "
    "OWN cited public output, proving this is deeply researched, not spam.\n"
    "  2. SHARED CONTEXT - an honestly-hedged hypothesis about why their current approach "
    "may be sensitive to out-of-distribution inputs or computational bottlenecks, framed as "
    "peer curiosity ('I ran into the same wall when...'), NEVER as an audit or unsolicited "
    "verdict on their work. Precise statistical language; 'if X, then likely Y'.\n"
    "  3. PROOF - the candidate's single most relevant capability with one concrete proof "
    "point, described in WORDS. NEVER include a URL or hyperlink: links in a first cold "
    "email are a primary spam trigger (2026 deliverability data). Instead, offer it: "
    "mention that a repo/benchmark/reproduction exists.\n"
    "     NO-INVENTION RULE (hard): every claim about the candidate must come verbatim "
    "from the CANDIDATE HEADLINE / PROOF POINT provided. NEVER invent an experience, "
    "project, dataset, or anecdote the profile does not contain - no 'I ran into the "
    "same issue building a <conveniently-matching system>' unless that system is in the "
    "provided materials. A fabricated anecdote discovered by the recipient ends the "
    "conversation permanently.\n"
    "  4. CTA - one low-friction PERMISSION question, e.g. 'Want me to send the repo "
    "over?' or a 15-minute call. Never ask for a job.\n\n"
    "HARD LENGTH LIMITS (2026 reply-rate evidence: response drops sharply past 125 words): "
    "body 60-125 words TOTAL. Subject: 4-7 words naming the specific technical topic - no "
    "clickbait, no 'quick question'.\n"
    "No employment vocabulary ('job', 'hire', 'resume', 'position', 'applicant'). No "
    "flattery. No fabricated claims about their systems or the candidate. NO URLs anywhere "
    "in the body. ACTIVE first person: 'I built', 'I measured' - never passive voice "
    "('a framework was developed'). Never recite the org's own published numbers back "
    "at them as the hook - they know their numbers; name their bottleneck in half a "
    "sentence and spend the words on what you bring. BANNED phrases: 'may be relevant', "
    "'might be of interest', 'could potentially'. The body ENDS with the permission "
    "question - never with a statement.\n\n"
    + _STYLE_RULES + "\n"
    "MEMO EXCEPTION to the style rules: never insert bracketed [ADD ...] prompts in a "
    "memo body - a memo must be sendable exactly as written. If a strengthening number "
    "is not in the provided materials, write the claim without the number.\n\n"
    "Output ONLY one strictly valid JSON object - double quotes on every key and "
    "string, no preamble, no markdown fences. Exact shape:\n"
    '{"subject": "specific technical subject line", "body": "the memo text"}'
)


_MEMO_VERIFY_PROMPT = (
    "You are a skeptical technical reviewer. You receive a draft memo, the ONLY "
    "evidence available about the target organization, and the ONLY candidate "
    "materials the sender can truthfully claim. Rewrite the memo so that every "
    "claim about the organization is directly supported by the evidence excerpts - "
    "delete or hedge anything unsupported. Hypotheses must be phrased as hypotheses. "
    "EQUALLY: delete any first-person claim about the candidate (experience, project, "
    "anecdote, number) that the candidate materials do not contain - an invented "
    "anecdote is the single most damaging failure mode. Delete any bracketed "
    "[ADD ...] placeholder; rephrase without the missing number. "
    "Keep the four-part structure (hook, shared-context hypothesis, proof, permission "
    "CTA), 60-125 words, subject 4-7 words, no employment vocabulary, and all style "
    "rules. PRESERVE active first-person voice for claims the materials DO support: "
    "'I built X' stays 'I built X' - converting a supported claim to passive voice "
    "is a rewrite failure, not caution. NEVER delete the closing permission question; "
    "if the draft lacks one, add it. A hedged closer ('It may be relevant to their "
    "work') is dead weight - replace it with the question. STRIP any URL or hyperlink from the body (spam trigger - the artifact is "
    "offered in words, sent only after the recipient says yes). If the draft is longer "
    "than 125 words, CUT it - brevity outranks completeness. If the draft is already "
    "fully grounded and within limits, return it unchanged.\n\n"
    + _STYLE_RULES + "\n\n"
    "Output ONLY one strictly valid JSON object - double quotes on every key and "
    'string, no preamble, no markdown fences: {"subject": "...", "body": "..."}'
)


_MEMO_CRITIC_PROMPT = (
    "You are an adversarial reviewer from a different model family than the "
    "drafter, with a KILL MANDATE. You are NOT asked to improve or polish the "
    "memo. Your only job is to decide whether it should be killed before a "
    "human wastes attention on it. Kill it if ANY of these hold:\n"
    "  - a claim about the organization is not directly supported by the "
    "evidence excerpts (hallucinated fact, wrong product, wrong problem);\n"
    "  - a first-person claim about the candidate does not appear in the "
    "candidate materials (invented experience, project, number, anecdote);\n"
    "  - the technical premise is wrong or embarrassing to a domain expert;\n"
    "  - it reads as templated LLM outreach a busy engineer would delete;\n"
    "  - it does not end with a direct question to the recipient, or closes "
    "on a hedge ('may be relevant', 'might be of interest').\n"
    "Judge the memo ON ITS OWN against the evidence - do not assume the "
    "drafter had good reasons. If uncertain about a factual claim, that is "
    "a kill, not a pass. A kill must name the specific offending sentence "
    "or claim in each issue. If nothing justifies a kill, verdict is "
    "'promote' with an empty issues list.\n\n"
    "Output ONLY one strictly valid JSON object: "
    '{"verdict": "promote" or "kill", "issues": ["specific problem", ...]}'
)

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["promote", "kill"]},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "issues"],
}


_MMR_LAMBDA = float(os.environ.get("JOB_SWARM_MMR_LAMBDA", "0.15"))

# ---------------------------------------------------------------------------
# Memo slot gate (2026-07-05 audit M1-M3). A memo slot is only spent on an
# org the human can actually email TODAY toward a job that can actually pay:
#   - academic research output (arXiv/grants) is excluded outright: the
#     audit found 77% of slots went to student "groups", and the outside
#     evidence found zero documented PI-cold-email -> industry conversions;
#   - orgs known ONLY through their ATS/regulatory exhaust (postings,
#     trials, 10-Ks, WARN) have no memo recipient - they belong in the
#     apply channel and INTEL;
#   - remaining orgs need a deliverable contact (email or HN username).
#     High scorers without one become find-contact chores instead, and a
#     manually discovered contact re-enters via the decisions log
#     ('contact' action -> meta manual_contacts).
# ---------------------------------------------------------------------------
_ACADEMIC_SOURCES = {"arxiv", "nsf", "nih", "sbir", "usaspending", "patents"}
_APPLY_ONLY_SOURCES = {"ats_jobs", "ats_seed", "remoteok", "usajobs",
                       "clinicaltrials", "ct_event", "tenk", "warn", "lca",
                       "formd_backfill", "sos_ny", "sos_co"}
_EMAILISH_RE = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")


def _deliverable_contact(contacts: dict) -> bool:
    for k, v in (contacts or {}).items():
        if not v:
            continue
        if k == "hn_user":
            return True   # messageable through the Who-is-Hiring thread
        if _EMAILISH_RE.search(str(v)):
            return True
    return False


def _memo_gate(org: dict, sources: set):
    """None if the org may hold a memo slot, else the human-readable reason."""
    if sources and sources <= _ACADEMIC_SOURCES:
        return "academic research output - no industry role to offer"
    if sources and sources <= _APPLY_ONLY_SOURCES:
        return "known only through postings/filings - apply channel, not cold email"
    if not _deliverable_contact(org.get("contacts")):
        return "no deliverable contact published"
    return None


def _mmr_select(pool: list, k: int) -> list:
    """
    Portfolio selection over the memo slots. The nightly memos are a weekly
    outreach PORTFOLIO, not k independent bets: pointwise top-k routinely
    put every slot in one sector. Greedy maximal-marginal-relevance -
    base score minus λ·(max cosine to an already-selected org's semantic
    state) - keeps the picks diversified across sectors/regimes at ~zero
    cost. Orgs whose embedding is unavailable diversify for free (sim 0).
    """
    if k <= 0 or not pool:
        return []
    conn = swarm_db.connect()
    vecs = {}
    for a in pool:
        v = swarm_db.org_recent_embedding(conn, a["org_key"])
        if v is not None:
            n = float(np.linalg.norm(v))
            if n > 0:
                vecs[a["org_key"]] = v / n
    conn.close()

    def base(a):
        return 0.6 * a.get("alignment_score", 0) + 0.4 * a.get("prescore", 0)

    remaining = list(pool)
    selected = []
    while remaining and len(selected) < k:
        best, best_val = None, -np.inf
        for a in remaining:
            penalty = 0.0
            va = vecs.get(a["org_key"])
            if va is not None and selected:
                sims = [float(va @ vecs[s["org_key"]]) for s in selected
                        if s["org_key"] in vecs]
                penalty = max(sims) if sims else 0.0
            val = base(a) - _MMR_LAMBDA * penalty
            if val > best_val:
                best, best_val = a, val
        selected.append(best)
        remaining.remove(best)
    return selected


async def _draft_one(org: dict, audit: dict, profile: dict) -> dict:
    proof = (profile.get("proof_points") or [{}])[0]
    links = profile.get("links") or {}
    evidence_txt = _evidence_block(org)
    ghost = org.get("predicted_role") or {}
    ghost_line = (
        f"PREDICTED UNPOSTED ROLE (from market archetype analysis - similar orgs "
        f"hire '{'; '.join(ghost.get('archetype_titles', []))}'): the memo's close "
        f"may gently propose defining such a role.\n"
        if ghost else ""
    )
    user_prompt = (
        f"TARGET: {org['display_name']}\n"
        f"THEIR RECENT PUBLIC OUTPUT (with document excerpts - the hook MUST "
        f"cite something from these excerpts):\n{evidence_txt}\n"
        f"AUDITED BOTTLENECK: {audit.get('bottleneck_diagnosis')}\n"
        f"INTERVENTION VECTOR: {audit.get('intervention_vector')}\n"
        f"{ghost_line}\n"
        f"CANDIDATE HEADLINE: {profile.get('headline')}\n"
        f"CANDIDATE PROOF POINT: {json.dumps(proof)}\n"
        f"CANDIDATE GITHUB (context ONLY - never put the URL in the memo; "
        f"offer to send it): {links.get('github') or '[ADD YOUR GITHUB URL TO profile/]'}"
    )
    response = await query_vllm(_MEMO_SYSTEM_PROMPT, user_prompt,
                                max_tokens=700, schema=_MEMO_SCHEMA)
    memo = _extract_json(response)
    if memo and memo.get("body"):
        # Devil's-advocate pass: strip/hedge any claim the evidence doesn't
        # support - on BOTH sides now: org claims against the evidence,
        # candidate claims against the profile materials (the 2026-07-05
        # audit caught an invented 'language retention model' anecdote that
        # the evidence-only check could not see). Cheap (≤12 extra calls).
        verify_user = (f"EVIDENCE:\n{evidence_txt}\n\n"
                       f"CANDIDATE MATERIALS (the only truthful claims):\n"
                       f"HEADLINE: {profile.get('headline')}\n"
                       f"PROOF POINT: {json.dumps(proof)}\n\n"
                       f"DRAFT SUBJECT: {memo.get('subject')}\n"
                       f"DRAFT BODY:\n{memo.get('body')}")
        verified = _extract_json(await query_vllm(
            _MEMO_VERIFY_PROMPT, verify_user, max_tokens=700, schema=_MEMO_SCHEMA))
        if verified and verified.get("body"):
            memo = {**memo, "subject": verified.get("subject") or memo["subject"],
                    "body": verified["body"]}
        # Cross-model critic (kill mandate) on the FINAL draft. A kill does
        # not drop the card - it escalates: the verdict rides the lint list,
        # the dossier, and the inbox card, and the human decides. Generator/
        # critic disagreement is exactly the signal worth human attention.
        critic_user = (f"EVIDENCE (the only permitted org facts):\n{evidence_txt}\n\n"
                       f"CANDIDATE MATERIALS (the only permitted candidate claims):\n"
                       f"HEADLINE: {profile.get('headline')}\n"
                       f"PROOF POINT: {json.dumps(proof)}\n\n"
                       f"MEMO SUBJECT: {memo.get('subject')}\n"
                       f"MEMO BODY:\n{memo.get('body')}")
        critic = _extract_json(await query_vllm(
            _MEMO_CRITIC_PROMPT, critic_user, max_tokens=500,
            schema=_CRITIC_SCHEMA, critic=True))
        if critic and critic.get("verdict"):
            memo["critic"] = {
                "verdict": critic["verdict"],
                "issues": [str(i) for i in (critic.get("issues") or [])][:4],
                "model": (os.path.basename(VLLM_CRITIC_MODEL)
                          if VLLM_CRITIC_URL else "same-model (no second server)"),
            }
    else:
        memo = {
            "subject": f"Technical note re: {org['display_name']}",
            "body": f"DRAFT PARSE FAILURE - raw model output:\n{response[:800]}",
        }
    # Lint the draft text AND the profile it was drafted from: a draft can be
    # free of literal markers yet still rest on synthetic credentials, so a
    # SYNTHETIC/smoketest marker anywhere in the profile flags every draft.
    memo["lint"] = _lint_draft(memo.get("subject"), memo.get("body"))
    if re.search(r"SYNTHETIC|smoketest", json.dumps(profile), re.I):
        memo["lint"] = memo["lint"] + ["profile is the synthetic smoke-test persona"]
    crit = memo.get("critic") or {}
    if crit.get("verdict") == "kill":
        first = (crit.get("issues") or ["unspecified"])[0]
        memo["lint"] = memo["lint"] + [
            f"second-model critic voted KILL: {first}"]
    memo["org_key"] = org["org_key"]
    memo["organization"] = org["display_name"]
    memo["contacts"] = org.get("contacts", {})
    memo["alignment_score"] = audit.get("alignment_score")
    memo["chosen_by"] = audit.get("chosen_by", "score")
    return memo


async def _discover_github_people(pool: list, shortlist: dict, run_date: str) -> dict:
    """Resolve GitHub logins + commit-email contacts for high-alignment orgs
    that lack a deliverable contact today. Merges a found email into the org's
    contacts (opening the memo gate) and persists the full people list to meta
    for the dossier. Returns {org_key: {login, people}}. Best-effort: any
    failure leaves the pipeline exactly as it was."""
    try:
        import github_engine
    except Exception as e:
        print(f"[GitHub] engine import failed (non-fatal): {e}")
        return {}
    conn = swarm_db.connect()
    src_map = {r["org_key"]: set(json.loads(r["sources"] or "[]"))
               for r in conn.execute("SELECT org_key, sources FROM orgs")}
    # Only orgs that could actually use a memo: above alignment, no deliverable
    # contact yet, and not known solely through academic or apply-only exhaust
    # (those have no cold-email recipient regardless of GitHub presence).
    targets = []
    for a in pool:
        k = a["org_key"]
        org = shortlist.get(k)
        if not org or a.get("alignment_score", 0) < MEMO_MIN_ALIGNMENT:
            continue
        if _deliverable_contact(org.get("contacts")):
            continue
        srcs = src_map.get(k, set())
        if srcs and (srcs <= _ACADEMIC_SOURCES or srcs <= _APPLY_ONLY_SOURCES):
            continue
        targets.append({"org_key": k, "display_name": org["display_name"],
                        "website": org.get("website")})
    if not targets:
        conn.close()
        return {}
    try:
        found = await github_engine.find_github_people(targets)
    except Exception as e:
        print(f"[GitHub] discovery skipped: {e}")
        conn.close()
        return {}
    # Persist discovered emails as manual_contacts (opens the gate now and on
    # future nights) and merge into the in-memory shortlist for this run.
    manual = json.loads(swarm_db.get_meta(conn, "manual_contacts", "{}"))
    for k, payload in found.items():
        email = next((p["email"] for p in payload["people"] if p.get("email")), None)
        if email:
            merged = dict(shortlist[k].get("contacts") or {})
            if not _deliverable_contact(merged):
                merged["email"] = email
                merged["contact_basis"] = (
                    "email from a recent public commit to the org's GitHub "
                    f"({payload['login']}) - verify against the linked commit")
                shortlist[k]["contacts"] = merged
                manual[k] = {"email": email,
                             "contact_basis": merged["contact_basis"]}
    swarm_db.set_meta(conn, "manual_contacts", json.dumps(manual))
    swarm_db.set_meta(conn, "github_people", json.dumps(found))
    conn.commit()
    conn.close()
    return found


async def strategy_synthesis(state: JobSwarmState):
    print("[Node] strategy_synthesis")
    with open(state["audit_path"]) as f:
        audit_payload = json.load(f)
    with open(state["shortlist_path"]) as f:
        shortlist = {o["org_key"]: o for o in json.load(f)["shortlist"]}
    loaded = profile_engine.load_profile(PROFILE_CACHE)

    run_date = audit_payload["run_date"]
    pool = sorted(
        (a for a in audit_payload["audits"] if a["org_key"] in shortlist),
        key=lambda a: 0.6 * a.get("alignment_score", 0) + 0.4 * a.get("prescore", 0),
        reverse=True,
    )

    # Slot gate: sources per org + manually discovered contacts (check-in
    # 'contact' decisions land in meta manual_contacts and re-open the gate).
    conn_g = swarm_db.connect()
    src_map = {r["org_key"]: set(json.loads(r["sources"] or "[]"))
               for r in conn_g.execute("SELECT org_key, sources FROM orgs")}
    manual_contacts = json.loads(
        swarm_db.get_meta(conn_g, "manual_contacts", "{}"))
    # Re-draft suppression (audit F1: the same org held a slot two nights
    # running): an unsent draft younger than 14 days already sits in the
    # tracker - drafting it again burns a slot and risks a double-send.
    recent_drafts = {r["org_key"] for r in conn_g.execute(
        "SELECT DISTINCT org_key FROM memos WHERE status = 'draft' "
        "AND run_date >= date('now', '-14 days') "
        "AND run_date < date('now')")}
    conn_g.close()
    for k, extra_contacts in manual_contacts.items():
        if k in shortlist and isinstance(extra_contacts, dict):
            merged = dict(shortlist[k].get("contacts") or {})
            merged.update({kk: vv for kk, vv in extra_contacts.items() if vv})
            shortlist[k]["contacts"] = merged

    # GitHub people channel: for high-alignment orgs that would otherwise be
    # gated on "no deliverable contact", resolve the org's GitHub login and
    # pull a named human + commit email. A discovered email opens the memo
    # gate exactly as a manually-found contact does; the full people list is
    # persisted for the dossier so the human can pick the right recipient.
    github_people = await _discover_github_people(pool, shortlist, run_date)

    gated, contact_chores = [], []
    for a in pool:
        if a["org_key"] in recent_drafts:
            continue    # fresh unsent draft exists - tracker already shows it
        reason = _memo_gate(shortlist[a["org_key"]], src_map.get(a["org_key"], set()))
        if reason is None:
            gated.append(a)
        elif (reason.startswith("no deliverable contact")
              and a.get("alignment_score", 0) >= MEMO_MIN_ALIGNMENT):
            org = shortlist[a["org_key"]]
            what = str(a.get("bottleneck_diagnosis") or "").split(". ")[0]
            contact_chores.append({
                "org_key": a["org_key"], "org": org["display_name"],
                "align": a.get("alignment_score"),
                "website": org.get("website"),
                "what": what[:220],
                "contacts": org.get("contacts") or {},
            })
    if len(gated) < len(pool):
        print(f"[Strategy] slot gate: {len(gated)}/{len(pool)} audited orgs "
              f"memo-eligible; {len(contact_chores)} need a contact found")
    pool = gated

    eligible = [a for a in pool if a.get("alignment_score", 0) >= MEMO_MIN_ALIGNMENT]
    explore_pool = [a for a in pool
                    if 0.30 <= a.get("alignment_score", 0) < MEMO_MIN_ALIGNMENT]

    # Removal channel: an org whose careers board just went dark is in a
    # hiring freeze, and an org that filed a WARN layoff notice (≤90d) is
    # shrinking - a memo to either is wasted for a month or two.
    conn_fz = swarm_db.connect()
    frozen = swarm_db.frozen_org_keys(conn_fz) | swarm_db.warn_org_keys(conn_fz)
    conn_fz.close()
    if frozen:
        n_skipped = sum(1 for a in eligible + explore_pool if a["org_key"] in frozen)
        if n_skipped:
            print(f"[Strategy] {n_skipped} memo candidates skipped - "
                  f"hiring freeze (dark board) or WARN layoff filing")
        eligible = [a for a in eligible if a["org_key"] not in frozen]
        explore_pool = [a for a in explore_pool if a["org_key"] not in frozen]

    # ε-greedy slots: deterministic per night (seeded by date) so re-runs of
    # the same evening pick the same orgs.
    picker = random.Random(run_date)
    explore = (picker.sample(explore_pool, min(EXPLORE_SLOTS, len(explore_pool)))
               if explore_pool and EXPLORE_SLOTS > 0 else [])
    exploit = _mmr_select(eligible, max(MEMO_TOP_M - len(explore), 0))
    for a in exploit:
        a["chosen_by"] = "score"
    for a in explore:
        a["chosen_by"] = "explore"
    finalists = exploit + [a for a in explore if a not in exploit]

    memos = await asyncio.gather(*[
        _draft_one(shortlist[a["org_key"]], a, loaded["profile"]) for a in finalists
    ])

    conn = swarm_db.connect()
    for m in memos:
        swarm_db.store_memo(conn, m["org_key"], run_date, m["subject"], m["body"], m["contacts"])
        conn.execute(
            "UPDATE orgs SET status = 'memo_drafted' WHERE org_key = ? "
            f"AND status NOT IN {swarm_db.HANDS_OFF_STATUSES!r}",
            (m["org_key"],),
        )
    # Log draft-time features: with the tracker's reply/drop outcomes this
    # becomes the dataset that replaces the hand-set 0.55/0.30/0.15 weights.
    for a in finalists:
        org = shortlist[a["org_key"]]
        swarm_db.log_outreach(
            conn, a["org_key"], a.get("alignment_score"), org.get("prescore"),
            org.get("hurdle_prob"), org.get("recency"), org.get("regime"),
            a.get("chosen_by", "score"))
    conn.commit()
    conn.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memo_path = os.path.join(TELEMETRY_DIR, f"memos_{timestamp}.json")
    with open(memo_path, "w") as f:
        json.dump({"run_date": run_date, "memos": memos,
                   "contact_chores": contact_chores[:3]}, f, indent=2)

    return {
        "messages": [{"role": "assistant", "name": "StrategySynthesis",
                      "content": f"Drafted {len(memos)} technical memos "
                                 f"({len(explore)} exploration picks; status: draft, "
                                 f"awaiting human review)."}],
        "memo_path": memo_path,
        "run_stats": {**state.get("run_stats", {}), "memos_drafted": len(memos),
                      "explore_memos": len(explore)},
    }


# =====================================================================
# NODE 5 - APPLICATION FORGE (direct openings -> tailored materials)
# =====================================================================

_APP_SYSTEM_PROMPT = (
    "You prepare application materials for ONE specific job posting, for a human who "
    "will edit and submit them personally. You receive the posting text and the "
    "candidate's structured profile. Ground every claim ONLY in the profile.\n\n"
    "Output ONLY one JSON object - no preamble, no explanation, no markdown fences, "
    "nothing before the opening brace or after the closing brace. Strict JSON: double "
    "quotes on every key and string, internal quotes escaped, no trailing commas. "
    "Exact shape:\n"
    '{\n'
    '  "note": "...",\n'
    '  "cover_letter": "...",\n'
    '  "resume_bullets": [{"theme": "...", "bullet": "..."}],\n'
    '  "keywords_matched": ["..."],\n'
    '  "keywords_missing": ["..."]\n'
    '}\n\n'
    "Field content:\n"
    "  note: 90-130 word application email. Structure: one sentence naming "
    "the specific thing in the posting that matches the candidate's strongest proof point; "
    "two or three sentences of evidence (concrete systems built, methods used); one-sentence "
    "close pointing to the GitHub link. No greeting fluff, no restating the resume.\n"
    "  cover_letter: 180-260 words, for portals that require one. Three "
    "paragraphs: (1) the specific problem/mandate in the posting and the one "
    "proof point that answers it; (2) evidence - systems built, methods, "
    "numbers from the profile (or [ADD REAL NUMBER: ...] prompts); (3) a "
    "direct close. Active first person, no 'I am writing to express', no "
    "flattery, nothing the resume bullets don't support.\n"
    "  resume_bullets: 3-5 items; theme = which experience this rewrites; bullet = resume "
    "bullet in the posting's own vocabulary, action verb first, with a "
    "number - or a bracketed [ADD REAL NUMBER: ...] prompt if the profile lacks one.\n"
    "  keywords_matched: posting keywords the candidate genuinely has evidence for.\n"
    "  keywords_missing: posting keywords the candidate does NOT have - list them "
    "honestly so the human knows the gap; never suggest faking them.\n"
    "  likely_interview_topics: 4-6 specific technical topics or question types "
    "this posting's interviews will probably probe, inferred from the posting text "
    "(named methods, systems, 'you will' duties) - so preparation is targeted. "
    "Include the missing keywords' topics: those are the gaps an interviewer "
    "finds first.\n\n"
    + _STYLE_RULES
)


async def _forge_one(posting: dict, profile: dict) -> dict:
    links = profile.get("links") or {}
    user_prompt = (
        f"POSTING: {posting['title']}\n"
        f"POSTING TEXT:\n{(posting.get('text') or '')[:3500]}\n\n"
        f"CANDIDATE PROFILE:\n{json.dumps({k: v for k, v in profile.items() if not k.startswith('_')}, indent=1)[:3500]}\n"
        f"CANDIDATE GITHUB: {links.get('github') or '[ADD YOUR GITHUB URL TO profile/]'}"
    )
    response = await query_vllm(_APP_SYSTEM_PROMPT, user_prompt,
                                max_tokens=1400, schema=_APP_SCHEMA)
    app = _extract_json(response) or {"note": f"FORGE PARSE FAILURE: {response[:400]}",
                                      "resume_bullets": [], "keywords_matched": [],
                                      "keywords_missing": []}
    app["posting_title"] = posting["title"]
    app["url"] = posting["url"]
    app["align"] = posting.get("align")
    app["days_open"] = posting.get("days_open")
    app["salary"] = (posting.get("extra") or {}).get("salary")
    app["location"] = (posting.get("extra") or {}).get("location")
    return app


async def application_forge(state: JobSwarmState):
    """Tailored note + resume bullet rewrites for the top direct openings."""
    print("[Node] application_forge")
    import trajectory_engine
    loaded = profile_engine.load_profile(PROFILE_CACHE)
    profile_vec = loaded["embedding"]

    conn = swarm_db.connect()
    postings = swarm_db.recent_docs_by_source(
        conn, ("ats_jobs", "remoteok", "hn_hiring", "usajobs"), days=10)
    facets = loaded.get("facets")
    # forge slots only go to the target geography (US/CA, Europe, AU/NZ),
    # and never to rejected orgs or postings the human already dismissed
    keep = _posting_filter(conn)
    postings = [p for p in postings
                if keep(p) and _geo_ok((p.get("extra") or {}).get("location"))]
    # Hard-constraint gate BEFORE ranking so forge slots aren't spent tailoring
    # materials for roles the candidate is disqualified from (8+ YoE, PhD-only,
    # active clearance) - the same gate compile_review applies, else the queue
    # buries a role that APPLICATIONS.md still writes a cover letter for.
    ftext = {}
    if postings:
        fqm = ",".join("?" for _ in postings)
        ftext = {r["doc_id"]: r["text"] for r in conn.execute(
            f"SELECT doc_id, text FROM docs WHERE doc_id IN ({fqm})",
            [p.get("doc_id") for p in postings])}
    taste_mat = trajectory_engine.taste_vectors(conn)
    for p in postings:
        p["align"] = trajectory_engine.alignment_score(p["embedding"], profile_vec, facets)
        hc_mult, hc_reasons = _hard_constraints(
            ftext.get(p.get("doc_id"), ""), p.get("title"))
        if hc_reasons:
            p.setdefault("extra", {})["hard_constraints"] = hc_reasons
        taste = trajectory_engine.taste_boost(p["embedding"], taste_mat)
        if taste > 1.0:
            p.setdefault("extra", {})["taste"] = round(taste, 3)
        p["score"] = (_score_posting(p) * _lca_demand_boost(conn, p["org_key"])
                      * hc_mult * taste)
    # fit × repost-forensics × seniority × LCA demand (see _score_posting - no
    # staleness term); winnable (non-senior) roles get the forge slots first
    # so materials aren't spent on Staff/Manager screens
    postings.sort(key=lambda p: p["score"], reverse=True)
    primary = [p for p in postings if not _SENIOR_RE.search(p["title"] or "")]
    primary_ids = {id(p) for p in primary}
    rest = [p for p in postings if id(p) not in primary_ids]
    top = (primary + rest)[:APP_TOP_K]
    # Full text lives in the docs table, not in recent_docs_by_source - fetch it
    for p in top:
        row = conn.execute("SELECT text FROM docs WHERE doc_id = ?", (p["doc_id"],)).fetchone()
        p["text"] = row["text"] if row else ""
    conn.close()

    apps = await asyncio.gather(*[_forge_one(p, loaded["profile"]) for p in top])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    app_path = os.path.join(TELEMETRY_DIR, f"applications_{timestamp}.json")
    with open(app_path, "w") as f:
        json.dump({"applications": list(apps)}, f, indent=2)

    return {
        "messages": [{"role": "assistant", "name": "ApplicationForge",
                      "content": f"Forged materials for {len(apps)} direct openings."}],
        "application_path": app_path,
        "run_stats": {**state.get("run_stats", {}), "applications_forged": len(apps)},
    }


# =====================================================================
# NODE 5.5 - ARTIFACT NOMINATOR (proof-of-work briefs, with provenance)
# =====================================================================
#
# The honest version of an "automated consultant": the swarm never sends
# analysis anywhere - it nominates the ≤3 finalists where TWO DAYS of the
# candidate's own H100 time can produce a small, real artifact (reproduce a
# figure, benchmark a released tool, extend a result one step), and writes a
# brief with a full provenance chain: every claimed problem cites the source
# document (engine, date, URL) plus a verbatim quote that is programmatically
# checked against the actual excerpts. The human backtraces, builds the
# artifact, and opens the email with work already done - the one move no
# LinkedIn applicant can match.

# Nightly volume capped at 2 (2026-07-07 matching-report edit d): only 1-2
# artifacts can ship per week, so >2 full briefs a night is compute the human
# can never consume. Two highest-evidence nominations, full specs.
ARTIFACT_TOP_N = int(os.environ.get("JOB_SWARM_ARTIFACT_N", "2"))
_ARTIFACT_POOL = 12

_CLUSTER_RESOURCES = (
    "SLURM cluster: multiple 4×H100-80GB GPU nodes, large CPU/RAM batch "
    "nodes, apptainer containers with PyTorch/CUDA, Numba CUDA, vLLM serving "
    "Llama-3.3-70B, sentence-transformers, hmmlearn/scikit-learn."
)

# 2026-07-07, per Liam: the system must NOT write code for him ("it coding
# projects for me honestly is a waste of time") - he builds every artifact
# himself. What he wants from the overnight pass is a PROJECT SPEC: why this
# project was chosen (full provenance he can verify), and a well-written
# description of what to build. No fenced code blocks, ever.
_PROJECT_SPEC_PROMPT = (
    "You write a PROJECT SPEC for a proof-of-work artifact a human "
    "statistician will design and code entirely himself. Your job is the "
    "reasoning and the description - NEVER the code. Produce a markdown "
    "document with EXACTLY these sections:\n"
    "## Why this project (the provenance) - how the pipeline found it: "
    "which of THEIR documents shows the problem (cite engine, date, URL, "
    "with a short verbatim quote per claim), why it matters to this org "
    "right now, and why this candidate specifically can win it.\n"
    "## What to build - plain prose, 150-300 words: the artifact, the ONE "
    "technical claim it demonstrates, and what 'done' looks like. Describe "
    "approach and design considerations in words; do NOT write code or "
    "pseudo-code.\n"
    "## Success criteria - what must be true for this to impress: which "
    "metrics to measure (never invent their values - the human measures "
    "everything), what a recipient would check first, and the honest bar "
    "for 'send it' vs 'not good enough'.\n"
    "## Suggested resources - public repos, datasets, papers, and which "
    "cluster resources fit (by name only - no invocation commands).\n"
    "## How to present it - one short paragraph: how the finished artifact "
    "plugs into the outreach email and what the subject line should "
    "promise.\n\n"
    "HARD RULES: no fenced code blocks anywhere; never invent a benchmark "
    "result, dataset size, or speedup; no emojis; assume ZERO access to "
    "the org's internal systems, products, or paid APIs - the project must "
    "be buildable from public models, repos, and data alone."
)

_NOMINATE_SCHEMA = {
    "type": "object",
    "properties": {
        "nominations": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"org_index": {"type": "integer"},
                                     "reason": {"type": "string"}},
                      "required": ["org_index", "reason"]},
        },
    },
    "required": ["nominations"],
}

_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "problem_statement": {"type": "string"},
        "evidence_quotes": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"quote": {"type": "string"},
                                     "why_it_matters": {"type": "string"}},
                      "required": ["quote", "why_it_matters"]},
        },
        "artifact_plan": {
            "type": "object",
            "properties": {"goal": {"type": "string"},
                           "steps": {"type": "array", "items": {"type": "string"}},
                           "estimated_hours": {"type": "number"},
                           "cluster_usage": {"type": "string"}},
            "required": ["goal", "steps", "estimated_hours", "cluster_usage"],
        },
        "email_hook": {"type": "string"},
        "why_you": {"type": "string"},
    },
    "required": ["title", "problem_statement", "evidence_quotes",
                 "artifact_plan", "email_hook", "why_you"],
}

_NOMINATE_SYSTEM_PROMPT = (
    "You triage organizations for PROOF-OF-WORK opportunities. An opportunity "
    "exists when an org's public output exposes a legible, tractable technical "
    "problem a strong candidate could visibly advance in ≤2 days on an HPC "
    "cluster: a paper with reproducible claims, a released tool with an "
    "unmeasured performance ceiling, a benchmark or method that invites one "
    "concrete extension or stress test. REJECT orgs whose evidence is vague "
    "descriptions, marketing copy, or problems needing private data or "
    "months of work.\n\n"
    "You receive numbered organizations with document excerpts and the "
    "candidate expertise matrix. Nominate AT MOST {n} - fewer is fine; an "
    "empty list is a valid answer. Judge only from the provided excerpts.\n\n"
    "Output ONLY one strictly valid JSON object: "
    '{{"nominations": [{{"org_index": 0, "reason": "..."}}]}}'
)

_BRIEF_SYSTEM_PROMPT = (
    "You write a proof-of-work ARTIFACT BRIEF: a plan for a candidate to build "
    "one small, real technical artifact addressing a problem visible in an "
    "organization's public output, using his own HPC cluster time. The "
    "candidate builds it himself; you only plan. Hard rules:\n"
    "- Ground EVERY claim about the org in the provided excerpts. Each "
    "evidence_quotes.quote must be a VERBATIM substring copied from one "
    "excerpt (they are checked mechanically; paraphrases are discarded).\n"
    "- The artifact must be completable in ≤2 days (≤16 hours) with the "
    "stated cluster resources and PUBLIC data/code only. ZERO access to the "
    "org's internal systems, products, or paid APIs may be assumed - if "
    "their product is closed, plan the adjacent public-stack version "
    "(public models, public repos, published benchmarks) and measure THAT. "
    "A plan step the candidate cannot literally run on his own cluster "
    "tonight is a planning failure.\n"
    "- email_hook: 2-4 sentences the candidate could send AFTER building it. "
    "Since results don't exist yet, write result placeholders as "
    "'[AFTER BUILD: e.g. observed scaling knee at N GPUs]'. Describe the "
    "artifact in WORDS and close with a permission question ('want me to "
    "send the repo over?') - NEVER include a URL: links in a first cold "
    "email are a primary spam trigger. No employment vocabulary ('job', "
    "'hire', 'resume', 'position').\n"
    "- why_you: 1-2 sentences mapping the plan onto the candidate's actual "
    "expertise matrix - no invented skills.\n\n"
    + _STYLE_RULES + "\n\n"
    "Output ONLY one strictly valid JSON object, exact shape:\n"
    '{"title": "...", "problem_statement": "...", '
    '"evidence_quotes": [{"quote": "...", "why_it_matters": "..."}], '
    '"artifact_plan": {"goal": "...", "steps": ["..."], '
    '"estimated_hours": 0, "cluster_usage": "..."}, '
    '"email_hook": "...", "why_you": "..."}'
)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _verify_quotes(brief: dict, org: dict) -> None:
    """Provenance check: each quote must be a verbatim substring of one of
    the org's evidence excerpts/titles. Attaches the matched source doc
    (engine, date, url) so the human can backtrace, and flags misses -
    an unverifiable quote is treated as a hallucination, never evidence."""
    evidence = org.get("evidence", [])
    for eq in brief.get("evidence_quotes", []):
        q = _norm_ws(eq.get("quote", ""))
        eq["verified"] = False
        if len(q) < 15:
            continue
        for e in evidence:
            hay = _norm_ws(f"{e.get('title', '')} {e.get('excerpt', '')}")
            if q in hay:
                eq["verified"] = True
                eq["source"] = e.get("source")
                eq["source_title"] = e.get("title")
                eq["source_date"] = e.get("date")
                eq["source_url"] = e.get("url")
                break


def _brief_markdown(brief: dict, org: dict, audit: dict) -> str:
    """One brief -> markdown with the full provenance chain."""
    plan = brief.get("artifact_plan") or {}
    lines = [
        f"## {org['display_name']} - {brief.get('title', 'artifact opportunity')}",
        "",
        f"**The problem (as diagnosed from their public output):** "
        f"{brief.get('problem_statement', '')}",
        "",
        "### Provenance - how the swarm found this (backtrace before building)",
        "",
        f"- Pipeline: prescore={org.get('prescore')} · alignment="
        f"{org.get('alignment')} · regime={org.get('regime')} "
        f"(hurdle={org.get('hurdle_prob')}) · LLM-audit alignment="
        f"{audit.get('alignment_score')}",
        f"- Audited bottleneck: {audit.get('bottleneck_diagnosis', 'n/a')}",
        "",
        "Direct quotes from their documents (VERIFIED = verbatim match checked "
        "mechanically; treat UNVERIFIED as model error and check the source "
        "yourself before repeating the claim):",
        "",
    ]
    for eq in brief.get("evidence_quotes", []):
        if eq.get("verified"):
            lines.append(
                f"- VERIFIED: \"{eq.get('quote', '').strip()}\" - "
                f"[{eq.get('source')}] ({eq.get('source_date')}) "
                f"[{eq.get('source_title')}]({eq.get('source_url')})")
        else:
            lines.append(f"- UNVERIFIED: \"{eq.get('quote', '').strip()}\"")
        why = eq.get("why_it_matters")
        if why:
            lines.append(f"  - why it matters: {why}")
    lines += ["", "Full evidence trail (every document behind this org's scores):", ""]
    for e in org.get("evidence", []):
        lines.append(f"- [{e['source']}] ({e['date']}) [{e['title']}]({e['url']})")
    lines += [
        "",
        f"### The artifact (≤2 days of your cluster time)",
        "",
        f"**Goal:** {plan.get('goal', '')}",
        "",
        *[f"{i}. {s}" for i, s in enumerate(plan.get("steps", []), 1)],
        "",
        f"**Estimated effort:** ~{plan.get('estimated_hours', '?')} h · "
        f"**Cluster usage:** {plan.get('cluster_usage', '')}",
        "",
        "### The email it becomes (draft - send only AFTER the artifact exists)",
        "",
        "```",
        brief.get("email_hook", ""),
        "```",
        "",
        f"**Why you:** {brief.get('why_you', '')}",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


async def artifact_nominator(state: JobSwarmState):
    print("[Node] artifact_nominator")
    with open(state["shortlist_path"]) as f:
        shortlist = {o["org_key"]: o for o in json.load(f)["shortlist"]}
    audits = {}
    if state.get("audit_path"):
        with open(state["audit_path"]) as f:
            audit_payload = json.load(f)
            audits = {a["org_key"]: a for a in audit_payload["audits"]}
    loaded = profile_engine.load_profile(PROFILE_CACHE)
    expertise = "\n".join(f"- {s}" for s in
                          loaded["profile"].get("expertise_matrix") or [])
    run_date = datetime.now().strftime("%Y-%m-%d")

    # Candidate pool: audited orgs with substantive evidence, minus orgs
    # holding a fresh brief. Deliberately NOT filtered by hiring signals -
    # a company not hiring at all is exactly the "I saw your problem"
    # cold-channel this brief exists for. Academic-only orgs ARE excluded
    # (same rule as _memo_gate, same owner decision): the deep lane is the
    # most expensive channel and it targets industry, not grant-funded groups.
    def _academic(org):
        srcs = {e.get("source") for e in (org.get("evidence") or [])
                if e.get("source")}
        return bool(srcs) and srcs <= _ACADEMIC_SOURCES

    conn = swarm_db.connect()
    recent = swarm_db.recently_nominated_orgs(conn, days=14)
    conn.close()
    pool = [
        (a, shortlist[k]) for k, a in audits.items()
        if k in shortlist and k not in recent
        and not _academic(shortlist[k])
        and a.get("alignment_score", 0) >= 0.30
        and any((e.get("excerpt") or "").strip()
                for e in shortlist[k].get("evidence", []))
    ]
    # Evidence-volume bonus: a 1-doc org yields briefs diagnosed from a
    # single page (the RightNow case, 2026-07-06) - prefer orgs the plan
    # can actually be grounded in.
    pool.sort(key=lambda t: 0.6 * t[0].get("alignment_score", 0)
              + 0.4 * t[0].get("prescore", 0)
              + 0.08 * min(len(t[1].get("evidence") or []), 3),
              reverse=True)
    pool = pool[:_ARTIFACT_POOL]
    if not pool:
        print("[Artifact] no eligible orgs tonight")
        return {"messages": [{"role": "assistant", "name": "ArtifactNominator",
                              "content": "No artifact opportunities tonight."}],
                "artifact_path": None}

    # ---- Selection call: which orgs expose a buildable problem? ----------
    listing = []
    for i, (a, org) in enumerate(pool):
        listing.append(f"[{i}] {org['display_name']}\n"
                       f"{_evidence_block(org, n=4, excerpt_chars=400)}")
    sel_user = (f"CANDIDATE EXPERTISE MATRIX:\n{expertise}\n\n"
                f"CLUSTER RESOURCES:\n{_CLUSTER_RESOURCES}\n\n"
                f"ORGANIZATIONS:\n\n" + "\n\n".join(listing))
    sel_raw = await query_vllm(
        _NOMINATE_SYSTEM_PROMPT.format(n=ARTIFACT_TOP_N), sel_user,
        max_tokens=500, schema=_NOMINATE_SCHEMA)
    sel = _extract_json(sel_raw) or {}
    picks = []
    for nom in (sel.get("nominations") or [])[:ARTIFACT_TOP_N]:
        try:
            idx = int(nom.get("org_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(pool):
            picks.append((pool[idx][0], pool[idx][1], nom.get("reason", "")))

    # ---- Detail call per pick: the actual brief ---------------------------
    async def _one_brief(audit, org, reason):
        user = (
            f"ORGANIZATION: {org['display_name']}\n"
            f"WHY NOMINATED: {reason}\n"
            f"THEIR DOCUMENTS (quotes must be verbatim substrings of these "
            f"excerpts):\n{_evidence_block(org, n=6, excerpt_chars=900)}\n\n"
            f"AUDITED BOTTLENECK: {audit.get('bottleneck_diagnosis')}\n\n"
            f"CANDIDATE EXPERTISE MATRIX:\n{expertise}\n\n"
            f"CLUSTER RESOURCES:\n{_CLUSTER_RESOURCES}"
        )
        raw = await query_vllm(_BRIEF_SYSTEM_PROMPT, user,
                               max_tokens=1400, schema=_BRIEF_SCHEMA)
        brief = _extract_json(raw)
        if not brief:
            return None
        _verify_quotes(brief, org)
        brief["org_key"] = org["org_key"]
        brief["organization"] = org["display_name"]
        return brief

    briefs = [b for b in await asyncio.gather(
        *[_one_brief(a, o, r) for a, o, r in picks]) if b]

    # ---- Overnight project spec (2026-07-07, replaces the code scaffold):
    # the cluster does the finding and the reasoning while the human sleeps;
    # the human does ALL the code. The spec carries provenance (how the
    # project was found, with citations into their documents) and a prose
    # description of what to build. Top 2 briefs, one call each.
    async def _one_scaffold(b):
        org = shortlist.get(b["org_key"], {})
        plan = b.get("artifact_plan") or {}
        user = (
            f"ORGANIZATION: {b['organization']}\n"
            f"PROBLEM STATEMENT: {b.get('problem_statement')}\n"
            f"ARTIFACT GOAL: {plan.get('goal')}\n"
            f"PLAN STEPS: {json.dumps(plan.get('steps') or plan)}\n"
            f"THEIR DOCUMENTS:\n"
            f"{_evidence_block(org, n=4, excerpt_chars=600) if org else 'n/a'}\n\n"
            f"CANDIDATE EXPERTISE MATRIX:\n{expertise}\n\n"
            f"CLUSTER RESOURCES:\n{_CLUSTER_RESOURCES}"
        )
        md = await query_vllm(_PROJECT_SPEC_PROMPT, user, max_tokens=3500)
        if md and not md.startswith("Error connecting"):
            b["scaffold_md"] = (
                f"# Project spec: {b.get('title', 'artifact')} - "
                f"{b['organization']}\n\n"
                "> The reasoning and the target are below - the design and "
                "every line of code are yours. Verify the cited quotes "
                "against the linked sources before building.\n\n" + md)
        return b

    top_briefs = briefs[:2]
    if top_briefs:
        await asyncio.gather(*[_one_scaffold(b) for b in top_briefs])
        print(f"[Artifact] {sum(1 for b in top_briefs if b.get('scaffold_md'))} "
              f"overnight scaffold(s) generated")

    conn = swarm_db.connect()
    for b in briefs:
        swarm_db.store_artifact_brief(conn, b["org_key"], run_date,
                                      b.get("title"), b)
    conn.commit()
    conn.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = os.path.join(TELEMETRY_DIR, f"artifacts_{timestamp}.json")
    with open(artifact_path, "w") as f:
        json.dump({"run_date": run_date, "briefs": briefs}, f, indent=2)

    n_verified = sum(1 for b in briefs
                     for eq in b.get("evidence_quotes", []) if eq.get("verified"))
    return {
        "messages": [{"role": "assistant", "name": "ArtifactNominator",
                      "content": f"{len(briefs)} proof-of-work briefs drafted "
                                 f"({n_verified} quotes provenance-verified)."}],
        "artifact_path": artifact_path,
        "run_stats": {**state.get("run_stats", {}),
                      "artifact_briefs": len(briefs)},
    }


# =====================================================================
# Listwise sliding-window re-rank (2026-07-07 matching-report edit b).
# Cosine + the deterministic posting score rank what is SEMANTICALLY NEAR;
# they cannot read "this role wants exactly your regime-modeling + HPC combo"
# vs "adjacent but you'd lose the screen". A RankGPT-style listwise pass over
# the already-filtered top postings lets the nightly model reason about fit
# ACROSS postings (the signal a pointwise 0-1 score structurally can't see).
# Windowed from the bottom up so strong items bubble toward the top; runs on
# the nightly A100 model, ~20 extra calls, additive to (not replacing) the
# org audit. Best-effort: any failure leaves the deterministic order intact.
# =====================================================================
_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["order"],
}

_RERANK_SYSTEM_PROMPT = (
    "You rank job postings by how well ONE candidate could win an interview, "
    "given the candidate's expertise matrix. Winning means the candidate "
    "clears the screen AND the role genuinely needs their differentiated "
    "skills - not mere topic overlap. Reward exact-fit early-career/graduate "
    "roles; penalize roles demanding seniority, credentials, or a specialty "
    "the candidate lacks.\n"
    "You receive the expertise matrix and a numbered list of postings. Return "
    "the posting NUMBERS ordered BEST-FIT FIRST.\n"
    # guided_json is best-effort (older vLLM 400s it and retries unguided), so
    # the prompt itself must demand the exact shape - the job-swarm db5375b
    # lesson: without this the unguided retry answers in prose and parses to
    # nothing.
    'Output ONLY JSON, no prose, exactly: {"order": [<numbers, best first, '
    "each posting number exactly once>]}"
)


async def _rank_window(items: list, expertise: str) -> list:
    """One window: returns a 0-based permutation (best-first) of len(items),
    or [] on failure (caller keeps the incoming order)."""
    listing = "\n".join(
        f"{i + 1}. {(it.get('title') or '').strip()[:90]} :: "
        f"{(it.get('excerpt') or '').strip()[:220]}"
        for i, it in enumerate(items))
    user = (f"CANDIDATE EXPERTISE MATRIX:\n{expertise}\n\n"
            f"POSTINGS (rank these {len(items)} best-fit first):\n{listing}")
    raw = await query_vllm(_RERANK_SYSTEM_PROMPT, user, max_tokens=200,
                           schema=_RERANK_SCHEMA)
    obj = _extract_json(raw) or {}
    order = obj.get("order") if isinstance(obj, dict) else None
    if not isinstance(order, list):
        return []
    seen, perm = set(), []
    for v in order:
        try:
            j = int(v) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= j < len(items) and j not in seen:
            seen.add(j)
            perm.append(j)
    for j in range(len(items)):          # append any the model dropped
        if j not in seen:
            perm.append(j)
    return perm


async def _rerank_postings(items: list, expertise: str,
                           window: int = 5, stride: int = 2,
                           iters: int = 2) -> list:
    """RankGPT sliding window: reorders `items` (best-first) in place across
    `iters` bottom-up passes. Returns the reordered list."""
    if len(items) <= 1 or not expertise:
        return items
    order = list(range(len(items)))
    for _ in range(iters):
        end = len(order)
        while end > 0:
            start = max(0, end - window)
            win = order[start:end]
            perm = await _rank_window([items[i] for i in win], expertise)
            if perm:
                order[start:end] = [win[j] for j in perm]
            if start == 0:
                break
            end -= stride
    return [items[i] for i in order]


# =====================================================================
# NODE 6 - COMPILE REVIEW QUEUE (terminal; output is for the HUMAN)
# =====================================================================

def _tech_lead_lines(leads) -> list:
    """Dossier fragment: who directs the relevant work, from public
    authorship. Names and papers only - find their published contact channel
    through the paper/org site yourself; the swarm doesn't harvest emails."""
    if not leads:
        return []
    lines = ["## Likely technical leads (from their recent publications)", ""]
    for ld in leads:
        works = "; ".join(
            f"[{w['title']}]({w['url']}) ({w['year']})" for w in ld["works"])
        lines.append(f"- **{ld['name']}** - senior/corresponding author on "
                     f"{ld['n_works']} recent papers: {works}")
    lines += ["", "> Address the memo to the person whose papers match the "
              "diagnosis - their published contact is on the paper itself.", ""]
    return lines


def _tenk_lines(row) -> list:
    """Dossier fragment: 10-K risk-factor language shift for public cos."""
    if not row:
        return []
    extra = json.loads(row["extra"] or "{}")
    delta = extra.get("ml_term_delta")
    if delta is None:
        return []
    direction = "+" if delta > 0 else ""
    return [
        f"**10-K signal:** ML/data vocabulary in Risk Factors moved "
        f"{direction}{delta} mentions YoY "
        f"([latest filing]({row['url']}), {row['date']}) - "
        + ("R&D language shifting toward your field."
           if delta > 0 else "no ML-ward drift in the latest filing."),
        "",
    ]


def _github_people_lines(gp) -> list:
    """Dossier fragment: named humans found on the org's public GitHub, with a
    deliverable commit email where one surfaced. The email that opened the memo
    gate is the first person carrying one; the rest are alternates the human
    can pick from. Every line links the commit/profile URL for backtracing."""
    if not gp or not gp.get("people"):
        return []
    lines = [f"## GitHub people ({gp['login']} - public org, verified by domain)",
             ""]
    for p in gp["people"]:
        who = f"**{p.get('name') or p.get('login')}**"
        if p.get("login") and p.get("name"):
            who += f" (@{p['login']})"
        bits = []
        if p.get("email"):
            bits.append(f"`{p['email']}`")
        if p.get("role_hint"):
            bits.append(p["role_hint"])
        tail = f" - {'; '.join(bits)}" if bits else ""
        src = f" [[commit/profile]({p['source_url']})]" if p.get("source_url") else ""
        lines.append(f"- {who}{tail}{src}")
    lines += ["", "> Commit emails are self-attested (the committer used them "
              "locally) - not SMTP-verified. Confirm the person still owns the "
              "role via the linked profile before sending.", ""]
    return lines


def _jobs_people_research_prompt(target_orgs: list, run_date: str) -> str:
    """Tomorrow's Gemini Deep Research homework for the JOBS desk (2026-07-07
    matching-report deliverable c). People-finding is the highest-value nightly
    ask: live web reconnaissance is Gemini's edge over the offline cluster.
    The human pastes this into Gemini, saves the output into jobs/research/,
    and the next nightly's research ingest binds the named people to orgs and
    opens the outreach gate. The structured trailer line is parsed mechanically
    - it MUST be emitted verbatim for each person or the ingest drops them."""
    orgs = [o for o in dict.fromkeys(target_orgs) if o][:12]
    org_block = "\n".join(f"- {o}" for o in orgs) or "- (no memo/apply targets tonight)"
    return f"""# Jobs Deep Research Prompt - {run_date}

Copy everything between the lines into Google Gemini Deep Research. Save the
full report into `jobs/research/` (the app's paste box on the jobs tab does
this for you). Tomorrow night's run ingests it and binds the people it finds to
these organizations, unlocking outreach the system currently gates off for lack
of a named contact.

---

Conduct an exhaustive public web reconnaissance on the following technology and
research organizations. My objective is to identify NAMED HUMANS in or near
their technical work - Engineering Leadership (CTO, VP/Head of Engineering,
Chief Scientist) and Senior/Staff/Lead Engineers and Research Scientists - so I
can write to a specific person about a specific technical problem.

Organizations:
{org_block}

Search these specific public sources: technical conference speaker schedules
(PyData, NeurIPS, ICML, KDD, USENIX, domain workshops) from 2024-2026;
engineering blog bylines on the org's own site; GitHub organization member and
top-contributor lists; recent podcast/webinar transcripts naming their staff as
guests; and press releases or talks quoting a named engineer. Do NOT search or
cite LinkedIn.

For each person you are confident about, END with one line in EXACTLY this
format (this is parsed automatically - do not vary the field names or the
pipes):
PERSON: <full name> | ROLE: <exact title> | ORG: <organization name, copied
from the list above> | EMAIL: <deliverable email if published publicly, else
none> | SOURCE: <URL where you verified the name-to-role-to-org link>

Rules: only people you can tie to the organization with a dated 2024-2026
public source; never guess an email (write none rather than infer); prefer the
person who would OWN a technical problem over a generalist recruiter. No emojis.

---
"""


def _research_people_lines(people) -> list:
    """Dossier fragment: named humans from the manual deep-research paste, each
    tied to a dated public source the human can verify."""
    if not people:
        return []
    lines = ["## People from deep research (verify each source before writing)", ""]
    for p in people[:8]:
        bits = [p.get("role")] if p.get("role") else []
        if p.get("email"):
            bits.append(f"`{p['email']}`")
        tail = f" - {'; '.join(bits)}" if bits else ""
        src = f" [[source]({p['source_url']})]" if p.get("source_url") else ""
        lines.append(f"- **{p.get('name')}**{tail}{src}")
    lines += ["", "> Found by Gemini from public web sources - confirm the "
              "name, role, and org against the linked source before sending.", ""]
    return lines


def _warm_path_lines(wp) -> list:
    """Dossier fragment: co-publication bridges from the home institution."""
    if not wp:
        return []
    lines = ["## Warm path - co-publications with your institution", ""]
    for b in wp["bridges"][:4]:
        home_side = ", ".join(b["home_authors"]) or "authors at your institution"
        lines.append(f"- ({b['year']}) *{b['title']}* - bridge: **{home_side}** "
                     f"co-published with {wp['institution']}. [{b['url']}]({b['url']})")
    lines += ["", "> A hallway intro through these names beats any cold memo - "
              "ask your professor first.", ""]
    return lines


def _ghost_role_lines(ghost, accel, hawkes=None) -> list:
    """Dossier fragment: predicted unposted role + hiring telemetry.
    Weak archetype matches are suppressed (audit K2: a cos=0.47 match once
    predicted a 'Director, Value Quantification' role for a student's
    single arXiv paper - misleading beats missing)."""
    lines = []
    if ghost and float(ghost.get("similarity") or 0) < 0.7:
        ghost = None
    if ghost:
        salary = (f"${ghost['market_median_salary']:,.0f} median at comparable orgs"
                  if ghost.get("market_median_salary") else "salary data pending")
        lines += [
            "## Ghost role (predicted, not yet posted)", "",
            f"Their semantic state sits nearest the **{ghost['archetype_titles'][0]}** "
            f"archetype (cos={ghost['similarity']}, {salary}, modeled on "
            f"{ghost['based_on_postings']} live postings). No matching role is posted - "
            "the memo proposes defining it.", "",
        ]
    if accel:
        lines += [
            f"**Hiring telemetry:** {accel['latest_total']} open roles on their board, "
            f"slope {accel['slope_per_day']:+.2f}/day over {accel['n_snapshots']} snapshots "
            f"({accel['latest_relevant']} currently relevant to you).", "",
        ]
    if hawkes:
        burst = hawkes["burst_ratio"]
        tag = ("ACTIVE BURST - hiring begets hiring; move now"
               if burst >= 2.0 else "near baseline")
        lines += [
            f"**Posting-arrival process (Hawkes):** intensity "
            f"{hawkes['intensity_now']}/day vs baseline {hawkes['baseline_per_day']}/day "
            f"(burst ratio ×{burst}, branching {hawkes['branching_ratio']}, "
            f"{hawkes['n_events']} arrivals/120d) - {tag}.", "",
        ]
    return lines


def _wclip(s, n: int) -> str:
    """Clip at a word boundary with an ellipsis - never mid-word."""
    s = (s or "").strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (cut or s[:n]) + "…"


def _loc_short(loc) -> str:
    """One readable location cell: first site +N, not nine cities."""
    if not loc:
        return "-"
    parts = [p.strip() for p in str(loc).split(";") if p.strip()]
    if len(parts) > 1:
        return f"{parts[0]} +{len(parts) - 1}"
    toks = [t.strip() for t in str(loc).split(",")]
    if len(toks) > 3:
        return f"{toks[0]} +{len(toks) - 1}"
    return ", ".join(toks[:2]) if len(toks) > 2 else str(loc)


def _fmt(x, nd: int = 2) -> str:
    """Human number: 0.557347 -> 0.56, 1.0 -> 1, None -> ?"""
    try:
        return f"{round(float(x), nd):g}"
    except (TypeError, ValueError):
        return "?" if x is None else str(x)


def _fmtlist(xs, nd: int = 3) -> str:
    if isinstance(xs, (list, tuple)) and xs:
        try:
            return "[" + ", ".join(f"{float(v):.{nd}f}" for v in xs) + "]"
        except (TypeError, ValueError):
            pass
    return str(xs)


# Target geography (Liam, 2026-07-05): US/Canada, Europe, Australia/NZ.
# Allow wins over block (multi-city rows count if ANY site qualifies);
# locations that match neither list pass through - better to show an
# ambiguous "2 Locations" row than to hide a real US role.
_GEO_ALLOW = re.compile(
    r"\b(united states|usa|u\.s\.|remote|canada|toronto|vancouver|montreal|"
    r"australia|sydney|melbourne|brisbane|new zealand|auckland|"
    r"united kingdom|uk|england|london|scotland|ireland|dublin|"
    r"germany|berlin|munich|france|paris|netherlands|amsterdam|"
    r"switzerland|zurich|zug|geneva|sweden|stockholm|denmark|copenhagen|"
    r"norway|oslo|finland|helsinki|spain|madrid|barcelona|italy|milan|rome|"
    r"austria|vienna|belgium|brussels|poland|warsaw|portugal|lisbon|"
    r"czech|prague|luxembourg|europe)\b", re.I)
_GEO_BLOCK = re.compile(
    r"\b(india|hyderabad|bangalore|bengaluru|mumbai|pune|chennai|gurgaon|"
    r"gurugram|noida|delhi|hong kong|singapore|japan|tokyo|osaka|china|"
    r"shanghai|beijing|shenzhen|hangzhou|korea|seoul|taiwan|taipei|"
    r"dubai|abu dhabi|uae|saudi|qatar|israel|tel aviv|brazil|sao paulo|"
    r"mexico|argentina|colombia|chile|philippines|manila|indonesia|jakarta|"
    r"vietnam|hanoi|thailand|bangkok|malaysia|kuala lumpur|nigeria|lagos|"
    r"kenya|nairobi|south africa|egypt|cairo|turkey|istanbul)\b", re.I)


def _geo_ok(loc) -> bool:
    if not loc:
        return True
    s = str(loc)
    if _GEO_ALLOW.search(s):
        return True
    return not _GEO_BLOCK.search(s)


# US preference (Liam, 2026-07-08): I'll take roles in the other allowed
# regions, but US ones should rank higher and non-US ones should clear a
# higher bar to surface at all. Two levers: _score_posting multiplies non-US
# roles by _NON_US_SCORE_MULT (ranking), and _geo_display_ok holds them to
# _NON_US_MIN_ALIGN before they earn a scarce queue/apply slot (appearance).
_NON_US_SCORE_MULT = float(os.environ.get("JOB_SWARM_NON_US_SCORE_MULT", "0.80"))
_NON_US_MIN_ALIGN = float(os.environ.get("JOB_SWARM_NON_US_MIN_ALIGN", "0.55"))
# US/remote tokens read as US-tier; the rest of the allow-list (Canada,
# Europe, AU/NZ) is explicitly non-US. Everything matching neither list is
# ambiguous and stays US-tier - the same lenient bias _geo_ok already takes.
_US_TIER = re.compile(r"\b(united states|usa|u\.s\.|u\.s|remote)\b", re.I)
_NON_US_ALLOW = re.compile(
    r"\b(canada|toronto|vancouver|montreal|"
    r"australia|sydney|melbourne|brisbane|new zealand|auckland|"
    r"united kingdom|uk|england|london|scotland|ireland|dublin|"
    r"germany|berlin|munich|france|paris|netherlands|amsterdam|"
    r"switzerland|zurich|zug|geneva|sweden|stockholm|denmark|copenhagen|"
    r"norway|oslo|finland|helsinki|spain|madrid|barcelona|italy|milan|rome|"
    r"austria|vienna|belgium|brussels|poland|warsaw|portugal|lisbon|"
    r"czech|prague|luxembourg|europe)\b", re.I)


def _is_us_tier(loc) -> bool:
    """US or remote -> US-tier (normal bar). An explicitly non-US allowed
    region -> non-US (strict bar). Unknown/ambiguous -> US-tier, matching
    _geo_ok's 'better to show an ambiguous US row' bias."""
    if not loc:
        return True
    s = str(loc)
    if _US_TIER.search(s):
        return True
    if _NON_US_ALLOW.search(s):
        return False
    return True


def _geo_display_ok(loc, align) -> bool:
    """Region gate for what SURFACES to the human. US-tier roles use the
    normal bar; non-US allowed roles must also clear _NON_US_MIN_ALIGN."""
    if not _geo_ok(loc):
        return False
    if _is_us_tier(loc):
        return True
    return (align or 0.0) >= _NON_US_MIN_ALIGN


def _abbr_money(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "?"
    if x >= 1e9:
        return f"${x / 1e9:.1f}B"
    if x >= 1e6:
        return f"${x / 1e6:.0f}M"
    return f"${x:,.0f}"


async def compile_review(state: JobSwarmState):
    print("[Node] compile_review")
    run_date = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(REPORTS_DIR, run_date)
    os.makedirs(out_dir, exist_ok=True)
    # Same-day rerun: clear numbered dossiers from the earlier pass, or a
    # stale org keeps a dead 'NN_' file next to tonight's (two 01_*.md).
    for old in glob.glob(os.path.join(out_dir, "[0-9][0-9]_*.md")):
        os.remove(old)

    with open(state["shortlist_path"]) as f:
        shortlist = {o["org_key"]: o for o in json.load(f)["shortlist"]}
    audits = {}
    if state.get("audit_path"):
        with open(state["audit_path"]) as f:
            audits = {a["org_key"]: a for a in json.load(f)["audits"]}
    memos, contact_chores = [], []
    if state.get("memo_path"):
        with open(state["memo_path"]) as f:
            _memo_doc = json.load(f)
            memos = _memo_doc["memos"]
            contact_chores = _memo_doc.get("contact_chores", [])

    # GitHub people discovered during strategy_synthesis (persisted to meta so
    # the terminal node need not re-hit the API). Keyed by org_key.
    _conn_gp = swarm_db.connect()
    github_people = json.loads(swarm_db.get_meta(_conn_gp, "github_people", "{}"))
    research_people = json.loads(swarm_db.get_meta(_conn_gp, "research_people", "{}"))
    _conn_gp.close()

    # Warm paths: co-publication bridges from the home institution to the
    # finalists + top watchlist (research-heavy orgs resolve; startups won't).
    import warmpath_engine
    warm_targets = [m["organization"] for m in memos] + [
        o["display_name"] for o in list(shortlist.values())[:15]]
    try:
        warm_paths = await warmpath_engine.find_warm_paths(list(dict.fromkeys(warm_targets)))
    except Exception as e:
        print(f"[WarmPath] skipped: {e}")
        warm_paths = {}
    # Likely technical leads (published authorship only) for the memo targets
    try:
        tech_leads = await warmpath_engine.find_technical_leads(
            list(dict.fromkeys(m["organization"] for m in memos)))
    except Exception as e:
        print(f"[WarmPath] technical leads skipped: {e}")
        tech_leads = {}

    # Sector hiring calendars (2026 research, section 3): outreach timed to
    # budget cycles beats outreach timed to your own enthusiasm.
    _SEASON_NOTES = {
        1: "Pharma/CRO budgets just opened (calendar-year headcount) - biostat outreach peaks now.",
        2: "Pharma headcount live; bank model-risk backfills surge post-bonus (Feb-Mar).",
        3: "Bank model-risk backfill window (post-bonus attrition) still open.",
        4: "Between cycles - event-driven targets (funding, trials, expansions) carry the month.",
        5: "Between cycles - event-driven targets carry the month.",
        6: "Between cycles - event-driven targets carry the month.",
        7: "Quant 2027 campus cycle is ALREADY OPEN (IMC Graduate QR BS/MS, "
           "Optiver, SIG, D.E. Shaw; Jane Street/Citadel/HRT open Jul-Aug). "
           "Rolling review, seats largely committed by late October - "
           "submitting these outranks everything else this month.",
        8: "Quant/prop campus cycle OPEN (Aug-Oct) - next-summer reqs fill by year-end.",
        9: "Quant campus cycle open; second bank model-risk window (September).",
        10: "Federal FY just started - national labs/FFRDCs rush Q1 requisitions (Oct-Nov). Clearance-path applications land best NOW.",
        11: "Federal/lab Q1 requisition surge still open.",
        12: "Corporate freeze month - favor event-driven targets; prep January pharma outreach.",
    }
    index_lines = [
        f"# Job Swarm Review Queue - {run_date}",
        "",
        "The swarm researched, scored, and drafted overnight. You review, edit, "
        "and **send from your own email** - plain text, Tue-Thu 9-11am recipient "
        "time, no links in a first email.",
        "",
        "**The morning order:** apply direct first, Day-1 postings first "
        "(recruiter review concentrates in the first 24-48h; late "
        "applications convert near zero) -> then the warm channel (follow-ups "
        "due, referral pass, outreach emails) -> then the deep block (the "
        "artifact build). Do as much or little as the day allows - but keep "
        "the order: the timing edge expires first.",
        "",
        f"**Season note:** {_SEASON_NOTES[datetime.now().month]}",
        "",
        "*Full raw feeds: [INTEL.md](INTEL.md).*",
        "",
    ]
    index_lines += _funnel_lines()
    intel_lines = [
        f"# Market Intel - {run_date}",
        "",
        "The raw feeds behind tonight's queue - everything the swarm drank, "
        "unabridged. Nothing here demands action; [REVIEW_QUEUE.md](REVIEW_QUEUE.md) "
        "already distilled it. Open this when you want the evidence behind a signal.",
        "",
    ]

    # ---- Follow-ups due: the highest-conversion 10 minutes of the day ------
    # Most replies come from the (single) follow-up. One per org, then let go.
    conn_fu = swarm_db.connect()
    due = swarm_db.followups_due(conn_fu, min_days=4)
    conn_fu.close()
    if due:
        index_lines += [
            "## Follow-ups due - contacted ≥4 days ago, no reply", "",
            "42% of all replies come from the follow-up (2026 benchmark data), "
            "and this system sends exactly ONE: short, with one new concrete "
            "detail - never 'just checking in' - then let go. After sending, "
            "mark **follow-up** in the outreach tracker or just say it in a "
            "check-in:", "",
        ]
        for o in due:
            sent = (o.get("contacted_at") or "")[:10]
            try:
                days = (datetime.now() - datetime.strptime(sent, "%Y-%m-%d")).days
                age = f"{days} days ago"
            except ValueError:
                age = "date unknown"
            index_lines.append(
                f"- **{o['display_name']}** (`{o['org_key']}`) - contacted {sent} ({age})"
            )
        index_lines += [""]

    # Memo section built separately so the queue can order channels by
    # expected yield (audit 2026-07-05): follow-ups, then direct openings
    # (the strongest channel), then memos.
    memo_q_lines = [
        f"## Tonight's memo drafts ({len(memos)}) - "
        "what they're stuck on, and your angle", "",
    ]

    # Latest 10-K doc per memo org (public companies only) for the dossier line
    conn_tk = swarm_db.connect()
    tenk_rows = {}
    for m in memos:
        row = conn_tk.execute(
            "SELECT url, date, extra FROM docs WHERE org_key = ? AND source = 'tenk' "
            "ORDER BY date DESC LIMIT 1", (m["org_key"],)).fetchone()
        if row:
            tenk_rows[m["org_key"]] = row
    conn_tk.close()

    for i, memo in enumerate(memos, 1):
        org = shortlist.get(memo["org_key"], {})
        audit = audits.get(memo["org_key"], {})
        slug = memo["org_key"][:40]
        dossier_name = f"{i:02d}_{slug}.md"

        contacts = memo.get("contacts", {})
        contact_lines = [
            f"- **{k}**: {', '.join(str(x) for x in v) if isinstance(v, list) else v}"
            for k, v in contacts.items() if v
        ] or ["- No published contact channel found - check the org website / paper PDFs."]
        evidence_lines = [
            f"- [{e['source']}] ({e['date']}) [{e['title']}]({e['url']})"
            for e in org.get("evidence", [])
        ]
        explore_note = (
            ["> **Exploration pick** - drawn from below the normal alignment "
             "threshold to gather calibration data. Send only if the dossier "
             "convinces you on its merits.", ""]
            if memo.get("chosen_by") == "explore" else [])
        lint_note = (
            [f"> **NOT SENDABLE AS WRITTEN** - {'; '.join(memo['lint'])}. "
             "Fix every flagged item before this leaves your outbox.", ""]
            if memo.get("lint") else [])
        crit = memo.get("critic") or {}
        critic_note = []
        if crit.get("verdict") == "kill":
            critic_note = [
                "> **CROSS-MODEL CRITIC: KILL** "
                f"({crit.get('model', '?')}) - the drafter and an independent "
                "model DISAGREE about this memo. Issues raised: "
                + "; ".join(crit.get("issues") or ["unspecified"])
                + ". Read the evidence trail yourself before deciding.", ""]
        elif crit.get("verdict") == "promote" and VLLM_CRITIC_URL:
            critic_note = [
                f"> Cross-model critic ({crit.get('model', '?')}): promoted - "
                "an independent model family found no grounds to kill it.", ""]
        dossier = "\n".join([
            f"# {memo['organization']}",
            "",
            *explore_note,
            *lint_note,
            *critic_note,
            f"**Alignment (LLM audit):** {_fmt(memo.get('alignment_score'))}   ",
            f"**Prescore (quant filter):** {_fmt(org.get('prescore'))}   ",
            f"**Evidence volume:** {org.get('n_docs')} doc(s) in corpus",
            "",
            "## Audited bottleneck",
            str(audit.get("bottleneck_diagnosis", "n/a")),
            "",
            f"**Intervention vector:** {audit.get('intervention_vector', 'n/a')}",
            "",
            "## Evidence trail",
            *evidence_lines,
            "",
            *_tenk_lines(tenk_rows.get(memo["org_key"])),
            "## Published contact channels",
            *contact_lines,
            "",
            *_github_people_lines(github_people.get(memo["org_key"])),
            *_research_people_lines(research_people.get(memo["org_key"])),
            *_tech_lead_lines(tech_leads.get(memo["organization"])),
            *_warm_path_lines(warm_paths.get(memo["organization"])),
            *_ghost_role_lines(org.get("predicted_role"), org.get("hiring_accel"),
                               org.get("hiring_hawkes")),
            "## DRAFT memo - edit before sending; verify every technical claim yourself",
            "",
            f"**Subject:** {memo['subject']}",
            "",
            "```",
            memo["body"],
            "```",
            "",
            "> Send manually from your own address. After sending, mark "
            f"**sent** for `{memo['org_key']}` in the outreach tracker - or "
            "just say it in a check-in (\"emailed "
            f"{memo['organization']}\") and it's recorded for you.",
        ])
        with open(os.path.join(out_dir, dossier_name), "w") as f:
            f.write(dossier)
        explore_tag = (" · exploration pick" if memo.get("chosen_by") == "explore"
                       else "")
        lint_tag = (" · NEEDS EDIT" if memo.get("lint") else "")
        memo_q_lines += [
            f"{i}. **{memo['organization'][:60]}** - align "
            f"{_fmt(memo.get('alignment_score'))} · prescore "
            f"{_fmt(org.get('prescore'))}{explore_tag}{lint_tag} - "
            f"[dossier + draft]({dossier_name})",
            f"   - *Stuck on:* {' '.join(str(audit.get('bottleneck_diagnosis', 'n/a')).split())}",
            f"   - *Your angle:* {' '.join(str(audit.get('intervention_vector', 'n/a')).split())}",
            "",
        ]

    # Watchlist: strong quant signal, below memo threshold - tomorrow's candidates
    watch = [o for k, o in shortlist.items()
             if k not in {m["org_key"] for m in memos}][:20]
    intel_lines += ["", "## Watchlist (high prescore, no memo yet)", ""]
    for o in watch:
        intel_lines.append(f"- {o['display_name'][:60]} - prescore={_fmt(o['prescore'])}, "
                           f"regime={o['regime']}")

    # ---- Direct openings: live postings from company ATS boards, ranked ----
    # These are the fast channel - fresher than aggregators. Ranked by
    # fit × repost-forensics × seniority (_score_posting; the old staleness
    # boost was red-team REFUTED - age predicts ghost jobs, and Day-1
    # freshness is what converts, so the queue surfaces fresh first).
    # Staff/Manager screens an MS new grad won't clear are down-weighted.
    # Comp: disclosed salary, else the employer's H-1B LCA median when
    # lca_engine.py has been run.
    import trajectory_engine
    loaded_pf = profile_engine.load_profile(PROFILE_CACHE)
    profile_vec = loaded_pf["embedding"]
    pf_facets = loaded_pf.get("facets")
    conn = swarm_db.connect()
    postings = swarm_db.recent_docs_by_source(
        conn, ("ats_jobs", "remoteok", "usajobs"), days=10)
    keep = _posting_filter(conn)
    postings = [p for p in postings if keep(p)]
    # Posting text for the hard-constraint gate (recent_docs_by_source omits
    # text). One batched read keyed by doc_id.
    ptext = {}
    if postings:
        pqm = ",".join("?" for _ in postings)
        ptext = {r["doc_id"]: r["text"] for r in conn.execute(
            f"SELECT doc_id, text FROM docs WHERE doc_id IN ({pqm})",
            [p.get("doc_id") for p in postings])}
    taste_mat = trajectory_engine.taste_vectors(conn)
    for p in postings:
        p["align"] = trajectory_engine.alignment_score(p["embedding"], profile_vec, pf_facets)
        hc_mult, hc_reasons = _hard_constraints(
            ptext.get(p.get("doc_id"), ""), p.get("title"))
        if hc_reasons:
            p.setdefault("extra", {})["hard_constraints"] = hc_reasons
        taste = trajectory_engine.taste_boost(p["embedding"], taste_mat)
        if taste > 1.0:
            p.setdefault("extra", {})["taste"] = round(taste, 3)
        p["score"] = (_score_posting(p) * _lca_demand_boost(conn, p["org_key"])
                      * hc_mult * taste)
    postings.sort(key=lambda p: p["score"], reverse=True)

    # Listwise re-rank of the top slice (LLM reasons about fit across postings;
    # the deterministic score picks WHICH postings are in contention, the
    # re-rank refines their ORDER). Excerpts come from the constraint-gate read
    # above. Best-effort - the deterministic order stands if the model is down.
    _RERANK_N = int(os.environ.get("JOB_SWARM_RERANK_N", "24"))
    if len(postings) >= 4:
        head = postings[:_RERANK_N]
        for p in head:
            p["excerpt"] = _posting_excerpt(ptext.get(p.get("doc_id")) or "",
                                            p.get("title"))
        try:
            expertise_rr = "\n".join(
                f"- {s}" for s in loaded_pf["profile"].get("expertise_matrix") or [])
            reranked = await _rerank_postings(head, expertise_rr)
            postings = reranked + postings[_RERANK_N:]
            print(f"[Rerank] listwise re-rank applied to top {len(head)} postings")
        except Exception as e:
            print(f"[Rerank] skipped (kept deterministic order): {e}")

    # Card enrichment (2026-07-06 review: "i have no idea what the company
    # is" / "shouldnt it explain the role?"): org display names, a plain-text
    # role excerpt, and the top profile facets each posting matches.
    # Labels aligned to pf_facets (expertise items + one per project file). Fall
    # back to expertise_matrix for caches built before _facet_labels existed.
    exp_names = (loaded_pf["profile"].get("_facet_labels")
                 or loaded_pf["profile"].get("expertise_matrix") or [])
    top_p = postings[:30]
    name_by_org, text_by_doc = {}, {}
    if top_p:
        qm = ",".join("?" for _ in top_p)
        name_by_org = {r["org_key"]: r["display_name"] for r in conn.execute(
            f"SELECT org_key, display_name FROM orgs WHERE org_key IN ({qm})",
            [p["org_key"] for p in top_p])}
        text_by_doc = {r["doc_id"]: r["text"] for r in conn.execute(
            f"SELECT doc_id, text FROM docs WHERE doc_id IN ({qm})",
            [p.get("doc_id") for p in top_p])}

    open_rows = []           # computed once, rendered twice (queue + intel)
    for p in postings[:30]:
        ex = p.get("extra", {})
        days_open = int(p.get("days_open") or 0)
        gflags = set(ex.get("ghost_flags") or [])
        open_tag = f"{days_open}d" + (
            " LONG-OPEN" if days_open >= 45 and ex.get("provider") != "usajobs"
            and "unmaintained_45d" not in gflags else "")
        if ex.get("close_date"):
            open_tag = f"closes {ex['close_date']}"
        reposts = int(ex.get("repost_count") or 0)
        kind = ex.get("repost_kind") or ("revised" if reposts else None)
        if ex.get("repost_mill") or kind == "evergreen":
            open_tag += " EVERGREEN"
        elif kind == "churn":
            open_tag += f" CHURN({int(ex.get('repost_gap_days') or 0)}d gap)"
        elif kind == "revised":
            open_tag += f" REPOST×{reposts}" if reposts > 1 else " REPOST"
            if ex.get("salary_up"):
                open_tag += " SALARY-RAISED"
        if gflags:
            open_tag += " GHOST"
        if ex.get("clearance_sponsor"):
            open_tag += " CLEARANCE-PATH"
        if ex.get("hard_constraints"):
            open_tag += " HARD-REQ(" + "; ".join(ex["hard_constraints"]) + ")"
        if ex.get("taste"):
            # taste_boost > 1: close to a role you liked (or to your own
            # why-note) - the visible receipt that liking a card did something
            open_tag += " YOUR-STYLE"
        title = (p["title"] or "").strip()
        if _SENIOR_RE.search(p["title"] or ""):
            title = "SENIOR: " + title
        comp = ex.get("salary")
        comp_short = comp
        if not comp:
            prior = swarm_db.salary_prior(conn, p["org_key"])
            comp = f"~${prior[0]:,.0f} (H-1B median, n={prior[1]})" if prior else "-"
            comp_short = f"~${prior[0] / 1000:,.0f}K est." if prior else "-"
        open_rows.append({
            "score": p["score"], "align": p["align"], "title": title,
            "tag": open_tag, "loc": ex.get("location") or "-",
            "comp": comp, "comp_short": comp_short,
            "provider": ex.get("provider", "link"), "url": p["url"],
            "org_key": p["org_key"], "doc_id": p.get("doc_id"),
            "days_open": round(float(p.get("days_open") or 0.0), 1),
            "org": name_by_org.get(p["org_key"]) or p["org_key"],
            "excerpt": _posting_excerpt(
                text_by_doc.get(p.get("doc_id")) or "", title),
            "fit": _facet_matches(p.get("embedding"), pf_facets, exp_names),
        })

    skip_tags = ("EVERGREEN", "GHOST")   # flagged as skips - no queue slot
    queue_rows = [r for r in open_rows
                  if not any(t in r["tag"] for t in skip_tags)
                  and _geo_display_ok(r["loc"], r["align"])][:12]
    # assembled now, appended to the queue AFTER the digest: memos (email
    # someone) are the primary channel, applications come second
    open_q_lines: list = []
    if queue_rows:
        open_q_lines += [
            "", f"## Direct openings - top {len(queue_rows)}, "
            "your geography only (US/CA · Europe · AU/NZ)", "",
            "*Straight from company ATS boards - fresher than aggregators, "
            "though some are also cross-posted elsewhere. Everything filtered "
            "out (other regions, EVERGREEN/GHOST) is in "
            "[INTEL.md](INTEL.md).*", "",
            "| Score | Role | Open | Where | Comp | Apply |",
            "|-------|------|------|-------|------|-------|",
        ]
        for r in queue_rows:
            open_q_lines.append(
                f"| {_fmt(r['score'])} | **{r['org']}** — {r['title']} | "
                f"{r['tag']} | {_loc_short(r['loc'])} | {r['comp_short']} | "
                f"[{r['provider']}]({r['url']}) |")
    else:
        open_q_lines += [
            "", "## Direct openings", "",
            "_Nothing in your geography worth a slot tonight - the unfiltered "
            "table is in [INTEL.md](INTEL.md)._",
        ]

    intel_lines += [
        "", "## Direct openings - full table", "",
        "Score = fit × repost-forensics × ghost-flags × clearance × seniority "
        "× H-1B demand. Tags: LONG-OPEN = 45+ days but still maintained "
        "(ambiguous: could be hard-to-fill, could be drifting toward ghost - "
        "age is displayed, not scored). REPOST = pulled and re-listed with "
        "changes (failed to fill; SALARY-RAISED = they raised the offer - "
        "apply immediately). EVERGREEN = perpetual pipeline ad (skip). CHURN = "
        "re-listed after a long gap (the last hire left - research the "
        "employer). GHOST = suspect posting (wide salary band / evergreen "
        "title / unmaintained / PERM visa-compliance ad). CLEARANCE-PATH = "
        "employer sponsors a first security clearance (US-citizen edge: far "
        "fewer applicants, $10-45k premium once cleared). SENIOR = "
        "senior-flagged title. Posting age is shown but NOT scored - old "
        "postings skew ghost (2026 evidence).",
        "",
        "| Score | Fit | Role | Open | Location | Comp | Apply |",
        "|-------|-----|------|------|----------|------|-------|",
    ]
    for r in open_rows:
        intel_lines.append(
            f"| {r['score']:.3f} | {r['align']:.3f} | {r['title']} | "
            f"{r['tag']} | {r['loc']} | {r['comp']} | "
            f"[{r['provider']}]({r['url']}) |")
    if not open_rows:
        intel_lines.append("_No live postings seen in the last 10 days._")

    # ---- Fresh capital: Form D raises this week - hiring in 30-90 days -----
    raises = swarm_db.recent_docs_by_source(conn, ("formd",), days=8)

    def _sold(r):
        try:
            return float(r.get("extra", {}).get("total_sold") or 0)
        except (TypeError, ValueError):
            return 0.0
    raises.sort(key=_sold, reverse=True)
    intel_lines += ["", "## Fresh capital (SEC Form D, last 8 days) - hiring follows money", ""]
    for r in raises[:20]:
        execs = ", ".join(r.get("contacts", {}).get("executives", [])[:3])
        intel_lines.append(f"- [{r['title'][:100]}]({r['url']}) - execs: {execs or 'n/a'}")
    if not raises:
        intel_lines.append("_No relevant Form D filings this week._")

    # ---- Pre-posting window: raised 21-150 days ago, not hiring YET --------
    # Red-team retiming (1d): capital deploys into hiring 60-120 days AFTER a
    # round closes, and day-of-announcement outreach lands in the lowest-
    # converting cohort (everyone pitches on the press release). The window
    # opens ~3 weeks post-filing and stays open while no postings exist.
    pre = conn.execute(
        "SELECT o.display_name, d.org_key, d.title, d.url, d.extra "
        "FROM docs d JOIN orgs o ON o.org_key = d.org_key "
        "WHERE d.source = 'formd' "
        "AND d.added_at <  datetime('now', '-21 days') "
        "AND d.added_at >= datetime('now', '-150 days') "
        "AND NOT EXISTS (SELECT 1 FROM docs p WHERE p.org_key = d.org_key "
        "  AND p.source = 'ats_jobs' "
        "  AND COALESCE(p.last_seen_at, p.added_at) >= datetime('now', '-4 days')) "
        "ORDER BY d.added_at DESC"
    ).fetchall()
    seen_pre, pre_rows = set(), []
    for r in pre:
        if r["org_key"] in seen_pre:
            continue
        seen_pre.add(r["org_key"])
        try:
            sold = float(json.loads(r["extra"] or "{}").get("total_sold") or 0)
        except (TypeError, ValueError):
            sold = 0.0
        pre_rows.append((sold, r))
    pre_rows.sort(key=lambda t: t[0], reverse=True)
    intel_lines += [
        "", "## Pre-posting window (raised 3 weeks to 5 months ago, "
        "no live openings)", "",
        "Money in the bank, roles still undefined, and the day-of-announcement "
        "pitch flood is over. Capital converts to hiring 60-120 days after a "
        "round - this is where the ghost-role pitch converts best.", "",
    ]
    for sold, r in pre_rows[:12]:
        amt = f"${sold:,.0f} raised" if sold else "amount undisclosed"
        intel_lines.append(f"- **{r['display_name'][:60]}** - {amt} · [filing]({r['url']})")
    if not pre_rows:
        intel_lines.append("_Every recent raiser already has live postings._")

    # ---- Phase transitions: sponsor's FIRST Phase III in the corpus --------
    # A first Phase III is when biostatistics headcount scales, months before
    # any posting exists - the regime change that matters for an MS Stats.
    p3 = conn.execute(
        "SELECT o.display_name, d.org_key, d.title, d.url, d.date "
        "FROM docs d JOIN orgs o ON o.org_key = d.org_key "
        "WHERE d.source = 'clinicaltrials' AND d.extra LIKE '%PHASE3%' "
        "AND d.added_at >= datetime('now', '-8 days') "
        "AND NOT EXISTS (SELECT 1 FROM docs d2 WHERE d2.org_key = d.org_key "
        "  AND d2.source = 'clinicaltrials' AND d2.extra LIKE '%PHASE3%' "
        "  AND d2.added_at < d.added_at)"
    ).fetchall()
    if p3:
        intel_lines += [
            "", "## First Phase III (sponsor entering pivotal trials) - "
            "biostat hiring follows in 3-6 months", "",
        ]
        seen_p3 = set()
        for r in p3:
            if r["org_key"] in seen_p3:
                continue
            seen_p3.add(r["org_key"])
            intel_lines.append(
                f"- **{r['display_name'][:60]}** - [{r['title'][:90]}]({r['url']}) "
                f"({r['date'] or 'date n/a'})")

    # ---- Phase II readouts: the go/no-go that precedes biostat hiring ------
    p2 = conn.execute(
        "SELECT o.display_name, d.org_key, d.title, d.url, d.date, d.extra "
        "FROM docs d JOIN orgs o ON o.org_key = d.org_key "
        "WHERE d.source = 'ct_event' "
        "AND d.added_at >= datetime('now', '-14 days') "
        "ORDER BY d.date DESC"
    ).fetchall()
    if p2:
        intel_lines += [
            "", "## Phase II readouts (completed / stopped recruiting, last 14 "
            "days) - funding catalyst + biostat hiring in 3-6 months", "",
        ]
        seen_p2 = set()
        for r in p2:
            if r["org_key"] in seen_p2:
                continue
            seen_p2.add(r["org_key"])
            status = json.loads(r["extra"] or "{}").get("status", "")
            intel_lines.append(
                f"- **{r['display_name'][:60]}** - [{r['title'][:90]}]({r['url']}) "
                f"({status}, {r['date'] or 'date n/a'})")

    # ---- Layoff radar: WARN filings - avoid the filers, court the rivals ---
    warn_rows = conn.execute(
        "SELECT o.display_name, d.title, d.url, d.date, d.extra "
        "FROM docs d JOIN orgs o ON o.org_key = d.org_key "
        "WHERE d.source = 'warn' AND d.added_at >= datetime('now', '-21 days') "
        "ORDER BY d.date DESC"
    ).fetchall()
    if warn_rows:
        by_industry: dict = {}
        for r in warn_rows:
            ex_w = json.loads(r["extra"] or "{}")
            by_industry.setdefault(ex_w.get("industry") or "Unclassified", []).append((r, ex_w))
        intel_lines += [
            "", "## Layoff radar (WARN filings, last 21 days)", "",
            "Memos to these orgs are auto-suppressed for 90 days (they're "
            "shrinking). The play is the OTHER side: their competitors just "
            "gained market share and freed-up talent budget - and their "
            "laid-off teams' employers-of-choice are about to get flooded, "
            "so move fast on anything you already have in flight nearby.", "",
        ]
        for industry, rows_i in sorted(by_industry.items(),
                                       key=lambda kv: -len(kv[1]))[:6]:
            intel_lines.append(f"**{industry}**")
            for r, ex_w in rows_i[:5]:
                n_aff = ex_w.get("employees_affected")
                loc = ", ".join(x for x in (ex_w.get("city"), ex_w.get("state")) if x)
                intel_lines.append(
                    f"- {r['display_name'][:55]} - {n_aff or '?'} affected "
                    f"({loc or 'n/a'}, {r['date'] or 'n/a'}) · "
                    f"[source]({ex_w.get('source_url') or r['url']})")
            intel_lines.append("")

    # ---- Expansion radar: companies legally registering in NY --------------
    # A foreign qualification = expansion into a new state, weeks before any
    # local posting. Corpus-tracked orgs first (we already know they fit),
    # then fresh names that smell like the candidate's sectors.
    sos_rows = conn.execute(
        "SELECT o.display_name, d.org_key, d.title, d.date, d.extra, "
        "  (SELECT COUNT(*) FROM docs d2 WHERE d2.org_key = d.org_key "
        "   AND d2.source NOT IN ('sos_ny', 'sos_co')) AS n_other_docs "
        "FROM docs d JOIN orgs o ON o.org_key = d.org_key "
        "WHERE d.source IN ('sos_ny', 'sos_co') "
        "AND d.added_at >= datetime('now', '-14 days') "
        "ORDER BY n_other_docs DESC, d.date DESC"
    ).fetchall()
    if sos_rows:
        _SECTOR_NAME_RE = re.compile(
            r"\b(AI|LABS?|BIO\w*|THERAPEUTIC|GENOMIC|DATA|ANALYTIC|ROBOTIC|"
            r"QUANT|CAPITAL|TECHNOLOG|COMPUT|INTELLIGEN|SCIENCE)\b", re.I)
        tracked = [r for r in sos_rows if r["n_other_docs"] > 0]
        fresh = [r for r in sos_rows if r["n_other_docs"] == 0
                 and _SECTOR_NAME_RE.search(r["display_name"] or "")]
        if tracked or fresh:
            intel_lines += [
                "", "## Expansion radar (new NY / CO registrations, last 14 days)", "",
                "A foreign-qualification filing = legally gearing up to operate "
                "(and hire) in that state, weeks before postings exist. Colorado "
                "doubles as the quantum-hub watch (Boulder/Denver).", "",
            ]
            for r in tracked[:8]:
                ex_s = json.loads(r["extra"] or "{}")
                intel_lines.append(
                    f"- **{r['display_name'][:55]}** (already in corpus, "
                    f"{r['n_other_docs']} docs) - from {ex_s.get('juris') or '?'}, "
                    f"filed {r['date'] or 'n/a'}")
            for r in fresh[:8]:
                ex_s = json.loads(r["extra"] or "{}")
                intel_lines.append(
                    f"- {r['display_name'][:55]} - from {ex_s.get('juris') or '?'}, "
                    f"filed {r['date'] or 'n/a'} (new name, sector-flavored)")

    # ---- 10-K language shifts: public cos drifting toward ML/data ----------
    tk = conn.execute(
        "SELECT o.display_name, d.url, d.date, d.extra "
        "FROM docs d JOIN orgs o ON o.org_key = d.org_key "
        "WHERE d.source = 'tenk' AND d.added_at >= datetime('now', '-30 days') "
        "AND d.extra LIKE '%ml_term_delta%'"
    ).fetchall()
    tk_rows = []
    for r in tk:
        delta = json.loads(r["extra"] or "{}").get("ml_term_delta")
        if delta is not None and delta > 0:
            tk_rows.append((int(delta), r))
    if tk_rows:
        tk_rows.sort(key=lambda t: t[0], reverse=True)
        intel_lines += [
            "", "## 10-K watch: risk-factor language drifting toward ML/data", "",
            "Public companies whose newest 10-K uses measurably more ML/data "
            "vocabulary than last year's - R&D budget moving toward your field "
            "before postings show it.", "",
        ]
        for delta, r in tk_rows[:10]:
            intel_lines.append(
                f"- **{r['display_name'][:60]}** - +{delta} ML/data mentions YoY "
                f"([filing]({r['url']}), {r['date']})")
    conn.close()

    # ---- Ghost roles: predicted, not-yet-posted positions -------------------
    ghosts = [o for o in shortlist.values() if o.get("predicted_role")]
    ghosts.sort(key=lambda o: o["predicted_role"]["similarity"], reverse=True)
    intel_lines += ["", "## Roles that don't exist yet (predicted from market archetypes)", ""]
    for o in ghosts[:15]:
        g = o["predicted_role"]
        sal = f", ~${g['market_median_salary']:,.0f}" if g.get("market_median_salary") else ""
        accel = o.get("hiring_accel") or {}
        accel_tag = (f", board growing {accel['slope_per_day']:+.2f} roles/day"
                     if accel.get("slope_per_day", 0) > 0 else "")
        hawkes = o.get("hiring_hawkes") or {}
        burst_tag = (f", HIRING BURST ×{hawkes['burst_ratio']} (Hawkes)"
                     if hawkes.get("burst_ratio", 0) >= 2.0 else "")
        intel_lines.append(
            f"- **{o['display_name'][:50]}** -> *{g['archetype_titles'][0]}* "
            f"(cos={g['similarity']}{sal}{accel_tag}{burst_tag}, regime={o['regime']})"
        )
    if not ghosts:
        intel_lines.append("_Taxonomy still warming up - needs ≥40 postings in the corpus._")

    # ---- Comp observatory: what the market discloses -------------------------
    import ghost_engine
    conn2 = swarm_db.connect()
    comp = ghost_engine.comp_observatory(conn2)
    conn2.close()
    if comp:
        q = comp["quartiles"]
        intel_lines += [
            "", "## Comp observatory (disclosed salaries in the corpus)", "",
            f"{comp['n_disclosed']} postings disclose pay. Quartiles: "
            f"P25 ${q[0]:,} · median ${q[1]:,} · P75 ${q[2]:,} · P90 ${q[3]:,}", "",
            "Top disclosed:",
        ]
        for t in comp["top_postings"]:
            intel_lines.append(f"- ${t['salary']:,} - [{t['title']}]({t['url']}) ({t['board']})")

    # ---- Market signals digest: the intel feeds, distilled for the queue ---
    digest = []
    if raises:
        top3 = []
        for r in raises[:3]:
            name = r["title"].replace("Form D: ", "").split(" raised ")[0]
            top3.append(f"{_wclip(name, 32)} ({_abbr_money(_sold(r))})")
        more = f", +{len(raises) - 3} more" if len(raises) > 3 else ""
        digest.append(f"- **Fresh capital** ({len(raises)} raises; hiring "
                      f"follows in 30-90 days): {', '.join(top3)}{more}")
    if pre_rows:
        names = ", ".join(r["display_name"][:32] for _, r in pre_rows[:3])
        more = f", +{len(pre_rows) - 3} more" if len(pre_rows) > 3 else ""
        digest.append(f"- **Pre-posting window** ({len(pre_rows)} funded orgs "
                      f"with no live openings - the best cold-outreach slot): "
                      f"{names}{more}")
    bursts = [o for o in ghosts
              if (o.get("hiring_hawkes") or {}).get("burst_ratio", 0) >= 2.0]
    if bursts:
        digest.append("- **Hiring bursts right now:** " + ", ".join(
            f"{o['display_name'][:32]} (×{o['hiring_hawkes']['burst_ratio']})"
            for o in bursts[:4]))
    if tk_rows:
        digest.append("- **R&D budgets drifting toward ML/data (10-K language):** "
                      + ", ".join(f"{r['display_name'][:28]} (+{d})"
                                  for d, r in tk_rows[:3]))
    n_p3 = len({r["org_key"] for r in p3})
    n_p2 = len({r["org_key"] for r in p2})
    if n_p3 or n_p2:
        digest.append(f"- **Biostat channel:** {n_p3} sponsors entered their "
                      f"first Phase III and {n_p2} Phase II readouts landed - "
                      "biostat hiring follows in 3-6 months")
    if warn_rows:
        digest.append(f"- **Layoff radar:** {len(warn_rows)} WARN filings "
                      "tracked; memos to those orgs are auto-suppressed")
    # Channel order by expected yield: openings first (fresh, real,
    # geo-filtered requisitions), memos second, market color last.
    index_lines += open_q_lines
    index_lines += [""] + memo_q_lines
    if digest:
        index_lines += [
            "", "## Market signals this week - the 30-second version", "",
            *digest,
        ]

    # ---- ARTIFACT_BRIEFS.md: proof-of-work nominations ----------------------
    briefs, apps = [], []      # also feed ITEMS.json (the dashboard inbox)
    if state.get("artifact_path") and os.path.exists(state["artifact_path"]):
        with open(state["artifact_path"]) as f:
            briefs = json.load(f)["briefs"]
        if briefs:
            brief_lines = [
                "# Proof-of-Work Artifact Briefs", "",
                "The swarm found these problems in public documents and planned a "
                "≤2-day artifact for each. **Backtrace the provenance chain first** "
                "(every VERIFIED quote links to the exact source document), build the "
                "artifact yourself on the cluster, and only then send the email - "
                "it opens with work already done, not a request. Never send an "
                "email whose claims you haven't personally verified and built.", "",
            ]
            for b in briefs:
                org = shortlist.get(b["org_key"], {})
                audit = audits.get(b["org_key"], {})
                brief_lines.append(_brief_markdown(b, org or
                                                   {"display_name": b["organization"],
                                                    "evidence": []}, audit))
                if b.get("scaffold_md"):
                    scaffold_name = f"SCAFFOLD_{b['org_key'][:40]}.md"
                    with open(os.path.join(out_dir, scaffold_name), "w") as f:
                        f.write(b["scaffold_md"])
                    brief_lines.append(
                        f"**Project spec ready:** [{scaffold_name}]"
                        f"({scaffold_name}) - the provenance, the what-to-"
                        "build description, and the success criteria; the "
                        "design and all the code are yours.\n")
            with open(os.path.join(out_dir, "ARTIFACT_BRIEFS.md"), "w") as f:
                f.write("\n".join(brief_lines))
            index_lines += ["", f"**Proof-of-work briefs:** "
                            f"[ARTIFACT_BRIEFS.md](ARTIFACT_BRIEFS.md) "
                            f"({len(briefs)} nominated - the highest-leverage "
                            f"hours this report offers)"]

    # ---- Entity hygiene: orgs sharing one website domain --------------------
    conn_m = swarm_db.connect()
    dup_groups = swarm_db.suggest_domain_merges(conn_m)
    conn_m.close()
    if dup_groups:
        intel_lines += [
            "", "## Possible duplicate orgs (same website domain)", "",
            "One company split across name variants dilutes its trajectory and "
            "evidence. Review, then fold: `python3 js_review.py merge <keep> <absorb>`", "",
        ]
        for domain, orgs in dup_groups[:8]:
            names = " · ".join(f"`{k}` ({n[:30]})" for k, n in orgs[:4])
            intel_lines.append(f"- **{domain}**: {names}")

    stats = state.get("run_stats", {})
    intel_lines += ["", "## Run stats", "", f"```json\n{json.dumps(stats, indent=2)}\n```"]

    # ---- APPLICATIONS.md: forged materials for the top direct openings ------
    if state.get("application_path") and os.path.exists(state["application_path"]):
        with open(state["application_path"]) as f:
            apps = json.load(f)["applications"]
        app_lines = [
            "# Application Materials - tailored per posting", "",
            "Everything below is a DRAFT grounded only in your profile documents. "
            "Fill every `[ADD REAL NUMBER: ...]` prompt with the true figure (or cut "
            "the claim), check the honest keyword gaps, then submit personally.", "",
        ]
        for a in apps:
            app_lines += [
                f"## {a['posting_title']}",
                f"align={a.get('align', 0):.3f} · open {int(a.get('days_open') or 0)}d · "
                f"{a.get('location') or '-'} · "
                f"{a.get('salary') or 'salary undisclosed'} · [posting]({a['url']})", "",
                "**Application note (edit, then send):**", "", "```",
                a.get("note", ""), "```", "",
            ]
            if a.get("cover_letter"):
                app_lines += ["**Cover letter (for portals that want one):**",
                              "", "```", a["cover_letter"], "```", ""]
            app_lines += [
                "**Resume bullets rewritten for this posting:**",
            ]
            for b in a.get("resume_bullets", []):
                app_lines.append(f"- *{b.get('theme', '')}* -> {b.get('bullet', '')}")
            app_lines += [
                "",
                f"**Keywords you can honestly claim:** {', '.join(a.get('keywords_matched', [])) or '-'}",
                f"**Honest gaps (don't fake these):** {', '.join(a.get('keywords_missing', [])) or 'none'}",
            ]
            topics = a.get("likely_interview_topics") or []
            if topics:
                app_lines += ["", "**Likely interview topics (prep here first):**"]
                app_lines += [f"- {t}" for t in topics]
            app_lines += ["", "---", ""]
        with open(os.path.join(out_dir, "APPLICATIONS.md"), "w") as f:
            f.write("\n".join(app_lines))
        index_lines += ["", f"**Tailored application materials:** [APPLICATIONS.md](APPLICATIONS.md) "
                        f"({len(apps)} postings)"]

    # Resume audit retired 2026-07-08 (hosted-model review replaced it).

    # ---- Config suggestions (regenerated whenever profile documents change) -
    cfg_src = os.path.join(PROFILE_CACHE, "CONFIG_SUGGESTIONS.md")
    if os.path.exists(cfg_src):
        with open(cfg_src) as f:
            cfg_text = f.read()
        with open(os.path.join(out_dir, "CONFIG_SUGGESTIONS.md"), "w") as f:
            f.write(cfg_text)
        index_lines += ["**Search-vocabulary suggestions for the new profile:** "
                        "[CONFIG_SUGGESTIONS.md](CONFIG_SUGGESTIONS.md) - review "
                        "and paste accepted keys into job_swarm_config.json"]

    # ---- ITEMS.json: the structured feed behind the dashboard inbox ---------
    # One entry per thing the human could DO tonight. The md files above are
    # the archive/evidence layer; this is the interface contract.
    app_by_url = {a.get("url"): a for a in apps if a.get("url")}
    items = []
    for i, memo in enumerate(memos, 1):
        org = shortlist.get(memo["org_key"], {})
        audit = audits.get(memo["org_key"], {})
        leads = [ld["name"] for ld in (tech_leads.get(memo["organization"]) or [])]
        summary = " ".join(str(audit.get("bottleneck_diagnosis", "")).split())
        if memo.get("lint"):
            summary = f"NEEDS EDIT ({'; '.join(memo['lint'])}) - {summary}"
        items.append({
            "id": f"memo:{memo['org_key']}", "kind": "email",
            "lane": "warm", "artifact_depth": "none",
            "critic": memo.get("critic"),
            "org_key": memo["org_key"], "org": memo["organization"],
            "title": f"Email {memo['organization']}",
            "lint": memo.get("lint") or [],
            "summary": summary,
            "angle": " ".join(str(audit.get("intervention_vector", "")).split()),
            "contacts": memo.get("contacts") or {}, "people": leads,
            "align": memo.get("alignment_score"), "prescore": org.get("prescore"),
            "regime": org.get("regime"),
            "explore": memo.get("chosen_by") == "explore",
            "subject": memo.get("subject"),
            "draft": memo.get("body"),
            "dossier": f"{i:02d}_{memo['org_key'][:40]}.md",
        })
    for r in open_rows:
        if any(t in r["tag"] for t in ("EVERGREEN", "GHOST")) \
                or not _geo_display_ok(r["loc"], r["align"]):
            continue
        a = app_by_url.get(r["url"])
        items.append({
            "id": "open:" + hashlib.sha1(r["url"].encode()).hexdigest()[:12],
            "kind": "apply", "lane": "breadth",
            "freshness_days": r.get("days_open"),
            "org_key": r["org_key"], "doc_id": r.get("doc_id"),
            "org": r.get("org"),
            "title": r["title"],
            "summary": (f"{r['loc']} · {r['comp']} · {r['tag']}"
                        + (f"\n{r['excerpt']}" if r.get("excerpt") else "")),
            "angle": r.get("fit") or None,
            "url": r["url"], "provider": r["provider"],
            "score": round(float(r["score"]), 3),
            # full posting text rides the card so the dashboard's tailored-
            # resume button has the real requirements to work from (Workday etc.
            # are JS-rendered, so a live fetch from the dashboard host would come back empty)
            "posting_text": (text_by_doc.get(r.get("doc_id")) or "")[:6000] or None,
            "draft": a.get("note") if a else None,
            "app": ({"note": a.get("note"),
                     "cover_letter": a.get("cover_letter"),
                     "bullets": a.get("resume_bullets"),
                     "topics": a.get("likely_interview_topics"),
                     "gaps": a.get("keywords_missing")} if a else None),
        })
    for b in briefs:
        plan = b.get("artifact_plan") or {}
        items.append({
            "id": f"brief:{b['org_key']}", "kind": "build",
            "lane": "deep", "artifact_depth": "deep",
            "scaffold": (f"SCAFFOLD_{b['org_key'][:40]}.md"
                         if b.get("scaffold_md") else None),
            "org_key": b["org_key"], "org": b.get("organization"),
            "title": f"Build for {b.get('organization')}: "
                     f"{b.get('title', 'proof-of-work artifact')}",
            "summary": (" ".join(str(b.get("problem_statement", "")).split())
                        + " THE BUILD: "
                        + " ".join(str(plan.get("goal", "")).split())),
            "angle": " ".join(str(plan.get("goal", "")).split()),
            "hours": plan.get("estimated_hours"),
            "dossier": "ARTIFACT_BRIEFS.md",
            "section": b.get("organization"),
        })
    # Find-contact chores (audit M2): high-scoring orgs the slot gate held
    # back for want of an address. The playbook is the 2026 contact-discovery
    # research; a found address re-enters via a check-in.
    for c in contact_chores:
        site = f" Start at {c['website']}." if c.get("website") else ""
        items.append({
            "id": f"chore:contact:{c['org_key']}", "kind": "chore",
            "lane": "warm",
            "org_key": c["org_key"], "org": c["org"],
            "title": f"Find a contact at {c['org']}",
            "summary": ((f"What they do: {c['what']} " if c.get("what") else "")
                        + f"Scores high (align {_fmt(c.get('align'))}) but "
                        f"publishes no email.{site} Playbook: guess "
                        "first.last@/first@ on their domain and SMTP-verify "
                        "(MillionVerifier); check GitHub commit emails "
                        "(append .patch to any commit URL); Prospeo on "
                        "LinkedIn. Found one? Say it in a check-in "
                        "('contact for this org is a@b.com') and tomorrow's "
                        "run drafts the memo."),
        })
    # Referral pass (audit lens 5): a referral converts to interview at
    # roughly 10x a cold application - manufacture one before applying.
    if queue_rows:
        conn_rn = swarm_db.connect()
        top_orgs = []
        for r in queue_rows[:3]:
            row = conn_rn.execute("SELECT display_name FROM orgs WHERE org_key = ?",
                                  (r["org_key"],)).fetchone()
            name = (row["display_name"] if row else r["org_key"])
            if name not in top_orgs:
                top_orgs.append(name)
        conn_rn.close()
        items.append({
            "id": f"chore:referral:{run_date}", "kind": "chore",
            "lane": "warm",
            "title": "Referral pass for tonight's top applications",
            "summary": (f"Before applying to {', '.join(top_orgs)}: search "
                        "LinkedIn's UGA alumni filter and the UGA Mentor "
                        "Program for someone inside. Ask for a 15-minute "
                        "chat, not a referral - alumni reply at ~40-45% and "
                        "about 1 in 5 chats turns into a referral, worth "
                        "~10x a cold application."),
        })
    # Standing one-time setup chores: fixed ids, so a single 'done' decision
    # greys them forever while they keep re-emitting until acted on.
    items.append({
        "id": "chore:setup:verifier", "kind": "chore", "lane": "ops",
        "title": "One-time: create an email-verifier account",
        "summary": ("MillionVerifier (~$0.45-1.50/1k, credits never expire) "
                    "or Clearout. Every outreach address gets SMTP-verified "
                    "before sending: at 10 emails/week a single bounce is a "
                    "10% bounce rate, which mailbox providers read as spam."),
    })
    items.append({
        "id": "chore:setup:ugamentor", "kind": "chore", "lane": "ops",
        "title": "One-time: join the UGA Mentor Program",
        "summary": ("3,500+ registered alumni searchable by industry, with a "
                    "built-in low-friction 'Quick Chat' ask - the cheapest "
                    "referral machine available to a UGA graduate."),
    })
    if os.path.exists(cfg_src):
        items.append({"id": f"chore:config:{run_date}", "kind": "chore",
                      "lane": "ops",
                      "title": "Review swarm vocabulary suggestions",
                      "summary": "Search-term proposals derived from your "
                                 "profile - approve into job_swarm_config.json.",
                      "dossier": "CONFIG_SUGGESTIONS.md"})
    # ---- Nightly people-finding Deep Research prompt (deliverable c) --------
    # Targets = tonight's memo orgs first, then the top direct-opening orgs -
    # exactly the orgs where a named human unlocks or strengthens outreach.
    dr_targets = [m["organization"] for m in memos] + [
        r["org"] for r in open_rows[:12]]
    dr_prompt = _jobs_people_research_prompt(dr_targets, run_date)
    dr_path = os.path.join(out_dir, "DEEP_RESEARCH_PROMPT.md")
    if os.path.exists(dr_path):   # same-day rerun: keep the earlier homework
        stamp = datetime.fromtimestamp(os.path.getmtime(dr_path)).strftime("%H%M%S")
        os.rename(dr_path, os.path.join(out_dir, f"DEEP_RESEARCH_PROMPT_{stamp}.md"))
    with open(dr_path, "w") as f:
        f.write(dr_prompt)
    # Chore card - only while today's paste is still missing (parallels the
    # quant desk). The app clears it when jobs/research/<date>_deepresearch.md
    # lands; research=True marks it as the copy-and-run-in-Gemini card.
    items.append({
        "id": f"chore:research:{run_date}", "kind": "chore", "lane": "ops",
        "title": "Run tonight's people-finding deep research",
        "summary": "Copy the prompt into Gemini Deep Research and paste the "
                   "result back on the jobs tab - it finds named people at "
                   "your top targets for tomorrow's outreach.",
        "dossier": "DEEP_RESEARCH_PROMPT.md", "research": True})

    with open(os.path.join(out_dir, "ITEMS.json"), "w") as f:
        json.dump({"run_date": run_date, "items": items}, f, indent=2)

    intel_path = os.path.join(out_dir, "INTEL.md")
    with open(intel_path, "w") as f:
        f.write("\n".join(intel_lines))

    # ---- Feasible universe: the role-discovery map (2026-07-06 blueprint).
    # Title-agnostic: profile embedding vs posting-requirement-text clusters.
    # One LLM call names the lanes; everything else is CPU arithmetic.
    # Best-effort by design - a discovery failure never costs the queue.
    try:
        import discovery_engine
        conn_u = swarm_db.connect()
        lanes = discovery_engine.build_universe(conn_u, profile_vec, _geo_ok)
        conn_u.close()
        if lanes:
            names = {}

            # Batched: one 24-lane response is too long for the slow
            # single-stream tier (2xA100 Llama ~5 tok/s blew the 300s HTTP
            # timeout on 2026-07-06 - "named 0/24"). 8 lanes/call keeps each
            # response short and the batches run concurrently.
            async def _name_batch(batch):
                raw = await query_vllm(
                    "You name job-market role families for a candidate-facing "
                    "report. Plain English, no hype, no emojis.",
                    discovery_engine.naming_prompt(batch),
                    max_tokens=1400, schema=discovery_engine.NAMING_SCHEMA)
                named = _extract_json(raw)
                if isinstance(named, list):   # bare array instead of {"lanes": ...}
                    named = {"lanes": named}
                lanes_out = (named or {}).get("lanes", [])
                if not lanes_out:
                    # Two blind nights (07-06/07) - always show WHAT came back.
                    print("WARN [Discovery] naming batch parsed to nothing; "
                          f"response head: {raw[:160]!r}")
                return lanes_out

            batches = [lanes[i:i + 8] for i in range(0, len(lanes), 8)]
            for batch_lanes in await asyncio.gather(
                    *[_name_batch(b) for b in batches]):
                for ln in batch_lanes:
                    if isinstance(ln, dict) and "cluster" in ln:
                        names[int(ln["cluster"])] = ln
            if len(names) < len(lanes):
                print(f"WARN [Discovery] naming calls named {len(names)}/"
                      f"{len(lanes)} lanes - exemplar-title fallback in use")
            universe_md, universe_q = discovery_engine.render(
                lanes, names, run_date)
            with open(os.path.join(out_dir, "FEASIBLE_UNIVERSE.md"), "w") as f:
                f.write(universe_md)
            index_lines += universe_q
    except Exception as e:      # noqa: BLE001 - report must still ship
        print(f"WARN [Discovery] universe map failed: {e}")

    index_path = os.path.join(out_dir, "REVIEW_QUEUE.md")
    with open(index_path, "w") as f:
        f.write("\n".join(index_lines))

    # Rewrite the persistent vim-editable tracker from the updated DB:
    # fresh day counts, tonight's new drafts, follow-ups due at the top.
    import tracker_engine
    conn_t = swarm_db.connect()
    tracker_engine.regenerate(conn_t)
    conn_t.close()

    print(f"[Review] Queue written -> {index_path}")
    return {
        "messages": [{"role": "assistant", "name": "ReviewCompiler",
                      "content": f"Review queue compiled at {index_path} "
                                 f"({len(memos)} dossiers, {len(watch)} watchlist)."}],
        "report_dir": out_dir,
    }


# =====================================================================
# GRAPH CONSTRUCTION
# =====================================================================

workflow = StateGraph(JobSwarmState)
workflow.add_node("profile_loader", profile_loader)
workflow.add_node("trajectory_filter", trajectory_filter)
workflow.add_node("llm_audit", llm_audit)
workflow.add_node("strategy_synthesis", strategy_synthesis)
workflow.add_node("application_forge", application_forge)
workflow.add_node("artifact_nominator", artifact_nominator)
workflow.add_node("compile_review", compile_review)

workflow.add_edge(START, "profile_loader")
workflow.add_edge("profile_loader", "trajectory_filter")
workflow.add_edge("trajectory_filter", "llm_audit")
workflow.add_edge("llm_audit", "strategy_synthesis")
workflow.add_edge("strategy_synthesis", "application_forge")
workflow.add_edge("application_forge", "artifact_nominator")
workflow.add_edge("artifact_nominator", "compile_review")
workflow.add_edge("compile_review", END)

app = workflow.compile()
