"""
discovery_engine.py - the feasible-universe map (role discovery).

The owner's design requirement (2026-07-06 deep-research blueprint): he does
NOT pre-pick a target job title. The system takes the profile embedding and
maps the FULL universe of role families he could plausibly win, by matching
against posting REQUIREMENT TEXT, never titles - titles are inflated and
inconsistent; requirement text is what the job actually is.

Method (deliberately transparent, no black boxes):
  1. Embeddings of every live posting seen in the last 60 days (already
     computed nightly for alignment scoring - reused, zero extra GPU).
  2. KMeans over the posting embeddings -> role-family clusters ("lanes"),
     same pattern ghost_engine.build_role_taxonomy uses for archetypes.
  3. Per lane: fit (cosine of the profile against the lane centroid),
     volume (postings, fresh-in-7d count), early-career openness (share of
     postings without a senior/staff/principal title gate), disclosed-comp
     median, and the freshest in-geography live examples.
  4. Lane rank = fit x (0.5 + 0.5 x openness) x log1p(volume): a lane you
     fit that hires juniors at volume outranks a perfect-fit lane with
     three senior seats. The formula is printed with the table so the
     human can argue with it.

The nightly LLM adds ONE call on top (js_graph does the call - this module
stays import-light): plain-English lane names plus a note on families the
candidate may not have considered. Everything else is CPU arithmetic.

Outcome coupling: lanes are named by their stable content hash, so when
funnel_events accumulates interviews per lane the day-90 reallocation
(jobs/PREREGISTRATION.md) can be computed against these same lane keys.
"""

from __future__ import annotations

import html
import re

import numpy as np

import swarm_db
from ghost_engine import (POSTING_SOURCES, _normalize, _role_title,
                          _salary_midpoint)

try:
    from sklearn.cluster import KMeans
except ImportError:      # pragma: no cover - agent_env always has sklearn
    KMeans = None

MIN_POSTINGS = 40
WINDOW_DAYS = 60

_SENIOR_RE = re.compile(
    r"\b(senior|staff|principal|lead|director|head of|vp|sr\.?|manager)\b",
    re.I)
_EARLY_RE = re.compile(
    r"\b(graduate|junior|entry[- ]level|new[- ]grad|early[- ]career|"
    r"associate|campus|intern(ship)?)\b", re.I)


def build_universe(conn, profile_vec, geo_ok) -> list:
    """-> ranked lanes [{cluster, rank_score, fit, openness, early_share,
    n_postings, n_fresh7, median_salary, rep_titles, examples}] or []."""
    if KMeans is None:
        return []
    postings = swarm_db.recent_docs_by_source(
        conn, POSTING_SOURCES + ("usajobs",), days=WINDOW_DAYS)
    if len(postings) < MIN_POSTINGS:
        print(f"[Discovery] only {len(postings)} postings - universe map "
              f"needs {MIN_POSTINGS}; skipping")
        return []

    X = _normalize(np.stack([p["embedding"] for p in postings]))
    pv = np.asarray(profile_vec, dtype=np.float32)
    pv = pv / (np.linalg.norm(pv) or 1.0)
    k = int(np.clip(len(postings) // 15, 6, 24))
    km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(X)
    centroids = _normalize(km.cluster_centers_)

    lanes = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        members = [postings[i] for i in idx]
        titles = [(m["title"] or "") for m in members]
        n = len(members)
        n_senior = sum(1 for t in titles if _SENIOR_RE.search(t))
        n_early = sum(1 for t in titles if _EARLY_RE.search(t))
        openness = 1.0 - n_senior / n
        fit = float(np.dot(centroids[c], pv))
        dists = np.linalg.norm(X[idx] - km.cluster_centers_[c], axis=1)
        rep = [members[i] for i in np.argsort(dists)[:3]]
        salaries = [s for s in (_salary_midpoint(m["extra"].get("salary"))
                                for m in members) if s]
        fresh_geo = sorted(
            (m for m in members
             if m["days_open"] <= 7 and geo_ok(m["extra"].get("location"))
             and not _SENIOR_RE.search(m["title"] or "")),
            key=lambda m: m["days_open"])
        lanes.append({
            "cluster": c,
            "fit": round(fit, 3),
            "openness": round(openness, 2),
            "early_share": round(n_early / n, 2),
            "n_postings": n,
            "n_fresh7": sum(1 for m in members if m["days_open"] <= 7),
            "median_salary": (float(np.median(salaries)) if salaries else None),
            "rep_titles": [html.unescape(_role_title(m)) for m in rep],
            "examples": [{"title": html.unescape((m["title"] or ""))[:70],
                          "url": m["url"],
                          "days_open": m["days_open"],
                          "loc": m["extra"].get("location") or "-"}
                         for m in fresh_geo[:3]],
            "rank_score": round(
                fit * (0.5 + 0.5 * openness) * float(np.log1p(n)), 3),
        })
    lanes.sort(key=lambda ln: ln["rank_score"], reverse=True)
    print(f"[Discovery] universe map: {k} lanes from {len(postings)} postings")
    return lanes


def naming_prompt(lanes: list) -> str:
    """User prompt for the single lane-naming LLM call (js_graph owns the
    call). The system names each lane in plain English and flags families
    the candidate likely has not considered."""
    rows = "\n".join(
        f"- cluster {ln['cluster']}: titles {ln['rep_titles']}, "
        f"{ln['n_postings']} postings, fit {ln['fit']}, "
        f"early-career share {ln['early_share']}"
        for ln in lanes)
    return (
        "Below are role-family clusters discovered by embedding live job "
        "postings (requirement text, not titles) and clustering them. For "
        "each cluster give a plain-English family NAME (2-5 words, the name "
        "a recruiter would use) and one NOTE (ONE sentence, 25 words max): "
        "what this family actually does day to day, and - only where true - "
        "flag 'adjacent family worth a look' when a statistics/HPC candidate "
        "might not have considered it.\n"
        # Guided decoding is best-effort (older vLLM 400s it and the call
        # retries unguided), so the prompt itself must demand JSON - without
        # this line the unguided retry answers in prose and parses to
        # nothing (0/24 lanes named, 2026-07-06 and 07-07).
        'Output ONLY JSON, no prose, exactly this shape: {"lanes": '
        '[{"cluster": <int>, "name": "...", "note": "..."}]} with one '
        "entry per cluster listed below.\n\n" + rows)


NAMING_SCHEMA = {
    "type": "object",
    "properties": {
        "lanes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster": {"type": "integer"},
                    "name": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["cluster", "name"],
            },
        },
    },
    "required": ["lanes"],
}


def _fmt_money(v) -> str:
    return f"~${v / 1000:,.0f}K" if v else "-"


def _fallback_name(ln: dict) -> str:
    """Exemplar-title fallback when the naming call fails: clip location
    suffixes and cap length so the table stays readable."""
    parts = []
    for t in ln["rep_titles"][:2]:
        t = t.split(" — ")[0].strip()[:48]
        if t and t not in parts:
            parts.append(t)
    return " / ".join(parts) or f"cluster {ln['cluster']}"


def render(lanes: list, names: dict, run_date: str) -> tuple:
    """-> (feasible_universe_md, queue_lines). names: {cluster: {name, note}}."""
    if not lanes:
        return "", []
    md = [
        f"# Feasible universe - {run_date}",
        "",
        "Role families discovered from live posting REQUIREMENT TEXT (not "
        "titles), ranked by `fit x (0.5 + 0.5 x openness) x log1p(volume)`. "
        "Fit is the profile-vs-lane-centroid cosine; openness is the share "
        "of postings without a senior/staff/principal gate. Argue with the "
        "formula, not the vibes - it is printed so you can.",
        "",
        "| # | Family | Fit | Open to juniors | Postings | Fresh 7d | Median comp |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, ln in enumerate(lanes, 1):
        nm = names.get(ln["cluster"], {})
        md.append(
            f"| {i} | {nm.get('name') or _fallback_name(ln)} "
            f"| {ln['fit']} | {int(ln['openness'] * 100)}% "
            f"(early {int(ln['early_share'] * 100)}%) | {ln['n_postings']} "
            f"| {ln['n_fresh7']} | {_fmt_money(ln['median_salary'])} |")
    md.append("")
    for i, ln in enumerate(lanes, 1):
        nm = names.get(ln["cluster"], {})
        md += [f"## {i}. {nm.get('name') or _fallback_name(ln)}",
               ""]
        if nm.get("note"):
            md += [nm["note"], ""]
        md.append(f"Representative titles: {'; '.join(ln['rep_titles'])}")
        if ln["examples"]:
            md.append("Freshest in-geography, junior-open examples:")
            md += [f"- [{e['title']}]({e['url']}) - {e['loc']} - "
                   f"open {e['days_open']:.0f}d" for e in ln["examples"]]
        else:
            md.append("No fresh in-geography junior-open posting this week - "
                      "a lane to watch, not to force.")
        md.append("")
    md += [
        "As interviews accrue in funnel_events, the day-90 rule "
        "(jobs/PREREGISTRATION.md) reallocates effort toward the lanes "
        "that actually convert - this map proposes, outcomes dispose.",
        "",
    ]

    top = lanes[:5]
    queue = [
        "## Feasible universe - where your profile actually lands",
        "",
        "Top role families by fit x junior-openness x live volume "
        "(full map + fresh examples: "
        "[FEASIBLE_UNIVERSE.md](FEASIBLE_UNIVERSE.md)):",
        "",
    ]
    for i, ln in enumerate(top, 1):
        nm = names.get(ln["cluster"], {})
        queue.append(
            f"{i}. **{nm.get('name') or _fallback_name(ln)}** - "
            f"fit {ln['fit']}, {ln['n_fresh7']} fresh this week, "
            f"{int(ln['openness'] * 100)}% junior-open, "
            f"{_fmt_money(ln['median_salary'])}")
    queue.append("")
    return "\n".join(md), queue
