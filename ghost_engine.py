"""
Job Swarm - Ghost Role Engine: predicting jobs that don't exist yet.

Three statistical signals combine to name a role before it is posted:

  1. ROLE ARCHETYPES - every live posting the swarm has ever collected is a
     labeled point in embedding space ("this semantic region hires this
     title at this salary"). K-means over the posting corpus yields a role
     taxonomy grounded in the actual market, refreshed nightly.

  2. SEMANTIC PROJECTION - for a target org with NO relevant posting, its
     recent semantic state (mean of its latest document embeddings - the same
     state the δ-shift GMM regime-classifies) is projected onto the archetype
     centroids. The nearest archetype is the role the org's technical output
     says it needs. An org in the Hurdle State whose trajectory is drifting
     toward the "ML Research Scientist" centroid is about to need one.

  3. HIRING ACCELERATION - nightly board_snapshots give each org a posting
     count time series. An OLS slope (postings/day) estimates hiring
     acceleration: a company adding engineers fast while posting zero
     statistics roles has a gap between its growth and its stated needs -
     the precise moment a founder-directed memo can define the role.

Output: `predicted_role` annotations on shortlist entries, consumed by the
memo synthesizer (the memo proposes the role) and the review queue's
"Roles that don't exist yet" section.
"""

import re

import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError as _e:
    raise ImportError("scikit-learn is required (already present in agent_env.sif).") from _e

import swarm_db
from ats_engine import salary_values

POSTING_SOURCES = ("ats_jobs", "remoteok", "hn_hiring")
MIN_POSTINGS_FOR_TAXONOMY = 40


def _salary_midpoint(salary_str):
    """'$150,000 - $200,000' -> 175000.0; handles '$293K - $385K' and bare
    '150k-200k' via the shared parser. None if unparseable."""
    nums = salary_values(salary_str)[:2] if salary_str else []
    return float(np.mean(nums)) if nums else None


def _role_title(posting: dict) -> str:
    """Human-usable role title for archetype labels. HN posts are
    'Company | Role | Location | ...'; ATS titles are 'Role - Location'."""
    t = posting.get("title") or ""
    if posting.get("source") == "hn_hiring" and "|" in t:
        parts = [x.strip() for x in t.split("|")]
        if len(parts) >= 2 and parts[1]:
            return f"{parts[1]} ({parts[0]})"[:60]
    return t.split("-")[0].strip()[:60]


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)


# =========================================================================
# 1. ROLE ARCHETYPE TAXONOMY
# =========================================================================

def build_role_taxonomy(conn) -> dict:
    """
    Clusters the accumulated posting corpus into role archetypes.
    Returns {"centroids": (k, d) array, "archetypes": [ {titles, median_salary,
    n_postings, example_urls} ]} or {} if the corpus is still too thin.
    """
    postings = swarm_db.recent_docs_by_source(conn, POSTING_SOURCES, days=60)
    if len(postings) < MIN_POSTINGS_FOR_TAXONOMY:
        print(f"[Ghost] only {len(postings)} postings - taxonomy needs "
              f"{MIN_POSTINGS_FOR_TAXONOMY}; skipping")
        return {}

    X = _normalize(np.stack([p["embedding"] for p in postings]))
    k = int(np.clip(len(postings) // 15, 6, 24))
    km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(X)

    archetypes = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        # Representative titles: the 3 members nearest the centroid
        dists = np.linalg.norm(X[idx] - km.cluster_centers_[c], axis=1)
        rep = idx[np.argsort(dists)[:3]]
        salaries = [s for s in (_salary_midpoint(postings[i]["extra"].get("salary"))
                                for i in idx) if s]
        archetypes.append({
            "cluster": c,
            "titles": [_role_title(postings[i]) for i in rep],
            "n_postings": int(len(idx)),
            "median_salary": float(np.median(salaries)) if salaries else None,
            "example_urls": [postings[i]["url"] for i in rep[:2]],
        })

    print(f"[Ghost] role taxonomy: {k} archetypes from {len(postings)} postings")
    return {"centroids": km.cluster_centers_, "archetypes": archetypes}


# =========================================================================
# 2 + 3. PER-ORG PREDICTION
# =========================================================================

def hawkes_hiring_burst(conn, org_key: str, beta: float = 1.0 / 7.0,
                        window_days: int = 120):
    """
    Self-exciting (Hawkes) model of posting arrivals with exponential kernel:

        λ(t) = μ + n·β Σ_{t_i<t} exp(-β(t - t_i)),   fixed β = 1/7 day⁻¹

    Hiring begets hiring - one posting means budget cleared, and a burst
    predicts more. The Theil-Sen slope on board snapshots sees only net
    headcount; the Hawkes intensity sees the ARRIVAL process, so it fires on
    a burst even when as many roles close as open. Fit by EM (μ = background
    rate, n = branching ratio ∈ [0,1)), pure numpy, ~50 iterations on the
    handful of events an org produces. Returns None below 6 events.

      burst_ratio = λ(now)/μ - how far above its own baseline the org's
      hiring process is running right now. >2 means an active burst.
    """
    ages = swarm_db.posting_arrival_ages(conn, org_key, window_days)
    if len(ages) < 6:
        return None
    t = np.sort(window_days - np.array(ages, dtype=np.float64))  # forward time
    T = float(window_days)
    n_ev = len(t)

    # Pairwise kernel matrix K[i, j] = β·exp(-β(t_i - t_j)) for j < i
    diff = t[:, None] - t[None, :]
    K = np.where(diff > 0, beta * np.exp(-beta * np.clip(diff, 0, None)), 0.0)
    edge = 1.0 - np.exp(-beta * (T - t))          # kernel mass inside window

    mu, br = n_ev / T * 0.7, 0.3                  # init: mostly background
    for _ in range(50):
        lam = mu + br * K.sum(axis=1)
        rho = mu / np.maximum(lam, 1e-12)         # P(event i is background)
        mu = float(np.sum(rho) / T)
        denom = float(np.sum(edge))
        br = float(np.clip(np.sum(1.0 - rho) / max(denom, 1e-9), 0.0, 0.95))
    lam_now = mu + br * float(np.sum(beta * np.exp(-beta * (T - t))))
    return {
        "n_events": n_ev,
        "baseline_per_day": round(mu, 4),
        "branching_ratio": round(br, 3),
        "intensity_now": round(lam_now, 4),
        "burst_ratio": round(lam_now / max(mu, 1e-9), 2),
    }


def hiring_acceleration(conn, org_key: str):
    """Theil-Sen slope of total postings/day over the last 28 days; None if
    <5 snapshots. OLS on a 3-point count series was pure noise; the median of
    pairwise slopes is robust to the single-day posting bursts ATS boards
    actually produce."""
    series = swarm_db.board_snapshot_series(conn, org_key, days=28)
    if len(series) < 5:
        return None
    t = -np.array([s[0] for s in series])     # forward time, days
    y = np.array([s[1] for s in series], dtype=np.float64)
    slopes = [(y[j] - y[i]) / (t[j] - t[i])
              for i in range(len(t)) for j in range(i + 1, len(t))
              if t[j] != t[i]]
    slope = float(np.median(slopes)) if slopes else 0.0
    return {"slope_per_day": round(slope, 3),
            "latest_total": int(y[-1]),
            "latest_relevant": int(series[-1][2]),
            "n_snapshots": len(series)}


def annotate_shortlist(conn, shortlist: list) -> dict:
    """
    Mutates shortlist entries in place, adding:
      hiring_accel   - board-growth estimate (if the org has a tracked board)
      predicted_role - nearest role archetype when the org has NO live
                       relevant posting (the ghost role)
    Returns the taxonomy dict for the review queue's comp observatory.
    """
    taxonomy = build_role_taxonomy(conn)
    centroids = _normalize(taxonomy["centroids"]) if taxonomy else None

    # Orgs that already show a relevant posting - no prediction needed there
    posting_org_keys = {p["org_key"] for p in
                        swarm_db.recent_docs_by_source(conn, POSTING_SOURCES, days=21)}

    for org in shortlist:
        accel = hiring_acceleration(conn, org["org_key"])
        if accel:
            org["hiring_accel"] = accel
        hawkes = hawkes_hiring_burst(conn, org["org_key"])
        if hawkes:
            org["hiring_hawkes"] = hawkes

        if centroids is None or org["org_key"] in posting_org_keys:
            continue
        series = swarm_db.org_doc_series(conn, org["org_key"])
        if not series:
            continue
        state = np.stack([d["embedding"] for d in series[-3:]]).mean(axis=0)
        norm = np.linalg.norm(state)
        if norm == 0:
            continue
        sims = centroids @ (state / norm)
        best = int(np.argmax(sims))
        arch = taxonomy["archetypes"][best]
        org["predicted_role"] = {
            "similarity": round(float(sims[best]), 4),
            "archetype_titles": arch["titles"],
            "market_median_salary": arch["median_salary"],
            "based_on_postings": arch["n_postings"],
        }

    n_pred = sum(1 for o in shortlist if o.get("predicted_role"))
    n_accel = sum(1 for o in shortlist if o.get("hiring_accel"))
    print(f"[Ghost] {n_pred} ghost roles predicted, {n_accel} orgs with hiring telemetry")
    return taxonomy


# =========================================================================
# COMP OBSERVATORY - market statistics from the accumulated posting corpus
# =========================================================================

def comp_observatory(conn) -> dict:
    """Salary distribution across all relevant postings with disclosed comp."""
    postings = swarm_db.recent_docs_by_source(conn, POSTING_SOURCES, days=60)
    rows = []
    for p in postings:
        mid = _salary_midpoint(p["extra"].get("salary"))
        if mid:
            rows.append((mid, p["title"], p["url"], p["extra"].get("board") or p["source"]))
    if not rows:
        return {}
    salaries = np.array([r[0] for r in rows])
    rows.sort(key=lambda r: r[0], reverse=True)
    return {
        "n_disclosed": len(rows),
        "quartiles": [round(float(q)) for q in np.percentile(salaries, [25, 50, 75, 90])],
        "top_postings": [{"salary": round(r[0]), "title": r[1][:80], "url": r[2],
                          "board": r[3]} for r in rows[:10]],
    }
