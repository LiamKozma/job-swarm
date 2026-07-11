"""
Job Swarm - Filtering Engine: high-dimensional state evaluation (GPU stage).

Quantitative phase of the filter, before any LLM sees a token:

  1. Embed every new document with sentence-transformers (CUDA when present -
     the H100s are already reserved for vLLM, so embedding rides along free).
  2. For each organization, order its accumulated corpus chronologically and
     compute the δ-shift series: cosine distance between consecutive semantic
     states. This is the discretized diffusion of the org's technical focus.
  3. Fit the K=2 Markov-Switching GMM (same estimator as the quant engine's
     return-regime model) on the δ-shift series. The high-variance component
     is the Hurdle State: rapid unstructured pivoting through terminology
     space, the signature of an org fighting a problem it can't yet solve.
  4. Run an Euler-Maruyama Monte Carlo on the fitted regime dynamics to score
     the probability the org's semantic volatility escalates over the next
     quarter (hurdle escalation probability).
  5. Score alignment: cosine similarity between the org's recent semantic
     state and the candidate profile embedding.

Orgs with thin histories (e.g. a YC startup with one description) skip the
SDE machinery and are scored on alignment + recency alone; their trajectories
densify automatically as the nightly corpus accumulates.
"""

import glob
import json
import math
import os
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as _e:
    raise ImportError("hmmlearn is required (already present in agent_env.sif).") from _e

import swarm_db

# =========================================================================
# CONSTANTS
# =========================================================================

EMBEDDING_MODEL_NAME = os.environ.get("JOB_SWARM_EMBED_MODEL", "all-MiniLM-L6-v2")
_FALLBACK_EMBED_MODEL = "all-MiniLM-L6-v2"
_CHUNK_WORDS = 350

K_REGIMES = 2
MIN_TRAJECTORY_DOCS = 5     # need K+3 shift observations for a stable GMM fit
N_GMM_RESTARTS = 5          # EM is multimodal on short series; keep best likelihood
N_MC_PATHS = 4_000
N_MC_STEPS = 90             # ~one quarter of daily semantic-drift steps
DT = 1.0 / 90.0

# δ-shift regimes are only meaningful on AUTHORED content - a single research
# voice moving through terminology space over time. Job postings and Form D
# boilerplate from the same org are cross-sectional variety, not pivoting;
# including them was what put every university and every ATS board in
# hurdle_state.
AUTHORED_SOURCES = {"nsf", "nih", "sbir", "arxiv", "yc", "usaspending",
                    "clinicaltrials", "patents", "tenk"}

# Prescore weights. 2026 red-team verdict demoted the Hurdle State: semantic
# volatility may signal distress+hiring-freeze rather than receptivity, so
# alignment dominates harder and hurdle is a tiebreaker until outreach_log
# outcomes settle the question empirically (the ε-greedy slots exist for
# exactly that). Hurdle framing stays in memos - the diagnosis angle is
# supported - it just doesn't drive the ranking.
W_ALIGNMENT, W_HURDLE, W_RECENCY = 0.65, 0.15, 0.20

_model_cache = {}


def _get_model():
    """Loads the sentence-transformer once; prefers CUDA if a GPU is visible.
    Fallback LADDER (configured -> bge-large -> MiniLM): if a newly configured
    embedder can't load (not downloaded, or the container's transformers is
    too old for its architecture), the nightly degrades to the PREVIOUS
    model, not all the way to MiniLM - the model-change guard would otherwise
    wipe and re-embed the whole corpus with the worst embedder we have."""
    if "model" not in _model_cache:
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        ladder = [EMBEDDING_MODEL_NAME, "BAAI/bge-large-en-v1.5",
                  _FALLBACK_EMBED_MODEL]
        seen = set()
        for name in [n for n in ladder if not (n in seen or seen.add(n))]:
            print(f"[Trajectory] Loading {name} on {device}")
            try:
                model = SentenceTransformer(name, device=device)
                # HARD sequence cap. bge truncated to 512 silently; Qwen3-
                # Embedding accepts 32k, and full-length docs × batch 64 on
                # the sliver of VRAM vLLM leaves free OOM'd the 2026-07-03
                # nightly mid-re-embed. Our texts are chunked to 350 words
                # anyway - nothing meaningful lives past 512 tokens.
                try:
                    model.max_seq_length = min(
                        int(getattr(model, "max_seq_length", 512) or 512), 512)
                except Exception:
                    pass
                if device == "cuda":
                    try:
                        model.half()   # halves weights+activations; cosine
                    except Exception:  # geometry is insensitive to fp16
                        pass
                _model_cache["model"] = model
                _model_cache["name"] = name
                break
            except Exception as e:
                print(f"[Trajectory] FAILED to load {name} ({e}); trying next")
        if "model" not in _model_cache:
            raise RuntimeError("no embedding model could be loaded")
    return _model_cache["model"]


def active_embed_model() -> str:
    """Name of the embedder actually in use (after any fallback)."""
    _get_model()
    return _model_cache["name"]


def embed_text(text: str) -> np.ndarray:
    """Chunked mean-pooled embedding (same scheme as edgar_engine)."""
    model = _get_model()
    words = (text or "").split()
    if not words:
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
    chunks = [" ".join(words[i: i + _CHUNK_WORDS]) for i in range(0, len(words), _CHUNK_WORDS)]
    emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return emb.mean(axis=0).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def alignment_score(state_vec: np.ndarray, profile_vec: np.ndarray,
                    facets: Optional[np.ndarray] = None) -> float:
    """
    Candidate-org alignment. The pooled profile vector sits BETWEEN the
    candidate's skill clusters (regime modeling ∩ CUDA/HPC ∩ distribution
    shift...), so pooled cosine under-scores orgs that match one facet hard.
    Blend: 0.5·pooled + 0.5·mean(top-2 facet cosines), where each facet is
    one expertise_matrix item embedded on its own (late-interaction style).
    """
    pooled = cosine_similarity(state_vec, profile_vec)
    if facets is None or len(facets) == 0:
        return pooled
    ns = float(np.linalg.norm(state_vec))
    if ns == 0.0:
        return pooled
    fn = facets / np.maximum(np.linalg.norm(facets, axis=1, keepdims=True), 1e-12)
    sims = fn @ (state_vec / ns)
    top = np.sort(sims)[-2:] if len(sims) >= 2 else sims
    return float(0.5 * pooled + 0.5 * float(np.mean(top)))


# =========================================================================
# TASTE (liked-role learning)
# =========================================================================
# The human likes a card in the dashboard, optionally saying why; the like
# lands in meta.liked_items via _absorb_decisions. Two vector kinds feed the
# signal: the liked posting's own corpus embedding (what the role IS) and a
# fresh embedding of the free-text why-note (what the human SAYS they value -
# often broader than the single posting, e.g. "on-site hardware work").
# Notes re-embed every run so they always live in the corpus's current
# embedding space and survive embed-model migrations for free.

def taste_vectors(conn) -> Optional[np.ndarray]:
    """Unit-row matrix of taste vectors, or None when nothing is liked yet."""
    liked = json.loads(swarm_db.get_meta(conn, "liked_items", "{}"))
    if not liked:
        return None
    vecs = []
    for item in liked.values():
        if item.get("url"):
            row = conn.execute(
                "SELECT embedding FROM docs WHERE url = ? "
                "AND embedding IS NOT NULL", (item["url"],)).fetchone()
            if row is not None and row["embedding"]:
                vecs.append(np.frombuffer(row["embedding"], dtype=np.float32))
    notes = [item["note"] for item in liked.values() if item.get("note")]
    if notes:
        model = _get_model()
        nv = model.encode(notes, convert_to_numpy=True, batch_size=8,
                          show_progress_bar=False)
        vecs.extend(np.asarray(v, dtype=np.float32) for v in nv)
    if not vecs:
        return None
    mat = np.stack(vecs)
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)


def taste_boost(state_vec: np.ndarray, taste_mat: Optional[np.ndarray]) -> float:
    """Score multiplier in [1.0, 1.15] from the closest taste vector.

    Max (not mean) over taste vectors: liking a quant role and a biotech role
    means BOTH styles should rank up, not their midpoint. Cosine <= 0.5 is
    neutral - in embedding space everything is mildly similar to everything,
    so only the top of the range carries signal. Capped at +15% so taste
    re-orders comparable candidates but never outshouts fit itself."""
    if taste_mat is None or len(taste_mat) == 0:
        return 1.0
    ns = float(np.linalg.norm(state_vec))
    if ns == 0.0:
        return 1.0
    sim = float(np.max(taste_mat @ (state_vec / ns)))
    return 1.0 + 0.15 * max(0.0, sim - 0.5) / 0.5


# =========================================================================
# MARKOV-SWITCHING GMM  (mirrors quant_engine.fit_markov_switching_gmm,
# vendored so job_swarm deploys standalone on /scratch)
# =========================================================================

def fit_markov_switching_gmm(series: np.ndarray, K: int = K_REGIMES) -> Optional[dict]:
    """Multi-restart EM: on 4-20 observations a single seed hands back local
    optima nightly; keep the best-likelihood fit across N_GMM_RESTARTS seeds."""
    X = series.reshape(-1, 1).astype(np.float64)
    best, best_ll = None, -np.inf
    for seed in range(N_GMM_RESTARTS):
        try:
            model = GaussianHMM(n_components=K, covariance_type="full",
                                n_iter=200, tol=1e-4, random_state=seed)
            model.fit(X)
            ll = float(model.score(X))
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = model, ll
    if best is None:
        return None
    state_seq = best.predict(X)
    return {
        "K": K,
        "regime_means": best.means_.flatten().tolist(),
        "regime_variances": best.covars_[:, 0, 0].tolist(),
        "transition_matrix": best.transmat_.tolist(),
        "current_regime": int(state_seq[-1]),
        "state_sequence": state_seq.tolist(),
    }


def _euler_maruyama_escalation(gmm: dict, rng: np.random.Generator) -> float:
    """
    Euler-Maruyama Monte Carlo on the fitted δ-shift dynamics:

        δ_{t+1} = δ_t + μ_k·Δt + σ_k·ε_t·√Δt,   k ~ Markov(transmat)

    Returns P(mean simulated future shift > current high-regime mean boundary):
    the probability the org's semantic volatility escalates into (or stays in)
    the Hurdle State over the next quarter.
    """
    means = np.array(gmm["regime_means"])
    variances = np.maximum(np.array(gmm["regime_variances"]), 1e-12)
    transmat = np.array(gmm["transition_matrix"])
    current = gmm["current_regime"]
    high = int(np.argmax(variances))

    # Pre-sample regime chains (vectorized inverse-CDF, as in quant_engine)
    cumprob = np.cumsum(transmat, axis=1)
    regimes = np.empty((N_MC_PATHS, N_MC_STEPS), dtype=np.int32)
    state = np.full(N_MC_PATHS, current, dtype=np.int32)
    for t in range(N_MC_STEPS):
        regimes[:, t] = state
        u = rng.random(N_MC_PATHS)
        state = np.argmax(u[:, None] < cumprob[state], axis=1).astype(np.int32)

    noise = rng.standard_normal((N_MC_PATHS, N_MC_STEPS))
    sqrt_dt = math.sqrt(DT)
    delta = np.full(N_MC_PATHS, float(means[current]))
    path_mean = np.zeros(N_MC_PATHS)
    for t in range(N_MC_STEPS):
        k = regimes[:, t]
        delta = delta + means[k] * DT + np.sqrt(variances[k]) * noise[:, t] * sqrt_dt
        path_mean += delta
    path_mean /= N_MC_STEPS

    boundary = float(means[high]) if means[high] > means.min() else float(means.mean())
    return float(np.mean(path_mean > boundary))


# =========================================================================
# PER-ORG TRAJECTORY METRICS
# =========================================================================

def _delta_series(authored: list) -> tuple:
    """
    (raw δ, per-√day normalized δ) for a time-ordered authored-doc series.
    Raw δ has no time units: a 0.3 jump over 2 days is violent, the same jump
    over 8 months is drift. Brownian scaling (δ/√Δt_days) makes the regime
    parameters per-day quantities, consistent with the Euler-Maruyama clock
    (Δt = 1 day). Undated gaps fall back to the series' median gap.
    """
    embs = np.stack([d["embedding"] for d in authored])
    raw = np.array([1.0 - cosine_similarity(embs[i], embs[i + 1])
                    for i in range(len(embs) - 1)])
    dates = []
    for d in authored:
        try:
            dates.append(datetime.strptime((d["date"] or "")[:10], "%Y-%m-%d"))
        except ValueError:
            dates.append(None)
    gaps = []
    for i in range(len(raw)):
        if dates[i] is not None and dates[i + 1] is not None:
            gaps.append(float(np.clip((dates[i + 1] - dates[i]).days, 1, 365)))
        else:
            gaps.append(None)
    known = [g for g in gaps if g is not None]
    fill = float(np.median(known)) if known else 30.0
    gap_arr = np.array([g if g is not None else fill for g in gaps])
    return raw, raw / np.sqrt(gap_arr)


def org_metrics(doc_series: list, profile_vec: np.ndarray,
                rng: np.random.Generator,
                facets: Optional[np.ndarray] = None) -> dict:
    """
    doc_series: time-ordered embedded docs for one org (from swarm_db.org_doc_series).
    Returns alignment, δ-shift statistics, regime classification, and prescore.
    Alignment/recency use the full series; the regime machinery uses only
    AUTHORED_SOURCES docs (see constant comment).
    """
    # Alignment: recent semantic state (last ≤3 docs mean-pooled) vs candidate
    # profile. AUTHORED docs only when any exist - the any-source state let
    # Form D legalese or posting boilerplate BE the org's identity for thin
    # orgs, and 0.65 of prescore rode on it (2026-07-05 audit F5). Orgs with
    # zero authored docs still score on what they have, with a flat penalty
    # acknowledging the state is boilerplate-derived.
    authored = [d for d in doc_series if d["source"] in AUTHORED_SOURCES]
    state_docs = authored if authored else doc_series
    recent_state = np.stack([d["embedding"] for d in state_docs[-3:]]).mean(axis=0)
    alignment = alignment_score(recent_state, profile_vec, facets)
    if not authored:
        alignment *= 0.85

    # Recency: exponential decay on days since last document (half-life 90d)
    last_date = None
    for d in reversed(doc_series):
        if d["date"]:
            try:
                last_date = datetime.strptime(d["date"][:10], "%Y-%m-%d")
                break
            except ValueError:
                continue
    if last_date is not None:
        age_days = max((datetime.now() - last_date).days, 0)
        recency = 0.5 ** (age_days / 90.0)
    else:
        recency = 0.3

    # δ-shift regimes: only over authored content, per-√day normalized
    metrics: dict = {
        "n_docs": len(doc_series),
        "n_authored": len(authored),
        "alignment": round(alignment, 6),
        "recency": round(recency, 6),
        "delta_series": None,
        "gmm": None,
        "regime": "insufficient_history",
        "hurdle_prob": 0.0,
        "escalation_prob": None,
    }

    if len(authored) >= MIN_TRAJECTORY_DOCS:
        raw_deltas, norm_deltas = _delta_series(authored)
        metrics["delta_series"] = [round(float(x), 6) for x in norm_deltas]
        gmm = fit_markov_switching_gmm(norm_deltas)
        if gmm is not None:
            high = int(np.argmax(gmm["regime_variances"]))
            in_hurdle = gmm["current_regime"] == high and \
                gmm["regime_variances"][high] > 1.5 * min(gmm["regime_variances"])
            escalation = _euler_maruyama_escalation(gmm, rng)
            metrics["gmm"] = {
                "regime_means": [round(m, 6) for m in gmm["regime_means"]],
                "regime_variances": [round(v, 8) for v in gmm["regime_variances"]],
                "transition_matrix": gmm["transition_matrix"],
                "current_regime": gmm["current_regime"],
                "high_variance_regime": high,
            }
            metrics["regime"] = "hurdle_state" if in_hurdle else "steady_state"
            metrics["escalation_prob"] = round(escalation, 4)
            metrics["hurdle_prob"] = round(
                0.6 * (1.0 if in_hurdle else 0.0) + 0.4 * escalation, 4
            )
    else:
        # Thin history: δ-shift regime unknowable; use mean RAW shift of
        # whatever authored pairs exist (raw keeps the old, capped scale)
        if len(authored) >= 2:
            raw_deltas, _ = _delta_series(authored)
            metrics["hurdle_prob"] = round(
                min(float(np.mean(raw_deltas)) * 2.0, 1.0) * 0.5, 4)
        metrics["regime"] = "insufficient_history"

    metrics["prescore"] = round(
        W_ALIGNMENT * max(alignment, 0.0)
        + W_HURDLE * metrics["hurdle_prob"]
        + W_RECENCY * recency,
        6,
    )
    return metrics


# =========================================================================
# TOP-LEVEL FILTER ENTRY POINT (called from the trajectory_filter node)
# =========================================================================

def run_filter_engine(ingest_payload_path: Optional[str], profile,
                      telemetry_dir: str, shortlist_size: int = 80) -> str:
    """
    1. Upserts newly ingested docs into the persistent corpus (SQLite, home dir)
    2. Embeds all docs still missing embeddings (GPU batch); auto-wipes and
       re-embeds the whole corpus if the embedding model changed
    3. Computes per-org trajectory metrics vs the candidate profile
    4. Writes the ranked shortlist telemetry JSON to Lustre; returns its path

    profile: dict {"embedding": vec, "facets": matrix-or-None} from
    profile_engine.load_profile (a bare ndarray is accepted for backcompat).
    """
    if isinstance(profile, np.ndarray):
        profile_vec, facets = profile, None
    else:
        profile_vec = profile["embedding"]
        facets = profile.get("facets")
    os.makedirs(telemetry_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conn = swarm_db.connect()
    rng = np.random.default_rng(42)

    # ---- 1. Corpus upsert -------------------------------------------------
    # Absorb EVERY payload not yet marked processed, not just the one passed
    # in - a one-time backfill (or two ingests between analyses) would
    # otherwise be silently lost when the next nightly overwrote "latest".
    n_new = 0
    pending = []
    if ingest_payload_path and os.path.exists(ingest_payload_path):
        raw_dir = os.path.dirname(os.path.abspath(ingest_payload_path))
        candidates = sorted(set(
            glob.glob(os.path.join(raw_dir, "ingest_*.json"))
            + [os.path.abspath(ingest_payload_path)]
        ))
        pending = [p for p in candidates
                   if not swarm_db.payload_processed(conn, os.path.basename(p))]
    for path in pending:
        with open(path) as f:
            payload = json.load(f)
        payload_docs = payload.get("docs", [])
        for doc in payload_docs:
            website = (doc.get("contacts") or {}).get("website")
            swarm_db.touch_org(conn, doc["org"], doc["source"], website)
            if swarm_db.upsert_doc(conn, doc):
                n_new += 1
        swarm_db.mark_payload_processed(conn, os.path.basename(path), len(payload_docs))
        conn.commit()
    print(f"[Trajectory] {len(pending)} payload(s) absorbed, "
          f"{n_new} new documents added to the corpus")

    # Split institution-level grant docs into PI-level orgs (idempotent)
    swarm_db.rekey_grant_docs_by_pi(conn)
    # Repair org names that leaked URLs before the HN parser stripped them
    swarm_db.rekey_url_mangled_orgs(conn)

    # ---- 1.5 Embedding-space consistency guard ------------------------------
    # Cosine geometry is only meaningful within ONE model's space. If the
    # embedder changed, wipe every stored embedding so the backlog step below
    # re-embeds the whole corpus consistently (a few minutes on an H100).
    current_model = active_embed_model()
    stored_model = swarm_db.get_meta(conn, "embed_model")
    n_embedded = conn.execute(
        "SELECT COUNT(*) FROM docs WHERE embedding IS NOT NULL").fetchone()[0]
    if stored_model is None and n_embedded:
        stored_model = _FALLBACK_EMBED_MODEL   # legacy corpus predates tracking
    if stored_model and stored_model != current_model:
        print(f"[Trajectory] EMBEDDING MODEL CHANGED ({stored_model} -> "
              f"{current_model}); wiping {n_embedded} embeddings for re-embed")
        conn.execute("UPDATE docs SET embedding = NULL")
    swarm_db.set_meta(conn, "embed_model", current_model)
    conn.commit()

    # ---- 2. Embed the backlog ----------------------------------------------
    # OOM-resilient: the embedder shares GPUs with a vLLM that pre-reserves
    # most of the VRAM. Batch starts at 16, not 64: with the 8B embedder a
    # failed 64-doc attempt leaves GiB of reserved-but-unallocated pages in
    # this process's caching allocator, and run 46807108 showed that squeeze
    # OOM-crashing the vLLM WORKER (the co-tenant) even though the halving
    # retry saved the embed pass itself. On CUDA OOM, halve and retry
    # (16 -> 8 -> 4) instead of killing the nightly.
    backlog = swarm_db.docs_missing_embeddings(conn)
    if backlog:
        model = _get_model()
        texts = [r["text"][:6000] for r in backlog]
        batch = 16
        while True:
            try:
                vectors = model.encode(texts, convert_to_numpy=True,
                                       batch_size=batch,
                                       show_progress_bar=False)
                break
            except Exception as e:
                if "out of memory" not in str(e).lower() or batch <= 4:
                    raise
                batch //= 2
                print(f"[Trajectory] CUDA OOM at batch {batch * 2} - "
                      f"retrying at {batch}")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        for row, vec in zip(backlog, vectors):
            swarm_db.store_embedding(conn, row["doc_id"], vec)
        conn.commit()
        # Hand the batch-activation pages back to the driver so the resident
        # footprint drops to ~weights for the rest of the DAG - the vLLM
        # co-tenant needs that slack for its own transient workspace.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"[Trajectory] {len(backlog)} documents embedded")

    # Removal channel: match tonight's fresh postings against recently-dead
    # ones - a reappearing posting is a role the org failed to fill.
    swarm_db.detect_reposts(conn)

    # ---- 3. Per-org metrics -------------------------------------------------
    org_rows = swarm_db.active_org_keys(conn)
    scored = []
    for org in org_rows:
        series = swarm_db.org_doc_series(conn, org["org_key"])
        if not series:
            continue
        m = org_metrics(series, profile_vec, rng, facets)
        # Merge contact channels published across this org's documents.
        # Evidence carries text EXCERPTS - the auditor and memo writer ground
        # their claims in these, not in titles alone.
        contacts: dict = {}
        evidence = []
        for d in series[-6:]:
            contacts.update({k: v for k, v in d["contacts"].items() if v})
            evidence.append({"date": d["date"], "title": d["title"],
                             "url": d["url"], "source": d["source"],
                             "excerpt": (d.get("excerpt") or "")[:700]})
        scored.append({
            "org_key": org["org_key"],
            "display_name": org["display_name"],
            "website": org.get("website"),
            "contacts": contacts,
            "evidence": evidence,
            **m,
        })
        swarm_db.update_org_scores(
            conn, org["org_key"], m["prescore"], m["alignment"],
            m["hurdle_prob"], m["regime"], "shortlisted" if m["prescore"] > 0.35 else "new",
        )
    conn.commit()

    scored.sort(key=lambda x: x["prescore"], reverse=True)
    shortlist = scored[:shortlist_size]

    # ---- 4. Ghost roles + hiring acceleration ------------------------------
    # Annotates shortlist entries with predicted_role / hiring_accel and
    # returns the role taxonomy for the review queue's comp observatory.
    import ghost_engine
    taxonomy = ghost_engine.annotate_shortlist(conn, shortlist)
    conn.close()

    telemetry = {
        "timestamp": timestamp,
        "embedding_model": active_embed_model(),
        "n_orgs_evaluated": len(scored),
        "n_new_docs": n_new,
        "weights": {"alignment": W_ALIGNMENT, "hurdle": W_HURDLE, "recency": W_RECENCY},
        "role_taxonomy": (taxonomy or {}).get("archetypes", []),
        "shortlist": shortlist,
    }
    path = os.path.join(telemetry_dir, f"shortlist_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"[Trajectory] {len(scored)} orgs evaluated -> top {len(shortlist)} -> {path}")
    return path
