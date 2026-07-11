"""
Job Swarm - Candidate Profile Engine.

Reads everything the candidate drops into the profile/ directory (resume,
project write-ups, thesis abstract - plain .txt or .md), then produces two
artefacts consumed by the rest of the swarm:

  1. profile.json  - an LLM-structured expertise matrix: skills, domains,
     proof points, and the links (GitHub, site) that outreach memos may cite.
  2. profile.npy   - the sentence-transformer embedding of the full corpus,
     the fixed point every org's semantic state is scored against.

Both are cached on scratch and rebuilt only when the profile files change
(SHA-1 over contents), so this costs nothing on ordinary nightly runs.
"""

import glob
import hashlib
import json
import os
import re

import numpy as np

# Write-ups carry a self-audit appendix (claims tables, lexicon scans, char
# counts) below a "Self-audit" / "Claims audit" heading - QA scaffolding meant
# for the human author, NOT profile content. Embedding or persona-extracting it
# dilutes every facet with citations like "README.md:3-10" and burns the budget
# on non-signal. Strip from the first such heading down; if a file has no audit
# heading, the whole file is kept (backward compatible).
_AUDIT_HEADING = re.compile(r'(?im)^\s*#{1,6}\s+(?:self-?audit|claims audit)\b')


def _strip_audit(text: str) -> str:
    return _AUDIT_HEADING.split(text, maxsplit=1)[0].rstrip()

PROFILE_DIR = os.environ.get(
    "JOB_SWARM_PROFILE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile"),
)

_CONFIG_ADVISOR_PROMPT = (
    "You tune the search vocabulary of an automated job-discovery pipeline to a "
    "candidate's actual expertise. You receive the candidate's structured profile and "
    "the pipeline's current search vocabulary. Propose the vocabulary that would "
    "surface the organizations most likely to need THIS candidate.\n\n"
    "Per-key rules:\n"
    "- grant_keywords: 5-8 phrases as they appear in NSF/NIH/federal grant abstracts "
    "(multi-word phrases fine; quoting is automatic).\n"
    "- arxiv_queries: 4-6 arXiv API queries in cat:XX.YY AND abs:\"phrase\" syntax; "
    "categories among cs.LG, stat.ML, stat.ME, q-fin.ST, q-bio.QM, math.ST unless the "
    "profile clearly points elsewhere.\n"
    "- usajobs_keywords: 4-6 federal job-search terms in OPM vocabulary "
    "('statistician', 'operations research'), not startup vocabulary.\n"
    "- hn_relevance_terms: 10-16 lowercase tokens/short phrases that would appear in "
    "a relevant startup hiring post.\n"
    "- ats_title_terms: lowercase SUBSTRINGS matched against job-posting titles; "
    "short and high-recall ('statist' catches statistician/biostatistics).\n"
    "- yc_tag_terms: 6-10 startup industry/tag words.\n"
    "Keep whatever in the current vocabulary still fits, drop what doesn't, add what "
    "is missing. Be conservative: every extra term is ingestion noise.\n\n"
    "Output ONLY one strictly valid JSON object - double quotes, no markdown fences - "
    "with exactly these keys: grant_keywords, arxiv_queries, usajobs_keywords, "
    "hn_relevance_terms, ats_title_terms, yc_tag_terms, rationale (list of one-line "
    "strings explaining each notable change)."
)


async def _advise_config(query_vllm, cache_dir: str, profile: dict) -> None:
    """LLM-proposed search vocabulary for THIS profile vs the active config.
    Suggest-only by design: a bad search term silently poisons ingestion, so a
    human pastes the block into job_swarm_config.json rather than the swarm
    rewriting its own eyes. Regenerated whenever the profile rebuilds."""
    from ingest_engines import load_config
    from ats_engine import DEFAULT_TITLE_TERMS
    cfg = load_config()
    vocab = {k: cfg.get(k) for k in
             ("grant_keywords", "arxiv_queries", "usajobs_keywords",
              "hn_relevance_terms", "yc_tag_terms")}
    vocab["ats_title_terms"] = cfg.get("ats_title_terms", DEFAULT_TITLE_TERMS)
    user = json.dumps({
        "candidate_profile": {k: profile.get(k) for k in
                              ("headline", "expertise_matrix", "domains")},
        "current_vocabulary": vocab,
    }, indent=1)
    response = await query_vllm(_CONFIG_ADVISOR_PROMPT, user)
    try:
        proposed = json.loads(response[response.find("{"): response.rfind("}") + 1])
    except Exception:
        print(f"[Profile] config advisor parse failure - skipped: {response[:200]}")
        return
    rationale = proposed.pop("rationale", [])
    lines = [
        "# Config suggestions - search vocabulary tuned to the new profile",
        "",
        "The profile documents changed, so the LLM re-derived the ingestion "
        "search vocabulary from your actual expertise matrix. Review each key, "
        "edit to taste, then paste the ones you accept into "
        "`job_swarm_config.json` (same directory as the swarm code). Nothing "
        "is applied automatically.",
        "",
        "## Why these changes",
        "",
        *[f"- {r}" for r in (rationale or ["(no rationale returned)"])],
        "",
        "## Proposed vocabulary (paste keys you accept into job_swarm_config.json)",
        "",
        "```json",
        json.dumps(proposed, indent=2),
        "```",
        "",
        "## Current vocabulary (for comparison)",
        "",
        "```json",
        json.dumps(vocab, indent=2),
        "```",
    ]
    with open(os.path.join(cache_dir, "CONFIG_SUGGESTIONS.md"), "w") as f:
        f.write("\n".join(lines))
    print("[Profile] CONFIG_SUGGESTIONS.md regenerated (profile changed)")


_PROFILE_SYSTEM_PROMPT = (
    "You are an elite technical talent analyst. You will receive a candidate's raw "
    "resume and project descriptions. Produce a strictly valid JSON object with keys:\n"
    "  'headline'         : one-line positioning statement (string)\n"
    "  'expertise_matrix' : list of 6-12 specific technical capabilities, most "
    "differentiated first (e.g. 'Markov-Switching GMM regime modeling', "
    "'Numba CUDA kernel engineering', 'SLURM/HPC orchestration')\n"
    "  'domains'          : list of application domains (strings)\n"
    "  'proof_points'     : list of {'claim': ..., 'evidence': ...} objects grounded "
    "ONLY in what the documents actually state - never invent accomplishments\n"
    "  'links'            : {'github': url-or-null, 'website': url-or-null} extracted "
    "from the documents\n"
    "Output ONLY the JSON object."
)


def _read_profile_files() -> list:
    """[(basename, text)] for each profile document, README skipped."""
    paths = sorted(
        glob.glob(os.path.join(PROFILE_DIR, "*.txt"))
        + glob.glob(os.path.join(PROFILE_DIR, "*.md"))
    )
    out = []
    for p in paths:
        if os.path.basename(p).upper().startswith("README"):
            continue
        with open(p, errors="replace") as f:
            out.append((os.path.basename(p), _strip_audit(f.read())))
    return out


def _read_profile_corpus() -> str:
    return "\n\n".join(f"===== {name} =====\n{text}"
                       for name, text in _read_profile_files())


def _corpus_hash(corpus: str) -> str:
    return hashlib.sha1(corpus.encode()).hexdigest()


async def build_candidate_profile(query_vllm, cache_dir: str) -> dict:
    """
    Returns {'profile_path', 'embedding_path', 'profile', 'rebuilt': bool}.
    query_vllm: async (system_prompt, user_prompt) -> str, injected by the graph
    so this module stays importable without a running LLM.
    """
    os.makedirs(cache_dir, exist_ok=True)
    corpus = _read_profile_corpus()
    if not corpus.strip():
        raise FileNotFoundError(
            f"No profile documents found in {PROFILE_DIR}. Drop your resume and "
            "project write-ups there as .txt or .md files (export PDF -> text first)."
        )

    import trajectory_engine

    digest = _corpus_hash(corpus)
    profile_path = os.path.join(cache_dir, "profile.json")
    embedding_path = os.path.join(cache_dir, "profile.npy")
    facets_path = os.path.join(cache_dir, "profile_facets.npy")
    embed_model = trajectory_engine.active_embed_model()

    # Cache hit: same corpus hash, same embedding model, all artefacts exist.
    # (The model check matters: a stale profile.npy from a different embedder
    # would silently break every cosine in the pipeline.)
    if os.path.exists(profile_path) and os.path.exists(embedding_path):
        with open(profile_path) as f:
            cached = json.load(f)
        if (cached.get("_corpus_sha1") == digest
                and cached.get("_embed_model") == embed_model
                and os.path.exists(facets_path)):
            return {"profile_path": profile_path, "embedding_path": embedding_path,
                    "profile": cached, "rebuilt": False}

    # ---- LLM structuring -------------------------------------------------
    response = await query_vllm(_PROFILE_SYSTEM_PROMPT, corpus[:24_000])
    structuring_ok = True
    try:
        profile = json.loads(response[response.find("{"): response.rfind("}") + 1])
    except Exception:
        structuring_ok = False
        profile = {
            "headline": "Computational statistician - profile structuring failed, using raw corpus",
            "expertise_matrix": [], "domains": [], "proof_points": [],
            "links": {}, "_llm_error": response[:500],
        }
    # Guard: a vLLM outage returns an error STRING that parses to no JSON, or a
    # valid-JSON-but-empty extraction. Either way, do NOT stamp the real corpus
    # hash onto a degraded profile - if we did, tonight's blip would cache-hit
    # forever (the resume never changes) and silently break every cosine in the
    # pipeline until the file is edited. Serve the degraded profile for THIS run
    # but poison the cache key so the next nightly re-runs structuring.
    if not structuring_ok or not (profile.get("expertise_matrix") or []):
        structuring_ok = False
    profile["_corpus_sha1"] = digest if structuring_ok else f"RETRY-{digest}"
    profile["_embed_model"] = embed_model

    # ---- Embedding --------------------------------------------------------
    # Embed the expertise matrix + corpus, weighting the distilled skills:
    # the profile vector should live where the candidate's edge lives.
    skills_text = ". ".join(profile.get("expertise_matrix") or [])
    corpus_vec = trajectory_engine.embed_text(corpus[:30_000])
    if skills_text:
        skills_vec = trajectory_engine.embed_text(skills_text)
        vec = 0.6 * skills_vec + 0.4 * corpus_vec
    else:
        vec = corpus_vec

    # ---- Facets: one embedding per expertise item AND per project doc -----
    # Alignment scoring blends pooled cosine with the best-matching facets, so
    # an org that hits one skill (or one project) hard isn't averaged into
    # mediocrity by the rest of the profile. Two facet families:
    #   - expertise_matrix items: the distilled skill vocabulary (short, dense)
    #   - per-file project write-ups: the full context of one project, embedded
    #     on its own so "your CUDA-kernel project matches their kernel team"
    #     surfaces even when the pooled profile leans elsewhere. Each file is
    #     embedded whole (embed_text chunks + mean-pools internally); the
    #     resume itself is skipped as a facet (it IS the pooled vector).
    facet_list = []
    facet_labels = []
    for s in (profile.get("expertise_matrix") or []):
        if not s:
            continue
        facet_list.append(trajectory_engine.embed_text(s))
        facet_labels.append(str(s))
    for name, text in _read_profile_files():
        low = name.lower()
        if not text.strip() or "resume" in low or "cv" in low:
            continue
        facet_list.append(trajectory_engine.embed_text(text[:12_000]))
        facet_labels.append(f"project: {os.path.splitext(name)[0]}")
    # Stored parallel to profile_facets.npy so a winning PROJECT facet renders a
    # real 'why you fit' label instead of being silently dropped (the facet
    # array is longer than expertise_matrix, so an expertise-only names list
    # loses exactly the project matches facet scoring exists to surface).
    profile["_facet_labels"] = facet_labels
    if facet_list:
        facet_vecs = np.stack(facet_list)
    else:
        facet_vecs = np.zeros((0, len(vec)), dtype=np.float32)
    np.save(facets_path, facet_vecs.astype(np.float32))

    np.save(embedding_path, vec.astype(np.float32))
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    # Resume audit retired 2026-07-08: the candidate reviews the resume with a
    # hosted frontier model instead; a 70B local pass added latency for advice
    # that was never load-bearing to matching.

    # ---- Config advisor - propose search vocabulary for the new profile ----
    try:
        await _advise_config(query_vllm, cache_dir, profile)
    except Exception as e:
        print(f"[Profile] config advisor failed (non-fatal): {e}")

    return {"profile_path": profile_path, "embedding_path": embedding_path,
            "profile": profile, "rebuilt": True}


def load_profile(cache_dir: str) -> dict:
    """Loads cached artefacts without rebuilding (raises if absent)."""
    profile_path = os.path.join(cache_dir, "profile.json")
    embedding_path = os.path.join(cache_dir, "profile.npy")
    facets_path = os.path.join(cache_dir, "profile_facets.npy")
    with open(profile_path) as f:
        profile = json.load(f)
    facets = np.load(facets_path) if os.path.exists(facets_path) else None
    if facets is not None and len(facets) == 0:
        facets = None
    return {"profile": profile, "embedding": np.load(embedding_path),
            "facets": facets,
            "profile_path": profile_path, "embedding_path": embedding_path}
