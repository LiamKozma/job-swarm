"""
Job Swarm - Alpha-source ingestion engines (CPU stage, batch partition).

Monitors the financial and intellectual precursors to hiring:

  Alpha Source 1 - Federal grant databases (NSF Awards API, NIH RePORTER,
                   SBIR.gov). Non-dilutive capital injection precedes
                   technical hiring by 60-120 days.
  Alpha Source 2 - arXiv pre-prints. IP dissemination precedes commercial
                   hiring; author/affiliation metadata identifies the labs
                   confronting hard high-dimensional problems right now.
  Alpha Source 3 - Startup telemetry: the Y Combinator directory (via the
                   yc-oss public JSON mirror - no keys, no scraping) and the
                   monthly Hacker News "Ask HN: Who is hiring?" thread (via
                   the public Algolia HN API).

All sources are public REST APIs hit with polite concurrency (semaphores +
delays) so a single UGA login-node IP never trips rate limits. Every fetcher
fails independently: one dead API never kills the nightly run.

Contact policy: only contact channels the owner deliberately published for
professional inquiry are recorded (NSF PI emails from public award records,
emails posted inside "Who is hiring?" posts, company websites). The swarm
never harvests emails from GitHub commit metadata and never sends mail.
"""

import asyncio
import hashlib
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

# =========================================================================
# CONFIGURATION
# =========================================================================

def _contact_email() -> str:
    """Operator contact for API User-Agents - never hardcoded in the repo.
    One line in ~/job_swarm/contact.email, or override the full string via
    the JOB_SWARM_USER_AGENT env var ("Agent/1.0 you@example.com")."""
    try:
        with open(os.path.expanduser("~/job_swarm/contact.email")) as fh:
            email = fh.read().strip()
        if email:
            return email
    except OSError:
        pass
    return "you@example.com"


_USER_AGENT = os.environ.get("JOB_SWARM_USER_AGENT") or (
    "JobSwarmResearch/1.0 " + _contact_email()
)
_HEADERS = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

# Global politeness gate across all engines
_NET_SEMAPHORE = asyncio.Semaphore(6)
_REQUEST_DELAY = 0.5  # seconds between requests within one engine

# Default search vocabulary - tuned to the candidate's expertise matrix.
# Override any of this via a JSON file pointed at by JOB_SWARM_CONFIG.
DEFAULT_CONFIG = {
    "grant_keywords": [
        "high-dimensional statistics",
        "distribution shift machine learning",
        "gaussian mixture model",
        "stochastic differential equation simulation",
        "out-of-distribution detection",
        "scientific machine learning HPC",
    ],
    "arxiv_queries": [
        'cat:cs.LG AND abs:"distribution shift"',
        'cat:cs.LG AND abs:"out-of-distribution"',
        'cat:stat.ML AND abs:"high-dimensional"',
        'cat:cs.LG AND abs:"catastrophic forgetting"',
        'cat:q-fin.ST AND abs:"regime switching"',
    ],
    "yc_tag_terms": [
        "machine learning", "ai", "artificial intelligence", "developer tools",
        "data engineering", "analytics", "fintech", "quant", "infrastructure",
    ],
    "hn_relevance_terms": [
        "machine learning", "ml", "statistics", "statistical", "quant",
        "hpc", "cuda", "gpu", "simulation", "time series", "inference",
        "distribution shift", "data science", "pytorch", "slurm",
    ],
    "usajobs_keywords": [
        "statistician", "data scientist", "machine learning",
        "operations research", "mathematical statistician",
    ],
    "max_per_source": 200,
    "lookback_days": 400,
}

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def _clean_hn_org(raw: str) -> str:
    """Company field of a Who-is-Hiring first line, minus decoration.
    Posters write 'Kog ( https://kog.ai )' - the URL must not leak into the
    org name, where it poisons the org_key ('koghttpsx2fx2fkogai')."""
    org = html.unescape(raw or "")
    org = re.sub(r"\(\s*(?:https?://|www\.)[^)]*\)", " ", org)
    org = re.sub(r"(?:https?://|www\.)\S+", " ", org)
    return re.sub(r"\s{2,}", " ", org).strip(" \t---|,;:")[:80]


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = os.environ.get("JOB_SWARM_CONFIG", "")
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg.update(json.load(f))
    return cfg


def _doc_id(source: str, native_id: str) -> str:
    return hashlib.sha1(f"{source}:{native_id}".encode()).hexdigest()


# Exponential backoff on 429/5xx - one login-node IP serves the whole swarm,
# so a polite retry ladder matters more than raw speed.
_MAX_RETRIES = 4


async def _request(session: aiohttp.ClientSession, method: str, url: str,
                   want: str = "json", **kwargs):
    headers = kwargs.pop("headers", _HEADERS)   # per-call override (API keys)
    last_err: Exception = RuntimeError("unreachable")
    for attempt in range(_MAX_RETRIES):
        try:
            async with _NET_SEMAPHORE:
                await asyncio.sleep(_REQUEST_DELAY)
                async with session.request(method, url, headers=headers, **kwargs) as resp:
                    if resp.status in (429, 500, 502, 503):
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status)
                    resp.raise_for_status()
                    return await (resp.json(content_type=None) if want == "json"
                                  else resp.text())
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError,
                asyncio.TimeoutError) as e:
            last_err = e
            status = getattr(e, "status", None)
            if status is not None and status not in (429, 500, 502, 503):
                raise
            await asyncio.sleep(3.0 * (2 ** attempt))  # 3s -> 6s -> 12s -> 24s
    raise last_err


async def _get_json(session, url, **kwargs):
    return await _request(session, "GET", url, want="json", **kwargs)


async def _get_text(session, url, **kwargs) -> str:
    return await _request(session, "GET", url, want="text", **kwargs)


async def _post_json(session, url, payload: dict):
    return await _request(session, "POST", url, want="json", json=payload)


# =========================================================================
# ALPHA SOURCE 1a - NSF AWARDS API
#   https://api.nsf.gov/services/v1/awards.json
#   PI contact info here is part of the public federal award record.
# =========================================================================

async def ingest_nsf(session: aiohttp.ClientSession, cfg: dict) -> list:
    docs = []
    fields = (
        "id,title,abstractText,awardeeName,awardeeCity,awardeeStateCode,"
        "date,startDate,fundsObligatedAmt,piFirstName,piLastName,piEmail"
    )
    pages = int(cfg.get("nsf_pages", 1))          # backfill overlays this to page deep
    date_start = (datetime.now(timezone.utc)
                  - timedelta(days=cfg["lookback_days"])).strftime("%m/%d/%Y")
    for keyword in cfg["grant_keywords"]:
        # Quote multi-word keywords - NSF ORs bare terms, which floods the
        # result set with unrelated awards (the filter engine would still
        # kill them on alignment, but there's no reason to ingest noise).
        kw = f'"{keyword}"' if " " in keyword else keyword
        awards = []
        for page in range(pages):
            try:
                data = await _get_json(
                    session,
                    "https://api.nsf.gov/services/v1/awards.json",
                    params={"keyword": kw, "printFields": fields, "rpp": 25,
                            "offset": page * 25 + 1, "dateStart": date_start},
                )
            except Exception as e:
                print(f"[NSF] query '{keyword}' page {page} failed: {e}")
                break
            batch = data.get("response", {}).get("award", []) or []
            awards.extend(batch)
            if len(batch) < 25:
                break
        for award in awards:
            institution = award.get("awardeeName") or ""
            if not institution:
                continue
            pi_name = f"{award.get('piFirstName', '')} {award.get('piLastName', '')}".strip()
            # PI-level org: a university's δ-trajectory mixes unrelated
            # departments; one PI's award stream is a single research voice.
            org = f"{pi_name} · {institution}" if pi_name else institution
            contacts = {"pi_name": pi_name}
            if award.get("piEmail"):
                contacts["pi_email"] = award["piEmail"]
                contacts["contact_basis"] = "PI email published in the public NSF award record"
            docs.append({
                "doc_id": _doc_id("nsf", str(award.get("id"))),
                "source": "nsf",
                "org": org,
                "date": _us_date_to_iso(award.get("startDate") or award.get("date")),
                "title": award.get("title"),
                "url": f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={award.get('id')}",
                "text": award.get("abstractText") or award.get("title") or "",
                "contacts": contacts,
                "extra": {"funds_obligated": award.get("fundsObligatedAmt"),
                          "institution": institution,
                          "matched_keyword": keyword},
            })
    print(f"[NSF] {len(docs)} award documents ingested")
    return docs[: cfg["max_per_source"]]


def _us_date_to_iso(d):
    """NSF returns mm/dd/yyyy; normalize to ISO for chronological sorting."""
    if not d:
        return None
    try:
        return datetime.strptime(d, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return d


# =========================================================================
# ALPHA SOURCE 1b - NIH RePORTER v2
#   POST https://api.reporter.nih.gov/v2/projects/search
# =========================================================================

async def ingest_nih(session: aiohttp.ClientSession, cfg: dict) -> list:
    docs = []
    since = (datetime.now(timezone.utc) - timedelta(days=cfg["lookback_days"])).strftime("%Y-%m-%d")
    for keyword in cfg["grant_keywords"]:
        payload = {
            "criteria": {
                "advanced_text_search": {
                    "operator": "and",
                    "search_field": "projecttitle,terms,abstracttext",
                    "search_text": keyword,
                },
                "award_notice_date": {"from_date": since},
            },
            "include_fields": [
                "ApplId", "ProjectTitle", "AbstractText", "Organization",
                "ContactPiName", "AwardAmount", "AwardNoticeDate",
            ],
            "offset": 0,
            # NIH caps limit at 500; the backfill overlay raises this
            "limit": min(int(cfg.get("nih_page_limit", 50)), 500),
        }
        try:
            data = await _post_json(session, "https://api.reporter.nih.gov/v2/projects/search", payload)
        except Exception as e:
            print(f"[NIH] query '{keyword}' failed: {e}")
            continue
        for proj in (data.get("results") or []):
            org_info = proj.get("organization") or {}
            institution = org_info.get("org_name") or ""
            if not institution:
                continue
            pi_name = (proj.get("contact_pi_name") or "").strip()
            org = f"{pi_name} · {institution}" if pi_name else institution
            docs.append({
                "doc_id": _doc_id("nih", str(proj.get("appl_id"))),
                "source": "nih",
                "org": org,
                "date": (proj.get("award_notice_date") or "")[:10] or None,
                "title": proj.get("project_title"),
                "url": f"https://reporter.nih.gov/project-details/{proj.get('appl_id')}",
                "text": proj.get("abstract_text") or proj.get("project_title") or "",
                "contacts": {"pi_name": pi_name},
                "extra": {"award_amount": proj.get("award_amount"),
                          "institution": institution,
                          "matched_keyword": keyword},
            })
    print(f"[NIH] {len(docs)} project documents ingested")
    return docs[: cfg["max_per_source"]]


# =========================================================================
# ALPHA SOURCE 1c - SBIR/STTR AWARDS
#   https://api.www.sbir.gov/public/api/awards
#   Phase II awardees are startups scaling prototypes into production -
#   the single highest-priority target class in the blueprint.
#
#   NOTE: this endpoint sits behind an aggressive WAF (429/403 seen from
#   residential IPs even with backoff). The engine fails soft; NSF's own
#   SBIR/STTR awards still arrive via the NSF Awards API, so losing this
#   source costs little. If it never succeeds from the UGA network either,
#   consider the bulk CSV exports on sbir.gov instead.
# =========================================================================

async def ingest_sbir(session: aiohttp.ClientSession, cfg: dict) -> list:
    docs = []
    for keyword in cfg["grant_keywords"]:
        try:
            data = await _get_json(
                session,
                "https://api.www.sbir.gov/public/api/awards",
                params={"keyword": keyword, "rows": 50, "format": "json"},
            )
        except Exception as e:
            print(f"[SBIR] query '{keyword}' failed: {e}")
            continue
        awards = data if isinstance(data, list) else data.get("results", []) or []
        for award in awards:
            org = award.get("firm") or ""
            if not org:
                continue
            native = award.get("contract") or award.get("award_link") or \
                f"{org}:{award.get('award_title')}:{award.get('award_year')}"
            contacts = {"poc_name": award.get("poc_name") or award.get("pi_name")}
            for email_field in ("poc_email", "pi_email"):
                if award.get(email_field):
                    contacts["poc_email"] = award[email_field]
                    contacts["contact_basis"] = "POC email published in the public SBIR award record"
                    break
            docs.append({
                "doc_id": _doc_id("sbir", str(native)),
                "source": "sbir",
                "org": org,
                "date": f"{award.get('award_year')}-01-01" if award.get("award_year") else None,
                "title": award.get("award_title"),
                "url": award.get("award_link") or "https://www.sbir.gov/awards",
                "text": award.get("abstract") or award.get("award_title") or "",
                "contacts": contacts,
                "extra": {"phase": award.get("phase"), "agency": award.get("agency"),
                          "award_amount": award.get("award_amount"),
                          "matched_keyword": keyword},
            })
    print(f"[SBIR] {len(docs)} award documents ingested")
    return docs[: cfg["max_per_source"]]


# =========================================================================
# ALPHA SOURCE 2 - ARXIV PRE-PRINTS (Atom API, stdlib XML parsing)
# =========================================================================

_ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# arXiv etiquette is ONE request per 3 seconds per IP; the global 0.5s pacing
# is too hot and 5/6 queries came back 429 on 2026-07-02. Serialize arXiv
# calls behind their own gate - other engines run concurrently regardless.
_ARXIV_GATE = asyncio.Lock()


async def _arxiv_query(session, params: dict) -> str:
    async with _ARXIV_GATE:
        raw = await _get_text(session, "https://export.arxiv.org/api/query",
                              params=params)
        await asyncio.sleep(3.0)
    return raw


async def ingest_arxiv(session: aiohttp.ClientSession, cfg: dict) -> list:
    docs = []
    depth = int(cfg.get("arxiv_depth", 60))       # backfill overlays this deeper
    step = min(depth, 100)                        # arXiv-recommended page size
    for query in cfg["arxiv_queries"]:
        entries = []
        for start in range(0, depth, step):
            try:
                raw = await _arxiv_query(
                    session,
                    params={
                        "search_query": query,
                        "start": start,
                        "max_results": step,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                    },
                )
                root = ET.fromstring(raw)
            except Exception as e:
                print(f"[arXiv] query '{query}' start={start} failed: {e}")
                break
            page_entries = root.findall("a:entry", _ATOM_NS)
            entries.extend(page_entries)
            if len(page_entries) < step:
                break
        for entry in entries:
            arxiv_id = (entry.findtext("a:id", "", _ATOM_NS) or "").rsplit("/", 1)[-1]
            authors, affiliations = [], []
            for author in entry.findall("a:author", _ATOM_NS):
                name = author.findtext("a:name", "", _ATOM_NS)
                if name:
                    authors.append(name)
                aff = author.findtext("arxiv:affiliation", "", _ATOM_NS)
                if aff:
                    affiliations.append(aff)
            # Affiliation is the org when present; otherwise group by lead author's lab
            org = affiliations[0] if affiliations else (f"{authors[0]} group" if authors else "")
            if not org:
                continue
            docs.append({
                "doc_id": _doc_id("arxiv", arxiv_id),
                "source": "arxiv",
                "org": org,
                "date": (entry.findtext("a:published", "", _ATOM_NS) or "")[:10] or None,
                "title": " ".join((entry.findtext("a:title", "", _ATOM_NS) or "").split()),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "text": " ".join((entry.findtext("a:summary", "", _ATOM_NS) or "").split()),
                "contacts": {"authors": authors[:8]},
                "extra": {"matched_query": query, "affiliations": affiliations[:8]},
            })
    try:
        await _resolve_arxiv_affiliations(session, docs,
                                          cap=int(cfg.get("arxiv_affil_lookups", 60)))
    except Exception as e:
        # Affiliation resolution is an enrichment - its failure must never
        # cost the night's papers (it did once: the 2026-07-02 DB lock).
        print(f"[arXiv] affiliation resolution failed (non-fatal): {e}")
    print(f"[arXiv] {len(docs)} papers ingested")
    return docs[: cfg["max_per_source"]]


def _miss_age_days(cached: str) -> int:
    try:
        return (datetime.now(timezone.utc).date()
                - datetime.strptime(cached[5:], "%Y-%m-%d").date()).days
    except ValueError:
        return 999


async def _openalex_institution(session, base: str) -> Optional[str]:
    """Institution of a paper's lead author, or None.
    Three-tier resolution (validated live 2026-07-02): arXiv preprints
    usually have institutions=[] AND raw_affiliation_strings=[] on the
    work itself, so the author's last-known institution is the workhorse."""
    mailto = _USER_AGENT.split()[-1]
    inst = None
    try:
        data = await _get_json(
            session,
            f"https://api.openalex.org/works/doi:10.48550/arxiv.{base}",
            params={"mailto": mailto},
        )
        first = (data.get("authorships") or [{}])[0]
        insts = first.get("institutions") or []
        if insts:
            inst = insts[0].get("display_name")
        if not inst:
            # NB: raws[0] on the (typical) empty list used to IndexError into
            # the blanket except, silently killing the author-record tier -
            # the real reason the first live run resolved 0/60.
            raws = first.get("raw_affiliation_strings") or []
            if raws:
                inst = (raws[0] or "").strip()[:80] or None
        if not inst:
            author_id = ((first.get("author") or {}).get("id") or "").rsplit("/", 1)[-1]
            if author_id:
                adata = await _get_json(
                    session,
                    f"https://api.openalex.org/authors/{author_id}",
                    params={"mailto": mailto},
                )
                known = adata.get("last_known_institutions") or []
                if known:
                    inst = known[0].get("display_name")
                if not inst:
                    # last_known_institutions is often null even when the
                    # author record carries a populated affiliations history
                    # (most-recent-first) - observed live 2026-07-02.
                    affs = adata.get("affiliations") or []
                    if affs:
                        inst = ((affs[0].get("institution") or {})
                                .get("display_name"))
    except Exception:
        inst = None
    return inst


async def _resolve_arxiv_affiliations(session, docs: list, cap: int = 60) -> None:
    """
    Most arXiv Atom entries carry no affiliation, which creates 'Lead Author
    group' pseudo-orgs (no contacts, wasted ATS probes). arXiv assigns DOIs
    (10.48550/arXiv.<id>), so OpenAlex can resolve the real institution.
    Results (including misses) are cached in the meta table; at most `cap`
    lookups per run. Papers still missing after this pass get retried from
    the corpus side by backfill_group_affiliations once OpenAlex has had
    time to index them.
    """
    # Lock discipline: engines share ONE SQLite file concurrently, and the
    # old shape (set_meta inside the lookup loop, one commit at the end) held
    # a write transaction across minutes of OpenAlex calls - on 2026-07-02 it
    # collided with the ATS probe loop's own long transaction and 'database
    # is locked' killed the whole arXiv source. Now: reads + network first
    # with NO write txn, then one short write burst; any DB failure degrades
    # to unresolved 'X group' orgs (the corpus-side backfill retries later)
    # instead of losing the night's papers.
    try:
        import swarm_db
        conn = swarm_db.connect()
    except Exception as e:
        print(f"[arXiv] OpenAlex resolution skipped (no DB): {e}")
        return
    today = datetime.now(timezone.utc).date()
    n_looked = n_fixed = 0
    writes = []   # (meta_key, value) applied in one short burst at the end
    try:
        for d in docs:
            if not d["org"].endswith(" group"):
                continue
            base = re.sub(r"v\d+$", "", (d["url"] or "").rsplit("/", 1)[-1])
            if not base:
                continue
            cached = swarm_db.get_meta(conn, f"oa_aff:{base}")
            if cached is not None:
                if cached.startswith("miss@"):
                    # OpenAlex indexes new papers with days-to-weeks of lag, so a
                    # miss is RETRYABLE - but only after a cool-off, or every run
                    # burns its whole lookup budget on the same fresh papers.
                    if _miss_age_days(cached) < 14:
                        continue
                    # cool-off elapsed -> fall through to a fresh lookup
                elif cached:
                    d["org"] = cached
                    n_fixed += 1
                    continue
                # else: legacy permanent-miss marker ("") -> fall through, retry once
            if n_looked >= cap:
                continue
            n_looked += 1
            inst = await _openalex_institution(session, base)
            if inst:
                # 'Lead Author · Institution' - same convention as PI-level grant
                # orgs: single research voice, warmpath searches the institution,
                # ATS probing skips it.
                lead = (d.get("contacts", {}).get("authors") or [None])[0]
                org_name = f"{lead} · {inst}" if lead else inst
                writes.append((f"oa_aff:{base}", org_name))
                d["org"] = org_name
                n_fixed += 1
            else:
                writes.append((f"oa_aff:{base}", f"miss@{today.isoformat()}"))
        for key, value in writes:
            swarm_db.set_meta(conn, key, value)
        conn.commit()
    except Exception as e:
        # Cache write lost, doc renames kept - strictly better than FAILING
        print(f"[arXiv] affiliation cache write failed (non-fatal): {e}")
    finally:
        conn.close()
    if n_looked or n_fixed:
        print(f"[arXiv] OpenAlex affiliations: {n_fixed} resolved "
              f"({n_looked} fresh lookups)")


async def backfill_group_affiliations(session, cap: int = 40,
                                      retry_after_days: int = 4) -> None:
    """
    Corpus-side retry of failed affiliation lookups. Day-fresh papers are
    almost never in OpenAlex yet (the live 2026-07-02 run resolved 0/60), and
    the nightly window only sees NEW papers - so without this pass an
    'X group' org would keep its pseudo-name forever. Each night: retry the
    oldest cached misses older than `retry_after_days`, re-key the docs of
    every hit to 'Lead Author · Institution', and drop emptied group orgs.
    """
    try:
        import swarm_db
        conn = swarm_db.connect()
    except Exception as e:
        print(f"[arXiv] affiliation backfill skipped (no DB): {e}")
        return
    rows = conn.execute(
        "SELECT d.doc_id, d.url, d.org_key, d.contacts, o.display_name "
        "FROM docs d JOIN orgs o ON o.org_key = d.org_key "
        "WHERE d.source = 'arxiv' AND o.display_name LIKE '% group' "
        "ORDER BY d.added_at ASC"
    ).fetchall()
    n_looked = n_fixed = 0
    emptied = set()
    for r in rows:
        if n_looked >= cap:
            break
        base = re.sub(r"v\d+$", "", (r["url"] or "").rsplit("/", 1)[-1])
        if not base:
            continue
        cached = swarm_db.get_meta(conn, f"oa_aff:{base}")
        if cached and not cached.startswith("miss@"):
            org_name = cached          # resolved earlier; doc just never moved
        else:
            if cached and _miss_age_days(cached) < retry_after_days:
                continue
            n_looked += 1
            inst = await _openalex_institution(session, base)
            if not inst:
                swarm_db.set_meta(
                    conn, f"oa_aff:{base}",
                    f"miss@{datetime.now(timezone.utc).date().isoformat()}")
                conn.commit()
                continue
            lead = (json.loads(r["contacts"] or "{}").get("authors") or [None])[0]
            org_name = f"{lead} · {inst}" if lead else inst
            swarm_db.set_meta(conn, f"oa_aff:{base}", org_name)
        new_key = swarm_db.touch_org(conn, org_name, "arxiv")
        if new_key != r["org_key"]:
            conn.execute("UPDATE docs SET org_key = ? WHERE doc_id = ?",
                         (new_key, r["doc_id"]))
            emptied.add(r["org_key"])
            n_fixed += 1
        conn.commit()   # short txns: the next iteration awaits the network
    # Group orgs whose every doc moved away are dead weight in org scans
    for key in emptied:
        left = conn.execute("SELECT 1 FROM docs WHERE org_key = ? LIMIT 1",
                            (key,)).fetchone()
        if left is None:
            conn.execute(
                "DELETE FROM orgs WHERE org_key = ? AND status NOT IN "
                f"{tuple(swarm_db.HANDS_OFF_STATUSES)!r}", (key,))
    conn.commit()
    conn.close()
    if n_looked or n_fixed:
        print(f"[arXiv] affiliation backfill: {n_fixed} group orgs re-keyed "
              f"({n_looked} retries of aged misses)")


# =========================================================================
# ALPHA SOURCE 3a - Y COMBINATOR DIRECTORY
#   via the yc-oss public JSON mirror (github.com/yc-oss/api):
#   plain GET, no API keys, refreshed daily from the official directory.
# =========================================================================

async def ingest_yc(session: aiohttp.ClientSession, cfg: dict) -> list:
    try:
        companies = await _get_json(session, "https://yc-oss.github.io/api/companies/all.json")
    except Exception as e:
        print(f"[YC] directory fetch failed: {e}")
        return []

    terms = [t.lower() for t in cfg["yc_tag_terms"]]
    docs = []
    for c in companies:
        haystack = " ".join([
            c.get("industry") or "", c.get("subindustry") or "",
            " ".join(c.get("tags") or []), c.get("one_liner") or "",
        ]).lower()
        if not any(t in haystack for t in terms):
            continue
        # Recent batches only - early-stage is the whole point. The batch label
        # ("Winter 2025", "Spring 2026") carries the cohort year; launched_at is
        # merely the launch-post date and can be years later.
        batch_year_match = re.search(r"(20\d{2})", c.get("batch") or "")
        if batch_year_match:
            if int(batch_year_match.group(1)) < datetime.now(timezone.utc).year - 2:
                continue
        if (c.get("status") or "Active") not in ("Active", "Public"):
            continue  # skip acquired/dead companies
        launched = c.get("launched_at") or 0
        name = c.get("name") or ""
        if not name:
            continue
        text = " ".join(filter(None, [c.get("one_liner"), c.get("long_description")]))
        docs.append({
            "doc_id": _doc_id("yc", c.get("slug") or name),
            "source": "yc",
            "org": name,
            "date": datetime.fromtimestamp(launched, tz=timezone.utc).strftime("%Y-%m-%d") if launched else None,
            "title": f"{name} ({c.get('batch', '?')}) - {c.get('one_liner', '')}",
            "url": c.get("url") or c.get("website"),
            "text": text or name,
            "contacts": {"website": c.get("website"), "yc_page": c.get("url")},
            "extra": {"batch": c.get("batch"), "team_size": c.get("team_size"),
                      "tags": c.get("tags"), "status": c.get("status"),
                      "is_hiring": c.get("isHiring") or c.get("is_hiring")},
        })
    print(f"[YC] {len(docs)} companies matched filter terms")
    return docs[: cfg["max_per_source"]]


# =========================================================================
# ALPHA SOURCE 3b - HACKER NEWS "WHO IS HIRING?" (Algolia HN API)
#   Emails found here were posted by the hirer explicitly to be contacted,
#   so recording them is squarely within the intended use.
# =========================================================================

async def _scan_hiring_thread(session, thread: dict, terms: list) -> list:
    """All relevant top-level job posts from one monthly thread."""
    story_id = thread["objectID"]
    docs = []
    for page in range(4):  # up to ~800 top-level comments
        data = await _get_json(
            session,
            "https://hn.algolia.com/api/v1/search_by_date",
            params={"tags": f"comment,story_{story_id}", "hitsPerPage": 200, "page": page},
        )
        hits = data.get("hits", [])
        if not hits:
            break
        for h in hits:
            if str(h.get("parent_id")) != str(story_id):
                continue  # only top-level job posts, not discussion replies
            text = re.sub(r"<[^>]+>", " ", h.get("comment_text") or "")
            text = html.unescape(text)
            low = text.lower()
            # Token-boundary matching: bare 'ml' must not match 'html'
            if sum(bool(re.search(
                    r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low))
                   for t in terms) < 2:
                continue  # require at least two relevance hits
            # HN convention: first line is "Company | Role | Location | ..."
            first_line = text.strip().split("\n")[0]
            org = _clean_hn_org(first_line.split("|")[0]) or "HN poster"
            emails = _EMAIL_RE.findall(text)
            contacts = {"hn_user": h.get("author")}
            if emails:
                contacts["email"] = emails[0]
                contacts["contact_basis"] = "email published by the hirer in the Who-is-Hiring post"
            docs.append({
                "doc_id": _doc_id("hn", h["objectID"]),
                "source": "hn_hiring",
                "org": org,
                "date": (h.get("created_at") or "")[:10] or None,
                "title": first_line[:200],
                "url": f"https://news.ycombinator.com/item?id={h['objectID']}",
                "text": text[:8000],
                "contacts": contacts,
                "extra": {"thread": thread.get("title")},
            })
    print(f"[HN] {len(docs)} relevant hiring posts from '{thread.get('title')}'")
    return docs


async def ingest_hn_hiring(session: aiohttp.ClientSession, cfg: dict) -> list:
    # hn_threads=1 nightly (current month); the backfill overlay raises it to
    # walk past monthly threads - repeat posters are a churn/growth signal.
    n_threads = int(cfg.get("hn_threads", 1))
    try:
        stories = await _get_json(
            session,
            "https://hn.algolia.com/api/v1/search_by_date",
            # author_whoishiring also posts "Who wants to be hired?" and
            # "Freelancer?" each month, so over-fetch and filter by title
            params={"tags": "story,author_whoishiring",
                    "hitsPerPage": max(10, n_threads * 4)},
        )
        threads = [h for h in stories.get("hits", [])
                   if "who is hiring" in (h.get("title") or "").lower()][:n_threads]
        if not threads:
            print("[HN] no 'Who is hiring?' thread found")
            return []
        docs = []
        terms = [t.lower() for t in cfg["hn_relevance_terms"]]
        for thread in threads:
            docs.extend(await _scan_hiring_thread(session, thread, terms))
        return docs[: cfg["max_per_source"]]
    except Exception as e:
        print(f"[HN] ingestion failed: {e}")
        return []


# =========================================================================
# ALPHA SOURCE 6 - USASPENDING.GOV (all federal grants + contracts)
#   NSF/NIH cover a fraction of federal science money. DOE, DARPA, ONR,
#   AFOSR, and NASA fund most scientific-ML and HPC work; USAspending's
#   keyless API covers all of it in one place. NSF/NIH results are skipped
#   here (richer metadata arrives via their native APIs).
# =========================================================================

async def ingest_usaspending(session: aiohttp.ClientSession, cfg: dict) -> list:
    pages = int(cfg.get("usaspending_pages", 1))
    lookback = min(int(cfg["lookback_days"]), 365)
    start = (datetime.now(timezone.utc) - timedelta(days=lookback)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    docs = []
    for keyword in cfg["grant_keywords"]:
        for page in range(1, pages + 1):
            payload = {
                "filters": {
                    "keywords": [keyword],
                    "time_period": [{"start_date": start, "end_date": end}],
                    # 02-05 = federal grant award types
                    "award_type_codes": ["02", "03", "04", "05"],
                },
                "fields": ["Award ID", "Recipient Name", "Description",
                           "Award Amount", "Start Date", "Awarding Agency",
                           "Awarding Sub Agency", "generated_internal_id"],
                "sort": "Start Date", "order": "desc",
                "page": page, "limit": 60,
            }
            try:
                data = await _post_json(
                    session,
                    "https://api.usaspending.gov/api/v2/search/spending_by_award/",
                    payload)
            except Exception as e:
                print(f"[USAspending] query '{keyword}' page {page} failed: {e}")
                break
            results = data.get("results") or []
            for r in results:
                agency = r.get("Awarding Agency") or ""
                sub = r.get("Awarding Sub Agency") or ""
                if (agency == "National Science Foundation"
                        or sub == "National Institutes of Health"):
                    continue  # native APIs carry these with abstracts + PI contacts
                org = (r.get("Recipient Name") or "").strip()
                desc = (r.get("Description") or "").strip()
                if not org or len(desc) < 40:
                    continue  # description-free awards embed as noise
                gid = r.get("generated_internal_id")
                docs.append({
                    "doc_id": _doc_id("usaspending", str(r.get("Award ID") or gid)),
                    "source": "usaspending",
                    "org": org.title() if org.isupper() else org,
                    "date": r.get("Start Date"),
                    "title": f"{sub or agency} award: {desc[:120]}",
                    "url": (f"https://www.usaspending.gov/award/{gid}"
                            if gid else "https://www.usaspending.gov"),
                    "text": desc,
                    "contacts": {},
                    "extra": {"award_amount": r.get("Award Amount"),
                              "agency": agency, "sub_agency": sub,
                              "matched_keyword": keyword},
                })
            if len(results) < 60:
                break
    print(f"[USAspending] {len(docs)} award documents ingested")
    return docs[: cfg["max_per_source"]]


# =========================================================================
# ALPHA SOURCE 7 - CLINICALTRIALS.GOV v2
#   A newly registered industry-sponsored Phase II/III trial means the
#   sponsor needs biostatisticians in 3-6 months - a leading indicator for
#   the classic MS-Stats channel (strong pay, sane hours), invisible to
#   job boards until far too late.
# =========================================================================

async def ingest_clinicaltrials(session: aiohttp.ClientSession, cfg: dict) -> list:
    days = int(cfg.get("ct_lookback_days", 30))
    pages = int(cfg.get("ct_pages", 1))
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    term = (f"AREA[StudyFirstPostDate]RANGE[{since},MAX] "
            f"AND AREA[LeadSponsorClass]INDUSTRY "
            f"AND (AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)")
    docs, token = [], None
    for _ in range(pages):
        params = {"query.term": term, "pageSize": 100}
        if token:
            params["pageToken"] = token
        try:
            data = await _get_json(session,
                                   "https://clinicaltrials.gov/api/v2/studies",
                                   params=params)
        except Exception as e:
            print(f"[ClinicalTrials] fetch failed: {e}")
            break
        for s in data.get("studies") or []:
            ps = s.get("protocolSection") or {}
            ident = ps.get("identificationModule") or {}
            nct = ident.get("nctId")
            sponsor = (((ps.get("sponsorCollaboratorsModule") or {})
                        .get("leadSponsor") or {}).get("name") or "").strip()
            if not nct or not sponsor:
                continue
            phases = (ps.get("designModule") or {}).get("phases") or []
            summary = (ps.get("descriptionModule") or {}).get("briefSummary") or ""
            conds = (ps.get("conditionsModule") or {}).get("conditions") or []
            status_mod = ps.get("statusModule") or {}
            first_post = ((status_mod.get("studyFirstPostDateStruct") or {})
                          .get("date"))
            title = ident.get("briefTitle") or ""
            docs.append({
                "doc_id": _doc_id("ct", nct),
                "source": "clinicaltrials",
                "org": sponsor,
                "date": first_post,
                "title": f"{'/'.join(phases) or 'Trial'}: {title[:150]}",
                "url": f"https://clinicaltrials.gov/study/{nct}",
                "text": (f"{title}. Phases: {', '.join(phases)}. "
                         f"Conditions: {', '.join(conds[:6])}. {summary}"),
                "contacts": {},
                "extra": {"phases": phases, "conditions": conds[:6],
                          "nct_id": nct,
                          "status": status_mod.get("overallStatus")},
            })
        token = data.get("nextPageToken")
        if not token:
            break

    # ---- Phase II readout sweep -------------------------------------------
    # A Phase II flipping to COMPLETED / ACTIVE_NOT_RECRUITING is the go/no-go
    # readout that precedes a funding catalyst and a biostat hiring surge by
    # 3-6 months (2026 research briefing). These are STATUS CHANGES on old
    # trials - invisible to the new-trials query above - so a separate sweep
    # on LastUpdatePostDate emits 'ct_event' docs. Sub-40-char text keeps
    # them out of the embedder (and thus all scoring); the review queue reads
    # them by source+extra, same pattern as WARN.
    ev_since = (datetime.now(timezone.utc)
                - timedelta(days=int(cfg.get("ct_event_lookback_days", 14)))
                ).strftime("%Y-%m-%d")
    ev_term = (f"AREA[LastUpdatePostDate]RANGE[{ev_since},MAX] "
               f"AND AREA[LeadSponsorClass]INDUSTRY "
               f"AND AREA[Phase]PHASE2 "
               f"AND (AREA[OverallStatus]COMPLETED "
               f"OR AREA[OverallStatus]ACTIVE_NOT_RECRUITING)")
    n_events = 0
    try:
        data = await _get_json(session,
                               "https://clinicaltrials.gov/api/v2/studies",
                               params={"query.term": ev_term, "pageSize": 100})
        for s in data.get("studies") or []:
            ps = s.get("protocolSection") or {}
            ident = ps.get("identificationModule") or {}
            nct = ident.get("nctId")
            sponsor = (((ps.get("sponsorCollaboratorsModule") or {})
                        .get("leadSponsor") or {}).get("name") or "").strip()
            if not nct or not sponsor:
                continue
            status = (ps.get("statusModule") or {}).get("overallStatus")
            title = ident.get("briefTitle") or ""
            docs.append({
                "doc_id": _doc_id("ct_evt", f"{nct}:{status}"),
                "source": "ct_event",
                "org": sponsor,
                "date": ((ps.get("statusModule") or {})
                         .get("lastUpdatePostDateStruct") or {}).get("date"),
                "title": f"Phase II readout ({status}): {title[:140]}",
                "url": f"https://clinicaltrials.gov/study/{nct}",
                "text": "Phase II readout event",
                "contacts": {},
                "extra": {"nct_id": nct, "status": status,
                          "trigger": "phase2_readout"},
            })
            n_events += 1
    except Exception as e:
        print(f"[ClinicalTrials] readout sweep failed (non-fatal): {e}")

    print(f"[ClinicalTrials] {len(docs) - n_events} new industry Phase II/III "
          f"trials + {n_events} Phase II readout events")
    return docs[: cfg["max_per_source"]]


# =========================================================================
# ALPHA SOURCE 8 - USAJOBS (federal postings; the citizen/WLB channel)
#   https://data.usajobs.gov/api/search - free API key registered to an
#   email; both must be sent as headers. Key lives OUTSIDE the repo:
#   env JOB_SWARM_USAJOBS_KEY or ~/job_swarm/usajobs.key (ZFS home, so it
#   survives the 30-day scratch purge). Fails soft when absent.
# =========================================================================

def _usajobs_key(cfg: dict) -> str:
    key = os.environ.get("JOB_SWARM_USAJOBS_KEY", "").strip()
    if key:
        return key
    keyfile = os.path.expanduser(
        cfg.get("usajobs_key_file", "~/job_swarm/usajobs.key"))
    if os.path.exists(keyfile):
        with open(keyfile) as f:
            return f.read().strip()
    return ""


async def ingest_usajobs(session: aiohttp.ClientSession, cfg: dict) -> list:
    key = _usajobs_key(cfg)
    if not key:
        print("[USAJOBS] no API key (JOB_SWARM_USAJOBS_KEY or "
              "~/job_swarm/usajobs.key) - source skipped")
        return []
    from ats_engine import DEFAULT_TITLE_TERMS
    title_terms = [t.lower() for t in cfg.get("ats_title_terms", DEFAULT_TITLE_TERMS)]
    headers = {
        "User-Agent": _USER_AGENT.split()[-1],   # the registered email
        "Authorization-Key": key,
        "Accept": "application/json",
    }
    days = min(int(cfg.get("usajobs_days", 30)), 60)   # API caps DatePosted at 60
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    docs, seen = [], set()
    for kw in cfg.get("usajobs_keywords", DEFAULT_CONFIG["usajobs_keywords"]):
        try:
            async with _NET_SEMAPHORE:
                await asyncio.sleep(_REQUEST_DELAY)
                async with session.get(
                    "https://data.usajobs.gov/api/search",
                    params={"Keyword": kw, "ResultsPerPage": 100,
                            "DatePosted": days, "SortField": "opendate",
                            "SortDirection": "desc"},
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
        except Exception as e:
            print(f"[USAJOBS] query '{kw}' failed: {e}")
            continue
        for item in ((data.get("SearchResult") or {})
                     .get("SearchResultItems") or []):
            d = item.get("MatchedObjectDescriptor") or {}
            cn = item.get("MatchedObjectId") or d.get("PositionID")
            title = d.get("PositionTitle") or ""
            if not cn or cn in seen:
                continue
            seen.add(cn)
            if not any(t in title.lower() for t in title_terms):
                continue
            close = (d.get("ApplicationCloseDate") or "")[:10]
            if close and close < today:
                continue  # already closed - not applyable
            org = (d.get("OrganizationName") or d.get("DepartmentName")
                   or "US Federal Government")
            loc = d.get("PositionLocationDisplay")
            salary = None
            for rem in d.get("PositionRemuneration") or []:
                if (rem.get("RateIntervalCode") or "").upper() != "PA":
                    continue  # annual rates only - hourly would skew comp stats
                try:
                    lo = float(rem.get("MinimumRange") or 0)
                    hi = float(rem.get("MaximumRange") or 0)
                except (TypeError, ValueError):
                    break
                if lo:
                    salary = f"${lo:,.0f}" + (f" - ${hi:,.0f}" if hi > lo else "")
                break
            ua_details = (d.get("UserArea") or {}).get("Details") or {}
            summary = ua_details.get("JobSummary") or ""
            quals = d.get("QualificationSummary") or ""
            apply_uri = d.get("ApplyURI")
            if isinstance(apply_uri, list):
                apply_uri = apply_uri[0] if apply_uri else None
            docs.append({
                "doc_id": _doc_id("usajobs", str(cn)),
                "source": "usajobs",
                "org": org,
                "date": (d.get("PublicationStartDate") or "")[:10] or None,
                "title": f"{title} - {org} ({loc})"[:200],
                "url": d.get("PositionURI"),
                "text": f"{title}. {summary} {quals}"[:8000],
                "contacts": {"apply_url": apply_uri or d.get("PositionURI"),
                             "contact_basis": "public federal job posting (USAJOBS)"},
                "extra": {"provider": "usajobs", "location": loc,
                          "salary": salary, "posting": True,
                          "close_date": close or None,
                          "department": d.get("DepartmentName"),
                          "matched_keyword": kw},
            })
    print(f"[USAJOBS] {len(docs)} open federal postings ingested")
    return docs[: cfg["max_per_source"]]


# =========================================================================
# ALPHA SOURCE 9 - PATENTS (USPTO Open Data Portal; R&D-direction signal)
#   PatentsView was sunset into the ODP (api.uspto.gov - verified live
#   2026-07-02: the old search.patentsview.org is NXDOMAIN and the legacy
#   endpoints 301 to the ODP transition guide). Free API key: MyUSPTO
#   account at data.uspto.gov -> API Manager; sent as X-API-KEY. A company
#   filing patent applications in the candidate's areas is investing R&D
#   headcount there months before hiring shows anywhere else. Application
#   titles are AUTHORED text, so they feed the δ-shift machinery too.
#   Fails soft when the key is absent (env JOB_SWARM_USPTO_KEY or
#   ~/job_swarm/uspto.key). The q grammar is config-overridable
#   (patents_query_template) so field-name corrections after the first
#   keyed run need no code change; errors print the full response body.
# =========================================================================

_USPTO_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"


def _uspto_key(cfg: dict) -> str:
    key = os.environ.get("JOB_SWARM_USPTO_KEY", "").strip()
    if key:
        return key
    keyfile = os.path.expanduser(
        cfg.get("uspto_key_file", "~/job_swarm/uspto.key"))
    if os.path.exists(keyfile):
        with open(keyfile) as f:
            return f.read().strip()
    return ""


async def ingest_patents(session: aiohttp.ClientSession, cfg: dict) -> list:
    key = _uspto_key(cfg)
    if not key:
        print("[Patents] no API key (JOB_SWARM_USPTO_KEY or "
              "~/job_swarm/uspto.key - free via MyUSPTO at data.uspto.gov) "
              "- source skipped")
        return []
    lookback = int(cfg.get("patents_lookback_days", 180))
    start = (datetime.now(timezone.utc) - timedelta(days=lookback)).strftime("%Y-%m-%d")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    keywords = cfg.get("patent_keywords") or cfg.get(
        "grant_keywords", DEFAULT_CONFIG["grant_keywords"])
    q_template = cfg.get("patents_query_template",
                         'applicationMetaData.inventionTitle:"{kw}"')
    headers = {**_HEADERS, "X-API-KEY": key}
    docs, seen = [], set()
    for kw in keywords:
        payload = {
            "q": q_template.format(kw=kw),
            "rangeFilters": [{"field": "applicationMetaData.filingDate",
                              "valueFrom": start, "valueTo": today}],
            "pagination": {"offset": 0,
                           "limit": int(cfg.get("patents_per_query", 50))},
            "sort": [{"field": "applicationMetaData.filingDate",
                      "order": "desc"}],
        }
        try:
            async with _NET_SEMAPHORE:
                await asyncio.sleep(_REQUEST_DELAY)
                async with session.post(_USPTO_SEARCH_URL, json=payload,
                                        headers=headers) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        # First keyed run may need a grammar fix - surface
                        # the server's own message, not just the status.
                        print(f"[Patents] query '{kw}' HTTP {resp.status}: "
                              f"{body[:300]}")
                        continue
                    data = json.loads(body)
        except Exception as e:
            print(f"[Patents] query '{kw}' failed: {e}")
            continue
        wrappers = (data.get("patentFileWrapperDataBag")
                    or data.get("patentBag") or data.get("results") or [])
        for w in wrappers:
            meta = w.get("applicationMetaData") or w
            app_no = (w.get("applicationNumberText")
                      or meta.get("applicationNumberText"))
            title = meta.get("inventionTitle") or ""
            if not app_no or app_no in seen or not title:
                continue
            seen.add(app_no)
            applicants = meta.get("firstApplicantName") or ""
            if not applicants:
                bag = meta.get("applicantBag") or []
                names = [a.get("applicantNameText") for a in bag
                         if isinstance(a, dict) and a.get("applicantNameText")]
                applicants = names[0] if names else ""
            if not applicants or len(applicants) < 3:
                continue  # individual inventors / unnamed - no org to target
            # Named inventors: the humans who actually did the technical work
            # named on this org's patent. Published in the public application
            # record - names only, no email (like arXiv authorship); the human
            # finds the channel through the org itself. First inventor is the
            # usual technical lead by USPTO convention.
            inventors = []
            for inv in (meta.get("inventorBag") or []):
                if isinstance(inv, dict):
                    nm = (inv.get("inventorNameText")
                          or " ".join(x for x in (inv.get("firstName"),
                                                  inv.get("lastName")) if x).strip())
                    if nm and nm not in inventors:
                        inventors.append(nm)
            first_inv = meta.get("firstInventorName")
            if first_inv and first_inv not in inventors:
                inventors.insert(0, first_inv)
            contacts = {}
            if inventors:
                contacts["inventors"] = inventors[:8]
                contacts["contact_basis"] = (
                    "inventor names published in the public USPTO application "
                    "record - find their channel through the org")
            docs.append({
                "doc_id": _doc_id("patents", app_no),
                "source": "patents",
                "org": applicants,
                "date": meta.get("filingDate"),
                "title": title[:200],
                "url": f"https://patentcenter.uspto.gov/applications/{app_no}",
                "text": f"{title}. Patent application by {applicants}, "
                        f"filed {meta.get('filingDate')}. "
                        + (f"Inventors: {', '.join(inventors[:8])}. "
                           if inventors else "")
                        + f"{meta.get('applicationStatusDescriptionText') or ''}",
                "contacts": contacts,
                "extra": {"matched_keyword": kw,
                          "application_number": app_no},
            })
    print(f"[Patents] {len(docs)} recent applications across "
          f"{len({d['org'] for d in docs})} applicant orgs")
    return docs[: cfg["max_per_source"]]


# =========================================================================
# ALPHA SOURCE 10 - WARN LAYOFF NOTICES (via warnfirehose.com)
#   Free keyless tier: 25 calls/day, 25 records/call (verified live
#   2026-07-03) - a nightly 4-day delta needs ~2-6 calls. 60-day forward
#   signal: (a) an org filing WARN is freezing/shrinking - outreach there is
#   wasted; (b) a competitor filing WARN means the survivors absorb its
#   market - outreach to THEM improves. Docs are stored with a sub-40-char
#   text so they are never embedded: invisible to all scoring paths until
#   the analyze-side consumption ships (repost-forensics batch), but the
#   history accumulates from tonight.
# =========================================================================

async def ingest_warn(session: aiohttp.ClientSession, cfg: dict) -> list:
    lookback = int(cfg.get("warn_lookback_days", 4))
    max_pages = int(cfg.get("warn_max_pages", 6))       # 6×25 = 150 records
    date_from = (datetime.now(timezone.utc)
                 - timedelta(days=lookback)).strftime("%Y-%m-%d")
    docs, offset = [], 0
    for _ in range(max_pages):
        try:
            data = await _get_json(
                session, "https://warnfirehose.com/api/records",
                params={"date_from": date_from, "limit": 25, "offset": offset})
        except Exception as e:
            print(f"[WARN] fetch failed at offset {offset}: {e}")
            break
        records = data.get("records") or []
        for r in records:
            rid = r.get("id")
            company = (r.get("company_name") or "").strip()
            if not rid or not company:
                continue
            n_aff = r.get("employees_affected")
            where = ", ".join(x for x in (r.get("city"), r.get("state")) if x)
            docs.append({
                "doc_id": _doc_id("warn", rid),
                "source": "warn",
                "org": company,
                "date": r.get("notice_date"),
                "title": (f"WARN: {company} - {n_aff or '?'} employees"
                          f" ({where or 'location n/a'}, effective "
                          f"{r.get('effective_date') or 'n/a'})")[:200],
                # Deliberately <40 chars: docs_missing_embeddings skips it,
                # so a layoff notice can never pollute an org's semantic
                # state or the posting sections. Analysis reads extra.
                "text": "WARN layoff notice",
                "contacts": {},
                "extra": {"employees_affected": n_aff,
                          "state": r.get("state"), "city": r.get("city"),
                          "layoff_type": r.get("layoff_type"),
                          "naics_code": r.get("naics_code"),
                          "industry": r.get("industry"),
                          "ticker": r.get("ticker"),
                          "effective_date": r.get("effective_date"),
                          "source_url": r.get("source_url")},
            })
        if len(records) < 25:
            break
        offset += 25
    print(f"[WARN] {len(docs)} layoff notices since {date_from}")
    return docs


# =========================================================================
# ALPHA SOURCE 11 - NY FOREIGN QUALIFICATIONS (Secretary of State filings)
#   data.ny.gov Socrata dataset 63wc-4exh (verified live 2026-07-03; free,
#   keyless, daily). A company filing an APPLICATION OF AUTHORITY is legally
#   registering to do business in New York - a deterministic expansion
#   signal that precedes local hiring by weeks (commercially validated:
#   KYB vendors sell exactly this feed). NY+CA are the only free states;
#   the CA half (calicodev.sos.ca.gov) waits on a registered API key.
#   Docs use the WARN pattern: sub-40-char text -> never embedded ->
#   invisible to scoring; the review queue reads them by source+extra.
# =========================================================================

_SOS_NY_URL = "https://data.ny.gov/resource/63wc-4exh.json"


def _sos_display_name(raw: str) -> str:
    """'HEARTWARD AI LLC' -> 'Heartward AI LLC' (keep initialisms readable)."""
    words = []
    for w in (raw or "").split():
        words.append(w if w in {"LLC", "INC.", "INC", "LP", "LLP", "PC",
                                "AI", "USA", "II", "III", "IV"} or len(w) <= 2
                    else w.title())
    return " ".join(words)


async def ingest_sos_ny(session: aiohttp.ClientSession, cfg: dict) -> list:
    lookback = int(cfg.get("sos_ny_lookback_days", 7))
    since = (datetime.now(timezone.utc)
             - timedelta(days=lookback)).strftime("%Y-%m-%dT00:00:00")
    params = {
        "$where": (f"date_filed > '{since}' AND entitytype like 'FOREIGN%' "
                   f"AND documenttype = 'APPLICATION OF AUTHORITY'"),
        "$order": "date_filed DESC",
        "$limit": str(int(cfg.get("sos_ny_max", 300))),
    }
    try:
        rows = await _get_json(session, _SOS_NY_URL, params=params)
    except Exception as e:
        print(f"[SoS-NY] fetch failed: {e}")
        return []
    docs = []
    for r in rows or []:
        name = (r.get("corp_name") or "").strip()
        film = r.get("film_num")
        if not name or not film:
            continue
        docs.append({
            "doc_id": _doc_id("sos_ny", film),
            "source": "sos_ny",
            "org": _sos_display_name(name),
            "date": (r.get("date_filed") or "")[:10] or None,
            "title": (f"NY expansion filing: {_sos_display_name(name)} "
                      f"(from {r.get('juris') or '?'})")[:200],
            "url": ("https://data.ny.gov/Economic-Development/"
                    "Corporations-and-Other-Entities-All-Filings/63wc-4exh"),
            "text": "NY foreign qualification",
            "contacts": {},
            "extra": {"juris": r.get("juris"),
                      "entitytype": r.get("entitytype"),
                      "county": r.get("cnty_prin_ofc"),
                      "film_num": film},
        })
    print(f"[SoS-NY] {len(docs)} foreign qualifications since {since[:10]}")
    return docs


# =========================================================================
# ALPHA SOURCE 12 - COLORADO BUSINESS ENTITIES (expansion signal #2)
#   data.colorado.gov Socrata dataset 4ykn-tg5h (verified live 2026-07-03,
#   via deep-research 4; 1.7M records, daily refresh, keyless). Filter:
#   out-of-state jurisdiction of formation = a foreign company registering
#   to operate in Colorado - Boulder/Denver is the US quantum-computing
#   hub (Quantinuum, Atom Computing), so this watches Liam's quantum lane
#   the way the NY feed watches finance/techbio. Same invisible-doc
#   pattern (sub-40-char text, never embedded).
# =========================================================================

_SOS_CO_URL = "https://data.colorado.gov/resource/4ykn-tg5h.json"


async def ingest_sos_co(session: aiohttp.ClientSession, cfg: dict) -> list:
    lookback = int(cfg.get("sos_co_lookback_days", 7))
    since = (datetime.now(timezone.utc)
             - timedelta(days=lookback)).strftime("%Y-%m-%dT00:00:00")
    params = {
        "$where": (f"entityformdate > '{since}' "
                   f"AND jurisdictonofformation != 'CO' "
                   f"AND entitystatus = 'Good Standing'"),
        "$order": "entityformdate DESC",
        "$limit": str(int(cfg.get("sos_co_max", 300))),
    }
    try:
        rows = await _get_json(session, _SOS_CO_URL, params=params)
    except Exception as e:
        print(f"[SoS-CO] fetch failed: {e}")
        return []
    docs = []
    for r in rows or []:
        name = (r.get("entityname") or "").strip()
        eid = r.get("entityid")
        if not name or not eid:
            continue
        docs.append({
            "doc_id": _doc_id("sos_co", eid),
            "source": "sos_co",
            "org": _sos_display_name(name),
            "date": (r.get("entityformdate") or "")[:10] or None,
            "title": (f"CO expansion filing: {_sos_display_name(name)} "
                      f"(from {r.get('jurisdictonofformation') or '?'}, "
                      f"{r.get('principalcity') or '?'})")[:200],
            "url": ("https://data.colorado.gov/Business/Business-Entities-in-"
                    "Colorado/4ykn-tg5h"),
            "text": "CO foreign registration",
            "contacts": {},
            "extra": {"juris": r.get("jurisdictonofformation"),
                      "entitytype": r.get("entitytype"),
                      "city": r.get("principalcity"),
                      "entity_id": eid},
        })
    print(f"[SoS-CO] {len(docs)} out-of-state registrations since {since[:10]}")
    return docs


# =========================================================================
# TOP-LEVEL ORCHESTRATOR (called by js_main.py --stage ingest)
# =========================================================================

async def run_all_ingestion(output_dir: str, backfill: bool = False) -> str:
    """
    Fires every alpha-source engine concurrently, merges results, and writes
    a single raw ingest payload to Lustre. Returns the payload path.

    backfill=True is the one-time deep-history sweep: the same engines run
    with wider windows and pagination so per-org δ-shift trajectories are
    dense on day one instead of day 30. Sources with no queryable history
    (ATS boards, RemoteOK) and the WAF-blocked SBIR endpoint are skipped.
    Idempotent - doc_ids are stable, so re-running only adds what's missing.
    """
    cfg = load_config()
    if backfill:
        cfg = {**cfg,
               "lookback_days":       int(cfg.get("backfill_lookback_days", 1100)),
               "max_per_source":      int(cfg.get("backfill_max_per_source", 2000)),
               "nsf_pages":           int(cfg.get("backfill_nsf_pages", 12)),
               "nih_page_limit":      500,
               "arxiv_depth":         int(cfg.get("backfill_arxiv_depth", 300)),
               "hn_threads":          int(cfg.get("backfill_hn_threads", 12)),
               "formd_lookback_days": int(cfg.get("backfill_formd_days", 45)),
               "formd_max_filings":   int(cfg.get("backfill_formd_max", 5000)),
               "usaspending_pages":   int(cfg.get("backfill_usaspending_pages", 3)),
               "ct_lookback_days":    int(cfg.get("backfill_ct_lookback_days", 365)),
               "ct_pages":            int(cfg.get("backfill_ct_pages", 5)),
               "patents_lookback_days": int(cfg.get("backfill_patents_days", 1100)),
               "patents_per_query":   int(cfg.get("backfill_patents_per_query", 200))}
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    from ats_engine import ingest_ats_boards, ingest_remoteok
    from formd_engine import ingest_formd
    from tenk_engine import ingest_tenk

    engines = [
        ("nsf", ingest_nsf), ("nih", ingest_nih), ("sbir", ingest_sbir),
        ("arxiv", ingest_arxiv), ("yc", ingest_yc), ("hn_hiring", ingest_hn_hiring),
        ("ats_jobs", ingest_ats_boards), ("remoteok", ingest_remoteok),
        ("formd", ingest_formd),
        ("usaspending", ingest_usaspending),
        ("clinicaltrials", ingest_clinicaltrials),
        ("usajobs", ingest_usajobs),
        ("patents", ingest_patents),
        ("tenk", ingest_tenk),
        ("warn", ingest_warn),
        ("sos_ny", ingest_sos_ny),
        ("sos_co", ingest_sos_co),
    ]
    if backfill:
        # usajobs: DatePosted caps at 60 days - no deep history to sweep.
        # tenk is budget-capped and self-backfilling (2 filings/org) - skip.
        # warn: free tier is 25 records/call - no budget for deep history.
        # sos_ny/sos_co: an old expansion filing is a stale signal - skip.
        skip = {"sbir", "ats_jobs", "remoteok", "usajobs", "tenk", "warn",
                "sos_ny", "sos_co"}
        engines = [(n, fn) for n, fn in engines if n not in skip]

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[fn(session, cfg) for _, fn in engines],
            return_exceptions=True,
        )
        # Corpus-side retry of 'X group' orgs whose OpenAlex lookup missed
        # while the paper was too fresh to be indexed (fail-soft).
        try:
            await backfill_group_affiliations(
                session, cap=int(cfg.get("arxiv_affil_backfill", 40)))
        except Exception as e:
            print(f"[arXiv] affiliation backfill failed: {e}")

    source_names = [n for n, _ in engines]
    docs, status = [], {}
    for name, result in zip(source_names, results):
        if isinstance(result, Exception):
            status[name] = f"FAILED: {result}"
        elif not result:
            # 0 docs is never a healthy nightly outcome for these sources -
            # per-request errors are swallowed inside the engines, so surface it
            status[name] = "EMPTY (0 docs - check per-query errors in the log)"
        else:
            status[name] = f"OK ({len(result)} docs)"
            docs.extend(result)

    payload = {
        "timestamp": timestamp,
        "backfill": backfill,
        "config": {k: v for k, v in cfg.items()},
        "source_status": status,
        "n_docs": len(docs),
        "docs": docs,
    }
    path = os.path.join(output_dir, f"ingest_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(payload, f)

    tag = "backfill" if backfill else "nightly"
    print(f"[Ingest] ({tag}) {len(docs)} total documents -> {path}")
    for name, s in status.items():
        print(f"  {name:10s} {s}")
    return path


def latest_ingest_payload(output_dir: str):
    """Newest ingest payload path, or None."""
    if not os.path.isdir(output_dir):
        return None
    files = sorted(f for f in os.listdir(output_dir) if f.startswith("ingest_") and f.endswith(".json"))
    return os.path.join(output_dir, files[-1]) if files else None
