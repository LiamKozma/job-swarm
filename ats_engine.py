"""
Job Swarm - ATS Shadow Board Engine (Alpha Source 4).

Most companies never post to LinkedIn/Indeed; they post to their own careers
page, which is overwhelmingly powered by a handful of providers that all
expose public, unauthenticated JSON APIs:

    Greenhouse       GET boards-api.greenhouse.io/v1/boards/<slug>/jobs?content=true
    Lever            GET api.lever.co/v0/postings/<slug>?mode=json
    Ashby            GET api.ashbyhq.com/posting-api/job-board/<slug>
    SmartRecruiters  GET api.smartrecruiters.com/v1/companies/<slug>/postings
    Workable         GET apply.workable.com/api/v1/widget/accounts/<slug>?details=true
    Recruitee        GET <slug>.recruitee.com/api/offers/
    Workday          POST <tenant>.<wd>.myworkdayjobs.com/wday/cxs/<tenant>/<site>/jobs

Two feeder streams decide which slugs to hit:
  1. A validated seed list (quant funds, AI labs, biotech, top-of-market tech -
     confirmed live 2026-07-01, several thousand postings).
  2. Every org the swarm's other alpha sources have ever surfaced: its name is
     slugified and probed against all three providers ONCE, with the outcome
     cached in SQLite (ats_boards), so discovery cost amortizes to zero.

Postings are filtered by title-relevance terms (statistics/quant/ML/research),
salary is extracted when disclosed (Ashby structured comp, Lever ranges, $-regex
otherwise), and each posting becomes a corpus doc - embedded and ranked against
the candidate profile like everything else, then surfaced in the review queue's
"Direct openings" section.

RemoteOK's public JSON feed rides along as a fourth stream for remote roles.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

import aiohttp

import swarm_db

# =========================================================================
# VALIDATED SEED BOARDS (probed live 2026-07-01; misses cost one 404)
# =========================================================================

SEED_BOARDS = {
    "greenhouse": [
        # Quant / prop trading (highest starting comp; brutal interviews+hours)
        "imc", "akunacapital", "jumptrading", "towerresearchcapital",
        "optiverus", "squarepointcapital", "point72", "virtu", "drw",
        "hudsonrivertrading", "radixtrading", "xtxmarkets", "gresearch",
        "twosigma", "citadel", "janestreet", "sig",
        # Quant additions (probe candidates - a miss costs one 404)
        "fiverings", "chicagotradingcompany", "flowtraders", "schonfeld",
        "headlandstech",
        # AI labs & ML infrastructure
        "anthropic", "databricks", "scaleai", "openai",
        # ML observability / eval - the single best employer class for a
        # thesis on LLM performance degradation under distribution shift
        "wandb", "arizeai",
        # Biotech / pharma statistics (strong pay, sane hours, MS-Stats demand)
        "recursionpharmaceuticals", "10xgenomics", "modernatx", "tempus",
        "insitro", "benchling", "grailbio",
        # TechBio neolabs (2026 deep-research recon: exact boards confirmed;
        # freshly funded, hire MS-level, PLM paper is the wedge)
        "isomorphiclabs", "profluent", "lilasciences",
        # Top-of-market tech
        "duolingo", "figma", "stripe",
    ],
    "lever": ["palantir", "mistral", "plaid", "voleon", "pdtpartners",
              # ML observability probe candidates
              "fiddler", "rungalileo", "whylabs"],
    "ashby": ["ramp", "notion", "openai", "cursor", "linear", "sierra",
              # LLM-eval probe candidates
              "patronusai", "braintrust",
              # TechBio neolabs (recon-confirmed Ashby boards)
              "chaidiscovery", "insitro", "genesis-molecular-ai",
              "cradlebio", "evolutionaryscale"],
    # New providers: no seeds yet - corpus orgs are slug-probed against them
    "smartrecruiters": [],
    "workable": [],
    "recruitee": [],
    # Workday runs almost all of big pharma, the FFRDCs, and legacy finance -
    # the exact employer class the grant/trial sources score but that
    # Greenhouse-style boards never see. Slug format: 'tenant:wdhost:site'
    # (all triples validated live 2026-07-02; probing can't guess them, so
    # Workday is seeds-only).
    "workday": [
        "novartis:wd3:Novartis_Careers",
        "pfizer:wd1:PfizerCareers",
        "bristolmyerssquibb:wd5:BMS",
        "msd:wd5:SearchJobs",                    # Merck
        "astrazeneca:wd3:Careers",
        "sanofi:wd3:SanofiCareers",
        "takeda:wd3:External",
        "amgen:wd1:Careers",
        "regeneron:wd1:Careers",
        "gilead:wd1:gileadcareers",
        "rand:wd5:External_Career_Site",         # FFRDC - strong WLB channel
        "stjude:wd1:STJUDE",
        "capitalone:wd12:Capital_One",
        "vanguard:wd5:vanguard_external",
        "mastercard:wd1:CorporateCareers",
    ],
}

# Title must contain one of these (lowercase substring) to enter the corpus.
DEFAULT_TITLE_TERMS = [
    "statist", "quant", "machine learning", " ml", "ml ", "data scien",
    "research", "applied scien", "algorithm", "model", "inference",
    "data engineer", "analytics", "actuar", "biostat", "simulation", "ai ",
]

# Money tokens: "$216,000", "$150k", "$380K", "1500000". The K-suffix branch
# matters at the very top of the market - Ashby comp summaries (OpenAI,
# Cursor, ...) are "$293K - $385K" and the old parser silently dropped them,
# biasing every comp statistic downward.
_MONEY_RE = re.compile(
    r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{2,3}(?:\.\d+)?\s?[kK](?![a-zA-Z])|\d{4,6})")
# Bare "150k-200k" ranges ($-less), common in HN Who-is-Hiring posts.
_BARE_RANGE_RE = re.compile(
    r"(?<![\w$.])(\d{2,3})\s?[kK]\s?[---]\s?\$?\s?(\d{2,3})\s?[kK](?![a-zA-Z])")

_PROBE_SEMAPHORE = asyncio.Semaphore(8)
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) JobSwarmResearch/1.0",
            "Accept": "application/json"}


def _doc_id(provider: str, native: str) -> str:
    return hashlib.sha1(f"ats:{provider}:{native}".encode()).hexdigest()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _money_to_num(tok: str) -> float:
    tok = tok.replace(",", "").strip()
    if tok and tok[-1] in "kK":
        return float(tok[:-1]) * 1000.0
    return float(tok)


def salary_values(text) -> list:
    """All plausible annual salary figures in a string, as floats.
    Also used by ghost_engine for archetype medians and comp quartiles."""
    vals = []
    for m in _MONEY_RE.finditer(str(text or "")):
        v = _money_to_num(m.group(1))
        # 401(k) mentions parse as 401000 - never a salary; genuine $401,000
        # offers are rare enough to sacrifice.
        if 30_000 <= v <= 2_000_000 and v != 401_000:
            vals.append(v)
        if len(vals) >= 4:
            break
    if not vals:
        m = _BARE_RANGE_RE.search(str(text or ""))
        if m:
            lo, hi = float(m.group(1)) * 1000, float(m.group(2)) * 1000
            if 30_000 <= lo <= 2_000_000 and hi >= lo:
                vals = [lo, hi]
    return vals


def _extract_salary(text: str):
    """Best-effort salary range from free text; returns 'low - high' string or None."""
    vals = salary_values(text)[:2]
    if not vals:
        return None
    lo = vals[0]
    hi = vals[1] if len(vals) > 1 and vals[1] > vals[0] else None
    return f"${lo:,.0f}" + (f" - ${hi:,.0f}" if hi else "")


async def _fetch(session: aiohttp.ClientSession, url: str):
    async with _PROBE_SEMAPHORE:
        await asyncio.sleep(0.3)
        async with session.get(url, headers=_HEADERS) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            return await resp.json(content_type=None)


# =========================================================================
# PER-PROVIDER FETCHERS - normalize to a common posting dict
# =========================================================================

async def fetch_greenhouse(session, slug: str):
    data = await _fetch(session, f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true")
    if data is None:
        return None
    postings = []
    for j in data.get("jobs", []):
        content = re.sub(r"<[^>]+>", " ", j.get("content") or "")
        postings.append({
            "native_id": f"gh:{slug}:{j.get('id')}",
            "title": j.get("title") or "",
            "location": (j.get("location") or {}).get("name"),
            "url": j.get("absolute_url"),
            # first_published, NOT updated_at: companies touch postings
            # constantly, and every touch was resetting the staleness clock -
            # the hard-to-fill signal and ghost-age heuristics both need
            # the TRUE original publish date (verified live 2026-07-03).
            "date": (j.get("first_published") or j.get("updated_at") or "")[:10] or None,
            "updated_at": (j.get("updated_at") or "")[:10] or None,
            "requisition_id": j.get("requisition_id"),
            "description": content[:6000],
            "salary": _extract_salary(content),
        })
    return postings


async def fetch_lever(session, slug: str):
    data = await _fetch(session, f"https://api.lever.co/v0/postings/{slug}?mode=json")
    if data is None or not isinstance(data, list):
        return None
    postings = []
    for j in data:
        desc = j.get("descriptionPlain") or ""
        salary = None
        sr = j.get("salaryRange") or {}
        if sr.get("min") and sr.get("max"):
            salary = f"${int(sr['min']):,} - ${int(sr['max']):,}"
        created = j.get("createdAt")
        postings.append({
            "native_id": f"lv:{slug}:{j.get('id')}",
            "title": j.get("text") or "",
            "location": (j.get("categories") or {}).get("location"),
            "url": j.get("hostedUrl"),
            "date": datetime.fromtimestamp(created / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if created else None,
            "description": desc[:6000],
            "salary": salary or _extract_salary(desc),
        })
    return postings


async def fetch_ashby(session, slug: str):
    data = await _fetch(
        session, f"https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true")
    if data is None:
        return None
    postings = []
    for j in data.get("jobs", []):
        desc = re.sub(r"<[^>]+>", " ", j.get("descriptionHtml") or j.get("descriptionPlain") or "")
        comp = j.get("compensation") or {}
        salary = (comp.get("compensationTierSummary")
                  or comp.get("scrapeableCompensationSalarySummary")
                  or _extract_salary(desc))
        postings.append({
            "native_id": f"ab:{slug}:{j.get('id')}",
            "title": j.get("title") or "",
            "location": j.get("location"),
            "url": j.get("jobUrl") or j.get("applyUrl"),
            "date": (j.get("publishedAt") or "")[:10] or None,
            "description": desc[:6000],
            "salary": salary,
        })
    return postings


async def fetch_smartrecruiters(session, slug: str):
    """SmartRecruiters public postings API. The list endpoint has no
    descriptions; details are fetched one level deeper for title-relevant
    postings only (capped) to keep probe cost sane."""
    data = await _fetch(session,
                        f"https://api.smartrecruiters.com/v1/companies/{slug}/postings?limit=100")
    if not isinstance(data, dict):
        return None
    content = data.get("content") or []
    if not content:
        return None
    title_terms = [t.lower() for t in DEFAULT_TITLE_TERMS]
    postings, details_fetched = [], 0
    for j in content[:100]:
        title = j.get("name") or ""
        desc = ""
        if details_fetched < 25 and any(t in title.lower() for t in title_terms):
            details_fetched += 1
            try:
                det = await _fetch(
                    session,
                    f"https://api.smartrecruiters.com/v1/companies/{slug}/postings/{j.get('id')}")
                sections = ((det or {}).get("jobAd") or {}).get("sections") or {}
                desc = " ".join(
                    re.sub(r"<[^>]+>", " ", (sections.get(k) or {}).get("text") or "")
                    for k in ("jobDescription", "qualifications", "additionalInformation"))
            except Exception:
                desc = ""
        loc = j.get("location") or {}
        postings.append({
            "native_id": f"sr:{slug}:{j.get('id')}",
            "title": title,
            "location": ", ".join(filter(None, [loc.get("city"), loc.get("country")])) or None,
            "url": f"https://jobs.smartrecruiters.com/{slug}/{j.get('id')}",
            "date": (j.get("releasedDate") or "")[:10] or None,
            "description": desc[:6000],
            "salary": _extract_salary(desc),
        })
    return postings


async def fetch_workable(session, slug: str):
    """Workable public widget API (details=true carries descriptions)."""
    data = await _fetch(
        session, f"https://apply.workable.com/api/v1/widget/accounts/{slug}?details=true")
    if not isinstance(data, dict):
        return None
    jobs = data.get("jobs") or []
    if not jobs:
        return None
    postings = []
    for j in jobs[:100]:
        desc = re.sub(r"<[^>]+>", " ", j.get("description") or "")
        postings.append({
            "native_id": f"wk:{slug}:{j.get('shortcode') or j.get('id')}",
            "title": j.get("title") or "",
            "location": ", ".join(filter(None, [j.get("city"), j.get("country")])) or None,
            "url": j.get("url") or j.get("shortlink"),
            "date": (j.get("published_on") or j.get("created_at") or "")[:10] or None,
            "description": desc[:6000],
            "salary": _extract_salary(desc),
        })
    return postings


async def fetch_recruitee(session, slug: str):
    """Recruitee public careers API ({slug}.recruitee.com/api/offers/)."""
    data = await _fetch(session, f"https://{slug}.recruitee.com/api/offers/")
    if not isinstance(data, dict):
        return None
    offers = data.get("offers") or []
    if not offers:
        return None
    postings = []
    for j in offers[:100]:
        desc = re.sub(r"<[^>]+>", " ",
                      (j.get("description") or "") + " " + (j.get("requirements") or ""))
        postings.append({
            "native_id": f"rc:{slug}:{j.get('id')}",
            "title": j.get("title") or "",
            "location": j.get("location") or j.get("city"),
            "url": j.get("careers_url") or j.get("careers_apply_url"),
            "date": (j.get("published_at") or j.get("created_at") or "")[:10] or None,
            "description": desc[:6000],
            "salary": _extract_salary(desc),
        })
    return postings


async def _wd_post(session, url: str, payload: dict):
    async with _PROBE_SEMAPHORE:
        await asyncio.sleep(0.3)
        async with session.post(url, json=payload, headers=_HEADERS) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            return await resp.json(content_type=None)


_WD_RELATIVE_RE = re.compile(r"Posted\s+(\d+)\+?\s+Days?\s+Ago", re.I)

# Workday boards carry thousands of postings (Novartis alone ~4k), so instead
# of paging the whole board we run its native full-text search with the
# candidate-relevant vocabulary and union the results.
_WD_SEARCHES = ["statistician", "biostatistics", "machine learning",
                "data scientist", "quantitative"]


def _wd_posted_to_iso(posted: str):
    """'Posted 7 Days Ago' -> ISO date (approx; '30+' floors at 30)."""
    p = (posted or "").strip().lower()
    if p == "posted today":
        days = 0
    elif p == "posted yesterday":
        days = 1
    else:
        m = _WD_RELATIVE_RE.search(posted or "")
        if not m:
            return None
        days = int(m.group(1))
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")


async def fetch_workday(session, slug: str):
    """Workday CXS JSON API (validated live 2026-07-02). slug is
    'tenant:wdhost:site'; descriptions come from a per-job detail call,
    fetched for title-relevant postings only (capped)."""
    try:
        tenant, host, site = slug.split(":")
    except ValueError:
        return None
    base = f"https://{tenant}.{host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}"
    title_terms = [t.lower() for t in DEFAULT_TITLE_TERMS]
    seen_paths = {}
    for search in _WD_SEARCHES:
        offset = 0
        while offset < 60:
            data = await _wd_post(session, f"{base}/jobs",
                                  {"appliedFacets": {}, "limit": 20,
                                   "offset": offset, "searchText": search})
            if data is None:
                return None if not seen_paths else list(seen_paths.values())
            jobs = data.get("jobPostings") or []
            for j in jobs:
                path = j.get("externalPath")
                if path and path not in seen_paths:
                    seen_paths[path] = j
            offset += 20
            if offset >= int(data.get("total") or 0) or not jobs:
                break
    postings, details_fetched = [], 0
    for path, j in seen_paths.items():
        title = j.get("title") or ""
        desc, date, url = "", _wd_posted_to_iso(j.get("postedOn")), None
        if details_fetched < 25 and any(t in title.lower() for t in title_terms):
            details_fetched += 1
            try:
                det = await _fetch(session, f"{base}{path}")
                info = (det or {}).get("jobPostingInfo") or {}
                desc = re.sub(r"<[^>]+>", " ", info.get("jobDescription") or "")
                date = (info.get("startDate") or "")[:10] or date
                url = info.get("externalUrl")
            except Exception:
                pass
        postings.append({
            "native_id": f"wd:{tenant}:{path}",
            "title": title,
            "location": j.get("locationsText"),
            "url": url or f"https://{tenant}.{host}.myworkdayjobs.com/{site}{path}",
            "date": date,
            "description": desc[:6000],
            "salary": _extract_salary(desc),
        })
    return postings


# Every fetch_* posting dict must carry these; enforced at the consumption site.
_REQUIRED_POSTING_KEYS = {"native_id", "title", "location", "url",
                          "date", "description", "salary"}

_FETCHERS = {"greenhouse": fetch_greenhouse, "lever": fetch_lever,
             "ashby": fetch_ashby, "smartrecruiters": fetch_smartrecruiters,
             "workable": fetch_workable, "recruitee": fetch_recruitee,
             "workday": fetch_workday}

# Providers whose board ids a slugified org name can actually hit. Workday
# needs a tenant:host:site triple no slug heuristic can produce - probing it
# would burn a request per org for a guaranteed miss.
_PROBEABLE = {p: f for p, f in _FETCHERS.items() if p != "workday"}


def _board_org_name(provider: str, slug: str) -> str:
    """Org display name for a board slug ('novartis:wd3:...' -> 'novartis')."""
    return slug.split(":")[0] if provider == "workday" else slug


# Evergreen/pipeline requisitions: resume-harvesting funnels, not real roles.
# ~30% of postings are ghost-ish (2026 deep-research briefing); these titles
# are the unambiguous tell.
_EVERGREEN_TITLE_RE = re.compile(
    r"talent (community|network|pool|pipeline)|general (application|interest)|"
    r"future (opportunit|opening)|evergreen|prospective", re.I)


def _ghost_flags(posting: dict) -> list:
    """Ghost-job heuristics computable at ingest (2026 research-validated):
    wide salary band (spread >50% = market-testing/evergreen), evergreen
    title, unmaintained posting (last content update >45d ago). Flags are
    ADVISORY - stored in extra, rendered for the human, and available to the
    repost-forensics scoring pass; nothing is silently dropped."""
    flags = []
    vals = salary_values(posting.get("salary"))[:2]
    if len(vals) == 2 and vals[0] > 0 and (vals[1] - vals[0]) / vals[0] > 0.50:
        flags.append("wide_salary_band")
    if _EVERGREEN_TITLE_RE.search(posting.get("title") or ""):
        flags.append("evergreen_title")
    upd = posting.get("updated_at")
    if upd:
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.strptime(upd, "%Y-%m-%d").replace(tzinfo=timezone.utc)).days
            if age > 45:
                flags.append("unmaintained_45d")
        except ValueError:
            pass
    # PERM labor-market-test ads (visa sponsorship compliance) are pre-filled
    # by the sponsored worker - near-zero external hire probability. "or
    # foreign equivalent" is the boilerplate tell (20 CFR 656.17 ads must
    # state objective degree requirements this way).
    if re.search(r"or\s+foreign\s+(degree\s+)?equivalent",
                 posting.get("description") or "", re.I):
        flags.append("perm_language")
    return flags


# Clearance-SPONSORSHIP tell: "ability to obtain" (vs "active clearance
# required") means the employer will sponsor a first clearance - for a US
# citizen this is the least-competitive high-pay lane there is (2026
# research: cleared MS-level ML runs $120k-$200k with a fraction of the
# applicants; Tier 3 processes in ~138-156 days under TW 2.0).
_CLEARANCE_SPONSOR_RE = re.compile(
    r"\b(ability|able|eligible|eligibility|willing|willingness)\s+to\s+obtain"
    r"\b.{0,80}?(security\s+clearance|clearance)", re.I)


# =========================================================================
# BOARD DISCOVERY - probe corpus orgs once, cache forever
# =========================================================================

async def _probe_org(session, org: dict, conn) -> list:
    """Tries all three providers for one unprobed org. Returns resolved boards."""
    slug = _slugify(org["display_name"])
    if len(slug) < 3:
        for provider in _PROBEABLE:
            swarm_db.store_ats_resolution(conn, org["org_key"], provider, slug, "miss")
        conn.commit()
        return []
    resolved = []
    for provider, fetcher in _PROBEABLE.items():
        try:
            postings = await fetcher(session, slug)
        except Exception:
            postings = None  # transient error -> recorded as miss; cheap to re-add later
        status = "resolved" if postings else "miss"
        swarm_db.store_ats_resolution(conn, org["org_key"], provider, slug, status)
        # Commit per write: engines run concurrently on ONE SQLite file, and a
        # write transaction held open across the next provider's network fetch
        # starves every other engine's writes ("database is locked" killed the
        # whole arXiv source on the 2026-07-02 nightly).
        conn.commit()
        if postings:
            resolved.append({"org_key": org["org_key"], "provider": provider, "slug": slug})
    return resolved


# =========================================================================
# TOP-LEVEL ENGINE (called from run_all_ingestion)
# =========================================================================

async def ingest_ats_boards(session: aiohttp.ClientSession, cfg: dict) -> list:
    title_terms = [t.lower() for t in cfg.get("ats_title_terms", DEFAULT_TITLE_TERMS)]
    probe_budget = int(cfg.get("ats_probe_per_night", 120))
    max_per_board = int(cfg.get("ats_max_per_board", 40))

    conn = swarm_db.connect()

    # ---- Ensure seed boards are registered --------------------------------
    for provider, slugs in SEED_BOARDS.items():
        for slug in slugs:
            org_name = _board_org_name(provider, slug)
            org_key = swarm_db.org_key_from_name(org_name)
            existing = swarm_db.ats_resolution(conn, org_key)
            if provider not in existing:
                swarm_db.touch_org(conn, org_name, "ats_seed")
                swarm_db.store_ats_resolution(conn, org_key, provider, slug, "resolved")
    conn.commit()

    # ---- Register conference-sponsor orgs as probe candidates -------------
    # NeurIPS/ICML sponsorship = an ML org with discretionary budget. Names
    # from config flow through touch_org into the normal slug-probe rotation,
    # so each costs one probe ever and resolved boards persist.
    for name in cfg.get("sponsor_orgs", []):
        if not conn.execute(
                "SELECT 1 FROM orgs WHERE org_key = ?",
                (swarm_db.org_key_from_name(name),)).fetchone():
            swarm_db.touch_org(conn, name, "sponsor_seed")
    conn.commit()

    # ---- Probe a rotating batch of never-checked corpus orgs --------------
    to_probe = swarm_db.unprobed_orgs(conn, probe_budget)
    newly_resolved = []
    for org in to_probe:
        newly_resolved.extend(await _probe_org(session, org, conn))
    conn.commit()
    print(f"[ATS] probed {len(to_probe)} orgs, {len(newly_resolved)} new boards resolved")

    # ---- Pull postings from every resolved board ---------------------------
    boards = swarm_db.resolved_boards(conn)
    docs = []
    seen_native = set()
    for board in boards:
        fetcher = _FETCHERS.get(board["provider"])
        if fetcher is None:
            continue
        try:
            postings = await fetcher(session, board["slug"]) or []
        except Exception as e:
            print(f"[ATS] {board['provider']}/{board['slug']} fetch failed: {e}")
            continue
        # A fetch_* that drops a key must not KeyError the whole engine
        # (2026-07-03: an edit to fetch_greenhouse lost "url" and killed the run).
        for p in postings:
            missing = _REQUIRED_POSTING_KEYS - p.keys()
            if missing:
                print(f"[ATS] WARNING {board['provider']}/{board['slug']}: "
                      f"posting missing {sorted(missing)} - filled with None")
                for k in missing:
                    p[k] = None
        postings = [p for p in postings if p["native_id"] and p["title"]]
        n_relevant_total = sum(
            1 for p in postings if any(t in p["title"].lower() for t in title_terms))
        swarm_db.record_board_snapshot(conn, board["org_key"], len(postings), n_relevant_total)
        conn.commit()   # keep write txns ms-scale - next board fetch is network
        kept = 0
        for p in postings:
            title_low = p["title"].lower()
            if not any(t in title_low for t in title_terms):
                continue
            if p["native_id"] in seen_native or kept >= max_per_board:
                continue
            seen_native.add(p["native_id"])
            kept += 1
            gflags = _ghost_flags(p)
            sponsors_clearance = bool(
                _CLEARANCE_SPONSOR_RE.search(p.get("description") or ""))
            docs.append({
                "doc_id": _doc_id(board["provider"], p["native_id"]),
                "source": "ats_jobs",
                "org": _board_org_name(board["provider"], board["slug"]),
                "date": p["date"],
                "title": f"{p['title']} - {p['location'] or 'location n/a'}",
                "url": p["url"],
                "text": f"{p['title']}. {p['description']}",
                "contacts": {"apply_url": p["url"],
                             "contact_basis": "public job posting on the company's own careers board"},
                "extra": {"provider": board["provider"], "board": board["slug"],
                          "location": p["location"], "salary": p["salary"],
                          "posting": True,
                          **({"updated_at": p["updated_at"]}
                             if p.get("updated_at") else {}),
                          **({"ghost_flags": gflags} if gflags else {}),
                          **({"clearance_sponsor": True}
                             if sponsors_clearance else {})},
            })
    conn.commit()
    conn.close()
    print(f"[ATS] {len(docs)} relevant postings across {len(boards)} boards")
    return docs


# =========================================================================
# REMOTEOK - public JSON feed of remote roles
# =========================================================================

async def ingest_remoteok(session: aiohttp.ClientSession, cfg: dict) -> list:
    title_terms = [t.lower() for t in cfg.get("ats_title_terms", DEFAULT_TITLE_TERMS)]
    try:
        data = await _fetch(session, "https://remoteok.com/api")
    except Exception as e:
        print(f"[RemoteOK] fetch failed: {e}")
        return []
    docs = []
    for j in (data or []):
        if not isinstance(j, dict) or not j.get("position"):
            continue  # first element is a legal notice
        title_low = j["position"].lower()
        if not any(t in title_low for t in title_terms):
            continue
        desc = re.sub(r"<[^>]+>", " ", j.get("description") or "")
        salary = None
        if j.get("salary_min") and j.get("salary_max"):
            salary = f"${int(j['salary_min']):,} - ${int(j['salary_max']):,}"
        docs.append({
            "doc_id": _doc_id("remoteok", str(j.get("id"))),
            "source": "remoteok",
            "org": j.get("company") or "unknown",
            "date": (j.get("date") or "")[:10] or None,
            "title": f"{j['position']} - REMOTE ({j.get('company')})",
            "url": j.get("url"),
            "text": f"{j['position']}. {desc[:6000]}",
            "contacts": {"apply_url": j.get("apply_url") or j.get("url"),
                         "contact_basis": "public job posting"},
            "extra": {"provider": "remoteok", "location": "Remote",
                      "salary": salary or _extract_salary(desc), "posting": True,
                      "tags": j.get("tags")},
        })
    print(f"[RemoteOK] {len(docs)} relevant remote postings")
    return docs
