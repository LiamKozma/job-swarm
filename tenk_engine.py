"""
Job Swarm - 10-K language-shift engine (cross-pollinated from the quant
swarm's edgar_engine, vendored lite so job_swarm deploys standalone).

The thesis, ported from the quant side: a public company whose 10-K risk-
factor language is shifting toward data/ML topics year-over-year is building
a team months before any posting exists. Instead of computing the shift here,
the engine simply INGESTS the Item 1A (Risk Factors) section of the two most
recent 10-Ks as 'tenk' docs - tenk is an AUTHORED source, so the existing
δ-shift GMM regime machinery measures the YoY semantic shift for free, and
the auditor/memo writer see real risk-factor excerpts as evidence. On top,
an explicit ML/data vocabulary count per filing gives the review queue a
human-readable "+N ML/data mentions YoY" line.

Which orgs: any corpus org whose normalized name matches an SEC ticker title
(that match IS the public-company test). CIK resolutions are cached in the
meta table (tenk_cik:<org_key>), so the bulk ticker map is consulted once per
org ever. Nightly budget: tenk_max_orgs (default 5) new orgs per run - each
costs one submissions-index fetch plus two multi-MB filing fetches.

EDGAR rate limit is 10 req/s; requests here ride the caller's session with a
descriptive User-Agent (SEC requirement).
"""

import html
import json
import re
from datetime import datetime

import swarm_db

EDGAR_WWW = "https://www.sec.gov"
EDGAR_DATA = "https://data.sec.gov"

_ML_TERMS = [
    "machine learning", "artificial intelligence", " ai ", "data science",
    "predictive model", "algorithm", "neural network", "large language model",
    "generative ai", "data analytics", "statistical model", "automation",
]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# dash LAST in the class = literal; the old ':--' parsed as a reversed
# range and crashed the first ingest under the Qwen3 container's python 3.11
_ITEM_1A_RE = re.compile(r"item\s*1a[.\s:-]", re.I)
_ITEM_1B_RE = re.compile(r"item\s*1b[.\s:-]|item\s*2[.\s:-]", re.I)

_TEXT_CAP = 25_000          # chars of Item 1A stored per filing (embedding input)
_SECTION_CAP = 150_000      # chars scanned for the ML-term count (full section)
_FRESH_DAYS = 300           # org already has a tenk doc newer than this -> skip


def _strip_html(raw: str) -> str:
    txt = _TAG_RE.sub(" ", raw)
    txt = html.unescape(txt)
    return _WS_RE.sub(" ", txt)


def _extract_item_1a(text: str) -> str:
    """Item 1A slice (full section, up to _SECTION_CAP). The TOC also says
    'Item 1A', so among all candidate starts keep the one with the LONGEST
    span to the next Item 1B/2 marker - that's the real section, not the
    TOC row. The ML-term count runs over this full section; only the stored
    text is capped tighter."""
    best = ""
    for m in _ITEM_1A_RE.finditer(text):
        nxt = _ITEM_1B_RE.search(text, m.end())
        span = text[m.end(): nxt.start()] if nxt else text[m.end(): m.end() + _SECTION_CAP]
        if len(span) > len(best):
            best = span
    return (best or text[:_SECTION_CAP])[:_SECTION_CAP].strip()


def _ml_term_count(text: str) -> int:
    low = f" {text.lower()} "
    return sum(low.count(t) for t in _ML_TERMS)


async def _get_json(session, url: str, headers: dict):
    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        return await resp.json(content_type=None)


async def _get_text(session, url: str, headers: dict) -> str:
    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        return await resp.text(errors="replace")


def _candidate_org_ciks(conn, cik_by_key: dict, cap: int) -> list:
    """(org_key, display_name, cik) for corpus orgs that resolve to a public
    company and have no fresh tenk docs. Misses are cached so each org is
    matched against the ticker map once ever."""
    out = []
    rows = conn.execute(
        "SELECT org_key, display_name FROM orgs "
        "WHERE display_name NOT LIKE '% · %' AND display_name NOT LIKE '% group' "
        f"AND status NOT IN {swarm_db.HANDS_OFF_STATUSES!r} "
        "ORDER BY COALESCE(latest_prescore, 0) DESC"
    ).fetchall()
    for r in rows:
        if len(out) >= cap:
            break
        cached = swarm_db.get_meta(conn, f"tenk_cik:{r['org_key']}")
        if cached == "miss":
            continue
        cik = cached or cik_by_key.get(r["org_key"])
        if cached is None:
            swarm_db.set_meta(conn, f"tenk_cik:{r['org_key']}", cik or "miss")
        if not cik:
            continue
        fresh = conn.execute(
            "SELECT 1 FROM docs WHERE org_key = ? AND source = 'tenk' "
            "AND added_at >= datetime('now', ?)",
            (r["org_key"], f"-{_FRESH_DAYS} days"),
        ).fetchone()
        if fresh:
            continue
        out.append((r["org_key"], r["display_name"], str(cik)))
    return out


async def ingest_tenk(session, cfg: dict) -> list:
    """Alpha-source engine: Item 1A of the two latest 10-Ks for up to
    tenk_max_orgs public corpus orgs per night. Fail-soft per org."""
    from ingest_engines import _doc_id, _USER_AGENT
    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    cap = int(cfg.get("tenk_max_orgs", 5))

    conn = swarm_db.connect()
    try:
        tickers = await _get_json(
            session, f"{EDGAR_WWW}/files/company_tickers.json", headers)
    except Exception as e:
        print(f"[10-K] ticker map fetch failed: {e}")
        conn.close()
        return []
    cik_by_key = {}
    for v in tickers.values():
        k = swarm_db.org_key_from_name(v.get("title") or "")
        cik_by_key.setdefault(k, f"{int(v['cik_str']):010d}")

    targets = _candidate_org_ciks(conn, cik_by_key, cap)
    conn.commit()

    docs = []
    for org_key, display_name, cik in targets:
        try:
            subs = await _get_json(
                session, f"{EDGAR_DATA}/submissions/CIK{cik}.json", headers)
            recent = subs.get("filings", {}).get("recent", {})
            filings = [
                (recent["filingDate"][i], recent["accessionNumber"][i],
                 recent["primaryDocument"][i])
                for i, form in enumerate(recent.get("form", []))
                if form == "10-K"
            ][:2]
            if len(filings) < 2:
                continue  # need two years for a shift
            counts = []
            for fdate, accession, primary in filings:
                acc = accession.replace("-", "")
                url = (f"{EDGAR_WWW}/Archives/edgar/data/{int(cik)}/{acc}/{primary}")
                raw = await _get_text(session, url, headers)
                item_1a = _extract_item_1a(_strip_html(raw))
                n_ml = _ml_term_count(item_1a)   # counted over the FULL section
                counts.append(n_ml)
                docs.append({
                    "doc_id": _doc_id("tenk", f"{cik}:{accession}"),
                    "source": "tenk",
                    "org": display_name,
                    "date": fdate,
                    "title": f"10-K Risk Factors ({fdate}) - {display_name}",
                    "url": url,
                    "text": item_1a[:_TEXT_CAP],
                    "contacts": {},
                    "extra": {"cik": cik, "form": "10-K",
                              "ml_term_count": n_ml},
                })
            if len(counts) == 2:
                # Newest filing carries the YoY delta the dossier prints
                docs[-2]["extra"]["ml_term_delta"] = counts[0] - counts[1]
        except Exception as e:
            print(f"[10-K] {display_name}: {e}")
            continue
    conn.close()
    print(f"[10-K] {len(docs)} risk-factor sections from "
          f"{len(docs) // 2} public corpus orgs")
    return docs
