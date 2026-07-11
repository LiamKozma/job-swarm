"""
Job Swarm - SEC Form D Ingestion Engine (Alpha Source 5).

Every private company raising capital under Regulation D must file a Form D
within 15 days of first sale. That makes the Form D stream the freshest
"this company just got money and will be hiring in 30-90 days" signal in
existence - earlier than TechCrunch, earlier than Crunchbase, and read by
approximately zero job seekers.

Pipeline (reuses the quant swarm's EDGAR conventions: compliant User-Agent,
10 req/s ceiling, semaphore at 8):

  1. Pull the EDGAR daily form index for the last N business days
     (~180 Form Ds/day; validated live 2026-07-01).
  2. Fetch each filing's primary_doc.xml (namespace-free XML).
  3. Keep only relevant industry groups - the raw stream is dominated by
     real-estate LLCs and pooled investment funds, which are noise here.
  4. Emit corpus docs carrying issuer, state, raise size, and the executives
     named in the public federal disclosure.

Form D docs are thin on text, so their embedding alignment is weak by design;
their power is signal-joining - a fresh raise on top of an org already in the
corpus (grant, paper, YC batch) is a strong escalation cue, and the review
queue surfaces the week's relevant raises in their own section.
"""

import asyncio
import hashlib
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import aiohttp

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


_SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT") or (
    "JobSwarmResearch/1.0 " + _contact_email())
_HEADERS = {"User-Agent": _SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_EDGAR_SEMAPHORE = asyncio.Semaphore(8)

# Industry groups worth tracking for a computational statistician.
DEFAULT_INDUSTRY_GROUPS = {
    "Other Technology", "Computers", "Telecommunications",
    "Biotechnology", "Pharmaceuticals", "Other Health Care",
    "Health Insurance", "Hospitals & Physicians",
    "Investing", "Investment Banking", "Other Banking & Financial Services",
    "Insurance", "Energy Conservation", "Other Energy",
}

# Pooled funds and real-estate vehicles never hire statisticians.
_EXCLUDED_GROUPS = {"Pooled Investment Fund", "Other Real Estate",
                    "Residential", "Commercial", "REITS & Finance",
                    "Construction", "Oil & Gas"}


def _doc_id(accession: str) -> str:
    return hashlib.sha1(f"formd:{accession}".encode()).hexdigest()


async def _get(session: aiohttp.ClientSession, url: str) -> bytes:
    async with _EDGAR_SEMAPHORE:
        await asyncio.sleep(0.15)
        async with session.get(url, headers=_HEADERS) as resp:
            resp.raise_for_status()
            return await resp.read()


def _quarter(dt: datetime) -> int:
    return (dt.month - 1) // 3 + 1


async def _fetch_daily_form_d(session, day: datetime) -> list:
    """Form D rows from one day's form index. Empty list on weekends/holidays."""
    url = (f"https://www.sec.gov/Archives/edgar/daily-index/"
           f"{day.year}/QTR{_quarter(day)}/form.{day.strftime('%Y%m%d')}.idx")
    try:
        raw = (await _get(session, url)).decode(errors="replace")
    except Exception:
        return []
    rows = []
    for line in raw.splitlines():
        parts = re.split(r"\s{2,}", line.strip())
        # Form Type | Company Name | CIK | Date Filed | File Name
        if len(parts) >= 5 and parts[0] == "D":
            rows.append({"company": parts[1], "cik": parts[2],
                         "date": parts[3], "path": parts[4]})
    return rows


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


async def _fetch_filing(session, row: dict):
    """Fetches and parses one Form D primary_doc.xml. Returns doc dict or None."""
    m = re.search(r"edgar/data/(\d+)/([\d-]+)\.txt", row["path"])
    if not m:
        return None
    cik, accession = m.group(1), m.group(2)
    url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
           f"{accession.replace('-', '')}/primary_doc.xml")
    try:
        root = _strip_ns(ET.fromstring(await _get(session, url)))
    except Exception:
        return None

    if (root.findtext(".//testOrLive") or "LIVE") != "LIVE":
        return None

    entity = root.findtext(".//primaryIssuer/entityName") or row["company"]
    industry = root.findtext(".//offeringData/industryGroup/industryGroupType") or "Unknown"
    state = root.findtext(".//primaryIssuer/issuerAddress/stateOrCountryDescription") or ""
    total_offering = root.findtext(".//offeringSalesAmounts/totalOfferingAmount") or ""
    total_sold = root.findtext(".//offeringSalesAmounts/totalAmountSold") or ""
    year_inc = root.findtext(".//primaryIssuer/yearOfInc/value") or ""
    new_entity = (root.findtext(".//primaryIssuer/yearOfInc/withinFiveYears") or "") == "true"

    execs = []
    for person in root.findall(".//relatedPersonsList/relatedPersonInfo"):
        first = person.findtext("relatedPersonName/firstName") or ""
        last = person.findtext("relatedPersonName/lastName") or ""
        roles = [r.text for r in person.findall("relatedPersonRelationshipList/relationship") if r.text]
        if first or last:
            execs.append(f"{first} {last}".strip() + (f" ({', '.join(roles)})" if roles else ""))

    def _fmt_amount(x):
        try:
            return f"${int(float(x)):,}"
        except (TypeError, ValueError):
            return "undisclosed" if x in ("", "Indefinite") else str(x)

    date_iso = datetime.strptime(row["date"], "%Y%m%d").strftime("%Y-%m-%d")
    return {
        "doc_id": _doc_id(accession),
        "source": "formd",
        "org": entity,
        "date": date_iso,
        "title": (f"Form D: {entity} raised {_fmt_amount(total_sold)} "
                  f"of {_fmt_amount(total_offering)} offering ({industry})"),
        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/primary_doc.xml",
        "text": (f"{entity} filed SEC Form D on {date_iso}. Industry: {industry}. "
                 f"State: {state}. Total offering: {_fmt_amount(total_offering)}; "
                 f"sold so far: {_fmt_amount(total_sold)}. "
                 f"{'Newly incorporated (' + year_inc + '). ' if new_entity else ''}"
                 f"Named executives: {'; '.join(execs[:6]) or 'n/a'}."),
        "contacts": {"executives": execs[:6],
                     "contact_basis": "names from the public federal Form D disclosure"},
        "extra": {"industry_group": industry, "state": state,
                  "total_offering": total_offering, "total_sold": total_sold,
                  "newly_incorporated": new_entity, "fresh_raise": True},
        "_industry": industry,
    }


async def ingest_formd(session: aiohttp.ClientSession, cfg: dict) -> list:
    lookback = int(cfg.get("formd_lookback_days", 4))
    max_filings = int(cfg.get("formd_max_filings", 250))
    keep_groups = set(cfg.get("formd_industry_groups", [])) or DEFAULT_INDUSTRY_GROUPS

    # ---- 1. Collect Form D rows from recent daily indexes ------------------
    today = datetime.now(timezone.utc)
    day_tasks = [_fetch_daily_form_d(session, today - timedelta(days=i))
                 for i in range(1, lookback + 1)]
    rows = [r for day in await asyncio.gather(*day_tasks) for r in day]
    print(f"[FormD] {len(rows)} Form D filings in the last {lookback} days")
    rows = rows[:max_filings]

    # ---- 2. Fetch + parse each filing --------------------------------------
    parsed = await asyncio.gather(*[_fetch_filing(session, r) for r in rows])

    # ---- 3. Industry filter --------------------------------------------------
    docs = []
    for doc in parsed:
        if doc is None:
            continue
        industry = doc.pop("_industry")
        if industry in _EXCLUDED_GROUPS:
            continue
        if industry not in keep_groups:
            continue
        docs.append(doc)
    print(f"[FormD] {len(docs)} relevant raises after industry filter")
    return docs
