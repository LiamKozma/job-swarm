"""
Job Swarm - Warm Path Engine (the social channel).

A warm introduction converts an order of magnitude better than any cold memo.
This engine finds the bridge automatically: for each finalist target, it asks
OpenAlex (the open scholarly graph - free, keyless, ~250M works) whether the
target institution has co-published with the candidate's home institution in
the last five years. A hit means someone at UGA - often a professor one email
away - has a direct working relationship with the target.

The dossier then says "ask Prof. X for an intro," which is a fundamentally
stronger move than any email the swarm could draft.

Startups mostly won't resolve in OpenAlex (no publications) - that's expected;
the engine shines for the national labs, university groups, and research-heavy
companies the grant/arXiv sources surface. Polite-pool etiquette: the mailto
parameter identifies the caller; concurrency is capped at 3.
"""

import asyncio
import os

import aiohttp

HOME_INSTITUTION = os.environ.get("JOB_SWARM_HOME_INSTITUTION", "University of Georgia")
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


_MAILTO = (os.environ.get("JOB_SWARM_USER_AGENT") or _contact_email()).split()[-1]
_SEM = asyncio.Semaphore(3)

_API = "https://api.openalex.org"


async def _get(session, url, params):
    params = {**params, "mailto": _MAILTO}
    async with _SEM:
        await asyncio.sleep(0.4)
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)


async def _institution_id(session, name: str):
    """Top OpenAlex institution match for a name, or None."""
    try:
        data = await _get(session, f"{_API}/institutions", {"search": name, "per-page": 1})
        hits = data.get("results") or []
        if hits and hits[0].get("works_count", 0) > 0:
            return hits[0]["id"].rsplit("/", 1)[-1], hits[0].get("display_name")
    except Exception:
        pass
    return None


async def _bridge_works(session, home_id: str, target_id: str):
    """Recent works co-authored by both institutions."""
    try:
        data = await _get(session, f"{_API}/works", {
            "filter": (f"authorships.institutions.lineage:{home_id},"
                       f"authorships.institutions.lineage:{target_id},"
                       f"from_publication_date:2021-01-01"),
            "per-page": 15,
            "sort": "publication_date:desc",
        })
    except Exception:
        return []
    works = []
    for w in data.get("results") or []:
        authorships = w.get("authorships") or []
        # Mega-consortium papers (Global Burden of Disease etc., 1000+
        # authors) make every institution "co-publish" with every other -
        # zero intro value. Real bridges are small-team collaborations.
        if len(authorships) > 25 or w.get("is_authors_truncated"):
            continue
        authors = [a.get("author", {}).get("display_name")
                   for a in authorships][:8]
        home_authors = [
            a.get("author", {}).get("display_name")
            for a in (w.get("authorships") or [])
            if any(home_id in (i.get("id") or "")
                   for i in (a.get("institutions") or []))
        ]
        works.append({
            "title": (w.get("title") or "")[:140],
            "year": w.get("publication_year"),
            "home_authors": [a for a in home_authors if a][:4],
            "all_authors": [a for a in authors if a],
            "url": w.get("doi") or w.get("id"),
        })
    return works[:5]


async def _org_recent_leads(session, target_id: str):
    """Likely technical leads at one institution: corresponding and last
    authors of its recent small-team papers. Published-for-inquiry authorship
    metadata ONLY - the swarm names names and papers, never harvests emails;
    the human finds the published contact channel through the paper itself."""
    try:
        data = await _get(session, f"{_API}/works", {
            "filter": (f"authorships.institutions.lineage:{target_id},"
                       f"from_publication_date:2024-01-01"),
            "per-page": 12,
            "sort": "publication_date:desc",
        })
    except Exception:
        return []
    tally: dict = {}
    for w in data.get("results") or []:
        authorships = w.get("authorships") or []
        if not authorships or len(authorships) > 25 or w.get("is_authors_truncated"):
            continue
        # Senior-author convention: the corresponding author(s), else the
        # last author, is who directs the work - the hiring-manager-shaped
        # person for a technical memo.
        leads = [a for a in authorships if a.get("is_corresponding")] \
            or [authorships[-1]]
        for a in leads:
            if not any(target_id in (i.get("id") or "")
                       for i in (a.get("institutions") or [])):
                continue  # corresponding author may be at the partner org
            name = (a.get("author") or {}).get("display_name")
            if not name:
                continue
            entry = tally.setdefault(name, {"name": name, "n_works": 0, "works": []})
            entry["n_works"] += 1
            if len(entry["works"]) < 2:
                entry["works"].append({
                    "title": (w.get("title") or "")[:120],
                    "year": w.get("publication_year"),
                    "url": w.get("doi") or w.get("id"),
                })
    ranked = sorted(tally.values(), key=lambda e: e["n_works"], reverse=True)
    return ranked[:3]


async def find_technical_leads(org_names: list, max_orgs: int = 12) -> dict:
    """{org_name: [lead, ...]} for research-resolvable finalists. Startups
    won't resolve (expected); labs, universities, and research-heavy
    companies return the people actually directing the relevant work."""
    results: dict = {}
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(
        timeout=timeout, headers={"User-Agent": f"JobSwarmResearch/1.0 ({_MAILTO})"}
    ) as session:
        async def process(name: str):
            search_name = name.split(" · ")[-1].strip() if " · " in name else name
            target = await _institution_id(session, search_name)
            if target is None:
                return
            leads = await _org_recent_leads(session, target[0])
            if leads:
                results[name] = leads

        await asyncio.gather(*[process(n) for n in org_names[:max_orgs]])
    print(f"[WarmPath] technical leads resolved for {len(results)}/"
          f"{min(len(org_names), max_orgs)} finalists")
    return results


async def find_warm_paths(org_names: list, max_orgs: int = 20) -> dict:
    """
    org_names: display names of finalist targets.
    Returns {org_name: {"institution": matched_name, "bridges": [works]}} for
    every target with at least one co-publication bridge to HOME_INSTITUTION.
    """
    results: dict = {}
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(
        timeout=timeout, headers={"User-Agent": f"JobSwarmResearch/1.0 ({_MAILTO})"}
    ) as session:
        home = await _institution_id(session, HOME_INSTITUTION)
        if home is None:
            print(f"[WarmPath] could not resolve home institution '{HOME_INSTITUTION}'")
            return {}
        home_id, home_name = home

        async def process(name: str):
            # PI-level orgs are 'PI Name · Institution' - OpenAlex needs the
            # institution part for the co-publication bridge.
            search_name = name.split(" · ")[-1].strip() if " · " in name else name
            target = await _institution_id(session, search_name)
            if target is None:
                return
            target_id, target_name = target
            if target_id == home_id:
                return
            bridges = await _bridge_works(session, home_id, target_id)
            if bridges:
                results[name] = {"institution": target_name, "bridges": bridges}

        await asyncio.gather(*[process(n) for n in org_names[:max_orgs]])

    print(f"[WarmPath] {len(results)}/{min(len(org_names), max_orgs)} targets have "
          f"a co-publication bridge to {HOME_INSTITUTION}")
    return results
