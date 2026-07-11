"""
Job Swarm - GitHub People Engine (the deliverable-contact channel).

The single most valuable thing the swarm can add is a NAMED, REACHABLE human
at a target org - and the memo gate (js_graph._memo_gate) blocks outreach to
any org without one. arXiv authorship covers the research-heavy orgs; this
engine covers the ones that live on GitHub instead: funded startups, tooling
companies, and lab groups that ship code but publish few papers.

Method, entirely public REST API, zero marginal cost, no scraping:

  1. Resolve org -> GitHub org login. Search /search/users?q=<name> type:org,
     then VERIFY the match: the candidate org's `blog` domain must equal the
     org's own website domain (when we know it), else the login is rejected.
     Name collisions ("apollo", "vercel"-lookalikes) die here rather than
     poisoning the contact database.
  2. People: public org members (logins), plus the authors of recent commits
     to the org's most-recently-pushed public repos.
  3. Emails: the commits-list JSON already carries commit.author.email - the
     raw git-config address the committer used locally, which is exactly what
     appending `.patch` would expose, at one request per repo-page instead of
     one per commit. GitHub's web-flow noreply masks are filtered out.
  4. Freshness/role binding: only repos pushed within 60 days are read, so an
     email that appears is an ACTIVE contributor to code the org ships today -
     the strongest role signal available without an account. Stale personal
     forks and long-dead repos never enter.

The engine NEVER sends mail and NEVER verifies an address by probing an SMTP
server (deliberately rejected: probing from the HPC IP risks blacklisting and
the address is already self-attested by the commit). It hands the human a name,
a login, an email, and the exact commit/profile URL to backtrace.

Auth: set JOB_SWARM_GITHUB_TOKEN (or drop it in a `.github_token` file beside
this module) for the 5,000 req/hr authenticated limit. Unauthenticated it
degrades to 60 req/hr and processes only the first few orgs, logging the rest
as skipped - never silently truncating.
"""

import asyncio
import os
import re
from urllib.parse import urlparse

import aiohttp

_API = "https://api.github.com"
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

# GitHub caps concurrent load hard for unauthenticated callers; stay polite.
_SEM = asyncio.Semaphore(4)

# Emails that are not a reachable human: GitHub's web-flow noreply masks, bot
# accounts, and no-reply service addresses. A per-user noreply
# (12345+login@users.noreply.github.com) is filtered too - it is not
# deliverable, and the login itself is the better handle in that case.
_SKIP_EMAIL_RE = re.compile(
    r"(users\.noreply\.github\.com|noreply|no-reply|@github\.com$|"
    r"\[bot\]|actions@|bot@|example\.com)", re.I)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

# Bots and automation logins that author commits but are not people.
_BOT_LOGIN_RE = re.compile(r"(\[bot\]|-bot$|^bot-|dependabot|renovate|"
                           r"greenkeeper|semantic-release|github-actions)", re.I)


def _token() -> str:
    tok = os.environ.get("JOB_SWARM_GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".github_token")
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return ""


def _domain(url: str) -> str:
    """Registrable-ish host for match verification: drop scheme, www, path."""
    if not url:
        return ""
    u = url if "//" in url else "//" + url
    host = (urlparse(u).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


class _GH:
    """Thin GitHub REST client: shared session, auth header, rate-limit aware.
    On a 403/429 with a zero remaining-limit it stops issuing new requests for
    the rest of the run (returns None) rather than hammering a closed window."""

    def __init__(self, session, token: str):
        self.session = session
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"JobSwarmResearch/1.0 ({_MAILTO})",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.exhausted = False

    async def get(self, path: str, params: dict = None):
        if self.exhausted:
            return None
        url = path if path.startswith("http") else f"{_API}{path}"
        async with _SEM:
            await asyncio.sleep(0.3)
            try:
                async with self.session.get(url, params=params or {},
                                            headers=self.headers) as resp:
                    if resp.status in (403, 429):
                        remaining = resp.headers.get("X-RateLimit-Remaining")
                        if remaining == "0":
                            self.exhausted = True
                            print("[GitHub] rate limit exhausted - "
                                  "remaining orgs skipped this run")
                        return None
                    if resp.status != 200:
                        return None
                    return await resp.json(content_type=None)
            except Exception:
                return None


async def _resolve_login(gh: _GH, name: str, website: str):
    """Best GitHub org login for a display name, VERIFIED against the org's
    own website domain when known. Returns login or None."""
    want_dom = _domain(website)
    # Prefer a domain-anchored search when we have the org's site; it collapses
    # collisions immediately (the org's GitHub `blog` usually is that domain).
    data = await gh.get("/search/users", {"q": f"{name} type:org", "per_page": 5})
    candidates = [(u.get("login")) for u in (data or {}).get("items", []) if u.get("login")]
    for login in candidates:
        detail = await gh.get(f"/orgs/{login}")
        if not detail:
            continue
        if not want_dom:
            # No site to verify against: accept only an exact-ish name match to
            # stay honest (login or org name equals the display name token).
            dn = (detail.get("name") or "").lower()
            token = re.sub(r"[^a-z0-9]", "", name.lower())
            if token and (token == re.sub(r"[^a-z0-9]", "", login.lower())
                          or token == re.sub(r"[^a-z0-9]", "", dn)):
                return login
            continue
        blog_dom = _domain(detail.get("blog") or "")
        if blog_dom and (blog_dom == want_dom or blog_dom.endswith("." + want_dom)
                         or want_dom.endswith("." + blog_dom)):
            return login
    return None


async def _user_person(gh: _GH, login: str) -> dict:
    """Name/role hint for a login from its public profile (one request)."""
    d = await gh.get(f"/users/{login}") or {}
    role = d.get("bio") or ""
    if d.get("company"):
        role = (role + " · " + d["company"]).strip(" ·")
    return {"name": d.get("name") or login, "login": login,
            "role_hint": role[:120] or None,
            "email": d.get("email") if _deliverable(d.get("email")) else None,
            "source_url": d.get("html_url") or f"https://github.com/{login}"}


def _deliverable(email) -> bool:
    email = (email or "").strip()
    return bool(email and _EMAIL_RE.match(email) and not _SKIP_EMAIL_RE.search(email))


async def _repo_commit_people(gh: _GH, login: str, max_repos: int = 3) -> dict:
    """{email: person} from recent commits to the org's freshest public repos.
    Only repos pushed within 60 days are read (active-code freshness gate)."""
    repos = await gh.get(f"/orgs/{login}/repos",
                         {"sort": "pushed", "per_page": max_repos, "type": "public"})
    people: dict = {}
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=60)
    for repo in (repos or []):
        pushed = repo.get("pushed_at") or ""
        try:
            if datetime.strptime(pushed, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc) < cutoff:
                continue
        except ValueError:
            continue
        commits = await gh.get(f"/repos/{login}/{repo['name']}/commits",
                               {"per_page": 30}) or []
        for c in commits:
            commit = c.get("commit") or {}
            author = commit.get("author") or {}
            email = author.get("email")
            if not _deliverable(email):
                continue
            gh_login = ((c.get("author") or {}) or {}).get("login")
            if gh_login and _BOT_LOGIN_RE.search(gh_login):
                continue
            if email in people:
                continue
            people[email] = {
                "name": author.get("name") or gh_login or email.split("@")[0],
                "login": gh_login,
                "email": email,
                "role_hint": f"commits to {login}/{repo['name']}",
                "source_url": c.get("html_url")
                or f"https://github.com/{login}/{repo['name']}",
            }
    return people


async def _one_org(gh: _GH, org: dict) -> tuple:
    name = org["display_name"]
    search_name = name.split(" · ")[-1].strip() if " · " in name else name
    login = await _resolve_login(gh, search_name, org.get("website") or "")
    if not login:
        return org["org_key"], None
    people: dict = {}
    # Commit-email people first: these carry a deliverable address AND an
    # active-contributor role binding.
    for email, person in (await _repo_commit_people(gh, login)).items():
        people[email.lower()] = person
    # Public members fill in named humans even where no email surfaced; the
    # human can reach them through the published GitHub profile.
    members = await gh.get(f"/orgs/{login}/public_members", {"per_page": 10}) or []
    for m in members[:6]:
        mlogin = m.get("login")
        if not mlogin or _BOT_LOGIN_RE.search(mlogin):
            continue
        if any(p.get("login") == mlogin for p in people.values()):
            continue
        person = await _user_person(gh, mlogin)
        key = (person.get("email") or f"login:{mlogin}").lower()
        if key not in people:
            people[key] = person
    ranked = sorted(people.values(),
                    key=lambda p: (p.get("email") is None, p.get("login") is None))
    return org["org_key"], {"login": login, "people": ranked[:8]}


async def find_github_people(orgs: list, max_orgs: int = 30) -> dict:
    """
    orgs: [{org_key, display_name, website}] finalist/shortlist targets.
    Returns {org_key: {"login": str, "people": [{name, login, email,
    role_hint, source_url}]}} for every org whose GitHub org was resolved and
    verified. Startups on GitHub resolve here where OpenAlex/arXiv find nothing.
    """
    token = _token()
    if not token:
        # 60 req/hr unauthenticated: each org costs ~1 search + 1 org detail +
        # ~3 repo pages + a few user lookups (~8-10 requests). Hold to a
        # handful and log the rest rather than half-populating on a dead limit.
        max_orgs = min(max_orgs, 5)
        print("[GitHub] no token (JOB_SWARM_GITHUB_TOKEN / .github_token) - "
              f"unauthenticated 60 req/hr, limiting to {max_orgs} orgs")
    results: dict = {}
    timeout = aiohttp.ClientTimeout(total=240)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        gh = _GH(session, token)
        batch = orgs[:max_orgs]
        for org_key, payload in await asyncio.gather(
                *[_one_org(gh, o) for o in batch]):
            if payload and payload["people"]:
                results[org_key] = payload
    skipped = max(0, len(orgs) - max_orgs)
    print(f"[GitHub] {len(results)}/{len(orgs[:max_orgs])} orgs yielded a "
          f"contact" + (f"; {skipped} orgs not attempted (limit)" if skipped else ""))
    return results
