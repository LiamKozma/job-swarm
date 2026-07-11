"""
Job Swarm - Deep Research people ingest (the manual Gemini loop, cluster side).

Each night the swarm emits DEEP_RESEARCH_PROMPT.md asking for NAMED PEOPLE at
the current top targets (js_graph._jobs_people_research_prompt). The human runs
it in Gemini, pastes the result on the jobs tab, and the publisher lands the
file in this run's research dir (emit_actions.absorb_jobs_research). This module
parses that paste and binds the people it names to organizations:

  - a deliverable EMAIL becomes a manual_contact, opening the outreach gate for
    that org exactly as a GitHub-found or human-found contact does;
  - every named person (with role + source URL) is stored under the org_key in
    the `research_people` meta store, which compile_review renders on the
    dossier so the human can address the memo to the right human.

Only the structured trailer line the prompt mandates is trusted:
  PERSON: <name> | ROLE: <title> | ORG: <org> | EMAIL: <email|none> | SOURCE: <url>
Prose is ignored - the paste is untrusted web text, so nothing that is not on a
verbatim PERSON: line ever enters the database. Additive and idempotent: files
already ingested (by content hash) are skipped, and re-running never duplicates.
"""

import glob
import hashlib
import json
import os
import re

import swarm_db

RESEARCH_DIR = os.environ.get(
    "JOB_SWARM_RESEARCH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "research"))

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
# One person per line, fields pipe-delimited, order fixed by the prompt.
_PERSON_RE = re.compile(
    r"PERSON:\s*(?P<name>.+?)\s*\|\s*ROLE:\s*(?P<role>.+?)\s*\|\s*"
    r"ORG:\s*(?P<org>.+?)\s*\|\s*EMAIL:\s*(?P<email>.+?)\s*\|\s*"
    r"SOURCE:\s*(?P<source>\S+)", re.I)


def _parse_people(text: str) -> list:
    people = []
    for m in _PERSON_RE.finditer(text or ""):
        name = m.group("name").strip()
        org = m.group("org").strip()
        if not name or not org or name.lower() == "<full name>":
            continue
        raw_email = m.group("email").strip()
        em = _EMAIL_RE.search(raw_email)
        people.append({
            "name": name[:120],
            "role": m.group("role").strip()[:140],
            "org": org[:160],
            "email": em.group(0) if em else None,
            "source_url": m.group("source").strip()[:400],
        })
    return people


def ingest_research(cache_note: str = "") -> dict:
    """Parse every research paste in RESEARCH_DIR, bind people to org_keys, and
    persist. Returns {n_files, n_people, n_emails, n_orgs}. Safe to call every
    nightly - already-ingested files (by content hash) are skipped."""
    if not os.path.isdir(RESEARCH_DIR):
        return {"n_files": 0, "n_people": 0, "n_emails": 0, "n_orgs": 0}
    conn = swarm_db.connect()
    seen = set(json.loads(swarm_db.get_meta(conn, "research_files", "[]")))
    known_orgs = {r["org_key"] for r in conn.execute("SELECT org_key FROM orgs")}
    research_people = json.loads(swarm_db.get_meta(conn, "research_people", "{}"))
    manual = json.loads(swarm_db.get_meta(conn, "manual_contacts", "{}"))

    files = sorted(glob.glob(os.path.join(RESEARCH_DIR, "*.md"))
                   + glob.glob(os.path.join(RESEARCH_DIR, "*.txt")))
    n_files = n_people = n_emails = 0
    touched_orgs = set()
    for path in files:
        try:
            with open(path, errors="replace") as f:
                body = f.read()
        except OSError:
            continue
        digest = hashlib.sha1(body.encode()).hexdigest()
        tag = f"{os.path.basename(path)}:{digest[:12]}"
        if tag in seen:
            continue
        seen.add(tag)
        n_files += 1
        for person in _parse_people(body):
            org_key = swarm_db.org_key_from_name(person["org"])
            # Bind only to organizations the pipeline already tracks: a person
            # at an org we have never ingested has no memo/audit to attach to,
            # and inventing an org from untrusted paste text is how a prompt
            # injection would smuggle a target in. Unknown orgs are dropped.
            if org_key not in known_orgs:
                continue
            n_people += 1
            touched_orgs.add(org_key)
            bucket = research_people.setdefault(org_key, [])
            if not any(p.get("name") == person["name"]
                       and p.get("source_url") == person["source_url"]
                       for p in bucket):
                bucket.append(person)
            if person["email"] and org_key not in manual:
                manual[org_key] = {
                    "email": person["email"],
                    "contact_basis": (
                        f"{person['name']} ({person['role']}) - from deep-"
                        f"research paste, verify at {person['source_url']}")}
                n_emails += 1

    if n_files:
        swarm_db.set_meta(conn, "research_files", json.dumps(sorted(seen)))
        swarm_db.set_meta(conn, "research_people", json.dumps(research_people))
        swarm_db.set_meta(conn, "manual_contacts", json.dumps(manual))
        conn.commit()
    conn.close()
    print(f"[Research] {n_files} new paste(s): {n_people} people bound to "
          f"{len(touched_orgs)} orgs, {n_emails} new deliverable email(s)")
    return {"n_files": n_files, "n_people": n_people,
            "n_emails": n_emails, "n_orgs": len(touched_orgs)}


def people_for(org_key: str) -> list:
    """Research-found people for an org (for dossier rendering)."""
    conn = swarm_db.connect()
    people = json.loads(swarm_db.get_meta(conn, "research_people", "{}"))
    conn.close()
    return people.get(org_key, [])
