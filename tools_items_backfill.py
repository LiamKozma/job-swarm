#!/usr/bin/env python3
"""Backfill ITEMS.json (the dashboard inbox feed) for report dirs generated
before js_graph emitted it natively. Parses the run's own md files - same
schema as the native emitter. Idempotent (overwrites ITEMS.json).
Usage: items_backfill.py <run_dir> [...]   Stdlib only, py3.9-safe."""
import hashlib
import json
import os
import re
import sys


def read(p):
    try:
        return open(p, encoding="utf-8").read()
    except OSError:
        return ""


def parse_dossier(run_dir, fname):
    text = read(os.path.join(run_dir, fname))
    if not text:
        return None
    org = (re.search(r"^# (.+)$", text, re.M) or [None, fname]).group(1).strip()
    slug = re.sub(r"^\d\d_", "", fname)[:-3]
    m = re.search(r"for `([^`]+)` in the outreach tracker", text)
    org_key = m.group(1) if m else slug
    def field(pat):
        m = re.search(pat, text, re.M)
        return m.group(1).strip() if m else None
    bot = re.search(r"## Audited bottleneck\s*\n(.+?)\n\s*\n", text, re.S)
    draft = re.search(r"```\n(.*?)\n```", text, re.S)
    contacts = dict(re.findall(r"^- \*\*(\w+)\*\*: (\S+)$", text, re.M))
    people = re.findall(r"^- \*\*(.+?)\*\* [—-]+ senior/corresponding", text, re.M)
    return {
        "id": "memo:" + org_key, "kind": "email", "org_key": org_key,
        "org": org, "title": "Email " + org,
        "summary": " ".join(bot.group(1).split()) if bot else "",
        "angle": field(r"\*\*Intervention vector:\*\*\s*(.+?)\s*$"),
        "contacts": contacts, "people": people,
        "align": field(r"\*\*Alignment \(LLM audit\):\*\*\s*([\d.]+)"),
        "prescore": field(r"\*\*Prescore \(quant filter\):\*\*\s*([\d.]+)"),
        "regime": field(r"\*\*δ-shift regime:\*\*\s*(\S+)"),
        "explore": "**Exploration pick**" in text,
        "subject": field(r"\*\*Subject:\*\*\s*(.+?)\s*$"),
        "draft": draft.group(1).strip() if draft else None,
        "dossier": fname,
    }


def parse_applications(run_dir):
    """{url: app dict} from APPLICATIONS.md."""
    text = read(os.path.join(run_dir, "APPLICATIONS.md"))
    apps = {}
    for chunk in text.split("\n## ")[1:]:
        m = re.search(r"\[posting\]\(([^)]+)\)", chunk)
        if not m:
            continue
        url = m.group(1)
        note = re.search(r"```\n(.*?)\n```", chunk, re.S)
        bullets = re.findall(r"^- \*(.+?)\* (?:→|->) (.+)$", chunk, re.M)
        topics_m = re.search(r"\*\*Likely interview topics.*?\*\*\n((?:- .+\n?)+)",
                             chunk)
        gaps = re.search(r"\*\*Honest gaps \(don't fake these\):\*\* (.+)", chunk)
        apps[url] = {
            "draft": note.group(1).strip() if note else None,
            "note": note.group(1).strip() if note else None,
            "bullets": [{"theme": t, "bullet": b} for t, b in bullets] or None,
            "topics": (re.findall(r"^- (.+)$", topics_m.group(1), re.M)
                       if topics_m else None),
            "gaps": ([g.strip() for g in gaps.group(1).split(",")]
                     if gaps and "none" not in gaps.group(1) else None),
        }
    return apps


def parse_openings(run_dir, apps):
    text = read(os.path.join(run_dir, "REVIEW_QUEUE.md"))
    m = re.search(r"## Direct openings.*?\n(.*?)(?=\n## |\n\*\*|\Z)", text, re.S)
    items = []
    if not m:
        return items
    for ln in m.group(1).splitlines():
        row = re.match(r"\|([^|]+)\|(.+)\|([^|]+)\|([^|]+)\|([^|]+)\|"
                       r"\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*$", ln)
        if not row:
            continue
        score, title, tag, where, comp, prov, url = (x.strip() for x in row.groups())
        if not re.match(r"[\d.]+$", score):
            continue
        a = apps.get(url)
        items.append({
            "id": "open:" + hashlib.sha1(url.encode()).hexdigest()[:12],
            "kind": "apply", "org_key": None, "title": title,
            "summary": "%s · %s · %s" % (where, comp, tag),
            "url": url, "provider": prov, "score": float(score),
            "draft": a.get("draft") if a else None,
            "app": a,
        })
    return items


def parse_briefs(run_dir, org_keys_by_name):
    text = read(os.path.join(run_dir, "ARTIFACT_BRIEFS.md"))
    items = []
    heads = list(re.finditer(r"^## (.+?) [—-]+ (.+)$", text, re.M))
    for i, m in enumerate(heads):
        org, title = m.group(1).strip(), m.group(2).strip()
        chunk = text[m.end():heads[i + 1].start() if i + 1 < len(heads)
                     else len(text)]
        hook = re.search(r"### The email it becomes.*?```\n(.*?)\n```",
                         chunk, re.S)
        key = org_keys_by_name.get(org) or re.sub(r"[^a-z0-9]", "", org.lower())
        items.append({
            "id": "brief:" + key, "kind": "build", "org_key": key, "org": org,
            "title": "Build for %s: %s" % (org, title),
            "summary": "Proof-of-work artifact (<=2 days on the cluster) - "
                       "backtrace the provenance, build, then email with "
                       "work already done.",
            "draft": hook.group(1).strip() if hook else None,
            "dossier": "ARTIFACT_BRIEFS.md",
            "section": org,
        })
    return items


def main(run_dir):
    run_date = os.path.basename(run_dir.rstrip("/"))
    items, keys = [], {}
    for f in sorted(os.listdir(run_dir)):
        if re.match(r"\d\d_.+\.md$", f):
            it = parse_dossier(run_dir, f)
            if it:
                items.append(it)
                keys[it["org"]] = it["org_key"]
    items += parse_openings(run_dir, parse_applications(run_dir))
    items += parse_briefs(run_dir, keys)
    if os.path.exists(os.path.join(run_dir, "RESUME_AUDIT.md")):
        items.append({"id": "chore:resume:" + run_date, "kind": "chore",
                      "title": "Read the resume audit",
                      "summary": "Fresh critique of your resume against this "
                                 "run's corpus.", "dossier": "RESUME_AUDIT.md"})
    if os.path.exists(os.path.join(run_dir, "CONFIG_SUGGESTIONS.md")):
        items.append({"id": "chore:config:" + run_date, "kind": "chore",
                      "title": "Review swarm vocabulary suggestions",
                      "summary": "Search-term proposals derived from your "
                                 "profile - approve into job_swarm_config.json.",
                      "dossier": "CONFIG_SUGGESTIONS.md"})
    with open(os.path.join(run_dir, "ITEMS.json"), "w") as f:
        json.dump({"run_date": run_date, "items": items}, f, indent=2)
    kinds = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print("ITEMS.json:", run_dir, kinds)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
