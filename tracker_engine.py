"""
Job Swarm - plain-text outreach tracker (the vim-over-sshfs workflow).

One persistent file, ~/job_swarm_reports/TRACKER.md, regenerated every
nightly run. Each org you could reach out to is one line with four
checkboxes; you edit the file in vim and put an x in a box:

    [x] [ ] [ ] [ ]  mayo_clinic | Mayo Clinic - sent 2026-06-25 (6d ago) FOLLOW-UP DUE

    box 1: first email sent        box 3: they replied
    box 2: follow-up sent          box 4: drop / not interested

At the start of every analyze stage the swarm reads your marks and syncs
them into the org status machine (sent -> 'contacted' + timestamp, so the
day counting starts; follow-up -> 'followed_up'; replied -> 'replied';
drop -> 'rejected'). At the end of the run it rewrites the file: boxes
reflect the database, annotations carry fresh day counts, new drafts
appear, and orgs due a follow-up float to the top.

Rules that keep a hand-edited file safe:
  - Only the four boxes and the org_key are parsed; everything after '|'
    is regenerated, so stray edits there are harmless.
  - Marks only move status FORWARD (you can't un-contact by clearing a
    box - use js_review.py if you ever need to rewind).
  - Unparseable lines are ignored, never destroyed silently: sync happens
    before the rewrite, so a mangled line at worst loses one mark.
  - js_review.py commands remain equivalent; the file re-renders from the
    DB either way.
"""

import os
import re
from datetime import datetime

import swarm_db

REPORTS_DIR = os.environ.get(
    "JOB_SWARM_REPORTS", os.path.expanduser("~/job_swarm_reports")
)
TRACKER_PATH = os.path.join(REPORTS_DIR, "TRACKER.md")

# 4 matches swarm_db.followups_due and the report copy - the two surfaces
# disagreed (4 vs 5) and could show different due lists (audit F4).
FOLLOWUP_DUE_DAYS = int(os.environ.get("JOB_SWARM_FOLLOWUP_DAYS", "4"))

_LINE_RE = re.compile(
    r"^\[([ xX])\]\s*\[([ xX])\]\s*\[([ xX])\]\s*\[([ xX])\]\s+([A-Za-z0-9_\-.]+)\s*\|"
)

# Forward-only transitions, strongest mark wins
_ORDER = {"new": 0, "shortlisted": 0, "audited": 0, "memo_drafted": 0,
          "contacted": 1, "followed_up": 2, "replied": 3, "rejected": 3}


def sync_marks(conn) -> int:
    """Reads TRACKER.md marks into the org status machine. Returns #changes."""
    if not os.path.exists(TRACKER_PATH):
        return 0
    changed = 0
    with open(TRACKER_PATH, errors="replace") as f:
        for line in f:
            m = _LINE_RE.match(line.strip())
            if not m:
                continue
            sent, fup, replied, drop = (c.strip().lower() == "x" for c in m.groups()[:4])
            org_key = m.group(5)
            target = None
            if drop:
                target = "rejected"
            elif replied:
                target = "replied"
            elif fup:
                target = "followed_up"
            elif sent:
                target = "contacted"
            if target is None:
                continue
            row = conn.execute(
                "SELECT status, contacted_at FROM orgs WHERE org_key = ?", (org_key,)
            ).fetchone()
            if row is None:
                continue
            if _ORDER.get(target, 0) <= _ORDER.get(row["status"], 0):
                continue  # forward-only
            # Any mark implies the first email went out - stamp once
            if row["contacted_at"] is None and target in ("contacted", "followed_up", "replied"):
                conn.execute(
                    "UPDATE orgs SET status = ?, contacted_at = ? WHERE org_key = ?",
                    (target, swarm_db._now(), org_key))
            else:
                conn.execute(
                    "UPDATE orgs SET status = ? WHERE org_key = ?", (target, org_key))
            # Terminal marks close the outreach_log row - the label for the
            # reply-probability calibration dataset.
            if target == "replied":
                swarm_db.record_outreach_outcome(conn, org_key, "replied")
            elif target == "rejected":
                swarm_db.record_outreach_outcome(conn, org_key, "dropped")
            changed += 1
    conn.commit()
    if changed:
        print(f"[Tracker] {changed} status change(s) absorbed from {TRACKER_PATH}")
    return changed


def _age_days(contacted_at) -> int:
    try:
        # clamp: the stamp is UTC, so a same-evening mark can look like "tomorrow"
        return max((datetime.now() - datetime.strptime((contacted_at or "")[:10], "%Y-%m-%d")).days, 0)
    except ValueError:
        return -1


def _line(org, annotation: str) -> str:
    s = org["status"]
    box = lambda on: "[x]" if on else "[ ]"
    return (f"{box(s in ('contacted', 'followed_up', 'replied'))} "
            f"{box(s == 'followed_up')} {box(s == 'replied')} {box(s == 'rejected')}  "
            f"{org['org_key']} | {org['display_name'][:52]} - {annotation}")


def regenerate(conn) -> str:
    """Rewrites TRACKER.md from the DB, grouped by what needs doing."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = [dict(r) for r in conn.execute(
        "SELECT o.org_key, o.display_name, o.status, o.contacted_at, "
        "o.latest_alignment, "
        "(SELECT MAX(m.run_date) FROM memos m WHERE m.org_key = o.org_key) "
        "AS draft_date "
        "FROM orgs o WHERE o.status IN "
        "('memo_drafted', 'contacted', 'followed_up', 'replied', 'rejected')"
    ).fetchall()]

    due, waiting, ready, fup_sent, closed = [], [], [], [], []
    for o in rows:
        age = _age_days(o.get("contacted_at"))
        sent_str = (o.get("contacted_at") or "")[:10] or "?"
        if o["status"] == "contacted":
            if age >= FOLLOWUP_DUE_DAYS:
                due.append((age, _line(o, f"sent {sent_str} ({age}d ago) - FOLLOW-UP DUE")))
            else:
                left = FOLLOWUP_DUE_DAYS - max(age, 0)
                waiting.append((age, _line(o, f"sent {sent_str} ({age}d ago) - follow-up in {left}d")))
        elif o["status"] == "memo_drafted":
            a = o.get("latest_alignment")
            a_s = f"{a:.2f}" if isinstance(a, (int, float)) else "?"
            where = (f"draft in report {o['draft_date']}" if o.get("draft_date")
                     else "draft ready")
            ready.append((-(a or 0), _line(o, f"{where} (align {a_s})")))
        elif o["status"] == "followed_up":
            fup_sent.append((age, _line(o, f"sent {sent_str}, followed up ({age}d since first email) - let it go")))
        else:  # replied / rejected
            closed.append((0, _line(o, "replied - yours now" if o["status"] == "replied" else "dropped")))

    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# Outreach tracker - regenerated {today} (edit boxes, save; swarm syncs nightly)",
        "#",
        "# [sent] [follow-up] [replied] [drop]   - mark with x. Marks only move forward.",
        "# Everything after '|' is rewritten nightly (day counts etc.) - edit only boxes.",
        "",
        f"## Follow-up due (one short follow-up, then stop) - {len(due)}", "",
        *(l for _, l in sorted(due, reverse=True)), "",
        f"## Ready to send (drafts in today's report) - {len(ready)}", "",
        *(l for _, l in sorted(ready)), "",
        f"## Waiting (contacted <{FOLLOWUP_DUE_DAYS}d ago) - {len(waiting)}", "",
        *(l for _, l in sorted(waiting, reverse=True)), "",
        f"## Followed up - no further action - {len(fup_sent)}", "",
        *(l for _, l in sorted(fup_sent, reverse=True)), "",
        f"## Closed - {len(closed)}", "",
        *(l for _, l in closed), "",
    ]
    with open(TRACKER_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"[Tracker] regenerated -> {TRACKER_PATH} "
          f"({len(due)} follow-ups due, {len(ready)} ready to send)")
    return TRACKER_PATH
