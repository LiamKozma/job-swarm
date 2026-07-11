"""
Job Swarm - review-queue CLI (run on the login node; no GPU needed).

  python3 js_review.py queue            # current drafted memos awaiting review
  python3 js_review.py show <org_key>   # full dossier data for one org
  python3 js_review.py contacted <org_key>   # mark as sent (by YOU, manually)
  python3 js_review.py followedup <org_key>  # follow-up sent (one per org, ever)
  python3 js_review.py replied  <org_key>    # they answered - you own it now
  python3 js_review.py rejected <org_key>    # drop an org from future runs
  python3 js_review.py followups            # contacted ≥4 days ago, no reply
  python3 js_review.py stats            # corpus / pipeline counts
  python3 js_review.py merges           # likely duplicate orgs (shared domain)
  python3 js_review.py merge <keep> <absorb>  # fold duplicate org into survivor
"""

import json
import sys

import swarm_db


def cmd_queue(conn):
    rows = conn.execute(
        "SELECT m.org_key, o.display_name, m.run_date, m.subject, o.latest_alignment, o.status "
        "FROM memos m JOIN orgs o USING (org_key) "
        "WHERE m.status = 'draft' AND o.status = 'memo_drafted' "
        "ORDER BY o.latest_alignment DESC"
    ).fetchall()
    if not rows:
        print("No drafts awaiting review.")
    for r in rows:
        print(f"[{r['run_date']}] align={r['latest_alignment']}  {r['org_key']}")
        print(f"    {r['display_name']} - {r['subject']}\n")


def cmd_show(conn, org_key):
    org = conn.execute("SELECT * FROM orgs WHERE org_key = ?", (org_key,)).fetchone()
    if org is None:
        sys.exit(f"Unknown org_key: {org_key}")
    print(json.dumps(dict(org), indent=2, default=str))
    memo = conn.execute(
        "SELECT * FROM memos WHERE org_key = ? ORDER BY run_date DESC LIMIT 1", (org_key,)
    ).fetchone()
    if memo:
        print(f"\n--- DRAFT MEMO ({memo['run_date']}) ---")
        print(f"Subject: {memo['subject']}\n\n{memo['body']}")
        print(f"\nContacts: {memo['contacts']}")
    audit = conn.execute(
        "SELECT * FROM audits WHERE org_key = ? ORDER BY run_date DESC LIMIT 1", (org_key,)
    ).fetchone()
    if audit:
        print(f"\n--- AUDIT ---\n{audit['bottleneck_diagnosis']}")
        print(f"Intervention: {audit['intervention_vector']}")


def cmd_set_status(conn, org_key, status):
    if status == "contacted":
        # Stamp the send date - the follow-up tracker keys off it
        cur = conn.execute(
            "UPDATE orgs SET status = ?, contacted_at = ? WHERE org_key = ?",
            (status, swarm_db._now(), org_key))
    else:
        cur = conn.execute("UPDATE orgs SET status = ? WHERE org_key = ?", (status, org_key))
    if cur.rowcount == 0:
        sys.exit(f"Unknown org_key: {org_key}")
    # Terminal marks also close the outreach_log row (calibration labels),
    # same as the TRACKER.md checkbox path.
    if status == "replied":
        swarm_db.record_outreach_outcome(conn, org_key, "replied")
    elif status == "rejected":
        swarm_db.record_outreach_outcome(conn, org_key, "dropped")
    conn.commit()
    print(f"{org_key} -> {status}")


def cmd_followups(conn):
    due = swarm_db.followups_due(conn, min_days=4)
    if not due:
        print("No follow-ups due. (Orgs appear here 4 days after 'contacted' "
              "until you mark them 'followedup', 'replied', or 'rejected'.)")
    for o in due:
        print(f"{(o.get('contacted_at') or '')[:10]}  {o['org_key']}  {o['display_name']}")


def cmd_merges(conn):
    groups = swarm_db.suggest_domain_merges(conn)
    if not groups:
        print("No shared-domain duplicates detected.")
        return
    print("Orgs sharing one website domain (likely the same company).\n"
          "Merge with: python3 js_review.py merge <keep_key> <absorb_key>\n")
    for domain, orgs in groups[:20]:
        print(f"{domain}:")
        for key, name in orgs:
            print(f"    {key:32s} {name}")


def cmd_merge(conn, keep_key, absorb_key):
    for key in (keep_key, absorb_key):
        if conn.execute("SELECT 1 FROM orgs WHERE org_key = ?", (key,)).fetchone() is None:
            sys.exit(f"Unknown org_key: {key}")
    n_docs = conn.execute("SELECT COUNT(*) FROM docs WHERE org_key = ?",
                          (absorb_key,)).fetchone()[0]
    answer = input(f"Fold '{absorb_key}' ({n_docs} docs) into '{keep_key}'? "
                   f"This is irreversible. [y/N] ")
    if answer.strip().lower() != "y":
        sys.exit("Aborted.")
    swarm_db.merge_orgs(conn, keep_key, absorb_key)
    print(f"{absorb_key} -> merged into {keep_key}")


def cmd_stats(conn):
    for label, q in [
        ("documents in corpus", "SELECT COUNT(*) FROM docs"),
        ("orgs tracked", "SELECT COUNT(*) FROM orgs"),
        ("audits stored", "SELECT COUNT(*) FROM audits"),
        ("memos drafted", "SELECT COUNT(*) FROM memos"),
        ("contacted (by you)", "SELECT COUNT(*) FROM orgs WHERE status='contacted'"),
    ]:
        print(f"{conn.execute(q).fetchone()[0]:6d}  {label}")
    print("\nBy status:")
    for r in conn.execute("SELECT status, COUNT(*) c FROM orgs GROUP BY status ORDER BY c DESC"):
        print(f"{r['c']:6d}  {r['status']}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    conn = swarm_db.connect()
    cmd = sys.argv[1]
    if cmd == "queue":
        cmd_queue(conn)
    elif cmd == "show" and len(sys.argv) == 3:
        cmd_show(conn, sys.argv[2])
    elif cmd in ("contacted", "rejected", "replied") and len(sys.argv) == 3:
        cmd_set_status(conn, sys.argv[2], cmd)
    elif cmd == "followedup" and len(sys.argv) == 3:
        cmd_set_status(conn, sys.argv[2], "followed_up")
    elif cmd == "followups":
        cmd_followups(conn)
    elif cmd == "merges":
        cmd_merges(conn)
    elif cmd == "merge" and len(sys.argv) == 4:
        cmd_merge(conn, sys.argv[2], sys.argv[3])
    elif cmd == "stats":
        cmd_stats(conn)
    else:
        sys.exit(__doc__)
    conn.close()


if __name__ == "__main__":
    main()
