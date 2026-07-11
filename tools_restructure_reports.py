#!/usr/bin/env python3
"""One-shot: restructure existing job_swarm report dirs into the new
REVIEW_QUEUE.md (morning read) + INTEL.md (raw feeds) split that
js_graph.py now generates, and swap the dossiers' SQL footer for the
tracker/check-in instruction. Idempotent: skips dirs that already have
INTEL.md. Usage: restructure_reports.py <report_dir> [<report_dir>...]
Stdlib only, py3.9-safe (runs on the cluster login node too).
"""
import os
import re
import sys

INTEL_SECTIONS = (
    "Watchlist", "Direct openings", "Fresh capital", "Pre-posting window",
    "First Phase III", "Phase II readouts", "Layoff radar", "Expansion radar",
    "10-K watch", "Roles that don't exist yet", "Comp observatory",
    "Possible duplicate orgs", "Run stats",
)


def wclip(s, n):
    s = (s or "").strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (cut or s[:n]) + "…"


def loc_short(loc):
    if not loc or loc in "—-":
        return "-"
    parts = [p.strip() for p in loc.split(";") if p.strip()]
    if len(parts) > 1:
        return "%s +%d" % (parts[0], len(parts) - 1)
    toks = [t.strip() for t in loc.split(",")]
    if len(toks) > 3:
        return "%s +%d" % (toks[0], len(toks) - 1)
    return ", ".join(toks[:2]) if len(toks) > 2 else loc


def split_sections(text):
    """-> (preamble_lines, [(title, lines)])."""
    pre, sections, cur = [], [], None
    for ln in text.splitlines():
        if ln.startswith("## "):
            cur = (ln[3:].strip(), [])
            sections.append(cur)
        elif cur is None:
            pre.append(ln)
        else:
            cur[1].append(ln)
    return pre, sections


def sec_is(title, key):
    return title.lower().startswith(key.lower())


def dossier_bits(path):
    """Extract (bottleneck, intervention_vector) from a dossier file."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return "", ""
    bot = ""
    m = re.search(r"## Audited bottleneck\s*\n(.+?)\n\s*\n", text, re.S)
    if m:
        bot = " ".join(m.group(1).split())
    vec = ""
    m = re.search(r"\*\*Intervention vector:\*\*\s*(.+)", text)
    if m:
        vec = m.group(1).strip()
    return bot, vec


def fix_dossier_footer(path):
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return
    m = re.search(r"^# (.+)$", text, re.M)
    org_name = m.group(1).strip() if m else "them"
    org_key = re.search(r"org_key='([^']+)'", text)
    key_s = " for `%s`" % org_key.group(1) if org_key else ""
    new_note = ("> Send manually from your own address. After sending, mark "
                "**sent**%s in the outreach tracker - or just say it in a "
                "check-in (\"emailed %s\") and it's recorded for you."
                % (key_s, org_name))
    out, changed = [], False
    for ln in text.splitlines():
        if "UPDATE orgs SET" in ln:
            changed = True
            continue
        if ln.startswith("> Send manually from your own address. After sending, mark it:"):
            out.append(new_note)
            changed = True
            continue
        out.append(ln)
    if changed:
        open(path, "w", encoding="utf-8").write("\n".join(out) + ("\n" if text.endswith("\n") else ""))


def build_memo_list(rows, run_dir):
    lines = [
        "## Tonight's memo drafts (%d) - what they're stuck on, and your angle" % len(rows), "",
        "Alignment is the 70B auditor's 0-1 rubric score; prescore is the "
        "quant filter that shortlisted the org. Trust the dossier's evidence "
        "over either number - `insufficient_history` means the corpus holds "
        "only a few documents on them.", "",
    ]
    for i, (org, align, prescore, regime, fname) in enumerate(rows, 1):
        try:
            pre_s = "%.2f" % float(prescore)
        except ValueError:
            pre_s = prescore
        bot, vec = dossier_bits(os.path.join(run_dir, fname))
        lines.append("%d. **%s** - align %s · prescore %s · %s - [dossier + draft](%s)"
                     % (i, org, align, pre_s, regime, fname))
        lines.append("   - *Stuck on:* %s" % (wclip(bot, 240) or "see dossier"))
        lines.append("   - *Your angle:* %s" % (wclip(vec, 120) or "see dossier"))
        lines.append("")
    return lines


def parse_memo_table(sec_lines):
    rows = []
    for ln in sec_lines:
        m = re.match(r"\|\s*(\d+)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|"
                     r"\s*\[([^\]]+)\]", ln)
        if m:
            rows.append(tuple(x.strip() for x in m.groups()[1:6]))
    return rows


def slim_openings(sec_lines):
    """Old full openings table -> slim top-12 rows (skip EVERGREEN/GHOST)."""
    out = []
    for ln in sec_lines:
        m = re.match(r"\|([^|]+)\|([^|]+)\|(.+)\|([^|]+)\|([^|]+)\|([^|]+)\|"
                     r"\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*$", ln)
        if not m:
            continue
        score, fit, title, opentag, loc, comp, prov, url = (x.strip() for x in m.groups())
        if not re.match(r"[\d.]+$", score):
            continue
        if "EVERGREEN" in opentag or "GHOST" in opentag:
            continue
        # titles were hard-cut at 70 chars at generation - drop the stub word
        if len(title) >= 69 and not title.endswith("…"):
            title = wclip(title.rsplit(" ", 1)[0], 70) if " " in title else title
            if not title.endswith("…"):
                title += "…"
        comp_short = comp
        cm = re.match(r"~\$([\d,]+)\s*\(H-1B median", comp)
        if cm:
            comp_short = "~$%dK est." % (int(cm.group(1).replace(",", "")) // 1000)
        try:
            score_s = "%.2f" % float(score)
        except ValueError:
            score_s = score
        out.append("| %s | %s | %s | %s | %s | [%s](%s) |"
                   % (score_s, title, opentag, loc_short(loc), comp_short, prov, url))
        if len(out) >= 12:
            break
    return out


def build_digest(sections):
    d = {t: v for t, v in sections}

    def sec(key):
        for t, v in sections:
            if sec_is(t, key):
                return v
        return []

    digest = []
    raises = [ln for ln in sec("Fresh capital") if ln.startswith("- [Form D:")]
    if raises:
        names = []
        for ln in raises[:3]:
            m = re.match(r"- \[Form D: (.+?) raised (\$[\d,]+)", ln)
            if m:
                amt = int(m.group(2).replace("$", "").replace(",", ""))
                a = ("$%.1fB" % (amt / 1e9) if amt >= 1e9 else
                     "$%dM" % (amt / 1e6) if amt >= 1e6 else "$%d" % amt)
                names.append("%s (%s)" % (wclip(m.group(1), 32), a))
        more = ", +%d more" % (len(raises) - 3) if len(raises) > 3 else ""
        digest.append("- **Fresh capital** (%d raises; hiring follows in 30-90 "
                      "days): %s%s" % (len(raises), ", ".join(names), more))
    pre = [ln for ln in sec("Pre-posting window") if ln.startswith("- **")]
    if pre:
        names = [re.sub(r"- \*\*(.+?)\*\*.*", r"\1", ln) for ln in pre[:3]]
        more = ", +%d more" % (len(pre) - 3) if len(pre) > 3 else ""
        digest.append("- **Pre-posting window** (%d funded orgs with no live "
                      "openings - the best cold-outreach slot): %s%s"
                      % (len(pre), ", ".join(names), more))
    bursts = [ln for ln in sec("Roles that don't exist yet")
              if "HIRING BURST" in ln]
    if bursts:
        names = [re.sub(r"- \*\*(.+?)\*\*.*", r"\1", ln) for ln in bursts[:4]]
        digest.append("- **Hiring bursts right now:** " + ", ".join(names))
    tks = [ln for ln in sec("10-K watch") if ln.startswith("- **")]
    if tks:
        names = []
        for ln in tks[:3]:
            m = re.match(r"- \*\*(.+?)\*\* [—-]+ \+(\d+)", ln)
            if m:
                names.append("%s (+%s)" % (wclip(m.group(1), 28), m.group(2)))
        if names:
            digest.append("- **R&D budgets drifting toward ML/data (10-K "
                          "language):** " + ", ".join(names))
    n_p3 = sum(1 for ln in sec("First Phase III") if ln.startswith("- **"))
    n_p2 = sum(1 for ln in sec("Phase II readouts") if ln.startswith("- **"))
    if n_p3 or n_p2:
        digest.append("- **Biostat channel:** %d sponsors entered their first "
                      "Phase III and %d Phase II readouts landed - biostat "
                      "hiring follows in 3-6 months" % (n_p3, n_p2))
    n_warn = sum(1 for ln in sec("Layoff radar") if ln.startswith("- "))
    if n_warn:
        digest.append("- **Layoff radar:** %d WARN filings tracked; memos to "
                      "those orgs are auto-suppressed" % n_warn)
    return digest


def restructure(run_dir):
    qpath = os.path.join(run_dir, "REVIEW_QUEUE.md")
    ipath = os.path.join(run_dir, "INTEL.md")
    if not os.path.exists(qpath):
        print("skip (no queue): %s" % run_dir)
        return
    if os.path.exists(ipath):
        print("skip (already restructured): %s" % run_dir)
        # still fix dossier footers if an earlier partial pass missed them
        for f in os.listdir(run_dir):
            if re.match(r"\d\d_.+\.md$", f):
                fix_dossier_footer(os.path.join(run_dir, f))
        return
    text = open(qpath, encoding="utf-8").read()
    run_date = os.path.basename(run_dir.rstrip("/"))
    pre, sections = split_sections(text)

    # preamble: drop the /scratch EXAMPLES pointer, add the split note
    pre = [ln for ln in pre if "EXAMPLES" not in ln and "quant_swarm" not in ln]
    while pre and pre[-1] == "":
        pre.pop()
    pre += ["",
            "*This file is the morning read: drafts to send, openings worth "
            "applying to, and a digest of tonight's market signals. The full "
            "raw feeds live in [INTEL.md](INTEL.md). Unfamiliar number? Tap "
            "it in the dashboard for a definition.*", ""]

    intel = ["# Market Intel - %s" % run_date, "",
             "The raw feeds behind tonight's queue - everything the swarm "
             "drank, unabridged. Nothing here demands action; "
             "[REVIEW_QUEUE.md](REVIEW_QUEUE.md) already distilled it. Open "
             "this when you want the evidence behind a signal.", ""]
    queue = list(pre)
    trailing = []          # the report pointers at the bottom of the old file
    memo_rows = parse_memo_table(pre)   # the memo table sits in the preamble
    openings_full = None

    for title, body in sections:
        if not memo_rows:
            memo_rows = parse_memo_table(body)
        if any(sec_is(title, k) for k in INTEL_SECTIONS):
            if sec_is(title, "Direct openings"):
                openings_full = (title, body)
                intel += ["", "## Direct openings - full table"] + body
            else:
                intel += ["", "## " + title] + body
            # pointer lines that were appended inside the run-stats section
            for ln in body:
                if re.match(r"\*\*(Proof-of-work|Tailored|Resume audit|"
                            r"Search-vocabulary)", ln):
                    trailing.append(ln)
        else:
            queue += ["", "## " + title] + body

    # memo table -> annotated list (drop the raw table from the queue)
    memo_tbl_re = re.compile(r"^\|.*\|$")
    queue = [ln for ln in queue if not memo_tbl_re.match(ln)]
    if memo_rows:
        queue += build_memo_list(memo_rows, run_dir)

    # slim openings for the queue
    if openings_full:
        rows = slim_openings(openings_full[1])
        if rows:
            queue += ["",
                      "## Direct openings - the top %d tonight (live on company "
                      "boards, not LinkedIn)" % len(rows), "",
                      "Ranked by fit and hiring-failure forensics; EVERGREEN/GHOST "
                      "postings are already filtered out here. The full table "
                      "(every row, every tag, the scoring formula) is in "
                      "[INTEL.md](INTEL.md).", "",
                      "| Score | Role | Open | Where | Comp | Apply |",
                      "|-------|------|------|-------|------|-------|"] + rows

    digest = build_digest(sections)
    if digest:
        queue += ["", "## Market signals this week - the 30-second version",
                  ""] + digest + ["", "Every bullet's full feed: [INTEL.md](INTEL.md)"]

    # de-dup the trailing pointer lines out of intel, keep them in the queue
    intel = [ln for ln in intel
             if not re.match(r"\*\*(Proof-of-work|Tailored|Resume audit|"
                             r"Search-vocabulary)", ln)]
    if trailing:
        queue += [""] + trailing

    open(ipath, "w", encoding="utf-8").write("\n".join(intel) + "\n")
    open(qpath, "w", encoding="utf-8").write("\n".join(queue) + "\n")
    for f in os.listdir(run_dir):
        if re.match(r"\d\d_.+\.md$", f):
            fix_dossier_footer(os.path.join(run_dir, f))
    print("restructured: %s (queue %d lines, intel %d lines)"
          % (run_dir, len(queue), len(intel)))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        restructure(d)
