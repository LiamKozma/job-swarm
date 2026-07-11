#!/usr/bin/env python3
"""Second pass over already-restructured report dirs: queue v2.

- header intro shortened, morning-read note -> one-line INTEL pointer
- metric-explainer paragraphs removed (tap-to-explain owns definitions)
- memo entries: clipped 'Stuck on'/'Your angle' replaced with FULL text
  re-extracted from each dossier
- openings section: geo-filtered (US/CA, Europe, AU/NZ), honest wording,
  moved AFTER the market-signals digest (email-first ordering)
- dossier headers: numbers rounded (prescore 0.557347 -> 0.56 etc.)

Idempotent. Usage: v2_pass.py <run_dir> [...]  Stdlib only, py3.9-safe.
"""
import os
import re
import sys

INTRO_NEW = ("The swarm researched, scored, and drafted overnight. You review, "
             "edit, and **send from your own email** - plain text, Tue-Thu "
             "9-11am recipient time, no links in a first email.")
OPEN_BLURB = ("*Straight from company ATS boards - fresher than aggregators, "
              "though some are also cross-posted elsewhere. Everything "
              "filtered out (other regions, EVERGREEN/GHOST) is in "
              "[INTEL.md](INTEL.md).*")

_GEO_ALLOW = re.compile(
    r"\b(united states|usa|u\.s\.|remote|canada|toronto|vancouver|montreal|"
    r"australia|sydney|melbourne|brisbane|new zealand|auckland|"
    r"united kingdom|uk|england|london|scotland|ireland|dublin|"
    r"germany|berlin|munich|france|paris|netherlands|amsterdam|"
    r"switzerland|zurich|zug|geneva|sweden|stockholm|denmark|copenhagen|"
    r"norway|oslo|finland|helsinki|spain|madrid|barcelona|italy|milan|rome|"
    r"austria|vienna|belgium|brussels|poland|warsaw|portugal|lisbon|"
    r"czech|prague|luxembourg|europe)\b", re.I)
_GEO_BLOCK = re.compile(
    r"\b(india|hyderabad|bangalore|bengaluru|mumbai|pune|chennai|gurgaon|"
    r"gurugram|noida|delhi|hong kong|singapore|japan|tokyo|osaka|china|"
    r"shanghai|beijing|shenzhen|hangzhou|korea|seoul|taiwan|taipei|"
    r"dubai|abu dhabi|uae|saudi|qatar|israel|tel aviv|brazil|sao paulo|"
    r"mexico|argentina|colombia|chile|philippines|manila|indonesia|jakarta|"
    r"vietnam|hanoi|thailand|bangkok|malaysia|kuala lumpur|nigeria|lagos|"
    r"kenya|nairobi|south africa|egypt|cairo|turkey|istanbul)\b", re.I)


def geo_ok(loc):
    if not loc:
        return True
    if _GEO_ALLOW.search(loc):
        return True
    return not _GEO_BLOCK.search(loc)


def fmt(x):
    try:
        return "%g" % round(float(x), 2)
    except (TypeError, ValueError):
        return "?"


def dossier_field(run_dir, fname, which):
    try:
        text = open(os.path.join(run_dir, fname), encoding="utf-8").read()
    except OSError:
        return ""
    if which == "bottleneck":
        m = re.search(r"## Audited bottleneck\s*\n(.+?)\n\s*\n", text, re.S)
        return " ".join(m.group(1).split()) if m else ""
    m = re.search(r"\*\*Intervention vector:\*\*\s*(.+)", text)
    return m.group(1).strip() if m else ""


def fix_dossier_numbers(path):
    text = open(path, encoding="utf-8").read()
    orig = text
    text = re.sub(r"(\*\*Alignment \(LLM audit\):\*\* )([0-9.eE+-]+)",
                  lambda m: m.group(1) + fmt(m.group(2)), text)
    text = re.sub(r"(\*\*Prescore \(quant filter\):\*\* )([0-9.eE+-]+)",
                  lambda m: m.group(1) + fmt(m.group(2)), text)
    text = re.sub(r"hurdle=([0-9.eE+-]+|None)",
                  lambda m: "hurdle=" + fmt(m.group(1)), text)
    text = re.sub(r"escalation=([0-9.eE+-]+|None)",
                  lambda m: "escalation=" + fmt(m.group(1)), text)
    if text != orig:
        open(path, "w", encoding="utf-8").write(text)


def fix_queue(run_dir):
    qp = os.path.join(run_dir, "REVIEW_QUEUE.md")
    if not os.path.exists(qp):
        return False
    lines = open(qp, encoding="utf-8").read().splitlines()
    out, open_sec, in_open = [], [], False
    cur_dossier = None
    for ln in lines:
        if ln.startswith("The swarm researched, scored, and drafted"):
            out.append(INTRO_NEW)
            continue
        if ln.startswith("*This file is the morning read") or \
           ln.startswith("*Full raw feeds:"):
            out.append("*Full raw feeds: [INTEL.md](INTEL.md).*")
            continue
        if ln.startswith("Alignment is the 70B auditor's"):
            continue
        if ln.startswith("Every bullet's full feed:"):
            continue
        if ln.startswith("## Direct openings"):
            in_open = True
            open_sec = []
            continue
        if in_open:
            if ln.startswith("## ") or ln.startswith("**Proof-of-work") or \
               ln.startswith("**Tailored") or ln.startswith("**Resume audit") or \
               ln.startswith("**Search-vocabulary"):
                in_open = False           # fall through to normal handling
            else:
                open_sec.append(ln)
                continue
        m = re.match(r"^\d+\. \*\*.*\[dossier \+ draft\]\(([^)]+)\)", ln)
        if m:
            cur_dossier = m.group(1)
            out.append(ln)
            continue
        m = re.match(r"^(\s+- \*(Stuck on|Your angle):\*) ", ln)
        if m and cur_dossier:
            which = "bottleneck" if m.group(2) == "Stuck on" else "vector"
            full = dossier_field(run_dir, cur_dossier, which)
            out.append("%s %s" % (m.group(1), full) if full else ln)
            continue
        out.append(ln)

    # rebuild the openings section: geo filter + new heading/blurb
    new_open = []
    if open_sec:
        rows = [ln for ln in open_sec if re.match(r"\|\s*[\d.]+\s*\|", ln)]
        kept = []
        for ln in rows:
            cells = [c.strip() for c in ln.split("|")]
            where = cells[4] if len(cells) > 5 else ""
            if geo_ok(where):
                kept.append(ln)
        if kept:
            new_open = [
                "", "## Direct openings - top %d, your geography only "
                "(US/CA · Europe · AU/NZ)" % len(kept), "", OPEN_BLURB, "",
                "| Score | Role | Open | Where | Comp | Apply |",
                "|-------|------|------|-------|------|-------|",
            ] + kept
        else:
            new_open = ["", "## Direct openings", "",
                        "_Nothing in your geography worth a slot tonight - the "
                        "unfiltered table is in [INTEL.md](INTEL.md)._"]

    # insert openings AFTER the digest section (or before trailing pointers)
    final = []
    inserted = False
    idx_digest = next((i for i, ln in enumerate(out)
                       if ln.startswith("## Market signals")), None)
    if idx_digest is not None:
        end = next((i for i in range(idx_digest + 1, len(out))
                    if out[i].startswith("## ") or out[i].startswith("**")),
                   len(out))
        final = out[:end] + new_open + out[end:]
        inserted = True
    if not inserted:
        idx_ptr = next((i for i, ln in enumerate(out)
                        if re.match(r"\*\*(Proof-of-work|Tailored|Resume audit|"
                                    r"Search-vocabulary)", ln)), len(out))
        final = out[:idx_ptr] + new_open + [""] + out[idx_ptr:]

    # collapse triple blank lines
    text, prev_blank = [], 0
    for ln in final:
        prev_blank = prev_blank + 1 if ln == "" else 0
        if prev_blank <= 2:
            text.append(ln)
    open(qp, "w", encoding="utf-8").write("\n".join(text) + "\n")
    return True


def main(run_dir):
    if not fix_queue(run_dir):
        print("skip (no queue):", run_dir)
        return
    for f in os.listdir(run_dir):
        if re.match(r"\d\d_.+\.md$", f):
            fix_dossier_numbers(os.path.join(run_dir, f))
    print("v2:", run_dir)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
