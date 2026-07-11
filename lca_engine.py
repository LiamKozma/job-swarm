"""
Job Swarm - H-1B LCA salary-prior builder (standalone; run once per quarter
on the LOGIN node - it needs internet and pandas, not the cluster containers).

The DOL publishes every H-1B Labor Condition Application quarterly, including
the ACTUAL salary the employer attests to paying, by employer × job title ×
location. That is comp ground truth nobody job-hunting uses. This script
distills it into per-employer medians for statistics/ML/quant titles and
writes them into the swarm DB's salary_priors table, where the review queue
uses them to impute comp for postings with no disclosed salary - directly in
service of the "$150k+ first job" goal.

Get the file (a few hundred MB .xlsx):
  https://www.dol.gov/agencies/eta/foreign-labor/performance
  -> "Disclosure Data" -> LCA Programs (H-1B, H-1B1, E-3) -> latest
    "LCA Disclosure Data FY20XX QX.xlsx"

Usage (login node):
  pip install --user pandas openpyxl        # once, if missing
  python3 lca_engine.py /path/to/LCA_Disclosure_Data_FY2026_Q2.xlsx
  python3 lca_engine.py 'https://www.dol.gov/.../LCA_Disclosure_Data_FY2026_Q2.xlsx'

Idempotent: re-running replaces the priors. Also writes
~/job_swarm_reports/LCA_TOP_EMPLOYERS.csv - browse it directly: it is a
ranked list of employers PROVEN to pay ≥$150k for your job titles.
"""

import os
import sys
import urllib.request

import swarm_db

# A filing counts if the job title or SOC title contains one of these.
RELEVANT_TERMS = [
    "statist", "data scien", "machine learning", "quant", "research scien",
    "biostat", "ml engineer", "applied scien", "algorithm", "actuar",
    "computational", "data engineer", "ai engineer", "research engineer",
]

_WAGE_COLS = ["WAGE_RATE_OF_PAY_FROM", "WAGE_UNIT_OF_PAY",
              "EMPLOYER_NAME", "JOB_TITLE", "SOC_TITLE", "FULL_TIME_POSITION"]

_ANNUALIZE = {"Year": 1.0, "Month": 12.0, "Bi-Weekly": 26.0,
              "Week": 52.0, "Hour": 2080.0}


def _load_frame(path_or_url: str):
    import pandas as pd
    path = path_or_url
    if path_or_url.startswith(("http://", "https://")):
        path = os.path.expanduser("~/job_swarm/lca_download.xlsx")
        print(f"Downloading {path_or_url} -> {path} (this is a big file)...")
        req = urllib.request.Request(
            path_or_url, headers={"User-Agent": "JobSwarmResearch/1.0"})
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    print(f"Reading {path} (usecols only - still takes a few minutes)...")
    return pd.read_excel(path, usecols=lambda c: c in _WAGE_COLS,
                         engine="openpyxl")


def build_priors(path_or_url: str) -> int:
    import pandas as pd

    df = _load_frame(path_or_url)
    n_total = len(df)

    hay = (df["JOB_TITLE"].fillna("") + " " + df["SOC_TITLE"].fillna("")).str.lower()
    mask = False
    for t in RELEVANT_TERMS:
        mask = mask | hay.str.contains(t, regex=False)
    df = df[mask].copy()

    factor = df["WAGE_UNIT_OF_PAY"].map(_ANNUALIZE)
    wage = pd.to_numeric(df["WAGE_RATE_OF_PAY_FROM"], errors="coerce") * factor
    df["annual"] = wage
    df = df[(df["annual"] >= 30_000) & (df["annual"] <= 2_000_000)]
    df = df[df.get("FULL_TIME_POSITION", "Y").fillna("Y") == "Y"]
    print(f"{len(df)} relevant full-time filings out of {n_total} total")

    grouped = (df.groupby(df["EMPLOYER_NAME"].str.strip())["annual"]
               .agg(["median", "count"]).reset_index())
    grouped = grouped[grouped["count"] >= 3]  # need a few filings for a stable median

    conn = swarm_db.connect()
    n_written = 0
    for _, row in grouped.iterrows():
        org_key = swarm_db.org_key_from_name(row["EMPLOYER_NAME"])
        swarm_db.set_salary_prior(conn, org_key, float(row["median"]),
                                  int(row["count"]))
        n_written += 1
    conn.commit()
    conn.close()
    print(f"{n_written} employer salary priors written to salary_priors")

    # The browsable artifact: employers proven to pay for these titles
    out_csv = os.path.expanduser("~/job_swarm_reports/LCA_TOP_EMPLOYERS.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    top = grouped.sort_values("median", ascending=False)
    top.columns = ["employer", "median_annual_salary", "n_filings"]
    top.to_csv(out_csv, index=False)
    n_150 = int((top["median_annual_salary"] >= 150_000).sum())
    print(f"Ranked employer list -> {out_csv}")
    print(f"{n_150} employers have a median ≥ $150,000 for your titles. "
          f"That CSV is a target list, not trivia.")
    return n_written


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    build_priors(sys.argv[1])
