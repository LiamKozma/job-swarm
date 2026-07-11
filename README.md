# Job Swarm - Predictive Strategic Placement on Sapelo2

An HPC-backed research swarm that inverts the job search: instead of applying
into ATS black boxes, it continuously monitors the **leading indicators of
hiring need** (grant disbursements, pre-prints, early-stage startup telemetry),
models each organization's semantic trajectory with the same Markov-Switching
GMM + Euler-Maruyama machinery as the quant swarm, has Llama-3-70B audit the
top matches against your expertise matrix, and drafts founder-directed
technical memos - which **you** review and send.

## Architecture

```
STAGE 1 - batch partition (16 CPU)               STAGE 2 - gpu_p (H100/A100)
┌─────────────────────────────┐                  ┌──────────────────────────────┐
│ ingest_engines.py           │                  │ vLLM: Qwen3-Next-80B (:8000) │
│                             │                  │  +Llama-3.3 critic :8001 on  │
│                             │                  │  4-GPU nights; A100 fallback │
│                             │                  │  tier serves Llama-3.3-AWQ   │
│  NSF / NIH / USAspending    │                  ├──────────────────────────────┤
│  arXiv (+OpenAlex affil.)   │   raw JSON       │ LangGraph DAG (js_graph.py)  │
│  YC dir · HN Who-is-hiring  │ ──on Lustre──>   │  profile_loader (+cfg advice)│
│  ClinicalTrials.gov v2      │                  │  trajectory_filter           │
│  USAJOBS (federal)          │                  │   - ST embeddings (CUDA)     │
│  ats_engine: 8 providers    │                  │   - δ-shift series per org   │
│   incl. Workday (pharma/    │                  │   - Markov-Switching GMM     │
│   FFRDC/finance)            │                  │   - Euler-Maruyama MC        │
│  formd_engine: SEC Form D   │                  │   - repost/freeze detection  │
│  tenk_engine: 10-K Item 1A  │                  │   - Hawkes arrival bursts    │
│  patents (USPTO ODP, keyed) │                  │  llm_audit (70B, JSON rubric)│
│  + sponsor seeds (NeurIPS)  │                  │  strategy_synthesis (memos,  │
└─────────────────────────────┘                  │    MMR-diversified)          │
                                                 │  application_forge (apps)    │
        SQLite corpus (home, backed up)          │  artifact_nominator (briefs) │
        grows every night -> trajectories         │  compile_review              │
        get denser over time                     └───────────────┬──────────────┘
                                                                 v
                                            ~/job_swarm_reports/YYYY-MM-DD/
                                            REVIEW_QUEUE.md + per-org dossiers
                                            -> YOU edit + send, then mark
                                              `js_review.py contacted <org>`
```

## The math (what the numbers in the report actually are)

All from `trajectory_engine.py`; the same Markov-switching machinery as the
quant swarm, applied to semantic drift instead of price returns.

**Embeddings.** Every document is split into 350-word chunks, embedded with
`all-MiniLM-L6-v2` (384-d), and mean-pooled to one vector `e_i` per doc.
Your `profile/` docs get the same treatment, pooled to one profile vector
`p`. Each org's docs are ordered by date.

**Alignment** (the `Align` column): 0.5 · cosine(recent state, pooled
profile `p`) + 0.5 · mean of the top-2 cosines against **facet**
embeddings. Facets are of two kinds (2026-07-07): each `expertise_matrix`
item embedded on its own, and each project write-up file embedded whole -
so "your CUDA project matches their kernel team" surfaces even when the
pooled profile leans elsewhere. The pooled vector sits between your skill
clusters, so pooled cosine alone under-scores an org that hits one skill
hard; the facet term fixes that. It answers: is
what this org is publishing *right now* close to what you do?

**δ-shift series.** For consecutive AUTHORED docs (grants, papers, YC/trial
descriptions - job postings and Form D boilerplate are excluded: a board's
posting variety is cross-sectional, not pivoting), `δ_i = (1 - cos(e_i,
e_{i+1})) / √Δt_days` - how far the org jumped in embedding space between
one public statement and the next, per √day (Brownian scaling: a 0.3 jump
over 2 days is violent, the same jump over 8 months is drift). NSF/NIH orgs
are keyed at the **PI level** (`PI · Institution`) so a university's regime
measures one lab's voice rather than cross-department heterogeneity. Stable voices take small steps in
a consistent direction; a team fighting a problem beyond its expertise pivots
through terminology space in large, erratic steps.

**Regime classification** (the `Regime` column; needs ≥5 authored docs = ≥4
shifts). A K=2 Gaussian HMM (hmmlearn, full covariance, 200 EM iterations,
best of 5 random restarts by likelihood) is fit to the normalized δ series,
yielding per-regime means/variances `(μ_k, σ_k²)` and a transition matrix
`A`. The high-variance regime is the **Hurdle State**; the org is labeled
`hurdle_state` iff its current decoded state is the high-variance one AND
`σ²_high > 1.5·σ²_low` (otherwise `steady_state`; fewer than 5 authored docs
-> `insufficient_history`).

**Escalation probability.** Euler-Maruyama Monte Carlo on the fitted
dynamics - 4,000 paths × 90 daily steps (Δt = 1/90, one quarter):

    δ_{t+1} = δ_t + μ_{k_t}·Δt + σ_{k_t}·√Δt·ε_t,   k_t ~ Markov(A),  ε_t ~ N(0,1)

`escalation = P(path mean > μ_high)`: the probability the org's semantic
volatility runs at Hurdle-State levels over the next quarter, regime
switching included.

**Hurdle probability.** `hurdle_prob = 0.6·1[in hurdle now] + 0.4·escalation`.
Thin-history orgs (2-4 docs) fall back to `min(2·mean(δ), 1)/2` - informative
but capped at 0.5 so no org is shortlisted on volatility it can't yet prove.

**Recency.** Exponential decay with a 90-day half-life on days since the
org's last dated document: `recency = 0.5^(age/90)` (0.3 if undated).

**Prescore** (the ranking):

    prescore = 0.65·max(alignment, 0) + 0.15·hurdle_prob + 0.20·recency

Alignment dominates (a volatile org you don't match is someone else's
opportunity); hurdle was DEMOTED to a tiebreaker by the 2026 red-team
(volatility can mean distress+freeze, not receptivity - the ε-greedy
outcomes will settle it), recency rewards active voices. Prescore > 0.35 ->
shortlisted; top 60 go to the 70B auditor - which sees real document
EXCERPTS (not titles) and none of the quant scores, so it stays an
independent second opinion. Audited alignment ≥ 0.45 -> draft memo (max
`MEMO_TOP_M` (default 8)/night AFTER the memo slot gate - see the
2026-07-06 layer below: academic-only orgs excluded, deliverable contact
required - of which up to `JOB_SWARM_EXPLORE_SLOTS` (default 2) are ε-greedy
picks from the 0.30-0.45 band - without exploration the weights above can
never be calibrated against reply outcomes). Every draft's features land in
`outreach_log`; tracker marks (replied/drop) close the loop, and once ~40-50
outcomes accumulate, fit a logistic regression on that table and let the
data set the weights. Each memo also gets a second **verification pass**
that deletes or hedges any claim the evidence excerpts don't support.
Hurdle State × high alignment = the target: a team publicly struggling with
a problem that looks like your resume.

**MMR portfolio selection (v4).** The memo slots are filled greedily by
`score - 0.15·max cos(candidate, already selected)` over org semantic
states: the nightly 12 are a diversified outreach portfolio, not 12
independent bets that can all land in one sector.

**Hawkes posting bursts (v4).** Per-org posting arrivals (first-seen times,
120-day window) fit a self-exciting process `λ(t) = μ + n·β Σ e^{-β(t-t_i)}`
(β = 1/7 day⁻¹, EM for μ and branching ratio n). The snapshot slope sees net
headcount; the Hawkes intensity sees the *arrival* process - `burst_ratio =
λ(now)/μ ≥ 2` means the org's hiring machine is running hot right now, even
if as many roles close as open. Needs ≥6 arrivals.

**Proof-of-work briefs (v4, `ARTIFACT_BRIEFS.md`).** The nominator picks ≤2
finalists (was 3; cut 2026-07-07) whose public documents expose a ≤2-day buildable problem and
plans an artifact (reproduce a figure, benchmark a tool on the H100s,
extend a result). Every claimed problem carries a **provenance chain**:
verbatim quotes mechanically checked against the source excerpts (VERIFIED/UNVERIFIED) with
links to the originating documents. Backtrace first, build second, email
third - the email then opens with work already done, which is the one move
no LinkedIn applicant can copy.

## Outreach policy (read this once)

The swarm **researches, scores, and drafts. It never sends.** Contact channels
recorded are public and self-attested: NSF PI emails in award records,
emails hirers post in Who-is-Hiring threads, company sites, and (since
2026-07-07, `github_engine.py`) commit-author emails on a target org's own
recent public repos - noreply/bot masks filtered, never probed against an
SMTP server, each with the commit URL to backtrace. No SMTP automation and
no bulk mail, ever: that violates GitHub's acceptable-use terms, torches
sender reputation, and converts worse than 5-10 personally-sent,
personally-verified memos per day. The review queue is designed for exactly that cadence.

Artifact briefs extend the same rule: the swarm plans the artifact, it never
produces or sends the "consulting". You backtrace the provenance chain,
build the thing yourself on the cluster, verify every claim, and only then
send an email describing work that actually exists. An UNVERIFIED quote
in a brief is a model error - never repeat it to a target.

## The GitHub copy (github.com/LiamKozma/job-swarm) and how to restore from it

The repo is an insurance copy of the CODE only. Everything reproducible or
private is gitignored: `data/` (raw payloads, rebuilt by any ingest run),
`inspect/`, reports, logs, the `*.sif` containers (11 GB, live in the parent
`quant_swarm/` dir), all `*.key` files, and `profile/` contents other than
its README (once the real resume lands there it must never be pushed).

If scratch gets purged, restore with:

```bash
mkdir -p /scratch/$USER/quant_swarm
cd /scratch/$USER/quant_swarm
git clone git@github.com:LiamKozma/job-swarm.git job_swarm
```

Then re-establish the pieces the repo does not carry:

1. **Containers** - copy `agent_env.sif` and `vllm_inference.sif` back into
   `/scratch/$USER/quant_swarm/` (parent dir), or rebuild them with
   `build_images.sh` / `build_vllm2.sbatch` from the quant swarm project.
2. **State** - `~/job_swarm/state/job_swarm.db` and the `~/job_swarm/*.key`
   files live on ZFS home, which survives the purge; nothing to do. If home
   is ever lost too, `backfill_ingest.sbatch` regrows ~3 years of corpus.
3. **Profile** - drop your resume + project write-ups into `profile/` as
   .txt/.md (see Phase 2 below).
4. Continue with the Runbook from Phase 1.

To push code changes back up: `git add -p && git commit && git push` from
the login node (the SSH deploy key is already on Sapelo2). Check
`git status` before every commit - anything under an ignored path showing
up as untracked means the .gitignore needs a new line, not a force-add.

## Runbook - follow top to bottom, no thinking required

The code already lives where the jobs expect it
(`/scratch/$USER/quant_swarm/job_swarm` - this directory). There is no
deploy/rsync step. Everything below is submitted from the login node
(apptainer only exists on COMPUTE nodes - never run it on the login node).

How the nightly is wired: `job_swarm_nightly.sh` is a thin submitter that
chains three real sbatch files - `nightly_ingest.sbatch` (CPU) ->
`nightly_analyze.sbatch` (GPU pool per job_swarm_nightly.sh: 4×H100 on
dual-model nights, JS_DUAL_MODEL=0 -> 2 GPU, A100 pair on the fallback
tier; a weekly heavy pass runs Saturday 04:00 UTC on 2x B200 once
`data/B200_OK` exists, and touching `~/job_swarm/state/FORCE_A100` pins
the A100 tier through H100 queue wedges; vLLM + DAG) -> `nightly_cleanup.sbatch`
(prints a PASS/FAIL verdict). Stage scripts must stay real files: heredocs
piped to sbatch stdin corrupted the submitted scripts (2026-07-02). The
analyze stage carries an `--exclude` list because only 4 of the 12 H100
nodes (ra5-2, ra7-2, ra8-3, ra8-4) have a driver new enough for the vLLM
container - rerun `probe_drivers.sh` and trim the list if GACRC upgrades.

### Phase 1 - backfill + first full run (synthetic profile) - PASSED 2026-07-02

```bash
cd /scratch/$USER/quant_swarm/job_swarm
mkdir -p ~/job_swarm/logs

# 1. One-time history backfill (~15-30 min, CPU). Gives every grant-funded
#    org ~3 years of trajectory so the GMM regime machinery works on day one.
sbatch backfill_ingest.sbatch          # DONE 2026-07-02

# 2. Full pipeline once, end to end:
bash job_swarm_nightly.sh              # DONE 2026-07-02 - verdict PASS
```

### Phase 1.5 - final smoke test of the v3+v4 changes (STILL the synthetic profile)

The 2026-07-02 evening round added Workday + USAJOBS + the removal channel
+ several fixes (v3), then the v4 creative round: the **artifact nominator**
(proof-of-work briefs with mechanically verified provenance chains), Hawkes
posting-arrival burst detection, MMR portfolio diversity in memo selection,
10-K risk-factor language ingestion for public corpus companies, H-1B LCA
filing volume as a demand boost, USPTO patent-application ingestion,
NeurIPS/ICML sponsor seeding, likely-technical-leads per finalist (published
authorship only), and per-posting likely-interview-topics. All validated
from the sandbox (unit + live-API where reachable) but not yet run
end-to-end on the cluster. Run Alex ONE more time and check the new
machinery before trusting it with your real profile:

```bash
# 0. One-time: the USAJOBS API key must exist (registered to the email in
#    JOB_SWARM_USER_AGENT). Already done 2026-07-02:
#      echo 'KEY' > ~/job_swarm/usajobs.key && chmod 600 ~/job_swarm/usajobs.key
#
# 0b. OPTIONAL (patents source stays silently skipped without it): free
#     USPTO Open Data key - MyUSPTO account at https://data.uspto.gov ->
#     API Manager, then:
#       echo 'KEY' > ~/job_swarm/uspto.key && chmod 600 ~/job_swarm/uspto.key
#     First keyed run: check the ingest log for "[Patents] query ... HTTP"
#     errors - if the q grammar needs a field-name fix, override
#     patents_query_template in job_swarm_config.json (no code change).

# 1. Run the pipeline:
cd /scratch/$USER/quant_swarm/job_swarm
bash job_swarm_nightly.sh
squeue --me

# 2. When the chain finishes, the verdict as usual:
cat $(ls -t ~/job_swarm/logs/cleanup_*.out | head -1)

# 3. v3 checklist - in the INGEST log (ls -t ~/job_swarm/logs/ingest_*.out):
#    [USAJOBS] N open federal postings         (N ≈ 30+; "skipped" = key missing)
#    [arXiv] affiliation backfill: N group orgs re-keyed   (N > 0 within a few
#        nights - day-fresh papers still miss; that is the cool-off working)
#    [ATS] postings count clearly above ~600   (15 Workday boards joined)
#    arXiv queries NOT 429ing                  (3s pacing gate)
#
# 4. v3 checklist - in the ANALYZE log:
#    [DB] 1 URL-mangled orgs re-keyed          (the kog fix, one-shot)
#    [DB] N postings flagged as reposts        (may be 0 for the first few
#        nights - reposts need boards to churn first)
#
# 5. v3 checklist - in the report (~/job_swarm_reports/$(date +%F)/):
#    - exactly ONE dossier per rank number (no duplicate 01_*.md)
#    - Direct openings: federal roles present with "closes YYYY-MM-DD",
#      Workday pharma roles present (novartis/pfizer/astrazeneca/...)
#    - new sections render: "Pre-posting window" and (when a sponsor
#      qualifies) "First Phase III"
#    - "X group" orgs start showing as "Author · Institution" over the
#      next few nights as the affiliation backfill catches up
#
# 6. v4 checklist - in the report:
#    - ARTIFACT_BRIEFS.md exists with ≤3 briefs; every brief has a
#      "Provenance" section; VERIFIED quotes link to real source docs (click one
#      and find the quote in the document - that IS the backtrace test);
#      UNVERIFIED lines are fine, that's the guard doing its job
#    - memo dossiers: "Likely technical leads" on research-heavy targets,
#      "Posting-arrival process (Hawkes)" where an org has ≥6 arrivals
#    - the 12 memos span >1 sector (MMR diversity - Alex's profile is
#      narrow, so expect at least fintech+biotech or similar)
#    - APPLICATIONS.md: each posting ends with "Likely interview topics"
#    - "10-K watch" section appears once tenk docs accumulate (needs the
#      analyze stage to have upserted them; may take a night)
#    - ingest log: "[10-K] N risk-factor sections" (N = 2×tenk_max_orgs
#      at steady state, less when few public cos are in the corpus yet),
#      "[Patents] ... skipped" unless you added the USPTO key,
#      "[WARN] N layoff notices" (keyless, ~5-40/night; accumulates
#      silently until the Layoff-radar section ships)
```

### Phase 2 - go live with your real profile

```bash
# 6. Restore the profile corpus (real resume + project write-ups, in place
#    since 2026-07-07): profile/ syncs from a private data repo - or just
#    drop your resume + project write-ups into profile/ as .txt/.md.
#    Include your GitHub URL somewhere in the documents. Historical note:
#    DO NOT send any memo drafted in the Alex Morgan smoke-test era (before
#    2026-07-07) - those cite the synthetic github.com/alexmorgan-smoketest;
#    the pre-send lint in js_graph.py blocks them mechanically.

# 7. Run once. The profile cache rebuilds automatically (SHA-1 over
#    profile/ contents): the 70B re-reads YOUR documents, re-embeds
#    everything, and regenerates CONFIG_SUGGESTIONS.md, where it proposes
#    search vocabulary derived from your actual expertise (grant keywords,
#    arXiv queries, USAJOBS terms, title filters):
bash job_swarm_nightly.sh

# 8. Read the report: review CONFIG_SUGGESTIONS.md and paste the keys you
#    accept into:
$EDITOR /scratch/$USER/quant_swarm/job_swarm/job_swarm_config.json
#    (Suggest-only by design - a bad search term silently mis-aims the
#    ingestion for weeks, so a human approves vocabulary changes.)

# 9. Run once more so the new vocabulary drives ingestion; this report is
#    the first fully-yours one:
bash job_swarm_nightly.sh

# 10. Automate - crontab -e on the login node, nightly at 04:00 UTC:
# 0 4 * * * /scratch/YOUR_USER/quant_swarm/job_swarm/job_swarm_nightly.sh >> $HOME/job_swarm/logs/cron_submit.log 2>&1
```

Reuses `agent_env.sif` and `vllm_inference.sif` from the quant swarm - every
dependency (langgraph, sentence-transformers, hmmlearn, numba, aiohttp) is
already in the image. No rebuild required.

**Source status (validated live 2026-07-01):** NSF (incl. PI emails from the
public award record), NIH RePORTER, arXiv, YC directory (1,300+ matched
companies, recent batches), and HN Who-is-Hiring (July 2026 thread) all pass.
SBIR is WAF-throttled - fails soft, and NSF's own SBIR/STTR awards arrive via
the NSF API anyway.

**Added 2026-07-02 (fail-soft, validate on first nightly):**
**USAspending.gov** - all federal grants beyond NSF/NIH (DOE, DARPA, ONR,
AFOSR, NASA fund most scientific-ML/HPC work; one keyless API).
**ClinicalTrials.gov v2** - newly registered industry Phase II/III trials:
the sponsor needs biostatisticians in 3-6 months, invisible to job boards
until far too late. **OpenAlex arXiv resolution** - papers without
affiliations resolve to real institutions via their arXiv DOIs (cached in
`meta`), killing the contactless "Lead Author group" pseudo-orgs.

**Alpha Source 4 - ATS shadow board** (`ats_engine.py`): seven providers'
public JSON APIs - Greenhouse/Lever/Ashby plus SmartRecruiters, Workable,
Recruitee (added 2026-07-02; mid-market and biotech boards the big three
miss), and **Workday** (added 2026-07-02 evening) - against (a) a seed list
spanning quant funds, AI labs, ML observability/eval startups (the best
employer class for a distribution-shift thesis: W&B, Arize, Fiddler,
Galileo, Patronus...), biotech, and top-of-market tech, and (b) every org the
corpus has ever surfaced, slug-probed once and cached in `ats_boards`.
Workday is the big-coverage one: nearly all of big pharma, the FFRDCs, and
legacy finance run on it - the exact employer class the grant/trial sources
score but that Greenhouse-style boards never see. 15 tenants validated live
(Novartis, Pfizer, BMS, Merck, AstraZeneca, Sanofi, Takeda, Amgen,
Regeneron, Gilead, RAND, St. Jude, Capital One, Vanguard, Mastercard);
Workday board ids are `tenant:wdhost:site` triples no slug heuristic can
guess, so it is **seeds-only** - add new triples to `SEED_BOARDS["workday"]`
after validating the CXS endpoint by hand. Postings track liveness
(`last_seen_at`): a role stays in the queue as long as it is actually on the
board, and its **days-open** count is tracked - but as DISPLAY and repost
forensics, not a boost: the 2026 red-team refuted the "long-open =
desperate team" theory (age predicts ghost jobs), and the application-
timing evidence runs the other way - interview odds are 2-3x in the first
24-48h and near zero past a week, so the queue surfaces Day-1 postings
first. Direct openings rank by **fit × repost-forensics × seniority**
(Staff/Manager/Supervisory titles an MS new grad won't clear are
down-weighted, "Graduate/Early Career" boosted), with comp imputed from
H-1B priors when undisclosed. Plus RemoteOK for remote roles.
Three later layers sharpen the ranking: a **hard-constraint clamp**
(2026-07-07) demotes postings whose text demands senior YOE, a PhD, or an
active clearance, with the reason flagged on the card; a **listwise
re-rank** (RankGPT-style sliding window over the top postings) lets the
70B reorder near-ties the scalar score can't split; and **taste learning**
(2026-07-08) embeds the cards you like in the app (plus your why-notes)
and multiplies posting scores by up to 1.15 toward the closest liked
vector - max over likes, neutral below cosine 0.5, tagged YOUR-STYLE on
the row. US-located roles rank first and non-US roles need a stricter
bar to surface at all (2026-07-09).

**Alpha Source 8 - USAJOBS** (added 2026-07-02 evening, validated live: 34
open postings, 28 with GS salary ranges): the federal channel - citizen-
eligible, WLB-sane, invisible to startup ATS providers. Free API key
registered to your email, read from `~/job_swarm/usajobs.key` (ZFS home -
survives the scratch purge; never lives in the repo) or env
`JOB_SWARM_USAJOBS_KEY`; fails soft when absent. Federal postings join
Direct openings and the application forge, but show `closes YYYY-MM-DD`
instead of days-open and are excluded from the staleness boost - federal
application windows are fixed by procedure, not by hiring failure.

**The removal channel** (added 2026-07-02 evening) - arrivals are only half
the signal; disappearances are the other half:
- **Repost detection** (`swarm_db.detect_reposts`): a posting that vanishes
  from a board and reappears days later under a new id is a role the org
  FAILED to fill - the repost resets the days-open clock, so without this
  it would masquerade as fresh. Embedding-matched (cos ≥ 0.92) per org;
  flagged REPOST in Direct openings with a score boost per repost.
- **Freeze filter** (`swarm_db.frozen_org_keys`): an org whose board went
  dark (0 postings now, ≥3 recently) is in a hiring freeze - memo drafting
  skips it rather than spending a slot on a door that just closed.
- **Pre-posting window** (queue section): orgs with a Form D raise ≤30 days
  old and NO live postings - money in the bank, roles still undefined, zero
  applicant competition. The single best cold-outreach slot; pairs with the
  ghost-role pitch.
- **First Phase III** (queue section): a sponsor's first pivotal trial in
  the corpus - biostatistics headcount scales 3-6 months later, long before
  any posting exists.

**Ghost Role Engine** (`ghost_engine.py`) - predicts jobs that don't exist
yet. K-means over the accumulated posting corpus builds a market-grounded role
taxonomy (title archetypes + median disclosed salary each). A target org with
no live posting gets its recent semantic state projected onto the archetype
centroids: the nearest archetype is the role its public output says it needs.
Nightly `board_snapshots` add a hiring-acceleration estimate (Theil-Sen
slope of posting counts - median of pairwise slopes, insensitive to single-day
posting bursts): a board growing fast with zero relevant roles is a gap the
memo can name. Surfaces in the queue's **Roles that don't exist yet** section
and feeds the memo synthesizer, plus a **Comp observatory** (quartiles of all
disclosed salaries). Validated on synthetic data: exact archetype recovery,
slope recovered to 3 decimal places.

**Warm Path Engine** (`warmpath_engine.py`) - for each finalist, queries
OpenAlex for co-publications between the target institution and your home
institution (default University of Georgia; `JOB_SWARM_HOME_INSTITUTION` to
change). A hit names the exact professors one intro away from the target.
Validated live: UGA<->Oak Ridge and UGA<->GTRI bridges found, including a 2026
UGA-ORNL distribution-shift paper. Warm intros ≫ cold memos - check this
section of the dossier before sending anything.

**Alpha Source 5 - SEC Form D** (`formd_engine.py`): daily EDGAR form index ->
`primary_doc.xml` per filing -> industry filter (drops pooled funds and real
estate). Fresh private raises with executive names from the federal
disclosure, ~15 days after first sale - hiring follows in 30-90 days. Live
test caught a $1.1B AI-infrastructure raise 2 days post-filing. Surfaces in
the **Fresh capital** section and joins the corpus for signal-stacking.

## One-time history backfill (run once, before or after the first nightly)

The GMM needs ≥5 dated docs per org; a single nightly leaves ~83% of orgs
with one. The backfill pulls ~3 years of NSF/NIH awards (paginated), arXiv
to depth 300/query, the past 12 HN Who-is-Hiring threads, and 45 days of
Form D raises (still inside their 30-90-day hiring window) - so trajectories
are dense on day one instead of day 30:

```bash
sbatch backfill_ingest.sbatch     # CPU only, ~15-30 min, safe to re-run
```

The next analyze stage absorbs it automatically (all pending payloads are
consumed and tracked in the `payloads` table - nothing is lost if several
ingests accumulate). The only signal that genuinely needs calendar time is
the hiring-acceleration slope: job boards have no public history.

## Daily loop (you, ~15 min)

Open **one file** in vim (over sshfs or on the login node) - it is your
whole workflow:

```bash
vim ~/job_swarm_reports/TRACKER.md
```

Every org is one line with four checkboxes - `[sent] [follow-up] [replied]
[drop]`. Put an `x` in a box when you do the thing; save; done. The nightly
run absorbs your marks (stamps the send date, stops re-targeting), then
rewrites the file with fresh day counts: follow-ups due float to the top,
new drafts appear under "Ready to send". Marks only move forward - clearing
a box rewinds nothing. Edit only the boxes; everything after `|` is
regenerated nightly.

## Reading a night's output (what each file is FOR)

**New here? Read `EXAMPLES/` first** - a real night's report (2026-07-02)
with every number annotated in place, plus a glossary README defining
alignment, prescore, regimes, hurdle/escalation, and both cosine scales.
Every nightly REVIEW_QUEUE.md links back to it. *(Note: the annotated
numbers predate the 2026-07-02 v2 engine changes - δ is now per-√day over
authored docs only, alignment blends facets, and the memo floor is 0.45 -
so expect the definitions to match but the magnitudes to differ.)*

Everything lands in `~/job_swarm_reports/$(date +%F)/`:

**`REVIEW_QUEUE.md`** - the strategic view. Top table = orgs that earned a
draft memo tonight (alignment ≥ 0.45), each linking to a numbered dossier
with the evidence, regime analysis, warm-path intros, and the memo text.
Below it: the watchlist (high prescore, no memo yet - `hurdle_state` means
their public output is pivoting fast, the hiring-need signature), direct
openings (live postings ranked by fit - HARD-TO-FILL, REPOST, salaries
when disclosed, `closes` dates on federal roles), fresh capital (who just
raised - hiring follows in 30-90 days), the pre-posting window (raised but
not yet hiring publicly), first-Phase-III sponsors, roles that don't exist
yet, and the comp observatory.

**`APPLICATIONS.md`** - the tactical view, for the top 8 postings. Per
posting: a short application note, your resume bullets REWRITTEN in that
posting's vocabulary, and `keywords_matched` / `keywords_missing`. This is
the resume-matching step: swap the suggested bullets into your master
resume, fill any `[ADD REAL NUMBER: ...]` placeholders with real figures,
apply. A missing keyword means either (a) you have the skill but your
resume doesn't say it - add it to the resume AND to `profile/` so future
scoring improves, or (b) you don't have it - leave it out; treat the list
as a study queue instead. The swarm never invents experience for you.

*(`RESUME_AUDIT.md` was retired 2026-07-08 - a hosted-model review of the
resume replaced the nightly 70B critique, which had gone stale between
profile edits.)*

**`CONFIG_SUGGESTIONS.md`** - regenerated whenever `profile/` changes: the 70B compares your new expertise matrix
against the pipeline's current search vocabulary and proposes replacements
(grant keywords, arXiv queries, USAJOBS terms, HN terms, title filters),
each change with a one-line rationale, plus a paste-ready JSON block.
Review it, paste the keys you accept into `job_swarm_config.json`, and the
NEXT night ingests with the new vocabulary. Suggest-only by design: the
swarm never rewrites its own eyes.

**`../TRACKER.md`** (one level up, persistent) - the state machine. See
"Daily loop" above: boxes are the only thing you edit.

CLI equivalents / extras:

```bash
python3 js_review.py show <org_key>                      # full dossier for one org
python3 js_review.py stats                               # corpus growth
# same effect as the tracker checkboxes (synced both ways):
python3 js_review.py contacted|followedup|replied|rejected <org_key>
python3 js_review.py followups                           # who's owed a follow-up
```

The rhythm that converts: 5-10 sends/day, every day, each one edited and
verified by you; one follow-up 5+ days later (most replies come from it);
walk the warm-path intros in person before any cold memo to the same org.

## Storage topology

| Tier | Path | Contents |
|------|------|----------|
| ZFS home (backed up) | `~/job_swarm/state/job_swarm.db` | corpus + embeddings + audits + memos (long-term memory) |
| Lustre scratch (30-day purge) | `/scratch/$USER/job_swarm/data/` | raw ingest payloads, shortlist/audit telemetry - all reproducible |
| Home | `~/job_swarm_reports/` | the human review queue |

## Tunables (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `JOB_SWARM_AUDIT_N` | 60 | shortlist depth sent to the 70B auditor |
| `JOB_SWARM_MEMO_M` | 8 | max memo drafts per night |
| `JOB_SWARM_MEMO_MIN_ALIGN` | 0.45 | LLM alignment floor for drafting (auditor is calibrated low; 0.55 left the memo budget ~empty) |
| `JOB_SWARM_EXPLORE_SLOTS` | 2 | ε-greedy memo slots from the 0.30-floor band (calibration data) |
| `JOB_SWARM_APP_K` | 8 | top direct openings that get per-posting application materials |
| `JOB_SWARM_EMBED_MODEL` | all-MiniLM-L6-v2 | swapping is now SAFE: the engine detects the change, wipes and re-embeds the whole corpus, and rebuilds the profile cache automatically. `nightly_analyze.sbatch` auto-selects the best cached embedder: Qwen3-Embedding-8B > Qwen3-Embedding-0.6B > bge-large-en-v1.5 |
| `JOB_SWARM_CONFIG` | - | path to the search-vocabulary JSON |
| `JOB_SWARM_USER_AGENT` | JobSwarmResearch/1.0 + `~/job_swarm/contact.email` | contact identity sent to SEC/OpenAlex/GitHub - put your email in the file once; no email ships in this repo |
| `JOB_SWARM_USAJOBS_KEY` | - | USAJOBS API key (fallback: `~/job_swarm/usajobs.key`); source skips itself when absent |

## Model upgrades (one-time downloads, login node)

Both upgrades are auto-detected by `nightly_analyze.sbatch` - until the
files exist, the old models keep the nightly alive:

```bash
pip install --user -U "huggingface_hub[cli]"

# 1. LLM: Llama-3.3-70B-AWQ - same size/args as Llama-3-70B, much better
#    instruction following and JSON discipline (fewer parse failures,
#    sharper audits). ~40 GB download.
huggingface-cli download casperhansen/llama-3.3-70b-instruct-awq \
  --local-dir /scratch/$USER/data/models/llama-3.3-70b-instruct-awq

# 2. Embedder: Qwen3-Embedding-8B is the standing model (~16 GB into the
#    HF cache; one-time re-embed of the corpus on first use):
sbatch download_qwen_embed_8b.sbatch
#    Fallbacks if it is absent: Qwen3-Embedding-0.6B, then bge-large-en-v1.5
#    (huggingface-cli download BAAI/bge-large-en-v1.5), then MiniLM.
```

The DAG also sends a JSON schema with every LLM call (vLLM guided decoding)
so audit/memo/application outputs cannot be malformed; on an older vLLM the
parameter is ignored gracefully and the parse ladder still applies.

## Comp ground truth - H-1B LCA priors (quarterly, 10 minutes)

The DOL publishes every H-1B LCA filing with the ACTUAL attested salary by
employer × job title. `lca_engine.py` distills the latest quarterly file
into per-employer medians for statistics/ML/quant titles:

```bash
# login node; needs pandas+openpyxl (pip install --user pandas openpyxl)
python3 lca_engine.py /path/to/LCA_Disclosure_Data_FY2026_Q2.xlsx
```

Two effects: postings with undisclosed salary get a "~$X (H-1B median)"
estimate in the Direct openings table, and
`~/job_swarm_reports/LCA_TOP_EMPLOYERS.csv` is a ranked list of employers
*proven* to pay ≥$150k for your titles - a target list, not trivia. File
source: dol.gov -> ETA -> Foreign Labor -> Performance -> Disclosure Data.

## The job-optimal layer (2026-07-06, from the deep-research blueprint)

Everything above still runs; this layer reorganizes the OUTPUT around lanes,
a fixed morning contract, and pre-registered decision rules. Design doc:
an internal deep-research blueprint (kept outside this repo).

**Lanes.** Every ITEMS.json card carries `lane`: `breadth` (Day-1 ATS
applications), `warm` (memos, follow-ups, contact/referral chores), `deep`
(artifact builds), `ops` (setup/config chores). The morning contract printed
atop REVIEW_QUEUE allocates 30/30/120 minutes in that order - the Day-1
timing edge expires first, so applications lead.

**Role discovery** (`discovery_engine.py`). Title-agnostic feasible-universe
map: KMeans over live posting requirement-text embeddings, profile-vs-
centroid cosine, lanes ranked by `fit x (0.5 + 0.5 x openness) x
log1p(volume)`. Writes `FEASIBLE_UNIVERSE.md` + a top-5 queue section.
Lane naming is one batched LLM pass (8 lanes/call - a single 24-lane
response outran the 300s HTTP timeout on the A100 tier); on failure the
exemplar-title fallback + a WARN line fire instead of silent degradation.

**Memo slot gate** (`_memo_gate`). Owner decision 2026-07-05: zero academic
memo slots (sources within `_ACADEMIC_SOURCES` are excluded outright),
deliverable contact (email/hn_user) required - otherwise the org becomes a
find-contact chore with a playbook; ATS/filing-only orgs route to the apply
channel. Two engines widen the deliverable-contact pool (2026-07-07):
`github_engine.py` resolves an org's GitHub login (domain-verified against
its website) and reads named committers on its freshest public repos, and
`research_engine.py` absorbs a nightly Gemini people-finding paste (strict
`PERSON:` trailer lines bound to known org keys); each run emits the
matching `DEEP_RESEARCH_PROMPT.md`. USPTO inventor names join as contacts
too. The artifact nominator applies the SAME academic exclusion
(2026-07-06): the deep lane targets industry.

**Cross-model critic.** `query_vllm(critic=True)` routes to a second model
family (VLLM_CRITIC_URL/_MODEL, Llama-3.3 on 4xH100 Qwen nights) with a
kill mandate on the FINAL memo draft, context-asymmetric (critic sees only
the draft). Kill -> lint line + dossier banner + card meta; disagreement
escalates to the human, never silently drops. Single-server nights label
verdicts "same-model (no second server)" - weaker evidence, honestly marked.

**Overnight scaffolds.** The top 2 artifact briefs get `SCAFFOLD_<org>.md` -
a PROJECT SPEC, never code (owner decision 2026-07-07): provenance cited
into the org's own documents, prose what-to-build, success criteria,
resources, presentation notes, under a no-fenced-code-blocks hard rule.
The design, every line of code, the run, and every published number stay
human.

**Funnel instrumentation.** App taps write DECISIONS.jsonl ->
`funnel_events` (UNIQUE ts,item_id,action - idempotent absorb). Cards carry
lane / freshness_days / artifact_depth / critic verdict; done cards offer
outcome chips (replied/screen/interview/offer). REVIEW_QUEUE renders a
per-lane tally once executed actions exist. Decision rules are
PRE-REGISTERED in `jobs/PREREGISTRATION.md` (data repo): day-30 execution
check, day-60 artifact kill switch (~15 sends < 5% reply -> kill deep lane),
day-90 lane reallocation by interviews-per-human-hour. Amendments append,
never edit.
