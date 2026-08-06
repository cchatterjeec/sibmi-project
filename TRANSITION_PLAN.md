# Project Continuity Plan — Post-Internship

Last updated: 2026-08-06
Author: Chandrima Chatterjee
Purpose: everything needed to keep this project alive after O2/Harvard access ends.

---

## 1. Project abstract (working draft)

Type 2 Diabetes (T2D) is a heterogeneous condition, yet subtyping and diagnosing
efforts have historically relied on single lab-drawn values such as HbA1c and
fasting plasma glucose rather than the rich temporal structure continuous glucose
monitoring (CGM) offers. Utilizing the AI-READI cohort (N=2,280; healthy,
pre-diabetic, oral medication, and insulin-dependent participants), we
investigated whether CGM-derived features could predict physiologically
meaningful subgroups within T2D, motivated by prior findings that CGM
metrics — time-in-range, coefficient of variation, entropy rate — are strong
predictors of HbA1c and outperform the standard GMI conversion. We applied
k-means/spectral clustering to a five-feature space (Age, A1c, BMI, Insulin,
C-peptide) and identified meaningful clusters at k=4 within the
insulin-dependent subgroup, suggesting greater heterogeneity within this
population than previously appreciated. Characteristics of 3 clusters aligned
closely with subgroups identified in Ahlqvist et al., including Severe Insulin
Deficient Diabetes (SIDD, elevated A1C with moderate insulin) and Severe
Insulin Resistant Diabetes (SIRD, high BMI and insulin), alongside a cluster
suggestive of possible LADA/beta-cell exhaustion (older age, lower BMI, low
insulin/C-peptide). We then tested whether CGM alone, without any of the
clustering input variables, could recover this cluster structure, and found
that a CGM-only classifier predicted membership in all four clusters with
AUPRC substantially above baseline prevalence, indicating that glycemic
dynamics captured by CGM encode information relevant to diabetes subtype
beyond what a single A1C or insulin value provides. These results position CGM
as a low-burden, physiologically informative tool for T2D stratification,
capable of recovering clinically meaningful heterogeneity — particularly
within the insulin-dependent population — using passively collected data.
Future work will examine whether these CGM-predicted subgroups relate to
differential complication risk, seeking validation in an independent cohort
(GRADE).

**Framing note:** GRADE is a validation/extension piece, not a central
requirement of the paper. The core result stands on AI-READI alone.

---

## 2. Do this before your O2 account is deactivated

Ask IT/your PI for the exact deactivation date and treat everything below as
due the week before it, not the day of.

- [ ] **Commit and push all code to GitHub.** Remote is already configured:
  `https://github.com/cchatterjeec/sibmi-project`. Local `HEAD` matched
  `origin/main` as of 2026-08-06, but:
  - `insulin_dependent_k4_complications/` (the new complications-by-cluster
    analysis — `common.py`, `complications_by_cluster.py`,
    `complications_summary.csv`, two plots) is **untracked** — not on GitHub yet.
  - `ai_readi/vscode.out` shows as modified.
  - `git fetch` failed in this session with a credential error
    (`could not read Username for 'https://github.com'`) — confirm you can
    still authenticate (PAT / SSH key) and that a real push succeeds, don't
    trust the locally cached `origin/main` ref alone.
- [ ] **Note what GitHub does *not* cover.** `.gitignore` excludes
  `*.csv`, `*.parquet`, `*.pkl`, `*.h5`, `*.log` — so `final_df.csv`, the
  `ai_readi/preprocessed/*.parquet` files, and every cluster/regression output
  CSV live *only* on O2 right now. Only 405 tracked files / 119MB of code and
  figures are actually backed up by git.
- [ ] **Copy the non-code project state to Dropbox.** Estimated at ~2.0GB
  (repo is 2.7GB total minus 610MB `.venv` minus 119MB git-tracked = ~2.0GB of
  data/results). Your Dropbox free tier currently has **1.85GB free** — this
  is a tight or losing fit. Before copying, either free up Dropbox space,
  upgrade the plan, or triage: at minimum prioritize `final_df.csv` (1.4MB),
  `ai_readi/preprocessed/*.parquet` (small), and the small per-analysis
  `*_assignments.csv` / `*_profiles.csv` / `*_results.csv` files over the
  bulkier intermediate artifacts (`.bak` files, `__pycache__`, `errs_and_outs/`
  logs — skip these).
- [ ] **`requirements.txt` generated** (2026-08-06, 54 packages, Python
  3.9.25) — already saved to `/n/groups/patel/chandrima/requirements.txt` and
  ready to commit. This is what lets you rebuild the `.venv` anywhere else.
- [ ] **AI-READI clinical/medication data already on Dropbox** (done
  2026-08-06): `clinical_data/` (all 7 OMOP tables incl. `observation.csv`
  for medications), `participants.tsv`/`.json`, `dataset_description.json` —
  155MB, see `Dropbox/AIREADI_data/`.
- [ ] **Decide about the imaging modalities.** Raw OCT/OCTA/retinal photography
  data was *not* moved (1.3TB / 1.1TB / 163GB respectively — far beyond
  Dropbox capacity). If the microvascular-complications angle (see §4) ends up
  needing OCT images specifically, that requires either a much larger
  storage plan or continued O2/collaborator access — flag this early with your
  PI rather than discovering it's needed after access ends.
- [ ] **Export/save your calendar of standing meetings** (e.g. the Friday
  11:00am check-in) if it's a Harvard calendar — set up a recurring instructions
  doc or personal calendar invite so the cadence doesn't just disappear.

---

## 3. Ongoing access you'll need to arrange

| What | Status | Action |
|---|---|---|
| **O2 / HPC compute** | Ends with internship | Ask PI (Patel lab) whether a collaborator/guest O2 account is possible post-departure. If not, plan to run everything from the GitHub code + Dropbox data on personal hardware or a cloud VM — `requirements.txt` is ready for this. |
| **AI-READI full dataset** | Only the clinical/medication subset is off O2 | If you need more later (imaging, raw wearable files), you'll likely need either continued lab access or your own data use agreement with the AI-READI/Bridge2AI data repository (FAIRhub) — check whether that's available to a non-Harvard-affiliated researcher, or whether it needs to stay routed through your PI. |
| **GRADE study data** | Not yet obtained — this is an open to-do from your notes | GRADE is NIH/NIDDK-funded; data access for NIDDK-funded studies typically runs through the **NIDDK Central Repository** (repository.niddk.nih.gov), which usually requires a Data Use Agreement and often institutional sponsorship. Confirm: (1) whether your Harvard PI can sponsor the DUA even after you leave, or (2) whether you need a new sponsoring institution/PI. Resolve this *before* assuming GRADE access is easy to get independently. |
| **GitHub repo** | Configured, needs a final verified push | See checklist above. |
| **Dropbox** | Set up, 1.85GB free as of 2026-08-06 | Decide on paid tier if the full data/results set won't fit. |
| **Continued mentorship / co-authorship contact** | Informal | Get a personal (non-Harvard) email on file with your PI and any collaborators now, while you still have a reliable way to reach them through Harvard channels. |

---

## 4. Open research questions / next steps (from working notes)

**GRADE integration (secondary to the main finding, not central):**
- Read the GRADE primary results paper (NEJM) before assuming what's in the dataset.
- Once access is obtained, check CGM coverage — GRADE may have substantially
  less/less-frequent CGM per participant than AI-READI, which could limit
  what the by-arm analysis below can show.
- Test whether the AI-READI-derived clusters (or the CGM→cluster relationship)
  show up differently across GRADE's randomized intervention arms
  (metformin + second agent: sulfonylurea / DPP-4i / GLP-1RA / basal insulin) —
  this is the natural "does this generalize" extension once GRADE is in hand.

**Drug/medication analysis (data now available, unblocked):**
- `Dropbox/AIREADI_data/clinical_data/observation.csv` has the medication
  survey fields: `cmtrt_a1c` (oral A1C meds), `cmtrt_insln` (insulin
  injection), `cmtrt_glcs` (other glucose-lowering injectable),
  `cmtrt_lfst` (lifestyle-only), plus OTC meds (`cm_act`, `cm_ant`, `cm_asp`,
  `cm_dcg`, `cm_ibp`, `cm_slp`).
- Next step: compute, per cluster (e.g. the k=4 insulin-dependent clusters),
  the proportion of participants on each medication vs. not — this is a
  direct within-cluster medication-prevalence table, analogous to the
  existing `complications_by_cluster.py` pattern in
  `insulin_dependent_k4_complications/`.

**Complications focus — microvascular / kidney (already started):**
- `insulin_dependent_k4_complications/` currently produces
  `complications_summary.csv` and neuropathy-related plots per cluster — this
  is the seed of the "central complication" angle from your notes.
- Small-vessel disease (microvascular) angle: kidney markers already live in
  `clinical_data` (creatinine, UACR — see the related `[[project-clusters-9-vars-kidney-cluster]]`
  finding, which found the pooled-cohort kidney-cluster signal did **not**
  hold up under study-group stratification — don't re-cite that as an
  established result without rechecking it in the insulin-dependent k=4
  clustering specifically, which is a different clustering than the one that
  finding was based on).
- OCT retinal imaging (1.3TB, not yet moved) would be the way to extend this
  from lab-marker proxies to actual imaged microvascular damage — this is the
  piece most likely to require continued O2/large-storage access; sequence it
  after resolving §3's OCT access question, not before.

**Writing:**
- Start from the abstract in §1 as the paper's core claim (AI-READI CGM →
  cluster recovery), with GRADE and the complications/drug analyses as
  supporting or follow-up sections, not prerequisites to a first draft.
- A reasonable first-draft outline: Intro (T2D heterogeneity + CGM
  motivation) → Methods (cohort, clustering, CGM feature set, CGM-only
  classifier) → Results (cluster characterization vs. Ahlqvist subtypes,
  CGM→cluster AUPRC) → Discussion (clinical implications, GRADE as
  independent-cohort validation, complications as translational relevance) →
  Limitations (CGM wear duration, imaging not yet incorporated).
- Consider drafting the brief abstract/talk version (mentioned in your notes)
  as a standalone one-pager first — it forces the "what's the one finding"
  framing before the full paper draft.

**Standing meeting:** Friday 11:00am check-in — bring this document.

---

## 5. Quick reference — where things live right now

- Code: `/n/groups/patel/chandrima/` → GitHub `cchatterjeec/sibmi-project`
- Feature table: `final_df.csv` (also referenced by nearly every
  regression/cluster/classification script as `DATA_PATH`)
- CGM pipeline: `cgm_ml_features.py` reads `ai_readi/preprocessed/cgm.parquet`
- AI-READI clinical + medication subset: `Dropbox/AIREADI_data/`
- Environment spec: `requirements.txt` (Python 3.9.25, 54 packages)
