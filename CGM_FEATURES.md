# CGM Feature Catalog - beyond Time-in-Range

Per-individual features extracted by **`cgm_ml_features.py`** from the AI-READI
Dexcom G6 series (`ai_readi/preprocessed/cgm.parquet`), for the machine-learning
regression pipeline. Each feature is one column in `cgm_ml_features.csv`;
`cgm_ml_features_dictionary.csv` carries the same `citation_key` codes used below.

Every DOI/PMID here was verified to resolve. This goes deliberately **beyond**
the consensus core (TIR / TAR / TBR / mean / SD / CV / GMI) into glycemic-
variability indices, risk scores, glucose **dynamics**, **signal complexity**,
**frequency-domain**, and **circadian** structure - the families that carry
information the standard percent-in-range summaries throw away.

> **Non-fasting note:** these are CGM *time-series* features and are valid
> regardless of fasting state - unlike the HOMA columns.
> CGM is the strongest glycemic phenotype AI-READI offers.

---

## 1. Central tendency & distribution shape

| Feature (column) | Definition | Citation |
|---|---|---|
| `mean_glucose`, `sd_glucose`, `cv_pct` | Mean, SD, coefficient of variation (CV >= 36% = "unstable"). | Rodbard 2009 |
| `iqr_glucose`, `mad_glucose` | Interquartile range; median absolute deviation (robust spread). | Rodbard 2009 |
| `min/max/range_glucose`, `p10/p25/median/p75/p90_glucose` | Extremes and percentiles of the glucose distribution. | Rodbard 2009 |
| `skewness`, `kurtosis` | 3rd/4th standardized moments - asymmetry and tail weight of the distribution. | Rodbard 2009 (generic moments) |

## 2. Time-in-range family

| Feature | Definition | Citation |
|---|---|---|
| `tir_70_180_pct`, `tbr_lt70_pct`, `tbr_lt54_pct`, `tar_gt180_pct`, `tar_gt250_pct` | Consensus 5-bin time-in/above/below-range. | Battelino 2019 |
| `tight_70_140_pct` | Time in tight range 70-140 mg/dL. | Battelino 2019 |
| `gmi_pct` | Glucose Management Indicator = 3.31 + 0.02392*mean. | Bergenstal 2018 |

## 3. Glycemic-variability indices

| Feature | Definition | Citation |
|---|---|---|
| `mage` | Mean Amplitude of Glycemic Excursions (swings > 1 SD). | **Service 1970**, *Diabetes* 19(9):644. DOI 10.2337/diab.19.9.644 - PMID 5469118 |
| `modd` | Mean Of Daily Differences: mean \|G(t) - G(t-24h)\|. | **Molnar 1972**, *Diabetologia* 8(5):342. DOI 10.1007/BF01218495 |
| `conga1/2/4/6` | Continuous Overlapping Net Glycemic Action: SD of readings *n* h apart. | **McDonnell 2005**, *Diabetes Technol Ther* 7(2):253. DOI 10.1089/dia.2005.7.253 - PMID 15857227 |
| `j_index` | J = 0.001*(mean + SD)^2 (mg/dL form). | **Wojcicki 1995**, *Horm Metab Res* 27(1):41. DOI 10.1055/s-2007-979906 - PMID 7729793 |
| `m_value` | Mean \|10*log10(BG/120)\|^3 - log deviation from target. | **Schlichtkrull 1965**, *Acta Med Scand* 177:95. DOI 10.1111/j.0954-6820.1965.tb01810.x - PMID 14251860 |
| `grade` | 425*[log10(log10(BG mmol/L)) + 0.16]^2, mean score. | **Hill 2007**, *Diabet Med* 24(7):753. DOI 10.1111/j.1464-5491.2007.02119.x - PMID 17459094 |
| `mag_per_hour` | Mean Absolute Glucose change per hour: sum(|dG|) / hours. | **Hermanides 2010**, *Crit Care Med* 38(3):838. DOI 10.1097/CCM.0b013e3181cc4be9 - PMID 20035218 |
| `mean_daily_range`, `within_day_sd`, `between_day_sd` | Daily amplitude; within-day SD (SDw); SD of daily means (SDb). | Rodbard 2009 (SD decomposition) |

## 4. Composite risk indices

| Feature | Definition | Citation |
|---|---|---|
| `lbgi` | Low Blood Glucose Index (Kovatchev risk transform, low side). | **Kovatchev 1997**, *Diabetes Care* 20(11):1655. DOI 10.2337/diacare.20.11.1655 - PMID 9353603 |
| `hbgi` | High Blood Glucose Index (high side). | **Kovatchev 1998**, *Diabetes Care* 21(11):1870. DOI 10.2337/diacare.21.11.1870 - PMID 9802735 |
| `adrr` | Average Daily Risk Range: mean of (daily max low-risk + max high-risk). | **Kovatchev 2006**, *Diabetes Care* 29(11):2433. DOI 10.2337/dc06-1085 - PMID 17065680 |
| `gri`, `gri_hypo_comp`, `gri_hyper_comp` | Glycemia Risk Index (0-100) = 3.0*Hypo + 1.6*Hyper. | **Klonoff 2022/2023**, *J Diabetes Sci Technol* 17(5):1226. DOI 10.1177/19322968221085273 - PMID 35348391 |

## 5. Glucose dynamics / rate of change

| Feature | Definition | Citation |
|---|---|---|
| `roc_mean_abs`, `roc_sd` | Mean absolute and SD of 5-min rate of change (mg/dL/min). | Rodbard 2009 |
| `pct_rapid_rise`, `pct_rapid_fall` | % of steps changing faster than +/-2 mg/dL/min. | Rodbard 2009 |

## 6. Excursion / episode burden

| Feature | Definition | Citation |
|---|---|---|
| `n_hypo_events`, `mean_hypo_dur_min`, `n_hyper_events`, `mean_hyper_dur_min` | Count/duration of episodes >= 15 min beyond 70 / 180 mg/dL. | Battelino 2019 |
| `hyper_auc_per_day`, `hypo_auc_per_day` | Mean daily area under/over the 180 / 70 threshold (trapezoidal). | Le Floch 1990, *Diabetes Care* 13(2):172. DOI 10.2337/diacare.13.2.172 - PMID 2351014 |

## 7. Signal complexity / nonlinear dynamics *(highest marginal value for ML)*

| Feature | Definition | Citation |
|---|---|---|
| `sample_entropy` | Sample entropy (m=2, r=0.2*SD); lower = more regular/"decomplexified". | Method: **Richman 2000**, *Am J Physiol* 278(6):H2039. DOI 10.1152/ajpheart.2000.278.6.H2039 - PMID 10843903. CGM: **Crenier 2016**, *JCEM* 101(4):1490. DOI 10.1210/jc.2015-4035 |
| `dfa_alpha` | Detrended Fluctuation Analysis exponent (long-range correlation). | Method: **Peng 1995**, *Chaos* 5(1):82. DOI 10.1063/1.166141 - PMID 11538314. CGM: **Signal 2013**, *J Diabetes Sci Technol* 7(6):1492. DOI 10.1177/193229681300700609 - PMID 24351175 |
| `poincare_sd1`, `poincare_sd2`, `poincare_ratio` | Poincare ellipse: short-term (SD1), long-term (SD2), ratio. | **Crenier 2014**, *Diabetes Technol Ther* 16(4):247. DOI 10.1089/dia.2013.0241 - PMID 24237387 |
| `autocorr_lag1` | Lag-1 (5-min) autocorrelation - glucose persistence/"memory". | Rodbard 2009; cf. Sugimoto 2025, *Commun Med* 5:103. DOI 10.1038/s43856-025-00819-5 |

*Also worth adding later:* **multiscale entropy (MSE)** - Costa 2002, *Phys Rev Lett* 89:068102 (DOI 10.1103/PhysRevLett.89.068102); CGM: Costa 2014, *Chaos* 24:033139 (DOI 10.1063/1.4894537).

## 8. Frequency domain

| Feature | Definition | Citation |
|---|---|---|
| `spectral_circadian_frac` | Fraction of FFT power in the 20-28 h (circadian) band. | Fico 2017, *J Diabetes Sci Technol* 11(4):773. DOI 10.1177/1932296816685717 - PMID 28627250 |
| `spectral_dominant_period_h` | Period of the dominant FFT component. | Miller 2007, *J Diabetes Sci Technol* 1(5):630. DOI 10.1177/193229680700100506 |
| `spectral_entropy` | Normalized entropy of the power spectrum (rhythm concentration vs spread). | Fico 2017 (PSD features) |

## 9. Circadian / temporal pattern

| Feature | Definition | Citation |
|---|---|---|
| `cosinor_mesor`, `cosinor_amplitude`, `cosinor_acrophase_h` | 24-h single-component cosinor: rhythm-adjusted mean, amplitude, peak hour. | Method: **Cornelissen 2014**, *Theor Biol Med Model* 11:16. DOI 10.1186/1742-4682-11-16 - PMID 24725531 |
| `dawn_rise` | Dawn effect: mean(06-09h) - mean(00-06h). | Monnier 2013, *Diabetes Care* 36(12):4057. DOI 10.2337/dc12-2127 - PMID 23996257; orig. Bolli & Gerich 1984, *NEJM* 310:746. PMID 6366551 |
| `overnight_mean`, `daytime_mean`, `nocturnal_hypo_pct` | Window-restricted means and overnight hypo burden. | Battelino 2019 (window definitions) |

## QC / coverage (gate before modeling; not used as model features)

`qc_n_egv`, `qc_wear_span_days`, `qc_n_days`, `qc_pct_active`. International
consensus minimum for reliable metrics: **>= 14 days & >= 70% active** (Battelino
2019). Metrics needing multiple days (MODD, between-day SD, cosinor, spectral)
return `NaN` when wear is too short.

---

## Reviews, consensus & software (cite these for methods)

- **Rodbard D 2009** - GV metric definitions + redundancy. *Diabetes Technol Ther* 11(S1):S55. DOI 10.1089/dia.2008.0132 - PMID 19469679.
- **ATTD / TIR consensus** - Battelino T 2019. *Diabetes Care* 42(8):1593. DOI 10.2337/dci19-0028 - PMID 31177185; Danne T 2017, DOI 10.2337/dc17-1600.
- **iglu (R)** - Broll 2021, *PLoS ONE* 16(4):e0248560. DOI 10.1371/journal.pone.0248560 - PMID 33793578.
- **cgmquantify (Python)** - Bent 2021, *IEEE OJEMB* 2:263. DOI 10.1109/OJEMB.2021.3105816 - PMID 35402978.
- **GLU (R)** - Millard 2020, *Int J Epidemiol* 49(3):744. DOI 10.1093/ije/dyaa004 - PMID 32737505.
- **cgmanalysis / EasyGV** - Vigers 2019 (DOI 10.1371/journal.pone.0216851); Hill 2011 (DOI 10.1089/dia.2010.0247).

## CGM features - ML precedent

- **Glucotypes** (unsupervised phenotyping) - Hall 2018, *PLoS Biol* 16(7):e2005143. DOI 10.1371/journal.pbio.2005143 - PMID 30040822.
- **Metabolic subphenotype prediction from CGM** - Metwally 2024, *Nat Biomed Eng*. DOI 10.1038/s41551-024-01311-6 - PMID 39715896 (curve-shape features predict muscle-IR AUC 0.95).

---

## Redundancy / feature-selection guidance (Rodbard 2009)

Most amplitude indices are highly collinear with total SD:

- **SD ~ CV ~ J-index ~ M-value ~ MAGE ~ CONGA** - all driven by within-day SD.
- **GMI ~ mean glucose** - deterministic function.
- **LBGI ~ TBR/hypo-AUC**; **HBGI ~ TAR/GMI/hyper-AUC**; **ADRR** ~ LBGI+HBGI.
- **GRI** is a linear combination of the TBR/TAR bins - expect collinearity with them.
- **MODD, CONGA-24, autocorr, MAG** capture between-day/temporal structure - more orthogonal to the SD family; higher marginal value.
- **Entropy, DFA, spectral, cosinor** are largely independent of amplitude metrics - the **highest-value additions** for ML.

This pipeline currently ships the **full family list above** into the CGM
ablation stage (Model 5a / Model 6a in `ablation_common.py`) rather than
pre-pruning; ElasticNetCV (L1-regularized) and XGBoost's tree-based
importance both do their own implicit selection downstream, and collinearity
is visible directly in the ElasticNet coefficient paths if it becomes a
problem. Revisit this file's guidance first if that pruning is ever needed.
