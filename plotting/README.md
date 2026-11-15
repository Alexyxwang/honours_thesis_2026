# Code for Chapters 3 and 4

This folder has the code behind every table, figure and number in Chapters 3 and 4 of the thesis. The scripts are
cleaned-up copies of the notebooks and scripts in the Honours folder, plus the scripts used to make the final figures.
The analysis scripts were re-run on the saved loss files and reproduce every table in the thesis exactly.

There are two stages:

1. **Loss pipeline** (`loss_*.py`): turns the TC tracks into losses with CLIMADA. This needs CLIMADA and the raw
   track data in `TC_datasets/`, and takes many hours. Its outputs (`impact/`, `hazard/`) are not stored in the repository as they are too large,
   so they need to be regenerated before running the analysis on the event losses.
2. **Analysis** (`ch3_*.py`, `ch4_*.py`): reads the saved losses and makes the tables and figures. It needs numpy,
   pandas, scipy and matplotlib, but not CLIMADA.

## How to run

Run from anywhere. Paths are relative to the repository root (the parent of `code/`); set `HONOURS_ROOT` to use
another location. Outputs go to `code/outputs/tables` and `code/outputs/figures`.

```
python ch3_loss_comparison.py          # Tables 3.1-3.4, Figures 3.2, 3.3, 3.5, 3.6           (~1 min)
python ch3_aal_maps.py                 # Figures 3.4, 3.7                                       (~1 min)
python ch4_scoring_rules.py            # Tables 4.1-4.10                                         (~5 min)
python ch4_weight_stability.py         # Figure 4.1  (run after ch4_scoring_rules.py)            (~30 min)
python ch4_robust_reinsurance.py       # Figure 4.2 and the numbers in Section 4.4               (~3 min)
```

## Files

| File | What it does |
|---|---|
| `common.py` | Paths, loaders for the ELTs and YLTs, the plot style and colours used in every figure |
| `loss_01_event_loss_tables.py` | ELT run: one hazard per track set, losses of every event → `impact/TC_AUS/TC_AUS_impact_<model>.csv` |
| `loss_02_year_hazards.py` | YLT run: one hazard file per simulated year, including the 200 counterfactual versions of each season → `hazard/` |
| `loss_03_year_loss_tables.py` | YLT run: annual loss of each hazard file → `results/annual_losses_v2.csv` |
| `ch3_loss_comparison.py` | Loss comparison of the synthetic models and the counterfactual sets |
| `ch3_aal_maps.py` | AAL per 0.5° cell maps |
| `ch4_scoring_rules.py` | Scoring-rule blend weights (full and conditional distributions, quantile windows) and the diagnostics quoted in the text |
| `ch4_weight_stability.py` | Subsampling of the conditional CRPS weights |
| `reinsurance_core.py` | Optimal reinsurance functions (from `robust_reinsurance.ipynb`) |
| `ch4_robust_reinsurance.py` | Optimal deductibles, CVaRs and the robustness frontier |

The Chapter 4 scripts import `scoring_rules_core.py` from the Honours folder, which holds the scoring rules, the
weight grid search and the subsampling.

## Where each table and figure comes from

| Thesis | Script | Output |
|---|---|---|
| Figure 3.1 (CLIMADA pipeline diagram) | drawn in LaTeX (TikZ) | – |
| Table 3.1 Event loss return periods, synthetic models | `ch3_loss_comparison.py` | `tab_elt_synthetic.csv` |
| Figure 3.2 Event loss RP curves | `ch3_loss_comparison.py` | `M_ELT_curve.png` |
| Table 3.2 Annual loss return periods, synthetic models | `ch3_loss_comparison.py` | `tab_ylt_synthetic.csv` |
| Figure 3.3 Annual loss RP curves | `ch3_loss_comparison.py` | `M_YLT_curve.png` |
| Figure 3.4 AAL map, synthetic models | `ch3_aal_maps.py` | `fig_aal_map_models.png` |
| Table 3.3 Alternative 42-year records, event losses | `ch3_loss_comparison.py` | `tab_elt_counterfactual.csv`, `tab_elt_counterfactual_pooled.csv` |
| Figure 3.5 Alternative records, event losses | `ch3_loss_comparison.py` | `CF_ELT_curve.png` |
| Table 3.4 Alternative 40-season records, annual losses | `ch3_loss_comparison.py` | `tab_ylt_counterfactual.csv` |
| Figure 3.6 Alternative records, annual losses | `ch3_loss_comparison.py` | `CF_YLT_curve.png` |
| Figure 3.7 AAL map, counterfactual sets | `ch3_aal_maps.py` | `fig_aal_map_counterfactual.png` |
| Tables 4.1, 4.2 YLT weights, full distribution (main rules, quantile windows) | `ch4_scoring_rules.py` | `sr_ylt_full.csv`, `tab_sr_ylt_full.tex`, `tab_sr_ylt_full_windows.tex` |
| Tables 4.3, 4.4 ELT weights, full distribution | `ch4_scoring_rules.py` | `sr_elt_full.csv`, `tab_sr_elt_full.tex`, `tab_sr_elt_full_windows.tex` |
| Tables 4.5, 4.7 YLT weights, conditional | `ch4_scoring_rules.py` | `sr_ylt_cond.csv`, `tab_sr_ylt_cond.tex`, `tab_sr_ylt_cond_windows.tex` |
| Table 4.6 Percentiles of annual losses below US$610m | `ch4_scoring_rules.py` | `tab_sr_ylt_cond_diag.csv` |
| Tables 4.8, 4.10 ELT weights, conditional | `ch4_scoring_rules.py` | `sr_elt_cond.csv`, `tab_sr_elt_cond.tex`, `tab_sr_elt_cond_windows.tex` |
| Table 4.9 Percentiles of event losses between US$1 and US$600m | `ch4_scoring_rules.py` | `tab_sr_elt_cond_diag.csv` |
| Figure 4.1 Stability of the weights | `ch4_weight_stability.py` | `fig_sr_stability.png`, `sr_{elt,ylt}_subsample.csv` |
| Figure 4.2 Robustness penalty | `ch4_robust_reinsurance.py` | `fig_robust_gamma.png`, `robust_frontier.csv` |

The `tab_*.tex` files are the table rows with the cell shading and bold maximum, ready to paste between `\midrule`
and `\bottomrule`.

## Numbers quoted in the text

The `text_*.csv` tables hold the numbers quoted in the text:

- **Section 3.3:**
  - `text_elt_spread`, `text_ylt_spread`: how many times larger the largest model is than the smallest, and each model's ratio to IBTrACS.
  - `text_elt_ratio_to_IBTrACS`.
  - `text_elt_tail_slope`: the exponent b in loss ∝ T^b between 1-in-50 and 1-in-500.
  - `text_elt_median`: median loss of the events above US$1.
  - `text_aal_by_model`.
- **Section 3.5:** the "share of records" rows of `tab_elt_counterfactual.csv` and `tab_ylt_counterfactual.csv`, e.g. the observed 1-in-5 year loss is exceeded in 89% of the IBTrACS_p records.
- **Sections 4.2–4.3:**
  - `text_sr_ylt_tail_penalty`: the share p_m of years below u and the log score penalty −log p_m.
  - `text_sr_ylt_crps_tail_term`: ∫_u^∞ (1 − F_m)².
  - `text_sr_{ylt,elt}_crps_by_quantile_level`: the CRPS of each model split by quantile level.
  - `text_sr_{ylt,elt}_vs_optimum`: each model alone and each equal two-model blend, as % above the optimal score.
- **Section 4.4:**
  - `text_reinsurance_single`: d*, CVaR with and without cover, VaR, S(0).
  - `text_reinsurance_cross`: each model's CVaR at the blend's d*.
  - `text_reinsurance_robust`: largest γ, d*, and the CVaR / variance trade-off.

## Settings

- **ELTs** (`loss_01`):
  - IBTrACS: 1980–2021, category 0–5, 1-hourly, N = 42 years.
  - STORM: 4 of 10 files for SI and SP, N = 4,000.
  - IRIS: n0–n39 for SI and SP, N = 4,000.
  - CHAZ: track files up to `2017_100.nc` with members 1–40, N = 3,900, frequency bias-corrected (×0.75).
  - IBTrACS_p: 30 perturbations per storm, seed 42.
  - WN-C: 30 members per storm; storms with fewer usable members are filled with copies of the historical track (repeat) or perturbed tracks (perturb).
  - All sets use the 0.25° grid over 110–155°E, 45–10°S, split to Meiler et al.'s Southern Hemisphere bounds.
  - Exposure is LitPop 300 arcsec, 'pc', 2020, lon ≤ 155°E. Vulnerability is Emanuel (2011): v_thresh = 25.7 m/s, v_half = 74.7 m/s.
- **YLTs** (`loss_02`, `loss_03`):
  - STORM: all 10 files, 10,000 years. IRIS: n0–n99, 10,000 years. CHAZ: 250 files × 40 members, 9,750 years, no frequency correction.
  - IBTrACS: Australian sub-basin storms by July–June season, 1980–2020.
  - Counterfactuals: 200 seeds × 40 seasons (1980–2019), each storm replaced with probability 0.7. WN-C uses usable members only.
- **Scoring rules** (`ch4_scoring_rules.py`):
  - Log score bins of US$10m; weight grid step 0.01.
  - Conditional thresholds: u = US$610m (YLT) and US$600m (ELT).
  - Quantile windows: three consecutive levels of 0.05, 0.1, 0.2, …, 0.9, 0.95.
  - Subsampling: 50 subsets of 60%, seed 0; events for the ELT, seasons for the YLT.
- **Reinsurance**: θ = 4, α = 5%, expected value premium, illustrative 50/50 STORM–CHAZ blend of annual losses.

## Not included

- The quantile-mapped WN-C sets (`MiniTC-QM`), which are not used in the thesis.
- The exploratory notebooks.
- The CSV conversion of the raw Mini WN-C files in `WN-C track processing and correction.ipynb`, which only renames columns before `read_simulations_AI` reads them.
