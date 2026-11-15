# Combining statistical, dynamical and AI cyclone catastrophe models for insurance applications

Code for the Honours thesis of Alex Wang (UNSW, School of Risk and Actuarial Studies). It covers the full pipeline behind the thesis results: generating tropical cyclone (TC) tracks, converting them to losses in Australia with [CLIMADA](https://github.com/CLIMADA-project/climada_python), and the scoring-rule blending and robust reinsurance analysis.

The clean, re-runnable scripts for every table and figure in Chapters 3 and 4 are in [`code/`](code/README.md), and the figures and tables exactly as they appear in the thesis are in [`thesis_results/`](thesis_results/README.md), with the script that generates each one. The notebooks and scripts elsewhere are the original working versions they were taken from.

## Pipeline

Run the steps in order. Scripts in `hazard_and_loss_generation/` use paths relative to that folder, so run them from inside it; the analysis notebooks in the root use paths relative to the root.

1. **Generate the tracks** (`TC hazard models/`) and copy the outputs into `TC_datasets/`:

   | Model | Run | Output files | Copy to |
   |---|---|---|---|
   | STORM | `STORM/MASTER.py` (South Indian and South Pacific basins) | `STORM_DATA_IBTRACS_{SI,SP}_1000_YEARS_{0-9}.txt` | `TC_datasets/STORM/` |
   | CHAZ | `CHAZ/Tutorial_chaz_downscaling_fixed.ipynb` (ERA5, 2017) | `2017_001.nc` ... `2017_250.nc` | `TC_datasets/CHAZ/` |
   | IRIS | `IRIS/run/run_iris.py`, then `IRIS/run/write_text_output.py` | `IRIS_{SI,SP}_100Y_n{0-99}.txt` | `TC_datasets/IRIS/` |
   | WN-C | `WNC_Mini_reforecast.ipynb` (Google Colab, writes to Google Drive `wn2_au_reforecast/tracks/`) | `{IBTrACS SID}.csv`, one per storm | `TC_datasets/MiniTC/` |

   Also download the IBTrACS archive `ibtracs.since1980.list.v04r01.csv` (NOAA NCEI) into the repository root.
2. **Prepare the WN-C tracks**: `hazard_and_loss_generation/WNC_track_conversion_and_QM.ipynb` converts the raw WN-C CSVs to the format read by `helper.py`, and writes a quantile-mapped copy to `TC_datasets/MiniTC-QM/`. The quantile-mapped set is not used in the thesis, but the original loss notebooks expect it.
3. **Event losses**: `hazard_and_loss_generation/event_loss_calculation.ipynb` builds one CLIMADA hazard per track set (Holland 2008 wind field, 0.25° grid; IBTrACS\_p generated here) and computes the event loss tables with LitPop exposure and the Emanuel (2011) impact function → `impact/TC_AUS/TC_AUS_impact_<model>.csv` and `aus_tc_loss_distributions.csv`.
4. **Year hazards**: `Hazard_STORM.py`, `Hazard_IRIS.py`, `Hazard_CHAZ.py` and `Hazard_IBTrACS.py` (which also builds the IBTrACS\_p and WN-C counterfactual seasons) write one hazard file per simulated year → `hazard/`.
5. **Annual losses**: `hazard_and_loss_generation/annual_loss_calculation.ipynb` → `results/annual_losses_v2.csv`.
6. **Analysis** (root):
   - `loss_comparison.ipynb`: loss comparison of the models and counterfactual sets (Chapter 3).
   - `blend_weights_YLT.ipynb` and `blend_weights_ELT.ipynb`: scoring-rule blend weights, using `scoring_rules_core.py` → `results/blend_runs/`, `results/blend_runs_ELT/`.
   - `blend_weights_results_viewer.ipynb`: summarises the blend runs, including the subsampling.
   - `robust_reinsurance.ipynb`: optimal reinsurance with the robustness penalty (Section 4.4).
   - `ENSO.csv`: ENSO index read by `scoring_rules_core.py`.

`hazard_and_loss_generation/WNC_track_visualisation.ipynb` plots the WN-C tracks against IBTrACS and is not needed for the results.

Steps 3–5 can instead be run with the cleaned scripts `code/loss_01_event_loss_tables.py`, `loss_02_year_hazards.py` and `loss_03_year_loss_tables.py`, which skip the unused quantile-mapped WN-C set (and hence step 2's quantile mapping, though the raw WN-C CSVs are still read through `helper.py`). The `code/ch3_*.py` and `code/ch4_*.py` scripts then reproduce every table and figure in Chapters 3 and 4 (see [`code/README.md`](code/README.md)).

## Data not included

The track datasets (`TC_datasets/`), hazard files (`hazard/`), event loss files (`impact/`), the IBTrACS archive (`ibtracs.since1980.list.v04r01.csv`, from NOAA NCEI) and the large inputs and outputs of the hazard models are too large for GitHub and are not included. The empty folders show where they go. The annual losses (`results/annual_losses_v2.csv`), the event loss distributions (`aus_tc_loss_distributions.csv`) and the final blend runs used in the thesis are included.

## Requirements

Python 3 with CLIMADA (for the loss pipeline), plus numpy, pandas, scipy, xarray and matplotlib. The hazard models have their own requirements (see their READMEs). The hazard scripts follow the code of Meiler et al. (2022), *Intercomparison of regional loss estimates from global synthetic tropical cyclone models*.
