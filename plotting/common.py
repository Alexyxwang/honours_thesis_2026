"""Shared paths, data loaders and plot style for the Chapter 3 and 4 scripts.

Every script is run from anywhere; paths are resolved relative to the Honours folder (the parent of this
`code` folder). Set the environment variable HONOURS_ROOT to point somewhere else.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("HONOURS_ROOT", Path(__file__).resolve().parents[1]))
IMPACT_DIR = ROOT / "impact" / "TC_AUS"                 # ELTs: CLIMADA Impact.write_csv output, one file per model
YLT_PATH = ROOT / "results" / "annual_losses_v2.csv"    # YLTs: one row per simulated year (model, loss, file)
ENSO_PATH = ROOT / "ENSO.csv"                           # only needed by scoring_rules_core.load_annual_losses
OUT_DIR = Path(os.environ.get("THESIS_OUT", Path(__file__).resolve().parent / "outputs"))
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# scoring_rules_core.py (scoring-rule blending) lives in the Honours folder
sys.path.insert(0, str(ROOT))

SYNTHETIC = ['STORM', 'CHAZ', 'IRIS']
COUNTERFACTUAL_ELT = ['IBTrACS_p', 'WN_MiniTC_repeat', 'WN_MiniTC_perturb']   # quantile-mapped (QM) WN-C sets not used
COUNTERFACTUAL_YLT = ['IBTrACS_p', 'MINI-WNC']

# ------------------------------------------------------------------------------------------------
# Plot style (all thesis figures)
# ------------------------------------------------------------------------------------------------
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COL = {'IBTrACS': INK, 'STORM': '#7b4fd6', 'CHAZ': '#c99a06', 'IRIS': '#d6337a', 'IBTrACS_p': '#2a78d6',
       'WN_MiniTC_repeat': '#eb6834', 'WN_MiniTC_perturb': '#1baf7a', 'MINI-WNC': '#eb6834'}
LAB = {'IBTrACS': 'IBTrACS (historical)', 'STORM': 'STORM', 'CHAZ': 'CHAZ', 'IRIS': 'IRIS', 'IBTrACS_p': 'IBTrACS_p',
       'WN_MiniTC_repeat': 'WN-C (repeat fill)', 'WN_MiniTC_perturb': 'WN-C (perturbed fill)', 'MINI-WNC': 'WN-C'}


def use_style(font_size=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': font_size, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    return plt


def loglog(ax, xlab, ylab):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
def load_elt(model, columns=('event_name', 'event_frequency', 'at_event')):
    """Event loss table of one model: event name, annual frequency and loss (at_event) of every event.

    The impact CSVs written by CLIMADA also contain per-exposure-point columns (exp_lat, exp_lon, eai_exp) below
    the event rows, so rows without an event frequency are dropped."""
    d = pd.read_csv(IMPACT_DIR / f"TC_AUS_impact_{model}.csv", low_memory=False, dtype={'event_name': str},
                    usecols=list(columns))
    d = d.dropna(subset=['event_frequency'])
    return d.astype({'event_frequency': float, 'at_event': float}).reset_index(drop=True)


def load_ylt():
    """Year loss tables of every model: columns model, loss, file (the per-year hazard file it came from)."""
    return pd.read_csv(YLT_PATH)


def save_table(df, name, fmt=None):
    """Write a results table to outputs/tables/<name>.csv and print it."""
    df.to_csv(TAB_DIR / f"{name}.csv")
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(f"\n=== {name}"); print(df.round(4) if fmt is None else df.map(fmt))


def latex_number(x):
    """Thesis number format for US$m values: 1{,}234 / 123 / 12 / 1.2 / 0.12."""
    if pd.isna(x):
        return '--'
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(',', '{,}')
    if abs(x) >= 10:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.1f}"
    return f"{x:.2f}"
