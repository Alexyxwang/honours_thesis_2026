"""Chapter 4 -- stability of the weights (Section 4.3.3, Figure 4.1).

The conditional CRPS weights are re-optimised on 50 random subsamples of 60% of each validation set (seed 0):
individual events for the ELT, whole seasons for the YLT (the counterfactual versions of a season are kept
together, as they come from the same historical storms). Same settings as ch4_scoring_rules.py.

Run ch4_scoring_rules.py first: the vertical marks (weights from the full validation sets) are read from
outputs/tables/sr_{elt,ylt}_cond.csv. `python ch4_weight_stability.py plot` redraws the figure from saved results.

Outputs (code/outputs):
    tables/sr_{elt,ylt}_subsample.csv    mean, SD and 2.5 / 50 / 97.5% of each weight across the subsamples
    figures/fig_sr_stability.png         Figure 4.1
"""
import numpy as np
import pandas as pd

from common import GRID, INK, INK2, FIG_DIR, TAB_DIR, use_style
from ch4_scoring_rules import setup
from scoring_rules_core import CRPS, expand_specs, subsample_sweep

import sys

PLOT_ONLY = 'plot' in sys.argv[1:]          # python ch4_weight_stability.py plot  -> redraw from the saved CSVs
sub = {}
for which in ('elt', 'ylt'):
    if PLOT_ONLY:
        sub[which] = pd.read_csv(TAB_DIR / f'sr_{which}_subsample.csv'); continue
    data, targets, base, cond, u, _ = setup(which)
    df = subsample_sweep(data, expand_specs(cond, target=targets), {'CRPS': CRPS()}, frac=0.6, n_iter=50,
                         by='row' if which == 'elt' else 'year', seed=0, min_obs=5)
    df.to_csv(TAB_DIR / f'sr_{which}_subsample.csv', index=False)
    sub[which] = df
    print(f'\n=== {which}'); print(df[['target', 'weight_STORM_mean', 'weight_STORM_sd', 'weight_STORM_q025',
                                     'weight_STORM_q975']].round(3).to_string())

# ------------------------------------------------------------------------------------------------
plt = use_style(12)
TCOL = {'IBTrACS': INK, 'IBTrACS_p': '#2a78d6', 'WN_MiniTC_repeat': '#eb6834', 'MINI-WNC': '#eb6834',
        'WN_MiniTC_perturb': '#1baf7a'}
LAB = {'IBTrACS': 'IBTrACS', 'IBTrACS_p': 'IBTrACS_p', 'WN_MiniTC_repeat': 'WN-C (repeat)', 'MINI-WNC': 'WN-C (repeat)',
       'WN_MiniTC_perturb': 'WN-C (perturbed)'}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
for ax, which, title in zip(axs, ('elt', 'ylt'), ('(a) ELT', '(b) YLT')):
    d = sub[which].reset_index(drop=True)
    full_w = pd.read_csv(TAB_DIR / f'sr_{which}_cond.csv')
    full_w = full_w[full_w.scoring_rule == 'CRPS'].set_index('target')['weight_STORM']
    for i, r in d.iterrows():
        yy = len(d) - 1 - i; c = TCOL[r.target]
        ax.plot([r.weight_STORM_q025, r.weight_STORM_q975], [yy, yy], color=c, lw=2, alpha=0.45, solid_capstyle='butt')
        lo, hi = max(0, r.weight_STORM_mean - r.weight_STORM_sd), min(1, r.weight_STORM_mean + r.weight_STORM_sd)
        ax.plot([lo, hi], [yy, yy], color=c, lw=7, solid_capstyle='butt')
        ax.plot(r.weight_STORM_mean, yy, 'o', color='white', mec=c, mew=2, ms=8, zorder=5)
        ax.plot(full_w[r.target], yy, marker='|', color=INK, ms=14, mew=2, zorder=6)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([LAB[t] for t in d.target][::-1]); ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlim(-0.03, 1.03); ax.set_xlabel('Weight on STORM'); ax.set_title(title, fontsize=12)
    ax.grid(True, axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=INK2, lw=2, alpha=0.45, label='2.5–97.5% range'), Line2D([], [], color=INK2, lw=6, label='Mean ± 1 SD'),
     Line2D([], [], marker='o', color='white', mec=INK2, mew=2, ms=7, lw=0, label='Mean'),
     Line2D([], [], marker='|', color=INK, ms=12, mew=2, lw=0, label='Full validation set')]
fig.legend(handles=h, loc='lower center', ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(FIG_DIR / 'fig_sr_stability.png', dpi=200); plt.close(fig)
print('saved', FIG_DIR / 'fig_sr_stability.png')
