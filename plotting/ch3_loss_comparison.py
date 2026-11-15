"""Chapter 3 -- loss comparison of the TC models (Sections 3.3 and 3.5).

Inputs (from the loss pipeline, see README):
    impact/TC_AUS/TC_AUS_impact_<model>.csv   event loss tables (ELTs)
    results/annual_losses_v2.csv              year loss tables (YLTs)

Outputs (code/outputs):
    tables/tab_elt_synthetic.csv              Table 3.1  event loss return periods, synthetic models
    tables/tab_ylt_synthetic.csv              Table 3.2  annual loss return periods, synthetic models
    tables/tab_elt_counterfactual.csv         Table 3.3  alternative 42-year records, event losses
    tables/tab_ylt_counterfactual.csv         Table 3.4  alternative 40-season records, annual losses
    tables/text_*.csv                         numbers quoted in the text (ratios, slopes, shares of records)
    figures/M_ELT_curve.png                   Figure 3.2
    figures/M_YLT_curve.png                   Figure 3.3
    figures/CF_ELT_curve.png                  Figure 3.5
    figures/CF_YLT_curve.png                  Figure 3.6
(The AAL maps, Figures 3.4 and 3.7, are made by ch3_aal_maps.py.)
"""
import re

import numpy as np
import pandas as pd

from common import (COL, COUNTERFACTUAL_ELT, FIG_DIR, INK, LAB, SYNTHETIC, latex_number, load_elt, load_ylt,
                    loglog, save_table, use_style)

ONE_IN = [2, 5, 10, 25, 42, 50, 100, 200, 500, 1000]
N_HIST_YEARS = 42                                       # IBTrACS ELT: 1980-2021

elt = {m: load_elt(m) for m in ['IBTrACS'] + SYNTHETIC + COUNTERFACTUAL_ELT}
ylt = load_ylt()


# ================================================================================================
# 1. Event losses (ELT), synthetic models -- Table 3.1
# ================================================================================================
def one_in_year_event(m, T):
    """1-in-T year event loss: the loss whose annual exceedance rate (the summed frequencies of all events at
    least as large) first reaches 1/T. Blank if T is beyond the length of the record."""
    d = elt[m][elt[m].at_event > 0].sort_values('at_event', ascending=False)
    rate = d.event_frequency.cumsum().to_numpy()
    k = np.searchsorted(rate, 1 / T - 1e-12)
    return d.at_event.iloc[k] if rate[0] <= 1 / T + 1e-12 and k < len(d) else np.nan


def elt_stats(m):
    L, f = elt[m].at_event.to_numpy(), elt[m].event_frequency.to_numpy()
    r = {f'1-in-{T}': one_in_year_event(m, T) / 1e6 for T in ONE_IN}
    r['Largest event'] = L.max() / 1e6
    r['Loss-causing events'] = (L > 0).sum()
    r['Loss-causing events per year'] = f[L > 0].sum()
    r['AAL'] = (L * f).sum() / 1e6
    return r


elt_table = pd.DataFrame({m: elt_stats(m) for m in ['IBTrACS'] + SYNTHETIC})
save_table(elt_table, 'tab_elt_synthetic')

# numbers in the text of Section 3.3.1
s = elt_table[SYNTHETIC].T
ratios = pd.DataFrame({
    'max/min model, 1-in-10': [s['1-in-10'].max() / s['1-in-10'].min()],
    'max/min model, 1-in-100': [s['1-in-100'].max() / s['1-in-100'].min()],
    'max/min model, 1-in-1000': [s['1-in-1000'].max() / s['1-in-1000'].min()]})
save_table(ratios, 'text_elt_spread')
save_table((s[['1-in-2', '1-in-5', '1-in-10', '1-in-25']] / elt_table['IBTrACS'][['1-in-2', '1-in-5', '1-in-10', '1-in-25']]),
           'text_elt_ratio_to_IBTrACS')
# tail slope: event loss ~ T^b between 1-in-50 and 1-in-500
save_table(pd.DataFrame({'b': np.log10(s['1-in-500'] / s['1-in-50'])}), 'text_elt_tail_slope')
# median loss of the loss-causing events (loss > 0, as in Chapter 3) of each model
save_table(pd.DataFrame({'median event loss (US$m)': {m: np.median(elt[m].at_event[elt[m].at_event > 0]) / 1e6
                                                      for m in ['IBTrACS'] + SYNTHETIC}}), 'text_elt_median')


# ================================================================================================
# 2. Annual losses (YLT), synthetic models -- Table 3.2
# ================================================================================================
def ylt_stats(m):
    x = ylt.loc[ylt.model == m, 'loss'].to_numpy(); n = len(x)
    # the 1-in-T year loss is the (1 - 1/T) quantile; for IBTrACS (41 seasons) the 1-in-42 level is the largest year
    r = {f'1-in-{T}': (np.quantile(x, 1 - 1 / T) if T <= n else (x.max() if T == 42 else np.nan)) / 1e6 for T in ONE_IN}
    r['Largest year'] = x.max() / 1e6
    r['Mean annual loss'] = x.mean() / 1e6
    r['Zero-loss years (%)'] = 100 * (x == 0).mean()
    r['Years'] = n
    return r


ylt_table = pd.DataFrame({m: ylt_stats(m) for m in ['IBTrACS'] + SYNTHETIC})
save_table(ylt_table, 'tab_ylt_synthetic')
s = ylt_table[SYNTHETIC].T
save_table(pd.DataFrame({
    'ratio to IBTrACS, 1-in-25': s['1-in-25'] / ylt_table.loc['1-in-25', 'IBTrACS'],
    'max/min model, 1-in-10': s['1-in-10'].max() / s['1-in-10'].min(),
    'max/min model, 1-in-100': s['1-in-100'].max() / s['1-in-100'].min(),
    'max/min model, 1-in-1000': s['1-in-1000'].max() / s['1-in-1000'].min()}), 'text_ylt_spread')


# ================================================================================================
# 3. Counterfactual event losses: 30 alternative 42-year records -- Table 3.3
# ================================================================================================
# Every historical storm has 30 counterfactual versions; taking the r-th version of every storm gives one
# alternative 1980-2021 record. Within a 42-year record the k-th largest event loss is exceeded k times in
# 42 years, so it sits at the 1-in-T year level T = 42 / k.
K_EV = 4 * N_HIST_YEARS                                    # ranks down to T = 0.25 years
T_EV = N_HIST_YEARS / np.arange(1, K_EV + 1)


def top_events(x):
    x = np.sort(np.asarray(x, float))[::-1][:K_EV]
    return np.pad(x, (0, K_EV - len(x)))


def alternative_event_records(m):
    """rows = realisation r; each row = that record's event losses (one version of every storm), largest first."""
    d = elt[m].copy(); n = d.event_name.astype(str)
    d['sid'] = n.str.split('_').str[0]                       # versions are named <SID>_gen<k>, <SID>_S<k>, <SID>_rep<k>
    if m == 'IBTrACS_p':
        d = d[n != d.sid]                                    # drop the original (unperturbed) track
    d['r'] = d.groupby('sid').cumcount()
    return np.array([top_events(g.at_event) for _, g in d.groupby('r')])


obs_ev = top_events(elt['IBTrACS'].at_event)
ev_records = {m: alternative_event_records(m) for m in COUNTERFACTUAL_ELT}
print('alternative event records per set:', {m: S.shape[0] for m, S in ev_records.items()})

rows = []
for k in (1, 2, 4, 6, 21, 42):
    r = {'1-in-T': N_HIST_YEARS / k, 'Rank': k, 'Observed': obs_ev[k - 1] / 1e6}
    for m, S in ev_records.items():
        c = S[:, k - 1]
        r[f'{m} median'], r[f'{m} 5%'], r[f'{m} 95%'] = np.median(c) / 1e6, np.quantile(c, .05) / 1e6, np.quantile(c, .95) / 1e6
        r[f'{m} share of records below observed'] = (c < obs_ev[k - 1]).mean()
        r[f'{m} share of records >= observed'] = (c >= obs_ev[k - 1]).mean()
    rows.append(r)
cf_elt = pd.DataFrame(rows).set_index('Rank')
save_table(cf_elt.T, 'tab_elt_counterfactual')
save_table(elt_table.reindex(['AAL', 'Loss-causing events per year']).join(
    pd.DataFrame({m: elt_stats(m) for m in COUNTERFACTUAL_ELT}).reindex(['AAL', 'Loss-causing events per year']))[['IBTrACS'] + COUNTERFACTUAL_ELT],
    'tab_elt_counterfactual_pooled')

# ================================================================================================
# 4. Counterfactual annual losses: 200 alternative 40-season records -- Table 3.4
# ================================================================================================
# The counterfactual YLT files are named ..._season<YYYY>_seed<NNNN>.hdf5: seed s of every season 1980-2019 forms
# one alternative 40-season record. The observed record is IBTrACS restricted to the same 40 seasons.
def season_of(f):
    return int(re.search(r'season(\d{4})', f).group(1))


ib = ylt[ylt.model == 'IBTrACS'].assign(season=lambda d: d.file.map(season_of))
obs_yr = np.sort(ib.loc[ib.season <= 2019, 'loss'].to_numpy())[::-1]
n_seasons = len(obs_yr)
T_YR = (n_seasons + 1) / np.arange(1, n_seasons + 1)

yr_records = {}
for m in ['IBTrACS_p', 'MINI-WNC']:
    d = ylt[ylt.model == m].assign(season=lambda d: d.file.map(season_of),
                                   seed=lambda d: d.file.str.extract(r'seed(\d+)')[0].astype(int))
    p = d.pivot(index='seed', columns='season', values='loss')
    assert p.shape == (200, n_seasons) and not p.isna().any().any()
    yr_records[m] = -np.sort(-p.to_numpy(), axis=1)

rows = []
for k in (1, 2, 4, 8, 20):
    r = {'1-in-T': (n_seasons + 1) / k, 'Rank': k, 'Observed': obs_yr[k - 1] / 1e6}
    for m, S in yr_records.items():
        c = S[:, k - 1]
        r[f'{m} median'], r[f'{m} 5%'], r[f'{m} 95%'] = np.median(c) / 1e6, np.quantile(c, .05) / 1e6, np.quantile(c, .95) / 1e6
        r[f'{m} share of records >= observed'] = (c >= obs_yr[k - 1]).mean()
    rows.append(r)
r = {'1-in-T': np.nan, 'Rank': 'Mean annual loss', 'Observed': obs_yr.mean() / 1e6}
for m, S in yr_records.items():
    c = S.mean(axis=1)
    r[f'{m} median'], r[f'{m} 5%'], r[f'{m} 95%'] = np.median(c) / 1e6, np.quantile(c, .05) / 1e6, np.quantile(c, .95) / 1e6
    r[f'{m} share of records >= observed'] = (c >= obs_yr.mean()).mean()
rows.append(r)
save_table(pd.DataFrame(rows).set_index('Rank').T, 'tab_ylt_counterfactual')


# ================================================================================================
# 5. Figures
# ================================================================================================
plt = use_style(10)


def draw(ax, m, T, loss):
    if m == 'IBTrACS':
        ax.plot(T, loss, lw=0, marker='o', ms=4, color=INK, label=LAB[m], zorder=5)
    else:
        ax.plot(T, loss, lw=1.6, color=COL[m], label=LAB[m])


def one_in_years(m):
    """ELT curve: T = 1 / (annual rate of events with loss >= l), from the event frequencies."""
    d = elt[m][elt[m].at_event > 0].sort_values('at_event', ascending=False)
    return 1 / d.event_frequency.cumsum().to_numpy(), d.at_event.to_numpy()


def one_in(x):
    """YLT curve: sorted annual losses at T = (n + 1) / rank."""
    x = np.sort(x)[::-1]
    return (len(x) + 1) / np.arange(1, len(x) + 1), x


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=200); plt.close(fig); print('saved', FIG_DIR / name)


# Figure 3.2: event loss RP curves, synthetic models
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for m in SYNTHETIC + ['IBTrACS']:
    T, x = one_in_years(m); k = x > 1e5
    draw(ax, m, T[k], x[k])
loglog(ax, '1-in-$T$ year', 'Event loss (US\\$)'); ax.set_xlim(0.2, 2e4); ax.set_ylim(1e5, 5e11)
ax.legend(loc='upper left', fontsize=9)
save(fig, 'M_ELT_curve.png')

# Figure 3.3: annual loss RP curves, synthetic models (CHAZ without frequency correction)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for m in SYNTHETIC + ['IBTrACS']:
    T, x = one_in(ylt.loc[ylt.model == m, 'loss'].to_numpy()); k = x > 0
    draw(ax, m, T[k], x[k])
loglog(ax, '1-in-$T$ year', 'Annual aggregate loss (US\\$)'); ax.set_xlim(1, 1.2e4); ax.set_ylim(1e6, 5e11)
ax.legend(loc='upper left', fontsize=9)
save(fig, 'M_YLT_curve.png')

pos = lambda x: np.where(x > 0, x, np.nan)

# Figure 3.5: alternative 42-year records, event losses (2 x 2, legend in the fourth panel)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(2, 2, figsize=(10, 8.2)); axs = axs.ravel()
for ax, m in zip(axs, COUNTERFACTUAL_ELT):
    S = ev_records[m]
    for rec in S:
        ax.plot(T_EV, pos(rec), color=COL[m], lw=0.5, alpha=0.25)
    ax.fill_between(T_EV, pos(np.quantile(S, .05, axis=0)), pos(np.quantile(S, .95, axis=0)), color=COL[m], alpha=0.18, lw=0)
    ax.plot(T_EV, pos(np.median(S, axis=0)), color=COL[m], lw=2)
    ax.plot(T_EV, pos(obs_ev), 'o', color=INK, ms=3.5, zorder=5)
    loglog(ax, '1-in-$T$ year (within a 42-year record)', 'Event loss (US\\$)')
    ax.set_xlim(0.25, 50); ax.set_ylim(1e5, 1e10); ax.set_title(LAB[m], loc='left', fontsize=13)
    ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20, 42]); ax.set_xticklabels(['0.25', '0.5', '1', '2', '5', '10', '20', '42'])
axs[3].axis('off')
h = [Line2D([], [], color='grey', lw=0.8, alpha=0.6), Patch(color='grey', alpha=0.25), Line2D([], [], color='grey', lw=2),
     Line2D([], [], lw=0, marker='o', color=INK, ms=5)]
axs[3].legend(h, ['Alternative record (one per $r$)', '5–95% of alternative records', 'Median alternative record',
                  'Observed record (IBTrACS)'], loc='center', fontsize=12, title='Line colour = counterfactual model',
              title_fontsize=11)
save(fig, 'CF_ELT_curve.png')

# Figure 3.6: alternative 40-season records, annual losses
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
for ax, m in zip(axs, ['IBTrACS_p', 'MINI-WNC']):
    S = yr_records[m]
    for rec in S:
        ax.plot(T_YR, pos(rec), color=COL[m], lw=0.4, alpha=0.15)
    ax.fill_between(T_YR, pos(np.quantile(S, .05, axis=0)), pos(np.quantile(S, .95, axis=0)), color=COL[m], alpha=0.2, lw=0,
                    label='5–95% range')
    ax.plot(T_YR, pos(np.median(S, axis=0)), color=COL[m], lw=2, label='Median of alternative records')
    ax.plot(T_YR, pos(obs_yr), 'o', color=INK, ms=4, zorder=5, label='IBTrACS (observed)')
    loglog(ax, '1-in-$T$ year (within a 40-season record)', '')
    ax.set_title(LAB[m], fontsize=12); ax.set_ylim(1e5, 2e10); ax.set_xlim(1, 50); ax.legend(loc='upper left', fontsize=9)
axs[0].set_ylabel('Annual aggregate loss (US\\$)')
save(fig, 'CF_YLT_curve.png')

# LaTeX rows of Tables 3.1 and 3.2 (US$m)
for name, tab in (('tab_elt_synthetic', elt_table), ('tab_ylt_synthetic', ylt_table)):
    with open(FIG_DIR.parent / 'tables' / f'{name}.tex', 'w') as f:
        for row, vals in tab.iterrows():
            f.write(f"{row} & " + ' & '.join(latex_number(v) for v in vals) + ' \\\\\n')
