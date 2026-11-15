"""Chapter 4 -- scoring-rule weights of the frequency blend (Sections 4.2 and 4.3).

Uses scoring_rules_core.py (in the Honours folder): the blend F_w = sum_m w_m F_m of the empirical loss
distributions of STORM, CHAZ and IRIS, with weights found by grid search over the simplex (step 0.01) to minimise
the mean score against each validation set.

    python ch4_scoring_rules.py ylt      annual losses (YLT), validation sets IBTrACS, IBTrACS_p, WN-C
    python ch4_scoring_rules.py elt      event losses (ELT), validation sets IBTrACS, IBTrACS_p, WN-C (repeat/perturbed)
    python ch4_scoring_rules.py          both (each takes several minutes)

Settings (as in the thesis):
    log score   bins of width US$10m
    full        YLT: every target season 1980-2020; ELT: every event with loss > US$1
    conditional YLT: blend scored conditional on X <= u, u = largest historical annual loss rounded up to the next
                     US$10m bin edge (US$610m), no lower bound so zero-loss years are kept
                ELT: target and components truncated to 1 < X < u and renormalised, u = largest historical event
                     loss rounded up to the next bin edge (US$600m)
    quantile windows: quantile scores on three consecutive levels of (0.05, 0.1, 0.2, ..., 0.9, 0.95)

Outputs (code/outputs/tables), for <w> = ylt / elt:
    sr_<w>_full.csv, sr_<w>_cond.csv          weights and optimal mean score for every target x rule
    tab_sr_<w>_{full,cond}{,_windows}.tex     LaTeX rows of the weight tables (Tables 4.1-4.5, 4.7, 4.8, 4.10)
    tab_sr_<w>_cond_diag.csv                  percentiles of the scored losses (Tables 4.6 and 4.9)
    text_sr_<w>_*.csv                         numbers quoted in the text (tail penalties, single-model scores, CRPS split)
"""
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from common import ENSO_PATH, IMPACT_DIR, TAB_DIR, YLT_PATH, save_table
import common  # noqa: F401  (adds the Honours folder to sys.path for scoring_rules_core)
from scoring_rules_core import (CRPS, BlendSpec, LogScore, QuantileWeightedCRPS, component_losses, expand_specs,
                               load_annual_losses, load_impact_dir, max_loss_threshold, observations,
                               quantile_window_rules, rule_for, single_model_scores, standard_rules, sweep,
                               target_table)

BW = 1e7                                   # log score bin width (US$10m)
C = ('STORM', 'CHAZ', 'IRIS')
NAME = {'IBTrACS': 'IBTrACS', 'IBTrACS_p': r'IBTrACS\_p', 'MINI-WNC': 'WN-C',
        'WN_MiniTC_repeat': 'WN-C (repeat)', 'WN_MiniTC_perturb': 'WN-C (perturbed)'}
W = ['weight_STORM', 'weight_CHAZ', 'weight_IRIS']


def setup(which):
    """Data, validation sets, base (full) spec, conditional spec and the threshold u."""
    if which == 'ylt':
        data = load_annual_losses(YLT_PATH, ENSO_PATH)
        targets = ['IBTrACS', 'IBTrACS_p', 'MINI-WNC']
        base = BlendSpec(targets[0], C, years=range(1980, 2021))
        u_max = max_loss_threshold(data, 'IBTrACS', years=range(1980, 2021))
        u = float(np.nextafter((np.floor(u_max / BW) + 1) * BW, -np.inf))          # just below US$610m
        cond = replace(base, condition_below=u)
        std = standard_rules(low_quantiles=(0.05, 0.1, 0.2), log_score_kwargs=dict(bin_width=BW))
    else:
        data = load_impact_dir(IMPACT_DIR)
        data = data[~data.model.str.contains('QM')]                               # quantile-mapped WN-C not used
        targets = ['IBTrACS', 'IBTrACS_p', 'WN_MiniTC_repeat', 'WN_MiniTC_perturb']
        base = BlendSpec(targets[0], C, loss_min=1)
        u_max = data.loc[data['model'] == 'IBTrACS', 'loss'].max()
        u = float((np.floor(u_max / BW) + 1) * BW)                                 # US$600m
        cond = replace(base, loss_max=u)
        std = standard_rules(low_quantiles=(0.7, 0.8, 0.9), log_score_kwargs=dict(bin_width=BW))
    print(f'{which}: largest historical loss {u_max:,.0f}; u = {u:,.0f}')
    rules = {**{k: std[k] for k in ('CRPS', 'LogScore', 'QuantileWeightedCRPS')}, **quantile_window_rules()}
    return data, targets, base, cond, u, rules


# ------------------------------------------------------------------------------------------------
# LaTeX rows of the weight tables (largest weight in bold, cell shading proportional to the weight)
# ------------------------------------------------------------------------------------------------
def cells(w):
    w = [round(float(x), 2) for x in w]; mx = max(w); out = []
    for x in w:
        s = f'{x:.2f}'
        if x == mx:
            s = rf'\textbf{{{s}}}'
        if x > 0:
            s = rf'\cellcolor{{wcol!{int(round(60 * x))}}}' + s
        out.append(s)
    return out


def main_rows(df, targets):
    rows = []
    for t in targets:
        c = []
        for r in ('LogScore', 'CRPS', 'QuantileWeightedCRPS'):
            c += cells(df[(df.target == t) & (df.scoring_rule == r)][W].iloc[0])
        rows.append(f'{NAME[t]} & ' + ' & '.join(c) + r' \\')
    return rows


def window_rows(df, targets):
    rows = []
    for r in [r for r in df.scoring_rule.unique() if r.startswith('Quantile (')]:
        c = []
        for t in targets:
            c += cells(df[(df.target == t) & (df.scoring_rule == r)][W].iloc[0])
        rows.append(f"{r[len('Quantile ('):-1]} & " + ' & '.join(c) + r' \\')
    return rows


# ------------------------------------------------------------------------------------------------
# Diagnostics quoted in the text
# ------------------------------------------------------------------------------------------------
A = (np.arange(4000) + 0.5) / 4000                      # quantile levels for the CRPS decomposition


def crps_by_band(sample, obs, bands=((0, .5), (.5, .8), (.8, .95), (.95, 1), (0, .8), (.8, 1))):
    """CRPS = 2 int_0^1 QS_alpha d alpha, split into bands of the quantile level alpha (mean over observations)."""
    q = np.quantile(sample, A, method='inverted_cdf')
    tot = np.zeros(len(A))
    for i in range(0, len(obs), 2000):
        d = obs[i:i + 2000, None] - q[None, :]
        tot += np.maximum(A * d, (A - 1) * d).sum(axis=0)
    per_alpha = 2 * tot / len(obs) / len(A)
    out = {f'{lo:g}-{hi:g}': per_alpha[(A >= lo) & (A < hi)].sum() for lo, hi in bands}
    out['total'] = per_alpha.sum()
    return out


def upper_tail_term(sample, u):
    """int_u^inf (1 - F(y))^2 dy of an empirical distribution: the part of the CRPS above u that does not depend
    on the observations when every observation lies below u."""
    x = np.sort(sample); n = len(x)
    S = 1 - np.arange(1, n + 1) / n                       # 1 - F on [x_k, x_{k+1})
    left = np.maximum(x, u); right = np.append(x[1:], x[-1])
    return float(np.sum(np.clip(right - left, 0, None) * S ** 2))


def pct(x):
    return {'Median': np.percentile(x, 50) / 1e6, '75th': np.percentile(x, 75) / 1e6,
            '90th': np.percentile(x, 90) / 1e6, '95th': np.percentile(x, 95) / 1e6}


def diagnostics(which, data, targets, base, cond, u, full, cnd):
    rules = {'CRPS': CRPS(), 'LogScore': LogScore(bin_width=BW), 'QuantileWeightedCRPS': QuantileWeightedCRPS()}

    # Tables 4.6 / 4.9: percentiles of the losses in the scored range
    rows, share = {}, {}
    comps = component_losses(data, cond)
    for m, c in zip(C, comps):
        if which == 'ylt':
            share[m] = (c <= u).mean(); c = c[c <= u]
        rows[m] = pct(c)
    for t in targets:
        o = observations(data, replace(cond, target=t)); rows[t] = pct(o)
        if which == 'ylt':
            share[t] = (target_table(data, replace(cond, target=t), condition=False)['loss'] <= u).mean()
    diag = pd.DataFrame(rows).T
    if which == 'ylt':
        diag[f'Years <= u (%)'] = 100 * pd.Series(share)
        # log score tail penalty -log p_m of each model alone (Section 4.2): p_m = share of years <= u
        save_table(pd.DataFrame({'p_m': pd.Series(share)[list(C)], '-log p_m': -np.log(pd.Series(share)[list(C)])}),
                   f'text_sr_{which}_tail_penalty')
        # the term int_u^inf (1 - F_m)^2 of the CRPS (US$m), which only depends on the model
        save_table(pd.DataFrame({'int_u^inf (1-F)^2 (US$m)': {m: upper_tail_term(c, u) / 1e6
                                                              for m, c in zip(C, component_losses(data, base))}}),
                   f'text_sr_{which}_crps_tail_term')
    save_table(diag, f'tab_sr_{which}_cond_diag')

    # single models and equal two-model blends against the optimum (% above the optimal mean score)
    out = []
    for label, spec, res in (('full', base, full), ('cond', cond, cnd)):
        for t in targets:
            s = replace(spec, target=t); obs = observations(data, s); cs = component_losses(data, s)
            sm = single_model_scores(data, s, rules)
            for r, rule in rules.items():
                best = res[(res.target == t) & (res.scoring_rule == r)]['min_mean_score'].iloc[0]
                row = {'distribution': label, 'target': t, 'rule': r, 'optimal score': best}
                rr = rule_for(s, rule)
                for m in C:
                    row[f'{m} alone'] = sm.loc[r, m]
                for a, b in (('STORM', 'CHAZ'), ('STORM', 'IRIS'), ('CHAZ', 'IRIS')):
                    row[f'equal {a}+{b}'] = rr.mixture_scores(None, cs, [1.0 * (m in (a, b)) for m in C], obs).mean()
                if r != 'LogScore':                       # log scores can be negative: no relative comparison
                    for k in [k for k in row if k.endswith('alone') or k.startswith('equal')]:
                        row[k + ' (% above optimum)'] = 100 * (row[k] / best - 1)
                out.append(row)
    save_table(pd.DataFrame(out).set_index(['distribution', 'target', 'rule']), f'text_sr_{which}_vs_optimum')

    # CRPS of each model alone, split by quantile level (US$m) -- full distribution
    rows = []
    for t in targets:
        obs = observations(data, replace(base, target=t))
        for m, c in zip(C, component_losses(data, base)):
            rows.append({'target': t, 'model': m, **{k: v / 1e6 for k, v in crps_by_band(c, obs).items()}})
    save_table(pd.DataFrame(rows).set_index(['target', 'model']), f'text_sr_{which}_crps_by_quantile_level')


def run(which):
    data, targets, base, cond, u, rules = setup(which)
    full = sweep(data, expand_specs(base, target=targets), rules, min_obs=5)
    full.to_csv(TAB_DIR / f'sr_{which}_full.csv', index=False)
    cnd = sweep(data, expand_specs(cond, target=targets), rules, min_obs=5)
    cnd.to_csv(TAB_DIR / f'sr_{which}_cond.csv', index=False)
    for label, df in (('full', full), ('cond', cnd)):
        print(f'\n=== {which} {label}')
        print(df[df.scoring_rule.isin(['LogScore', 'CRPS', 'QuantileWeightedCRPS'])][['target', 'scoring_rule'] + W + ['min_mean_score', 'n_obs']].to_string())
        for kind, rows in (('', main_rows(df, targets)), ('_windows', window_rows(df, targets))):
            (TAB_DIR / f'tab_sr_{which}_{label}{kind}.tex').write_text('\n'.join(rows) + '\n')
    diagnostics(which, data, targets, base, cond, u, full, cnd)


if __name__ == '__main__':
    for w in (sys.argv[1:] or ['ylt', 'elt']):
        run(w)
