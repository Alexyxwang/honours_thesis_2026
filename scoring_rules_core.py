"""
Blend weights for TC loss models under proper scoring rules (shared by the blend_weights_ELT / blend_weights_YLT notebooks).

A blend is the mixture F_w = sum_m w_m F_m of empirical loss distributions (e.g. STORM, CHAZ, IRIS). The weights
w minimise the mean score of F_w against a target sample (e.g. IBTrACS, IBTrACS_p, WN / MINI-WNC), found by
grid search over the weight simplex.

    data  = load_impact_dir()          # ELT: one row per event      (model, loss, year=NaN, enso=NaN)
    data  = load_annual_losses()       # YLT: one row per simulated year (model, loss, file, year, enso)
    spec  = BlendSpec('IBTrACS_p', ('STORM', 'CHAZ', 'IRIS'), years=range(1980, 2021))
    df    = sweep(data, expand_specs(spec, target=[...]), standard_rules())

Sections: 1 data, 2 blend problems, 3 scoring rules, 4 optimisation, 5 uncertainty, 6 saving, 7 diagnostics.
"""
import itertools
import json
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde


# ==================================================================================================
# 1. Data
# ==================================================================================================
def load_impact_dir(impact_dir="impact/TC_AUS", prefix="TC_AUS_impact_"):
    """ELT: one row per event with loss > 0 from every <prefix><model>.csv (CLIMADA Impact.write_csv, at_event)."""
    files = sorted(Path(impact_dir).glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"no {prefix}*.csv in {Path(impact_dir).resolve()}")
    parts = []
    for f in files:
        loss = pd.to_numeric(pd.read_csv(f, usecols=['at_event'])['at_event'], errors='coerce').to_numpy()
        parts.append(pd.DataFrame({'model': f.stem[len(prefix):], 'loss': loss[loss > 0]}))
        print(f"{f.stem[len(prefix):]:22s} kept {(loss > 0).sum():>7d} of {len(loss):>7d} events")
    return pd.concat(parts, ignore_index=True).assign(year=np.nan, enso=np.nan)


def load_annual_losses(path="results/annual_losses_v2.csv", enso_path="ENSO.csv"):
    """YLT: one row per simulated year. Real seasons ('season1980' in the hazard file name) get a calendar year and
    ENSO phase; synthetic years ('year0042') get NaN."""
    df = pd.read_csv(path)
    df['year'] = pd.to_numeric(df['file'].str.extract(r'season(\d{4})', expand=False), errors='coerce')
    header = next(i for i, l in enumerate(Path(enso_path).read_text().splitlines()) if l.lower().startswith('year'))
    enso = pd.read_csv(enso_path, skiprows=header)[['year', 'enso']].astype({'year': float})
    return df.merge(enso, on='year', how='left')


def summary(data):
    """One row per model: size, loss level, zero share and calendar-year coverage."""
    return data.groupby('model').agg(n=('loss', 'size'), mean=('loss', 'mean'), median=('loss', 'median'),
                                     max=('loss', 'max'), frac_zero=('loss', lambda s: (s == 0).mean()),
                                     year_min=('year', 'min'), year_max=('year', 'max'), n_years=('year', 'nunique'))


# ==================================================================================================
# 2. Blend problems
# ==================================================================================================
@dataclass(frozen=True)
class BlendSpec:
    """One blend problem.
    years / exclude_years / enso_phase  filter the TARGET only (component catalogues are synthetic).
    loss_min / loss_max                 keep a < X < b in the target AND every component (each renormalised).
    condition_below=u                   drop target values above u and score the blend conditional on X <= u;
                                        the weights reported are still those of the full blend."""
    target: str
    components: Tuple[str, ...] = ('STORM', 'CHAZ', 'IRIS')
    years: Optional[Tuple[int, ...]] = None
    exclude_years: Optional[Tuple[int, ...]] = None
    enso_phase: Optional[str] = None            # 'Nino', 'Nina', 'Neutral' or a tuple of them
    loss_min: Optional[float] = None
    loss_max: Optional[float] = None
    condition_below: Optional[float] = None
    note: Optional[str] = None                  # free-text label carried into the results (e.g. 'own q=1.0')

    def __post_init__(self):
        object.__setattr__(self, 'components', tuple(self.components))
        for f in ('years', 'exclude_years'):
            if getattr(self, f) is not None:
                object.__setattr__(self, f, tuple(sorted({int(y) for y in np.atleast_1d(getattr(self, f))})))
        if self.enso_phase is not None and not isinstance(self.enso_phase, str):
            object.__setattr__(self, 'enso_phase', tuple(self.enso_phase))
        if len(self.components) < 2:
            raise ValueError("a blend needs at least two components")

    @property
    def enso_label(self):
        return 'All' if self.enso_phase is None else '+'.join(np.atleast_1d(self.enso_phase))

    @property
    def loss_label(self):
        lo, hi = self.loss_min, self.loss_max
        if lo is None and hi is None:
            return 'All'
        return ' < '.join(([f'{lo:.3g}'] if lo is not None else []) + ['X'] + ([f'{hi:.3g}'] if hi is not None else []))

    @property
    def label(self):
        parts = [self.target, '+'.join(self.components)]
        if self.years:
            parts.append(f"years {self.years[0]}-{self.years[-1]}")
        if self.exclude_years:
            parts.append(f"excl {list(self.exclude_years)}")
        if self.enso_phase is not None:
            parts.append(f"ENSO={self.enso_label}")
        if self.loss_label != 'All':
            parts.append(self.loss_label)
        if self.condition_below is not None:
            parts.append(f"X<={self.condition_below:.3g}")
        if self.note:
            parts.append(self.note)
        return ' | '.join(parts)


def _in_range(x, lo, hi):
    return (x > (-np.inf if lo is None else lo)) & (x < (np.inf if hi is None else hi))


def model_losses(data, model):
    if not (data['model'] == model).any():
        raise KeyError(f"unknown model {model!r}; available: {sorted(data['model'].unique())}")
    return data.loc[data['model'] == model, 'loss'].to_numpy(float)


def target_table(data, spec, condition=True):
    """Target rows after the year / ENSO / loss-range filters and (unless condition=False) loss <= condition_below."""
    model_losses(data, spec.target)
    t = data[data['model'] == spec.target]
    if spec.years is not None:
        t = t[t['year'].isin(spec.years)]
    if spec.exclude_years:
        t = t[~t['year'].isin(spec.exclude_years)]
    if spec.enso_phase is not None:
        t = t[t['enso'].isin(np.atleast_1d(spec.enso_phase))]
    t = t[_in_range(t['loss'].to_numpy(float), spec.loss_min, spec.loss_max)]
    if condition and spec.condition_below is not None:
        t = t[t['loss'] <= spec.condition_below]
    return t


def observations(data, spec):
    return target_table(data, spec)['loss'].to_numpy(float)


def component_losses(data, spec):
    """Loss sample of each component: loss range applied, never year-filtered (ConditionalBelow truncates at u)."""
    out = []
    for m in spec.components:
        x = model_losses(data, m)
        out.append(x[_in_range(x, spec.loss_min, spec.loss_max)])
    return out


def expand_specs(base, **grid):
    """Every combination of the listed alternatives, e.g. expand_specs(base, target=[...], years=[range(...), ...]).
    loss_range=[(a, b), ...] sets loss_min and loss_max together."""
    for k, v in grid.items():
        if not isinstance(v, list):
            raise TypeError(f"{k}= must be a list of alternatives")
    specs = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(grid, combo))
        if 'loss_range' in kw:
            kw['loss_min'], kw['loss_max'] = kw.pop('loss_range')
        specs.append(replace(base, **kw))
    return specs


def model_subsets(models, sizes=(2, 3)):
    return [c for k in sizes for c in itertools.combinations(models, k)]


def year_windows(first, last, width, step=1):
    """Sliding windows: year_windows(2000, 2019, 10, 2) -> 2000-2009, 2002-2011, ..., 2010-2019."""
    return [range(s, s + width) for s in range(first, last - width + 2, step)]


def expanding_windows(first, last, min_width=5, step=1, grow='forward'):
    """Windows growing from `first` forward, or from `last` backward."""
    if grow == 'forward':
        return [range(first, e + 1) for e in range(first + min_width - 1, last + 1, step)]
    return [range(s, last + 1) for s in range(last - min_width + 1, first - 1, -step)]


def loss_bands(edges):
    """[(e0, e1), (e1, e2), ...] for expand_specs(loss_range=...)."""
    return list(zip(edges[:-1], edges[1:]))


def loss_quantile_bounds(data, model, q_min=None, q_max=None):
    """(loss_min, loss_max) at quantiles of one model's losses; either may be None."""
    x = model_losses(data, model)
    return tuple(None if q is None else float(np.quantile(x, q)) for q in (q_min, q_max))


def own_threshold_specs(data, base, targets, quantiles=(1.0,), how='condition_below'):
    """One spec per (target, q) whose upper threshold u is the q-quantile of THAT target's own losses (over
    base.years for annual data), instead of a threshold shared by all targets.
    how='condition_below': drop target values above u and score the blend conditional on X <= u (YLT section 3).
    how='loss_max':        truncate target and components to X <= u, each renormalised (ELT loss ranges)."""
    specs = []
    for t in targets:
        for q in quantiles:
            u = max_loss_threshold(data, t, years=base.years, q=q)
            kw = (dict(condition_below=u) if how == 'condition_below'
                  else dict(loss_max=float(np.nextafter(u, np.inf))))      # loss_max is strict: keep X == u
            specs.append(replace(base, target=t, note=f'own q={q:g}', **kw))
    return specs


def max_loss_threshold(data, model='IBTrACS', years=None, q=1.0):
    """u for condition_below: the q-quantile (default: maximum) of a model's losses, optionally over some years."""
    t = data[data['model'] == model]
    return float(np.quantile(t[t['year'].isin(years)]['loss'] if years is not None else t['loss'], q))


# ==================================================================================================
# 3. Scoring rules (negatively oriented: lower is better)
# ==================================================================================================
class ScoringRule:
    """Scores the blend at every observation. prepare() runs once per optimisation and puts all components on one
    common support xk, so the blend's pmf at weights w is just w @ P."""
    def prepare(self, comps, obs):
        xk = np.unique(np.concatenate(comps))
        P = np.vstack([np.bincount(np.searchsorted(xk, c), minlength=len(xk)) / len(c) for c in comps])
        return xk, P

    def mixture_scores(self, context, comps, weights, obs):
        xk, P = self.prepare(comps, obs) if context is None else context
        w = np.asarray(weights, float)
        return self.score(xk, (w / w.sum()) @ P, np.asarray(obs, float))

    def score(self, xk, pk, y):
        raise NotImplementedError


class CRPS(ScoringRule):
    """Exact CRPS of a discrete distribution: E|X - y| - E|X - X'| / 2."""
    def score(self, xk, pk, y):
        F, Fx = np.cumsum(pk), np.cumsum(pk * xk)
        e_xx = 2.0 * np.sum(np.diff(xk) * F[:-1] * (1.0 - F[:-1]))
        j = np.searchsorted(xk, y, side='right') - 1
        F_y = np.where(j >= 0, F[np.clip(j, 0, None)], 0.0)
        Fx_y = np.where(j >= 0, Fx[np.clip(j, 0, None)], 0.0)
        return 2 * y * F_y - 2 * Fx_y + Fx[-1] - y - 0.5 * e_xx


class ThresholdWeightedCRPS(CRPS):
    """CRPS after the chaining map v(z) = (z-t) Phi((z-t)/s) + s phi((z-t)/s), which upweights losses above t."""
    def __init__(self, threshold, scale):
        self.t, self.s = float(threshold), float(scale)

    def v(self, z):
        z = np.asarray(z, float)
        a = (z - self.t) / self.s
        return (z - self.t) * stats.norm.cdf(a) + self.s * stats.norm.pdf(a)

    def score(self, xk, pk, y):
        return super().score(self.v(xk), pk, self.v(y))


class QuantileScore(ScoringRule):
    """Mean pinball loss over the quantile levels `taus` (default 0.05, 0.10, ..., 0.95)."""
    def __init__(self, taus=None):
        self.taus = np.arange(0.05, 1.0, 0.05) if taus is None else np.asarray(taus, float)

    def pinball(self, xk, pk, y):
        F = np.cumsum(pk) / pk.sum()
        q = xk[np.clip(np.searchsorted(F, self.taus), 0, len(xk) - 1)]
        d = y[:, None] - q[None, :]
        return np.maximum(self.taus * d, (self.taus - 1.0) * d)

    def score(self, xk, pk, y):
        return self.pinball(xk, pk, y).mean(axis=1)


class QuantileWeightedCRPS(QuantileScore):
    """Upper-tail weighted CRPS: 2 * integral over alpha of the pinball loss times alpha^2."""
    def __init__(self):
        super().__init__(np.linspace(0.0, 1.0, 101))

    def score(self, xk, pk, y):
        trapz = getattr(np, 'trapezoid', None) or np.trapz
        return 2.0 * trapz(self.pinball(xk, pk, y) * self.taus ** 2, self.taus, axis=1)


LOGSCORE_BIN_WIDTH = 1e7                                   # default LogScore bin width (loss units)
_ZERO, _OVER = np.iinfo(np.int64).min, np.iinfo(np.int64).max


class LogScore(ScoringRule):
    """Discrete log score -log sum_m w_m P_m(bin(y)), losses binned on the raw scale; minimising it maximises the
    binned likelihood. Its value is a log probability, so weights (not mean scores) compare across bin widths.
    bin_width: bins [k*bw, (k+1)*bw).  edges: explicit bin edges instead (values outside share an end bin).
    cap: losses >= cap share one bin.  zero_atom: exact zeros get their own bin."""
    def __init__(self, bin_width=LOGSCORE_BIN_WIDTH, edges=None, cap=None, zero_atom=False):
        self.bin_width, self.cap, self.zero_atom = float(bin_width), cap, zero_atom
        self.edges = None if edges is None else np.asarray(edges, float)

    @property
    def label(self):
        s = f"{len(self.edges) - 1} edge bins" if self.edges is not None else f"bin={self.bin_width:.0e}"
        s += f", cap={self.cap:.0e}" if self.cap is not None else ""
        return f"LogScore ({s}{', zero atom' if self.zero_atom else ''})"

    def bin_codes(self, x):
        x = np.asarray(x, float)
        c = (np.searchsorted(self.edges, x, side='right') - 1 if self.edges is not None
             else np.floor(x / self.bin_width)).astype(np.int64)
        if self.cap is not None:
            c[x >= self.cap] = _OVER
        if self.zero_atom:
            c[x == 0] = _ZERO
        return c

    def prepare(self, comps, obs):
        """(n_models, n_obs) matrix: probability each component gives to each observation's bin."""
        y, rows = self.bin_codes(obs), []
        for c in comps:
            codes, counts = np.unique(self.bin_codes(c), return_counts=True)
            i = np.clip(np.searchsorted(codes, y), 0, len(codes) - 1)
            rows.append(np.where(codes[i] == y, counts[i] / len(c), 0.0))
        return np.vstack(rows)

    def mixture_scores(self, context, comps, weights, obs):
        P = self.prepare(comps, obs) if context is None else context
        w = np.asarray(weights, float)
        return -np.log(np.maximum((w / w.sum()) @ P, 1e-300))


class ConditionalBelow(ScoringRule):
    """Any rule applied to X | X <= u. Components are truncated at u and re-weighted by w_m P_m(X <= u), which is
    exactly the conditional of the full blend, so the optimised w are still full-blend weights."""
    def __init__(self, inner, u):
        self.inner, self.u = inner, float(u)

    def prepare(self, comps, obs):
        if np.any(np.asarray(obs) > self.u):
            raise ValueError("drop observations above u before conditional scoring")
        trunc = [c[c <= self.u] for c in comps]
        mass = np.array([len(t) / len(c) for t, c in zip(trunc, comps)])
        if np.any(mass == 0):
            raise ValueError(f"a component has no losses <= {self.u:.4g}")
        return trunc, mass, self.inner.prepare(trunc, obs)

    def mixture_scores(self, context, comps, weights, obs):
        trunc, mass, inner = self.prepare(comps, obs) if context is None else context
        return self.inner.mixture_scores(inner, trunc, np.asarray(weights, float) * mass, obs)


def rule_for(spec, rule):
    """`rule`, wrapped in ConditionalBelow when the spec conditions on X <= u."""
    return rule if spec.condition_below is None else ConditionalBelow(rule, spec.condition_below)


def standard_rules(low_quantiles=(0.05, 0.1, 0.2), twcrps=None, log_score_kwargs=None):
    """Default rule set. twcrps=(threshold, scale) adds ThresholdWeightedCRPS; log_score_kwargs configure LogScore."""
    rules = {'CRPS': CRPS(), 'LogScore': LogScore(**(log_score_kwargs or {})), 'Quantile': QuantileScore(),
             f'Quantile {tuple(low_quantiles)}': QuantileScore(low_quantiles),
             'Quantile (0.8, 0.9, 0.95)': QuantileScore([0.8, 0.9, 0.95]),
             'QuantileWeightedCRPS': QuantileWeightedCRPS()}
    if twcrps is not None:
        rules['ThresholdWeightedCRPS'] = ThresholdWeightedCRPS(*twcrps)
    return rules


def quantile_window_rules(levels=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95), size=3):
    """QuantileScore on sliding windows of consecutive levels: how the weights move from low to high losses."""
    windows = [tuple(levels[i:i + size]) for i in range(len(levels) - size + 1)]
    return {f'Quantile {w}': QuantileScore(w) for w in windows}


def log_score_bin_rules(bin_widths=(1e5, 1e6, 1e7, 1e8, 1e9), **kwargs):
    """One LogScore per bin width, for sensitivity analysis. kwargs (cap=, zero_atom=) go to every rule."""
    rules = [LogScore(bin_width=b, **kwargs) for b in bin_widths]
    return {r.label: r for r in rules}


def twcrps_params(data, target, q=0.9):
    """(threshold, scale) for ThresholdWeightedCRPS: the target's q-quantile and standard deviation."""
    x = model_losses(data, target)
    return float(np.quantile(x, q)), float(x.std())


# ==================================================================================================
# 4. Optimisation
# ==================================================================================================
GRID_STEP = 0.01   # weight resolution: 5,151 grid points for 3 models, 101 for 2


class InsufficientDataError(ValueError):
    pass


def weight_grid(n, step=GRID_STEP):
    """Every weight vector on the simplex with spacing `step`."""
    k = int(round(1 / step))
    head = [c for c in itertools.product(range(k + 1), repeat=n - 1) if sum(c) <= k]
    return np.array([(*c, k - sum(c)) for c in head], float) / k


def optimize_weights(comps, obs, rule, step=None):
    """Exhaustive grid search for the weights minimising the mean score. Returns (weights, mean score)."""
    context = rule.prepare(comps, obs)
    grid = weight_grid(len(comps), step or GRID_STEP)
    scores = np.array([rule.mixture_scores(context, comps, w, obs).mean() for w in grid])
    best = int(np.argmin(scores))
    return grid[best], float(scores[best])


def _weight_columns(components, w):
    return {**{f'weight_{m}': wi for m, wi in zip(components, w)},
            'dominant_model': components[int(np.argmax(w))], 'at_corner': bool(w.max() > 0.999)}


def run_blend(data, spec, rule, rule_name=None, step=None, min_obs=1):
    """Optimal weights for one spec under one rule, as one results row."""
    t, comps = target_table(data, spec), component_losses(data, spec)
    if len(t) < max(min_obs, 1) or min(map(len, comps)) < 2:
        raise InsufficientDataError(f"{spec.label}: {len(t)} observations, component sizes {list(map(len, comps))}")
    w, score = optimize_weights(comps, t['loss'].to_numpy(float), rule_for(spec, rule), step)
    yrs = t['year'].dropna()
    return {'target': spec.target, 'scoring_rule': rule_name or type(rule).__name__,
            **_weight_columns(spec.components, w), 'min_mean_score': score,
            'components': '+'.join(spec.components),
            'years': f"{yrs.min():.0f}-{yrs.max():.0f}" if len(yrs) else None, 'n_years': yrs.nunique() or None,
            'excluded_years': ','.join(map(str, spec.exclude_years)) if spec.exclude_years else None,
            'enso_phase': spec.enso_label, 'loss_range': spec.loss_label, 'condition_below': spec.condition_below,
            'note': spec.note,
            'n_obs': len(t), 'n_above_dropped': len(target_table(data, spec, condition=False)) - len(t),
            **{f'n_events_{m}': len(c) for m, c in zip(spec.components, comps)}}


RESULT_LEAD = ['target', 'scoring_rule']   # then weight_* columns, then everything else (see order_columns)


def order_columns(df, models=()):
    """target, scoring_rule, weight_* (in `models` order where given), then the rest in their original order."""
    weights = sorted((c for c in df.columns if c.startswith('weight_')),
                     key=lambda c: list(models).index(c[7:]) if c[7:] in models else len(models))
    lead = [c for c in RESULT_LEAD if c in df.columns] + weights
    return df[lead + [c for c in df.columns if c not in lead]]


def sweep(data, specs, rules, min_obs=1, verbose=True):
    """Runs every spec under every rule: one results row per (spec, rule). Columns that are empty for every row
    (e.g. years for event data) are dropped. Skipped specs: df.attrs['skipped']."""
    specs = list(specs)
    jobs = [(spec, name, rule) for spec in specs for name, rule in rules.items()]
    rows, skipped, t0 = [], [], time.time()
    for i, (spec, name, rule) in enumerate(jobs, 1):
        try:
            rows.append(run_blend(data, spec, rule, name, min_obs=min_obs))
        except InsufficientDataError as e:
            skipped.append(str(e))
        if verbose and (i % max(1, len(jobs) // 10) == 0 or i == len(jobs)):
            print(f"{i}/{len(jobs)} ({time.time() - t0:.0f}s)")
    models = list(dict.fromkeys(m for s in sorted(specs, key=lambda s: -len(s.components)) for m in s.components))
    df = order_columns(pd.DataFrame(rows).dropna(axis=1, how='all'), models)
    df.attrs['skipped'] = sorted(set(skipped))
    if skipped:
        print(f"skipped {len(df.attrs['skipped'])} spec(s): see df.attrs['skipped']")
    return df


# ==================================================================================================
# 5. Uncertainty and influence
# ==================================================================================================
def subsample_weights(data, spec, rule, frac=0.6, n_iter=50, by='year', seed=0):
    """Re-optimise on random subsets of the target: whole calendar years (by='year'; counterfactual seasons of one
    year are not independent) or single rows (by='row', the only option for event data). One row per subset."""
    t, comps, rule = target_table(data, spec), component_losses(data, spec), rule_for(spec, rule)
    units = t['year'].dropna().unique() if by == 'year' else t.index.to_numpy()
    if len(units) == 0:
        raise InsufficientDataError(f"{spec.label}: no target {'calendar years' if by == 'year' else 'observations'}")
    rng, rows = np.random.default_rng(seed), []
    for _ in range(n_iter):
        pick = rng.choice(units, max(1, round(len(units) * frac)), replace=False)
        obs = t.loc[t['year'].isin(pick) if by == 'year' else pick, 'loss'].to_numpy(float)
        w, s = optimize_weights(comps, obs, rule)
        rows.append({**_weight_columns(spec.components, w), 'min_mean_score': s, 'n_obs': len(obs)})
    return pd.DataFrame(rows)


def subsample_sweep(data, specs, rules, frac=0.6, n_iter=50, by='year', seed=0, min_obs=1, verbose=True):
    """subsample_weights for every spec x rule. One row per (spec, rule): mean, sd and 2.5 / 50 / 97.5% of each
    weight across the subsamples, plus the share of subsamples whose optimum sits on a corner. Specs with fewer
    than min_obs target observations are skipped (df.attrs['skipped'])."""
    jobs = [(spec, name, rule) for spec in specs for name, rule in rules.items()]
    models = list(dict.fromkeys(m for s, _, _ in sorted(jobs, key=lambda j: -len(j[0].components))
                                for m in s.components))
    rows, skipped, t0 = [], [], time.time()
    for i, (spec, name, rule) in enumerate(jobs, 1):
        try:
            if len(target_table(data, spec)) < max(min_obs, 1):
                raise InsufficientDataError(f"{spec.label}: fewer than {max(min_obs, 1)} observations")
            sub = subsample_weights(data, spec, rule, frac, n_iter, by, seed)
        except InsufficientDataError as e:
            skipped.append(str(e))
            continue
        row = {'target': spec.target, 'scoring_rule': name}
        for c in [c for c in sub.columns if c.startswith('weight_')]:
            q = sub[c].quantile([0.025, 0.5, 0.975]).to_numpy()
            row.update({f'{c}_mean': sub[c].mean(), f'{c}_sd': sub[c].std(),
                        f'{c}_q025': q[0], f'{c}_median': q[1], f'{c}_q975': q[2]})
        row.update({'frac_at_corner': sub['at_corner'].mean(), 'components': '+'.join(spec.components),
                    'years': f"{spec.years[0]}-{spec.years[-1]}" if spec.years else None,
                    'loss_range': spec.loss_label, 'condition_below': spec.condition_below, 'note': spec.note,
                    'by': by, 'frac': frac, 'n_iter': n_iter})
        rows.append(row)
        if verbose:
            print(f"{i}/{len(jobs)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows).dropna(axis=1, how='all')
    wcols = sorted((c for c in df.columns if c.startswith('weight_')),
                   key=lambda c: models.index(c[7:].rsplit('_', 1)[0]) if c[7:].rsplit('_', 1)[0] in models else len(models))
    lead = ['target', 'scoring_rule'] + wcols
    df = df[lead + [c for c in df.columns if c not in lead]] if len(df) else df
    df.attrs['skipped'] = sorted(set(skipped))
    if skipped:
        print(f"skipped {len(df.attrs['skipped'])} spec(s): see df.attrs['skipped']")
    return df


def influence_analysis(data, spec, rule, max_drop=2, max_runs=2000):
    """Re-optimise with every combination of up to max_drop target years removed; one row per combination."""
    t = target_table(data, spec)
    t = t[t['year'].notna()]
    years = sorted(t['year'].astype(int).unique())
    combos = [d for k in range(min(max_drop, len(years) - 1) + 1) for d in itertools.combinations(years, k)]
    if len(combos) > max_runs:
        raise ValueError(f"{len(combos)} optimisations needed (max_runs={max_runs}); lower max_drop or the year range")
    comps, rule, rows = component_losses(data, spec), rule_for(spec, rule), []
    for dropped in combos:
        w, s = optimize_weights(comps, t.loc[~t['year'].isin(dropped), 'loss'].to_numpy(float), rule)
        rows.append({'n_dropped': len(dropped), 'dropped_years': dropped, 'min_mean_score': s,
                     **_weight_columns(spec.components, w)})
    return pd.DataFrame(rows)


def plot_influence(df):
    """Weight on the baseline-dominant model after dropping each single year, and how often dropped-year sets
    containing each year move the solution off a corner."""
    base = df[df['n_dropped'] == 0].iloc[0]
    col = f"weight_{base['dominant_model']}"
    single = df[df['n_dropped'] == 1]
    years = sorted({y for d in df['dropped_years'] for y in d})
    off = [1 - df.loc[df['dropped_years'].apply(lambda d: y in d), 'at_corner'].mean() for y in years]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].bar([str(d[0]) for d in single['dropped_years']], single[col])
    ax[0].set(title=f"{col} after dropping one year", ylim=(0, 1.05))
    ax[1].bar([str(y) for y in years], off)
    ax[1].set(title='share of dropped sets with this year that leave the corner', ylim=(0, 1.05))
    for a in ax:
        a.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.show()
    k = df.loc[~df['at_corner'], 'n_dropped']
    print(f"baseline {base['dominant_model']} = {base[col]:.2f}; fewest dropped years leaving the corner: "
          f"{k.min() if len(k) else 'none'}")


# ==================================================================================================
# 6. Saving
# ==================================================================================================
def save_run(df, tag, out_dir, notes=""):
    """Saves df to <out_dir>/<timestamp>__<tag>/results.csv (+ meta.json) and appends to <out_dir>/manifest.csv."""
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(out_dir) / f"{stamp}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', tag)}"
    run_dir.mkdir(parents=True)
    df.to_csv(run_dir / 'results.csv', index=False)
    meta = {'tag': tag, 'notes': notes, 'n_rows': len(df), 'skipped': df.attrs.get('skipped', [])}
    (run_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    manifest = Path(out_dir) / 'manifest.csv'
    pd.DataFrame([{'timestamp': stamp, 'tag': tag, 'n_rows': len(df), 'folder': str(run_dir), 'notes': notes}]
                 ).to_csv(manifest, mode='a', header=not manifest.exists(), index=False)
    print(f"saved {len(df)} rows -> {run_dir}")
    return run_dir


# ==================================================================================================
# 7. Diagnostics
# ==================================================================================================
def plot_weight_sweep(df, x, rules=None):
    """Stacked optimal weights along column x ('components', 'years', 'loss_range', ...), one panel per rule.
    Each (x, rule) pair must be unique, so filter df first if it varies something else too."""
    rules = rules or list(df['scoring_rule'].unique())
    wcols = [c for c in df.columns if c.startswith('weight_')]
    fig, axes = plt.subplots(len(rules), 1, figsize=(9, 3.2 * len(rules)), sharex=True, squeeze=False)
    for ax, r in zip(axes[:, 0], rules):
        sub = df[df['scoring_rule'] == r].set_index(x)[wcols].fillna(0).rename(columns=lambda c: c[7:])
        sub.plot(kind='bar', stacked=True, ax=ax, width=0.85, legend=False, ylim=(0, 1), title=r, ylabel='weight')
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.setp(axes[-1, 0].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout(); plt.show()


def plot_bin_sensitivity(df):
    """Optimal weights against LogScore bin width, one panel per target (filter df to one spec per target)."""
    d = df[df['scoring_rule'].str.startswith('LogScore')].copy()
    d['bin_width'] = d['scoring_rule'].str.extract(r'bin=([0-9.e+-]+)', expand=False).astype(float)
    wcols = [c for c in d.columns if c.startswith('weight_') and d[c].notna().any()]
    groups = list(d.dropna(subset=['bin_width']).groupby('target', sort=False))
    fig, axes = plt.subplots(len(groups), 1, figsize=(7, 3.2 * len(groups)), squeeze=False)
    for ax, (target, g) in zip(axes[:, 0], groups):
        g = g.sort_values('bin_width')
        for c in wcols:
            ax.plot(g['bin_width'], g[c], marker='o', label=c[7:])
        ax.set(xscale='log', ylim=(-0.02, 1.02), title=target, xlabel='bin width', ylabel='weight')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout(); plt.show()


def log_score_diagnostics(data, spec, rules):
    """Per LogScore rule: number of bins the observations occupy, and the share of observations whose bin has no
    mass in each component (and in all of them)."""
    obs, comps = observations(data, spec), component_losses(data, spec)
    if spec.condition_below is not None:
        comps = [c[c <= spec.condition_below] for c in comps]
    rows = []
    for name, r in rules.items():
        P = r.prepare(comps, obs)
        rows.append({'rule': name, 'n_obs': len(obs), 'n_obs_bins': len(np.unique(r.bin_codes(obs))),
                     **{f'unsupported_{m}': (p == 0).mean() for m, p in zip(spec.components, P)},
                     'unsupported_all': (P.sum(axis=0) == 0).mean()})
    return pd.DataFrame(rows)


def single_model_scores(data, spec, rules):
    """Mean score of each component on its own (weight 1) against the spec's target, for each rule."""
    obs, comps = observations(data, spec), component_losses(data, spec)
    return pd.DataFrame({name: [rule_for(spec, rule).mixture_scores(None, [c], [1.0], obs).mean() for c in comps]
                         for name, rule in rules.items()}, index=list(spec.components)).T


def analyse_scores(data, spec, rule, weights=None, logx=True):
    """Per-observation score of the blend at fixed weights (default equal) and of each component alone, against the
    observed loss; histogram of the observations underneath."""
    rule = rule_for(spec, rule)
    obs, comps = np.sort(observations(data, spec)), component_losses(data, spec)
    w = np.ones(len(comps)) if weights is None else np.asarray(weights, float)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    for m, c in zip(spec.components, comps):
        a1.scatter(obs, rule.mixture_scores(None, [c], [1.0], obs), s=8, alpha=0.4, label=m)
    scores = rule.mixture_scores(None, comps, w, obs)
    a1.scatter(obs, scores, s=14, marker='x', color='k', label=f'blend {np.round(w / w.sum(), 2)}')
    a1.set(ylabel='score (lower is better)', title=spec.label)
    a1.legend()
    a2.hist(obs, bins=np.geomspace(max(obs.min(), 1), obs.max(), 60) if logx else 60)
    a2.set(xlabel='observed loss', ylabel='count', xscale='log' if logx else 'linear')
    plt.tight_layout(); plt.show()
    return scores


def plot_loss_distributions(data, models, years=None):
    """KDE of log10(loss) per model (zeros excluded; zero share in the legend). `years` filters real seasons."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in models:
        t = data[data['model'] == m]
        if years is not None and t['year'].notna().any():
            t = t[t['year'].isin(years)]
        x = t['loss'].to_numpy(float)
        lx = np.log10(x[x > 0])
        grid = np.linspace(lx.min() - 0.5, lx.max() + 0.5, 500)
        ax.plot(grid, gaussian_kde(lx)(grid), lw=2, label=f"{m} (n={len(x)}, {np.mean(x == 0):.1%} zero)")
    ax.set(xlabel='log10(loss)', ylabel='density')
    ax.legend(); plt.tight_layout(); plt.show()


def compare_exceedance(data, models=('STORM', 'CHAZ', 'IRIS'), targets=(),
                       rps=(2, 5, 10, 20, 50, 100, 200, 500, 1000), rp_unit='years', plot=True):
    """Loss at return periods (Weibull plotting positions, log-log interpolation), in units of sample rows
    (rp_unit: 'years' for annual data, 'events' for event data). `targets` are dashed reference curves, excluded
    from the highest/lowest columns."""
    curves = {}
    for m in [*models, *targets]:
        x = np.sort(model_losses(data, m))
        curves[m] = ((len(x) + 1) / np.arange(len(x), 0, -1), x)       # ascending return period
    if plot:
        fig, ax = plt.subplots(figsize=(9, 7))
        for m, (rp, x) in curves.items():
            ax.plot(rp, x, ls='--' if m in targets else '-', label=f"{m} (n={len(x)})")
        ax.set(xscale='log', yscale='log', xlabel=f'return period ({rp_unit})', ylabel='loss')
        ax.legend(); ax.grid(which='both', alpha=0.3); plt.show()
    interp = lambda rp, x: np.exp(np.interp(np.log(rps), np.log(rp), np.log(np.clip(x, 1, None))))
    table = pd.DataFrame({m: interp(*c) for m, c in curves.items()}, index=pd.Index(rps, name='return_period'))
    table['highest'], table['lowest'] = table[list(models)].idxmax(axis=1), table[list(models)].idxmin(axis=1)
    return table


def plot_year_boxplots(data, models=('IBTrACS_p', 'MINI-WNC-QM'), reference='IBTrACS', years=None):
    """Spread of counterfactual annual losses per calendar year, with the reference model's loss marked."""
    ref = data[data['model'] == reference].set_index('year')['loss']
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 9), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        t = data[(data['model'] == m) & data['year'].notna()]
        if years is not None:
            t = t[t['year'].isin(years)]
        yrs = sorted(t['year'].unique())
        ax.boxplot([np.log10(t.loc[t['year'] == y, 'loss'].clip(lower=1)) for y in yrs], vert=False,
                   positions=range(len(yrs)), widths=0.6, flierprops=dict(marker='.', markersize=3, alpha=0.4))
        ax.set_yticks(range(len(yrs)), [int(y) for y in yrs])
        marks = [(np.log10(max(ref[y], 1)), i) for i, y in enumerate(yrs) if y in ref.index]
        if marks:
            ax.scatter(*zip(*marks), color='red', marker='D', s=40, zorder=5, label=f'actual {reference}')
        ax.set(title=m, xlabel='log10(annual loss)')
    axes[0, 0].legend(loc='lower right'); plt.tight_layout(); plt.show()


def year_composition_by_band(data, target, n_bands=20, years=None):
    """Share of each calendar year among the target's losses in each of n_bands equal-count loss bands (plotted);
    returns each band's count and loss range."""
    t = data[(data['model'] == target) & data['year'].notna()]
    if years is not None:
        t = t[t['year'].isin(years)]
    band = pd.qcut(t['loss'].rank(method='first'), n_bands, labels=False) + 1
    props = pd.crosstab(band, t['year'].astype(int), normalize='index')
    plt.figure(figsize=(14, 7))
    plt.imshow(props.T, aspect='auto', cmap='Greys', interpolation='nearest')
    plt.colorbar(label='share of band')
    plt.yticks(range(props.shape[1]), props.columns)
    plt.xticks(range(n_bands), props.index)
    plt.xlabel(f'loss band (1 = lowest {100 / n_bands:.0f}%)'); plt.ylabel('calendar year'); plt.title(target)
    plt.tight_layout(); plt.show()
    return t.groupby(band)['loss'].agg(['count', 'min', 'max'])
