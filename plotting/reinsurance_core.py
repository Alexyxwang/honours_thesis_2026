"""Optimal stop-loss reinsurance with a robustness penalty (Section 4.4).

Functions copied from robust_reinsurance.ipynb (cells 0 and 4-7; the frontier plot is replaced by
ch4_robust_reinsurance.py). Each model's annual losses (YLT) form a discrete loss distribution LossDist(xk, pk).
For a deductible d and loading theta, the insurer's total cost is T_d = min(X, d) + pi(d), with the expected value
premium pi(d) = (1 + theta) E[(X - d)+]. The objective is
    CVaR_alpha(T_d of the blend) + gamma * variance across models of CVaR_alpha(T_d of model i),
and the optimal deductible satisfies sum_i w_i (1 + 2 gamma delta_i) S_i(d*) = 1 / (1 + theta),
delta_i = pi_i(d*) - sum_j w_j pi_j(d*)  (closed_form_deductible / closed_form_deductible_check).
"""
import pandas as pd

from common import YLT_PATH


def load_tc_losses(models=('IRIS', 'STORM', 'CHAZ')):
    """{model: YLT rows} with each simulated year given frequency 1 / (number of years)."""
    all_agg_annual_losses = pd.read_csv(YLT_PATH)
    tc_losses = {}
    for model in models:
        df_sub = all_agg_annual_losses[all_agg_annual_losses['model'] == model].copy()
        df_sub['frequency'] = 1 / len(df_sub)
        tc_losses[model] = df_sub
    return tc_losses


import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ---- units ----------------------------------------------------------------
# Losses are multiplied by LOSS_SCALE when the distributions are built.
#   1.0  -> dollars (default, easiest to read)
#   1e-6 -> $m,   1e-9 -> $bn
LOSS_SCALE = 1.0
# The variance penalty has units of 1/(loss unit), so gamma depends on LOSS_SCALE.
# Gammas below are written "per $bn"; multiply by PER_BN to convert to the current units.
PER_BN = 1e-9 / LOSS_SCALE


from collections import namedtuple
LossDist = namedtuple('LossDist', ['xk', 'pk'])

def create_rv_discrete(tc_annual_loss_freq_table, scale=None):
    scale = LOSS_SCALE if scale is None else scale
    aggregated = (
        tc_annual_loss_freq_table
        .groupby('loss', as_index=False)['frequency']
        .sum()
        .sort_values('loss')
    )
    xk = aggregated['loss'].to_numpy() * scale
    pk = aggregated['frequency'].to_numpy()
    return LossDist(xk, pk / pk.sum())

def calc_VaR(rv, alpha, tol=1e-12):
    order = np.argsort(rv.xk)
    x, p = rv.xk[order], rv.pk[order]
    cdf = np.cumsum(p)
    idx = np.searchsorted(cdf, (1.0 - alpha) - tol, side='left')
    return float(x[min(idx, len(x) - 1)])


def calc_CVaR(rv, alpha):
    var_val = calc_VaR(rv, alpha)
    excess = np.maximum(rv.xk - var_val, 0.0)
    return float(var_val + np.sum(excess * rv.pk) / alpha)


def get_TC_model_array(models, tc_dict=None):
    """
    Takes a list of model name strings and a dictionary of model DataFrames,
    returning a list of scipy.stats.rv_discrete objects in the specified order.
    """
    tc_dict = load_tc_losses() if tc_dict is None else tc_dict
    rv_array = []
    for model in models:
        if model not in tc_dict:
            raise KeyError(f"Model '{model}' not found in provided dictionary.")
        
        df = tc_dict[model]
        rv_obj = create_rv_discrete(df)
        rv_array.append(rv_obj)
        
    return rv_array


PREMIUM_PRINCIPLE = 'ev'    # 'ev' or 'ph'
PH_C = 0.7                  # PH-transform index c in (0, 1]; only used when PREMIUM_PRINCIPLE == 'ph' (c = 1 gives 'ev')

def ph_layer_integral(rv, deductible: float, c: float):
    """
    int_d^inf S(x)^c dx for a discrete loss distribution (S is a step function).
    With c = 1 this equals E[(X - d)+].
    """
    order = np.argsort(rv.xk)
    x, p = rv.xk[order], rv.pk[order]
    S = np.clip(1.0 - np.cumsum(p), 0.0, 1.0)          # S(x) on [x_k, x_{k+1})
    left = np.concatenate([[0.0], x[:-1]])              # intervals [0, x_0), [x_0, x_1), ...
    right = x
    S_int = np.concatenate([[1.0], S[:-1]])             # S = 1 below the smallest loss
    width = np.clip(right - np.maximum(left, deductible), 0.0, None)
    return float(np.sum(width * S_int ** c))

def calc_stop_loss_premium(rv, deductible: float, loading: float):
    """
    'ev': (1 + loading) * E[(X-d)+]
    'ph': (1 + loading) * int_d^inf S(x)^c dx   (Wang 1995 PH transform, c = PH_C)
    """
    if PREMIUM_PRINCIPLE == 'ph':
        return float((1 + loading) * ph_layer_integral(rv, deductible, PH_C))

    if PREMIUM_PRINCIPLE == 'ev':
        return float((1 + loading) * np.sum(np.maximum(0.0, rv.xk - deductible) * rv.pk))

    raise ValueError(f"Unknown principle: {PREMIUM_PRINCIPLE}")

def total_insurer_loss(rv, loss, deductible, loading):
    retained = np.minimum(loss, deductible)
    return (retained + calc_stop_loss_premium(rv, deductible, loading))

def delta(rv_array, model_i, d, loading):
    premiums = [calc_stop_loss_premium(rv, d, loading) for rv in rv_array]
    return (premiums[model_i] - np.mean(premiums))


def single_model_total_risk_cvar(rv, d, loading, alpha):
    d = max(0.0, float(d))
    premium = calc_stop_loss_premium(rv, d, loading)
    total = LossDist(np.minimum(rv.xk, d) + premium, rv.pk)
    return calc_CVaR(total, alpha)

def mixture_cvar(rv_array, weights, d, loading, alpha):
    """CVaR_alpha of the mixture of per-model total-risk distributions."""
    weights = np.asarray(weights, dtype=float)
    d = max(0.0, float(d))

    mixed = LossDist(
        np.concatenate([rv.xk for rv in rv_array]),
        np.concatenate([w * rv.pk for rv, w in zip(rv_array, weights)])
    )
    premium = calc_stop_loss_premium(mixed, d, loading)
    total = LossDist(np.minimum(mixed.xk, d) + premium, mixed.pk)
    return calc_CVaR(total, alpha)


def blend_and_variance_cvar(rv_array, weights, d, loading, alpha):
    """CVaR of the blend (priced on the blend), plus variance of the per-model CVaRs."""
    weights = np.asarray(weights, dtype=float)
    cvars = np.array([
        single_model_total_risk_cvar(rv, d, loading, alpha) for rv in rv_array
    ])
    mix_c = mixture_cvar(rv_array, weights, d, loading, alpha)    
    var_c = np.sum(weights * (cvars - np.sum(weights * cvars)) ** 2)
    return mix_c, var_c, cvars

# ---- deductible grid + minimize_scalar refinement ----
def make_grid(rv_array, weights, loading, alpha, n=400, min_frac=1e-8):
    """Coarse log-spaced deductible grid with (blend CVaR, variance CVaR) at each point."""
    dmax = float(max(rv.xk.max() for rv in rv_array))
    d_grid = np.concatenate([[0.0], np.geomspace(min_frac * dmax, dmax, n)])
    vals = np.array([
        blend_and_variance_cvar(rv_array, weights, d, loading, alpha)[:2]
        for d in d_grid
    ])
    return d_grid, vals

def optimize_deductible_mean_variance(rv_array, weights, loading, alpha, gamma, grid=None):
    """
    Solves: min_d mean_CVaR(d) + gamma * variance_CVaR(d).

    Returns the optimal deductible only if all effective weights
    are non-negative:
        1 + 2 * gamma * delta_i(d*) >= 0 for every model.
    """
    max_possible_loss = float(max(np.max(rv.xk) for rv in rv_array))

    def objective(d):
        blend_c, var_c, _ = blend_and_variance_cvar(
            rv_array, weights, d, loading, alpha
        )
        return blend_c + gamma * var_c

    if grid is None:
        grid = make_grid(rv_array, weights, loading, alpha)
    d_grid, vals = grid

    i = int(np.argmin(vals[:, 0] + gamma * vals[:, 1]))
    lo = d_grid[max(i - 1, 0)]
    hi = d_grid[min(i + 1, len(d_grid) - 1)]

    opt_result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method='bounded'
    )

    d_star = float(opt_result.x)

    # keep the grid point if the refinement is somehow worse
    if objective(d_star) > vals[i, 0] + gamma * vals[i, 1]:
        d_star = float(d_grid[i])
    blend_star, var_star, cvars_star = blend_and_variance_cvar(
        rv_array, weights, d_star, loading, alpha
    )

    premiums = np.array([
        calc_stop_loss_premium(rv, d_star, loading)
        for rv in rv_array
    ])
    delta = premiums - np.sum(np.asarray(weights) * premiums)

    # Check effective-weight non-negativity for every model
    effective_weight_factors = 1 + 2 * gamma * delta
    valid_weights = np.all(effective_weight_factors >= -1e-10)

    if not valid_weights:
        return None

    return {
        'gamma': gamma,
        'optimal_deductible': d_star,
        'blend_CVaR': blend_star,
        'variance_CVaR': var_star,
        'per_model_CVaR': cvars_star,
        'effective_weight_factors': effective_weight_factors,
        'effective_weights': np.asarray(weights) * effective_weight_factors,
    }


def find_max_gamma(rv_array, weights, loading, alpha, grid=None, rtol=1e-3, max_doublings=60, verbose=True):
    """
    Largest gamma for which the optimal deductible still gives non-negative effective weights
    w_i (1 + 2 gamma delta_i(d*)) for every model, i.e. the end of the valid frontier.

    1. Start from the closed-form cap at the gamma = 0 optimum: gamma_0 = min over delta_i < 0 of -1/(2 delta_i).
    2. If gamma_0 is still valid (delta changes with d along the frontier), double it until it is not.
    3. Bisect between the last valid and first invalid gamma until they agree to rtol.
    Assumes validity is lost only once as gamma grows (true when the most optimistic model's
    effective weight keeps falling), which is the usual case.
    """
    w = np.asarray(weights, dtype=float)
    if grid is None:
        grid = make_grid(rv_array, w, loading, alpha)

    def valid(g):
        return optimize_deductible_mean_variance(rv_array, w, loading, alpha, g, grid=grid) is not None

    base = optimize_deductible_mean_variance(rv_array, w, loading, alpha, 0.0, grid=grid)
    d0 = base["optimal_deductible"]
    P = np.array([calc_stop_loss_premium(rv, d0, loading) for rv in rv_array])
    dl = P - np.sum(w * P)
    if not np.any(dl < 0):
        if verbose:
            print("all models give the same premium at d*: no variance trade-off (max gamma = 0)")
        return 0.0
    g0 = float(np.min(-1 / (2 * dl[dl < 0])))

    lo, hi = 0.0, g0
    for _ in range(max_doublings):
        if not valid(hi):
            break
        lo, hi = hi, 2 * hi
    else:
        if verbose:
            print(f"warning: still valid after {max_doublings} doublings; returning gamma = {lo:.3e}")
        return lo

    while (hi - lo) > rtol * hi:
        mid = 0.5 * (lo + hi)
        if valid(mid):
            lo = mid
        else:
            hi = mid

    if verbose:
        end = optimize_deductible_mean_variance(rv_array, w, loading, alpha, lo, grid=grid)
        k = int(np.argmin(end["effective_weights"]))
        print(f"max gamma = {lo:.4e} ({lo / PER_BN:.4f} per $bn; closed-form cap at gamma = 0: {g0 / PER_BN:.4f} per $bn); "
              f"lowest effective weight there: model {k} = {end['effective_weights'][k]:.4f}")
    return lo


def generate_mean_variance_frontier(
    rv_array, weights, loading, alpha, max_gamma=None, n_points=25
):
    """
    Sweeps gamma from 0 to max_gamma and keeps only records with valid effective weights.
    max_gamma=None (default) finds the end of the valid frontier automatically (find_max_gamma),
    so the sweep covers the full frontier.
    """
    grid = make_grid(rv_array, weights, loading, alpha)
    if max_gamma is None:
        max_gamma = find_max_gamma(rv_array, weights, loading, alpha, grid=grid)
    gamma_vec = np.linspace(0, max_gamma, n_points)

    records = [
        optimize_deductible_mean_variance(
            rv_array, weights, loading, alpha, gamma, grid=grid
        )
        for gamma in gamma_vec
    ]

    records = [record for record in records if record is not None]

    return pd.DataFrame(records)

def closed_form_deductible(rv_array, wts, loading, adj=None):
    """
    Closed-form deductible (valid for d below every model's VaR) for the objective
        CVaR_alpha(blend total risk) + gamma * variance of per-model CVaRs,
    whose first-order condition is

        S_blend(d)^c + sum_i w_i * adj_i * S_i(d)^c = 1/(1+loading),
        S_blend = sum_i w_i S_i,   adj_i = 2*gamma*delta_i,   delta_i = pi_i - sum_j w_j pi_j,
        c = 1 for 'ev', c = PH_C for 'ph'.

    'ev' (c = 1): reduces to sum_i w_i (1 + adj_i) S_i(d) = 1/(1+loading)  (reweighted Cai & Tan).
    gamma = 0  : reduces to S_blend(d)^c = 1/(1+loading)  (single-model result on the blend).
    Returns the smallest loss value d at which the left-hand side is <= 1/(1+loading).
    """
    if PREMIUM_PRINCIPLE == 'ev':
        c = 1.0
    elif PREMIUM_PRINCIPLE == 'ph':
        c = PH_C
    else:
        raise ValueError("closed form only derived for 'ev' and 'ph'")
    target = 1.0 / (1.0 + loading)
    wts = np.asarray(wts, dtype=float)
    adj = np.zeros(len(rv_array)) if adj is None else np.asarray(adj, dtype=float)
    cand = np.unique(np.concatenate([[0.0]] + [rv.xk for rv in rv_array]))
    S_blend = np.zeros_like(cand)
    pen = np.zeros_like(cand)
    for rv, wi, ai in zip(rv_array, wts, adj):
        order = np.argsort(rv.xk)
        x, cdf = rv.xk[order], np.cumsum(rv.pk[order])
        k = np.searchsorted(x, cand, side='right') - 1          # P(X <= d) = cdf[k]
        S = np.clip(np.where(k >= 0, 1.0 - cdf[np.clip(k, 0, None)], 1.0), 0.0, 1.0)
        S_blend += wi * S
        pen += wi * ai * S ** c
    h = S_blend ** c + pen
    below = np.nonzero(h <= target + 1e-12)[0]
    return float(cand[below[0]] if len(below) else cand[-1])


def closed_form_deductible_check(frontier_df, rv_array, weights, loading, alpha,
                                 max_iter=100, rtol=1e-12):
    """
    For each gamma in the frontier, solve the first-order condition (see closed_form_deductible)
        S_blend(d*)^c + 2*gamma * sum_i w_i delta_i S_i(d*)^c = 1/(1+loading)
    and compare with the numerical optimum. For 'ev' this is the effective-weight blend
    sum_i w_i (1 + 2*gamma*delta_i) S_i(d*) = 1/(1+loading).

    d_closed_form : uses 2*gamma*delta_i stored in the frontier (taken at the numerical d*)
    d_fixed_point : iterates d -> closed_form_deductible(2*gamma*delta(d)) until it stops changing
                    (delta depends on d through the premiums)
    beats_no_reinsurance : objective at d_fixed_point < objective with no reinsurance
                    (needed under 'ph': the objective falls back towards the no-reinsurance
                     value deep in the tail, so the FOC point is only a local optimum)
    """
    w = np.asarray(weights, dtype=float)
    min_var = min(calc_VaR(rv, alpha) for rv in rv_array)   # closed form needs d* below every model's VaR
    d_none = float(max(rv.xk.max() for rv in rv_array))      # deductible = max loss  <=> no reinsurance

    def adj(d, gamma):
        P = np.array([calc_stop_loss_premium(rv, d, loading) for rv in rv_array])
        return 2 * gamma * (P - np.sum(w * P))

    def objective(d, gamma):
        b, v, _ = blend_and_variance_cvar(rv_array, w, d, loading, alpha)
        return b + gamma * v

    rows = []
    for _, r in frontier_df.iterrows():
        g, d_num = r["gamma"], r["optimal_deductible"]

        d_cf = closed_form_deductible(rv_array, w, loading,
                                      adj=np.asarray(r["effective_weight_factors"], dtype=float) - 1)

        d_fp = d_cf
        for _ in range(max_iter):
            d_new = closed_form_deductible(rv_array, w, loading, adj=adj(d_fp, g))
            converged = abs(d_new - d_fp) <= rtol * max(abs(d_fp), 1e-300)
            d_fp = d_new
            if converged:
                break

        obj_num, obj_fp, obj_none = objective(d_num, g), objective(d_fp, g), objective(d_none, g)
        rows.append({
            "gamma": g,
            "d_numerical": d_num,
            "d_closed_form": d_cf,
            "d_fixed_point": d_fp,
            "rel_diff_closed_form": (d_cf - d_num) / d_num,
            "rel_diff_fixed_point": (d_fp - d_num) / d_num,
            "objective_numerical": obj_num,
            "objective_fixed_point": obj_fp,
            "objective_no_reinsurance": obj_none,
            "closed_form_not_worse": obj_fp <= obj_num * (1 + 1e-12),
            "beats_no_reinsurance": obj_fp < obj_none,
            "d_below_all_VaR": d_fp < min_var,
        })

    out = pd.DataFrame(rows)
    print(f"max |rel diff| closed form vs numerical: {out['rel_diff_closed_form'].abs().max():.2e}")
    print(f"max |rel diff| fixed point vs numerical: {out['rel_diff_fixed_point'].abs().max():.2e}")
    print(f"closed form objective <= numerical objective in {out['closed_form_not_worse'].sum()} "
          f"of {len(out)} rows;  d below every model's VaR in all rows: {out['d_below_all_VaR'].all()};  "
          f"beats no reinsurance in all rows: {out['beats_no_reinsurance'].all()}")
    return out
