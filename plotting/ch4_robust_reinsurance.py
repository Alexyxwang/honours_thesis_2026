"""Chapter 4 -- optimal reinsurance with multiple CAT models and the robustness penalty (Section 4.4).

Problem of Cai & Tan (2007): choose the stop-loss deductible d minimising CVaR_alpha(min(X, d) + pi(d)), with
pi(d) = (1 + theta) E[(X - d)+], theta = 4, alpha = 5%, where X is the annual loss (YLT) of a model. The blend is
the illustrative 50/50 frequency blend of STORM and CHAZ.

Outputs (code/outputs):
    tables/text_reinsurance_single.csv   d*, CVaR without and with cover, VaR and S(0) for STORM, CHAZ, the blend
    tables/text_reinsurance_cross.csv    each model's CVaR at the blend's d*
    tables/robust_frontier.csv           gamma, d*, effective weights, blend CVaR, variance of model CVaRs
    tables/text_reinsurance_robust.csv   largest admissible gamma and the trade-off at that gamma
    figures/fig_robust_gamma.png         Figure 4.2
"""
import numpy as np
import pandas as pd

from common import COL, FIG_DIR, GRID, INK, TAB_DIR, save_table, use_style
from reinsurance_core import (LossDist, blend_and_variance_cvar, calc_CVaR, calc_VaR, closed_form_deductible,
                              closed_form_deductible_check, generate_mean_variance_frontier, get_TC_model_array,
                              single_model_total_risk_cvar)

theta, alpha = 4, 0.05
storm, chaz = get_TC_model_array(['STORM', 'CHAZ'])
w = np.array([0.5, 0.5])
blend = LossDist(np.concatenate([storm.xk, chaz.xk]), np.concatenate([w[0] * storm.pk, w[1] * chaz.pk]))

# ---- each model and the blend on its own: d* solves S(d*) = 1 / (1 + theta) (the 1-in-5 year annual loss) ----
res = {}
for name, rv in [('STORM', storm), ('CHAZ', chaz), ('blend', blend)]:
    d = closed_form_deductible([rv], [1.0], theta)
    res[name] = {'d* (US$m)': d / 1e6,
                 'CVaR with optimal cover (US$m)': single_model_total_risk_cvar(rv, d, theta, alpha) / 1e6,
                 'CVaR without cover (US$m)': calc_CVaR(rv, alpha) / 1e6,
                 f'VaR_{alpha:g} (US$m)': calc_VaR(rv, alpha) / 1e6,
                 'S(0)': float(rv.pk[rv.xk > 0].sum())}
save_table(pd.DataFrame(res).T, 'text_reinsurance_single')

# ---- CVaR of each model, and of the blend, at the blend's deductible ----
d_blend = closed_form_deductible([storm, chaz], w, theta)
b, v, cv = blend_and_variance_cvar([storm, chaz], w, d_blend, theta, alpha)
save_table(pd.DataFrame({'blend d* (US$m)': [d_blend / 1e6], 'blend CVaR (US$m)': [b / 1e6],
                         'STORM CVaR (US$m)': [cv[0] / 1e6], 'CHAZ CVaR (US$m)': [cv[1] / 1e6],
                         'variance of model CVaRs': [v]}), 'text_reinsurance_cross')

# ---- robustness penalty: frontier over gamma up to the largest admissible value ----
fr = generate_mean_variance_frontier([storm, chaz], w, theta, alpha, n_points=100)
chk = closed_form_deductible_check(fr, [storm, chaz], w, theta, alpha)
fr['w_STORM'] = fr.effective_weights.apply(lambda x: x[0]); fr['w_CHAZ'] = fr.effective_weights.apply(lambda x: x[1])
fr['cvar_STORM'] = fr.per_model_CVaR.apply(lambda x: x[0]); fr['cvar_CHAZ'] = fr.per_model_CVaR.apply(lambda x: x[1])
fr['d_fixed_point'] = chk['d_fixed_point'].to_numpy()
fr = fr.drop(columns=['per_model_CVaR', 'effective_weight_factors', 'effective_weights'])
fr.to_csv(TAB_DIR / 'robust_frontier.csv', index=False)
f0, f1 = fr.iloc[0], fr.iloc[-1]
save_table(pd.DataFrame({
    'largest admissible gamma (x 1e-10 per US$)': [f1.gamma * 1e10],
    'd* at gamma = 0 (US$m)': [f0.optimal_deductible / 1e6], 'd* at largest gamma (US$m)': [f1.optimal_deductible / 1e6],
    'w_STORM at largest gamma': [f1.w_STORM], 'w_CHAZ at largest gamma': [f1.w_CHAZ],
    'change in blend CVaR (%)': [100 * (f1.blend_CVaR / f0.blend_CVaR - 1)],
    'change in variance of model CVaRs (%)': [100 * (f1.variance_CVaR / f0.variance_CVaR - 1)]}).T, 'text_reinsurance_robust')

# ---- Figure 4.2 ----
plt = use_style(13)
g = fr.gamma * 1e10
fig, ax = plt.subplots(1, 3, figsize=(11, 4.4))
for m in ('STORM', 'CHAZ'):
    ax[0].plot(g, fr[f'w_{m}'], color=COL[m], lw=2, label=m)
ax[0].set(ylim=(-0.02, 1.02), ylabel=r'Effective weight $\tilde{w}_i$', title='(a) Effective model weights')
ax[0].legend(loc='center right')
ax[1].plot(g, fr.optimal_deductible / 1e6, color=INK, lw=2)
ax[1].set(ylabel=r'$d^*$ (US\$m)', title='(b) Optimal deductible')
for a in ax[:2]:
    a.set_xlabel(r'$\gamma$ ($\times 10^{-10}$ per US\$)'); a.set_xlim(0, g.max() * 1.02)
x = 100 * (fr.variance_CVaR / fr.variance_CVaR.iloc[0] - 1); y = 100 * (fr.blend_CVaR / fr.blend_CVaR.iloc[0] - 1)
ax[2].plot(x, y, color=INK, lw=2)
ax[2].plot([x.iloc[0], x.iloc[-1]], [y.iloc[0], y.iloc[-1]], 'o', color=INK, ms=6)
ax[2].annotate(r'$\gamma = 0$', (x.iloc[0], y.iloc[0]), xytext=(-6, 8), textcoords='offset points', ha='right')
ax[2].annotate(rf'$\gamma = {fr.gamma.iloc[-1] * 1e10:.1f}\times10^{{-10}}$', (x.iloc[-1], y.iloc[-1]), xytext=(8, -3),
               textcoords='offset points', ha='left')
ax[2].set(xlabel='Change in variance of\nmodel CVaRs (%)', ylabel='Change in blend CVaR (%)', title='(c) Trade-off')
for a in ax:
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.title.set_fontsize(13)
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig_robust_gamma.png', dpi=200); plt.close(fig)
print('saved', FIG_DIR / 'fig_robust_gamma.png')
