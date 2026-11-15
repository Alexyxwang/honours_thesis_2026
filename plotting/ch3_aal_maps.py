"""Chapter 3 -- where losses occur: AAL maps (Figures 3.4 and 3.7).

The AAL of each exposure point is CLIMADA's expected annual impact (eai_exp column of the impact CSVs, i.e. the
sum over events of frequency x loss at that point). The points are summed into 0.5 degree cells.

Outputs (code/outputs):
    tables/aal_cells_0p5deg.csv          AAL per 0.5 degree cell and model
    tables/text_aal_by_model.csv         total AAL of each model (US$m)
    figures/fig_aal_map_models.png       Figure 3.4  IBTrACS and the synthetic models
    figures/fig_aal_map_counterfactual.png  Figure 3.7  IBTrACS and the counterfactual sets
"""
import numpy as np
import pandas as pd

from common import FIG_DIR, IMPACT_DIR, TAB_DIR, save_table, use_style

MODELS = ['IBTrACS', 'STORM', 'CHAZ', 'IRIS', 'IBTrACS_p', 'WN_MiniTC_repeat', 'WN_MiniTC_perturb']
CELL = 0.5


def aal_cells(model):
    d = pd.read_csv(IMPACT_DIR / f"TC_AUS_impact_{model}.csv", usecols=['exp_lat', 'exp_lon', 'eai_exp'],
                    low_memory=False).dropna(subset=['exp_lat'])
    d['lat_c'] = np.floor(d.exp_lat / CELL) * CELL + CELL / 2
    d['lon_c'] = np.floor(d.exp_lon / CELL) * CELL + CELL / 2
    g = d.groupby(['lat_c', 'lon_c']).agg(eai=('eai_exp', 'sum'), n_exp=('eai_exp', 'size')).reset_index()
    return g.assign(model=model)


df = pd.concat([aal_cells(m) for m in MODELS], ignore_index=True)
df.to_csv(TAB_DIR / 'aal_cells_0p5deg.csv', index=False)
save_table(pd.DataFrame({'AAL (US$m)': df.groupby('model').eai.sum() / 1e6}).reindex(MODELS), 'text_aal_by_model')

# ------------------------------------------------------------------------------------------------
plt = use_style(10)
import matplotlib
from matplotlib.colors import ListedColormap, LogNorm

CITIES = {'Darwin': (130.84, -12.46), 'Cairns': (145.77, -16.92), 'Port Hedland': (118.61, -20.31),
          'Brisbane': (153.03, -27.47), 'Perth': (115.86, -31.95), 'Sydney': (151.21, -33.87)}
LEFT = {'Brisbane', 'Sydney', 'Cairns'}                     # labels drawn to the left of the marker
NORM = LogNorm(1e0, 1e8)
CMAP = plt.get_cmap('magma_r').copy(); CMAP.set_under('#fff8c4')
LATS, LONS = np.arange(-44.75, -9.5, CELL), np.arange(112.25, 154.5, CELL)


def panel(ax, m, title):
    d = df[df.model == m]
    G = np.full((LATS.size, LONS.size), np.nan); Z = np.full_like(G, np.nan)
    for la, lo, e in zip(d.lat_c, d.lon_c, d.eai):
        i, j = int(round((la - LATS[0]) / CELL)), int(round((lo - LONS[0]) / CELL))
        if 0 <= i < LATS.size and 0 <= j < LONS.size:
            G[i, j] = 1                                      # cell with exposure
            if e > 0:
                Z[i, j] = e                                  # cell with loss
    ax.pcolormesh(LONS, LATS, G, cmap=ListedColormap(['#e6e6e6']), shading='nearest')
    im = ax.pcolormesh(LONS, LATS, np.ma.masked_invalid(Z), cmap=CMAP, norm=NORM, shading='nearest')
    for c, (x, y) in CITIES.items():
        ax.plot(x, y, 'o', ms=5, mfc='white', mec='black', mew=1.3, zorder=5)
        ax.annotate(c, (x, y), xytext=(-5 if c in LEFT else 5, 0), textcoords='offset points',
                    ha='right' if c in LEFT else 'left', va='center', fontsize=11, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6))
    ax.set_title(f"{title}\nAAL US\\${d.eai.sum() / 1e6:,.0f}m", loc='left', fontsize=13)
    ax.set_xlim(112, 155); ax.set_ylim(-45, -9.5); ax.set_aspect(1.1); ax.axis('off')
    return im


def figure(panels, name, ncols=2):
    n = len(panels); nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.3 * nrows + 0.8))
    axs = np.atleast_1d(axs).ravel()
    for ax, (m, t) in zip(axs, panels):
        im = panel(ax, m, t)
    for ax in axs[n:]:
        ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.12, wspace=0.02, hspace=0.18)
    cax = fig.add_axes([0.18, 0.06, 0.64, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='min')
    cb.set_label('Average annual loss per 0.5° cell (US\\$/yr); every cell with loss > 0 shown', fontsize=11)
    fig.savefig(FIG_DIR / name, dpi=200); plt.close(fig); print('saved', FIG_DIR / name)


HIST = ('IBTrACS', 'Historical tracks (IBTrACS, 1980–2021)')
figure([HIST, ('STORM', 'STORM'), ('CHAZ', 'CHAZ'), ('IRIS', 'IRIS')], 'fig_aal_map_models.png')
figure([HIST, ('IBTrACS_p', 'IBTrACS_p'), ('WN_MiniTC_repeat', 'WN-C (repeat fill)'),
        ('WN_MiniTC_perturb', 'WN-C (perturbed fill)')], 'fig_aal_map_counterfactual.png')
