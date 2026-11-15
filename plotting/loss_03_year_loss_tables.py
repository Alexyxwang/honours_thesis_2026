"""Loss pipeline, step 3 -- year loss tables (YLTs) from the per-year hazards of step 2.

Clean copy of `hazard_and_loss_generation/annual_loss_calculation.ipynb`. Requires CLIMADA.
The annual loss of a simulated year is the sum of the losses of all events in its hazard file, with the same
exposure (LitPop, 300 arcsec, 'pc', 2020, lon <= 155) and impact function (Emanuel 2011) as the ELTs.

Output: results/annual_losses_v2.csv  (model, loss, file), one row per hazard file. A year whose storms miss the
exposure has zero loss. For CHAZ, the 250 x 40 = 10,000 (file, member) combinations include 250 without tracks,
which have no file, leaving 9,750 years.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from climada.engine import ImpactCalc
from climada.entity import Exposures
from climada.entity.exposures import LitPop
from climada.entity.impact_funcs import ImpactFuncSet, ImpfTropCyclone
from climada.hazard import TropCyclone

ROOT = Path(os.environ.get("HONOURS_ROOT", Path(__file__).resolve().parents[1]))
HAZ_DIR = ROOT / "hazard"

exp1 = LitPop.from_countries('AUS', 300, (1, 1), 'pc', reference_year=2020)
exp = Exposures(exp1.gdf[exp1.geometry.x <= 155])
exp.check()
exp.gdf['impf_TC'] = 1
if_set = ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])
if_set.check()


def annual_losses(folder):
    files = sorted((HAZ_DIR / folder).glob("*.hdf5"))
    if not files:
        raise ValueError(f"no hazard files in {HAZ_DIR / folder}")
    losses = []
    for i, f in enumerate(files, 1):
        haz = TropCyclone.from_hdf5(f)
        haz.frequency = np.ones(haz.event_id.size)       # some files carry a malformed frequency; it does not affect at_event
        losses.append(ImpactCalc(exp, if_set, haz).impact(save_mat=False).at_event.sum())
        if i % 500 == 0 or i == len(files):
            print(f"[{folder}] {i}/{len(files)}")
    return pd.DataFrame({'loss': losses, 'file': [str(f) for f in files]})


SETS = {'STORM': 'STORM', 'IRIS': 'IRIS', 'CHAZ': 'CHAZ', 'IBTrACS_p': 'Counterfactuals/IBTrACS_p/prob070',
        'IBTrACS': 'IBTrACS_AU', 'MINI-WNC': 'Counterfactuals/Mini-TC/prob070'}
out = pd.concat([annual_losses(folder).assign(model=m) for m, folder in SETS.items()], ignore_index=True)
out[['model', 'loss', 'file']].to_csv(ROOT / "results" / "annual_losses_v2.csv", index=False)
