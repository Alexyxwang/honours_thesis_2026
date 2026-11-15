"""Loss pipeline, step 1 -- event loss tables (ELTs) for every track set (Section 3.1, ELT run).

Clean copy of `hazard_and_loss_generation/event_loss_calculation.ipynb` (cells 0-15; the
exploratory plots of the top events are left out). Requires CLIMADA and the track data in TC_datasets/.

Each track set becomes one hazard (Holland 2008 wind field on a 0.25 degree grid over 110-155E, 45-10S), split to
Meiler et al.'s Southern Hemisphere bounds. Each event gets annual frequency 1/N, N = number of years in the set:
    IBTrACS    1980-2021, category 0-5, 1-hourly                                   N = 42
    STORM      SI + SP, 4 of the 10 files of 1,000 years                            N = 4,000
    CHAZ       global 2017 set, track files up to 2017_100.nc, ensemble members 1-40  N = 9,750 x 100 / 250 = 3,900
               then frequency bias-corrected so its region-hitting rate equals IBTrACS (factor ~0.75)
    IRIS       SI + SP, realisations n0-n39 (100 years each)                        N = 4,000
    IBTrACS_p  CLIMADA perturbed tracks, 30 per storm (seed 42), plus the originals  N = 42 x 31
    WN-C       30 Mini WeatherNext Cyclones members per storm; storms with fewer usable members are filled with
               copies of the historical track (repeat) or perturbed tracks (perturb)   N = 42 x 30
Losses: LitPop exposure of Australia (300 arcsec, produced capital 'pc', 2020, lon <= 155), Emanuel (2011)
impact function (CLIMADA ImpfTropCyclone.from_emanuel_usa: v_thresh = 25.7 m/s, v_half = 74.7 m/s).

Output: impact/TC_AUS/TC_AUS_impact_<model>.csv (Impact.write_csv), the input of every Chapter 3 and 4 script.
Wind fields are cached in hazard/TC_AUS_windfields/ (set REBUILD_HAZARDS = True to rebuild).
"""
import copy
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from climada.engine import Impact
from climada.entity import Exposures
from climada.entity.exposures import LitPop
from climada.entity.impact_funcs import ImpactFuncSet, ImpfTropCyclone
from climada.hazard import Centroids, TCTracks, TropCyclone

ROOT = Path(os.environ.get("HONOURS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "hazard_and_loss_generation"))
from helper import read_simulations_AI, read_simulations_iris  # noqa: E402

TC_DATA = ROOT / "TC_datasets"
HAZARD_DIR = ROOT / "hazard" / "TC_AUS_windfields"
IMPACT_DIR = ROOT / "impact" / "TC_AUS"
REBUILD_HAZARDS = False
reg = 'AUS'

# ------------------------------------------------------------------------------------------------
# Exposure and centroids
# ------------------------------------------------------------------------------------------------
exp1 = LitPop.from_countries('AUS', 300, (1, 1), 'pc', reference_year=2020)
exp = Exposures(exp1.gdf[exp1.geometry.x <= 155])
exp.check()
centroids = Centroids.from_pnt_bounds((110, -45, 155, -10), res=0.25)

# ------------------------------------------------------------------------------------------------
# Meiler et al. (2022) basin split
# ------------------------------------------------------------------------------------------------
BASIN_BOUNDS = {'AP': [-180.0, 0.0, 0.0, 65.0], 'IO': [30.0, 100.0, 0.0, 40.0],
                'SH': [-180.0, 180.0, -60.0, 0.0], 'WP': [100.0, 180.0, 0.0, 65.0]}   # lonmin, lonmax, latmin, latmax
reg_id = {'AP': 5000, 'IO': 5001, 'SH': 5002, 'WP': 5003}


def basin_split_haz(hazard, basin):
    """Keep the centroids inside the basin bounds (region_id is None in CLIMADA 5, so it is built here)."""
    x_min, x_max, y_min, y_max = BASIN_BOUNDS[str(basin)]
    c = hazard.centroids
    basin_idx = (c.lat > y_min) & (c.lat < y_max) & (c.lon > x_min) & (c.lon < x_max)
    region_id = np.zeros(c.size, dtype=int) if c.region_id is None else np.asarray(c.region_id).copy()
    region_id[basin_idx] = reg_id[basin]
    hazard.centroids.gdf['region_id'] = region_id
    return hazard.select(reg_id=reg_id[basin])


# ------------------------------------------------------------------------------------------------
# Track sets
# ------------------------------------------------------------------------------------------------
def drop_nan_tracks(tc_tracks):
    tc_tracks.data = [tr for tr in tc_tracks.data
                      if not (np.isnan(tr.lat.values).any() or np.isnan(tr.lon.values).any())]
    return tc_tracks


# IBTrACS (Meiler et al.: global, CLIMADA default provider, category 0-5, 1-hourly without land parameters)
HIST_YEAR_RANGE = (1980, 2021)
_tracks = TCTracks.from_ibtracs_netcdf(year_range=HIST_YEAR_RANGE)
historical_TCs = TCTracks()
for i in range(0, 6):
    historical_TCs.data.extend(_tracks.subset({'category': i}).data)
historical_TCs.equal_timestep(time_step_h=1., land_params=False)
del _tracks
N_YEARS_HIST = HIST_YEAR_RANGE[1] - HIST_YEAR_RANGE[0] + 1

# STORM: South Indian + South Pacific, 4 of the 10 1,000-year files per basin
N_SLICES, YEARS_PER_SLICE = 4, 1000
STORM_TCs = TCTracks()
for f in [TC_DATA / "STORM" / f"STORM_DATA_IBTRACS_{b}_1000_YEARS_{i}.txt" for b in ("SP", "SI") for i in range(N_SLICES)]:
    STORM_TCs.data.extend(TCTracks.from_simulations_storm(str(f)).data)
N_YEARS_STORM = N_SLICES * YEARS_PER_SLICE

# CHAZ: global 2017 set, the first 100 of the 250 track files, intensity ensemble members 1-40
chaz_files = sorted(glob.glob(str(TC_DATA / "CHAZ" / "*.nc")))
chaz_files = chaz_files[:next(i for i, f in enumerate(chaz_files) if f.endswith("2017_100.nc")) + 1]
CHAZ_TCs = TCTracks()
for f in chaz_files:
    CHAZ_TCs.data += TCTracks.from_simulations_chaz(f, ensemble_nums=list(range(1, 41))).data
N_YEARS_CHAZ = 9750 * len(chaz_files) / 250            # 9,750 track-file x member years in all 250 files


# IRIS: SI + SP realisations n0-n39 (100 years each; SI and SP realisations with the same n are the same years)
def _iris_files(basin, max_n=39):
    files = glob.glob(str(TC_DATA / "IRIS" / f"IRIS_{basin}_100Y_n*.txt"))
    n = lambda p: int(re.search(r"_n(\d+)\.txt$", os.path.basename(p)).group(1))
    return sorted((f for f in files if n(f) <= max_n), key=n)


iris_si, iris_sp = _iris_files("SI"), _iris_files("SP")
IRIS_TCs = TCTracks()
for f in iris_si + iris_sp:
    IRIS_TCs.data += read_simulations_iris(f)
N_YEARS_IRIS = len(iris_si) * 100

# IBTrACS_p: CLIMADA perturbed tracks of the historical set (1-point tracks dropped)
NB_SYNTH_TRACKS = 30
IBTRACS_p_TCs = copy.deepcopy(historical_TCs)
IBTRACS_p_TCs.data = [x for x in IBTRACS_p_TCs.data if x.time.size > 1]
IBTRACS_p_TCs.calc_perturbed_trajectories(nb_synth_tracks=NB_SYNTH_TRACKS, seed=42)
N_YEARS_IBTRACS_p = N_YEARS_HIST * (NB_SYNTH_TRACKS + 1)


# WN-C: Mini WeatherNext Cyclones ensembles, one CSV per storm (TC_datasets/MiniTC/<SID>.csv)
def sh_historical_from_tracks(hist_tcs, basin='SH'):
    """Every IBTrACS track above with a point inside the SH bounds; its 'season' is its SID year."""
    x_min, x_max, y_min, y_max = BASIN_BOUNDS[basin]
    hist_by_sid = {}
    for tr in hist_tcs.data:
        lat, lon = tr.lat.values, tr.lon.values
        lon = np.where(lon > 180, lon - 360, lon)
        if ((lat > y_min) & (lat < y_max) & (lon > x_min) & (lon < x_max)).any():
            hist_by_sid[tr.sid] = tr
    sids = sorted(hist_by_sid)
    return sids, {sid: int(sid[:4]) for sid in sids}, hist_by_sid


def _fill_gap(hist_track, n_missing, mode, seed):
    """n_missing substitute tracks for a storm with too few usable AI members."""
    if mode == "repeat":                                  # copies of the historical track
        out = []
        for k in range(n_missing):
            tr = hist_track.copy(deep=True); tr.attrs["sid"] = f"{hist_track.sid}_rep{k}"; out.append(tr)
        return out
    if mode == "perturb":                                 # CLIMADA perturbed versions of the historical track
        tmp = TCTracks(); tmp.data = [hist_track.copy(deep=True)]
        tmp.equal_timestep(time_step_h=1.)
        tmp.calc_perturbed_trajectories(nb_synth_tracks=n_missing, seed=seed)
        return [tr for tr in tmp.data if not tr.attrs.get("orig_event_flag", True)]
    raise ValueError(mode)


def build_wn_tracks(ai_folder, sids, sid_to_season, hist_by_sid, fill_mode="repeat", n_ens=30, max_dist_deg=8.0, seed=42):
    """Up to n_ens usable members per storm (a member is usable if read_simulations_AI matches it to the storm:
    a forecast track starting within 8 degrees of the observed genesis), topped up to n_ens by fill tracks."""
    WN_TCs, n_filled = TCTracks(), 0
    for sid in sids:
        hist_track = hist_by_sid.get(sid)
        if hist_track is None or hist_track.time.size < 2:
            continue
        matched, path = [], os.path.join(ai_folder, f"{sid}.csv")
        if os.path.exists(path):
            event = pd.Series({"SID": sid, "LAT": float(hist_track.lat.values[0]), "LON": float(hist_track.lon.values[0]),
                               "ISO_TIME": pd.Timestamp(hist_track.time.values[0])})
            matched = read_simulations_AI(path=path, IBTrACS_event=event, max_dist_deg=max_dist_deg)
        matched = list(matched)[:n_ens]
        if n_ens - len(matched) > 0:
            matched += _fill_gap(hist_track, n_ens - len(matched), fill_mode, seed); n_filled += 1
        WN_TCs.data.extend(matched)
    n_years = len({sid_to_season[s] for s in sids}) * n_ens
    print(f"[{os.path.basename(ai_folder)} | {fill_mode}] {len(WN_TCs.data)} tracks, {n_filled}/{len(sids)} storms filled")
    return WN_TCs, n_years


sids, sid_to_season, hist_by_sid = sh_historical_from_tracks(historical_TCs)
WN = {mode: build_wn_tracks(str(TC_DATA / "MiniTC"), sids, sid_to_season, hist_by_sid, fill_mode=mode)
      for mode in ("repeat", "perturb")}

historical_TCs, STORM_TCs, CHAZ_TCs, IRIS_TCs, IBTRACS_p_TCs = map(
    drop_nan_tracks, (historical_TCs, STORM_TCs, CHAZ_TCs, IRIS_TCs, IBTRACS_p_TCs))
WN = {mode: (drop_nan_tracks(t), n) for mode, (t, n) in WN.items()}


# ------------------------------------------------------------------------------------------------
# Hazards (batched for memory, cached as hdf5)
# ------------------------------------------------------------------------------------------------
def build_tc_hazard(tc_tracks, centroids, frequency_per_event, batch_size=200, label="", save_name=None):
    haz_path = HAZARD_DIR / f"TC_{reg}_025res_{save_name}.hdf5"
    if save_name is not None and haz_path.is_file() and not REBUILD_HAZARDS:
        return TropCyclone.from_hdf5(haz_path)
    tc_list = []
    for i in range(0, tc_tracks.size, batch_size):
        batch = TCTracks(); batch.data = tc_tracks.data[i:i + batch_size]
        tc_batch = TropCyclone.from_tracks(batch, centroids=centroids)
        tc_batch.frequency = np.full(tc_batch.event_id.size, frequency_per_event)
        tc_list.append(tc_batch)
        print(f"[{label}] {min(i + batch_size, tc_tracks.size)}/{tc_tracks.size}")
    tc_hazard = TropCyclone.concat(tc_list)
    if save_name is not None:
        HAZARD_DIR.mkdir(parents=True, exist_ok=True); tc_hazard.write_hdf5(haz_path)
    return tc_hazard


hazard_dict = {
    'IBTrACS': build_tc_hazard(historical_TCs, centroids, 1 / N_YEARS_HIST, 'IBTrACS', 'IBTrACS_SH'),
    'STORM': build_tc_hazard(STORM_TCs, centroids, 1 / N_YEARS_STORM, 'STORM', 'STORM'),
    'CHAZ': build_tc_hazard(CHAZ_TCs, centroids, 1 / N_YEARS_CHAZ, 'CHAZ', 'CHAZ_SH'),
    'IBTrACS_p': build_tc_hazard(IBTRACS_p_TCs, centroids, 1 / N_YEARS_IBTRACS_p, 'IBTrACS_p', 'IBTrACS_p_SH'),
    'IRIS': build_tc_hazard(IRIS_TCs, centroids, 1 / N_YEARS_IRIS, 'IRIS', 'IRIS'),
    **{f'WN_MiniTC_{mode}': build_tc_hazard(t, centroids, 1 / n, f'WN_MiniTC_{mode}', f'WN_MiniTC_{mode}')
       for mode, (t, n) in WN.items()},
}
hazard_dict = {name: basin_split_haz(haz, 'SH') for name, haz in hazard_dict.items()}


# CHAZ frequency bias correction (Sobel et al. 2019; Meiler et al. 2022): scale CHAZ so that its annual rate of
# events reaching the exposure region equals the IBTrACS rate (8.1 events per year)
def region_hits(hazard):
    return hazard.intensity.max(axis=1).toarray().ravel() > 0


def freq_bias_corr(hazard, years):
    num_tracks = region_hits(hazard).sum()
    ib = hazard_dict['IBTrACS']
    freq_IB = ib.frequency[region_hits(ib)].sum()
    cor = freq_IB / (num_tracks / years)
    return np.ones(hazard.event_id.size) * cor / years, cor


hazard_dict['CHAZ'].frequency, cor_c = freq_bias_corr(hazard_dict['CHAZ'], N_YEARS_CHAZ)
print(f"CHAZ frequency correction factor: {cor_c:.4f}")

# ------------------------------------------------------------------------------------------------
# Impacts -> ELTs
# ------------------------------------------------------------------------------------------------
if_set = ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])
if_set.check()
exp.gdf['impf_TC'] = 1
IMPACT_DIR.mkdir(parents=True, exist_ok=True)
for name, haz in hazard_dict.items():
    impact = Impact()
    exp.assign_centroids(haz)
    impact.calc(exp, if_set, haz, save_mat=True)
    impact.write_csv(IMPACT_DIR / f"TC_{reg}_impact_{name}.csv")
    print(f"[{name}] AAL: {impact.aai_agg:,.0f}")
