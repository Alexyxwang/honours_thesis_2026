"""Loss pipeline, step 2 -- one hazard file per simulated year, for the year loss tables (Section 3.1, YLT run).

Clean copy of Hazard_STORM.py, Hazard_IRIS.py, Hazard_CHAZ.py and Hazard_IBTrACS.py in
`hazard_and_loss_generation/` (the resume-from-file-k shortcuts used while running them in chunks are removed).
Requires CLIMADA and the track data in TC_datasets/.

    python loss_02_year_hazards.py STORM | IRIS | CHAZ | IBTrACS | counterfactual

Each simulated year gets one hazard (Holland 2008 wind field, 0.25 degree grid over 110-155E, 45-10S), written to
hazard/<set>/. Years without storms get no file (they are zero-loss years).
    STORM      all 10 files of 1,000 years for SI and SP; storms of the same synthetic year in SI and SP are combined
               (STORM's own year column is read from the raw txt file)                        10,000 years
    IRIS       realisations n0-n99 for SI and SP; realisation n, year y of SI and SP is one year  10,000 years
    CHAZ       one year per (track file, intensity ensemble member), 250 files x 40 members,
               split to the SH bounds; combinations without tracks are skipped                9,750 years
    IBTrACS    storms with a point in the Australian sub-basins (WA, EA), by SH season (Jul-Jun)  1980-2020
    counterfactual  200 alternative versions (seeds 1-200) of each season 1980-2019: every historical storm is
               independently replaced with probability 0.7 by one of its counterfactual versions, chosen at
               random (IBTrACS_p: 30 CLIMADA perturbations, seed 42; WN-C: the usable Mini WN-C members only,
               so storms without one keep their historical track)                            8,000 years each
"""
import copy as cp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from climada.hazard import TCTracks, TropCyclone

ROOT = Path(os.environ.get("HONOURS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "hazard_and_loss_generation"))
from helper import basin_split_haz, get_AUS_centroids, read_simulations_AI, read_simulations_iris  # noqa: E402

TC_DATA = ROOT / "TC_datasets"
HAZARD_DIR = ROOT / "hazard"
reg = 'SH'
cent = get_AUS_centroids()                    # 0.25 degree grid, (110, -45, 155, -10)


def drop_nan_tracks(tc_tracks):
    tc_tracks.data = [tr for tr in tc_tracks.data
                      if not (np.isnan(tr.lat.values).any() or np.isnan(tr.lon.values).any())]
    return tc_tracks


def write_year(tracks, path, equal_timestep=False, split=False):
    batch = TCTracks(); batch.data = tracks
    if equal_timestep:
        batch.equal_timestep(time_step_h=1.)
    tc_hazard = TropCyclone.from_tracks(batch, centroids=cent)
    if split:
        tc_hazard = basin_split_haz(tc_hazard, reg)
    tc_hazard.check()
    tc_hazard.write_hdf5(path)


# ------------------------------------------------------------------------------------------------
# STORM
# ------------------------------------------------------------------------------------------------
def storm_years(n_loops=10, years_per_loop=1000):
    """CLIMADA's STORM reader dates every track to 1980, so each track's synthetic year (column 0 of the raw file)
    is matched in file order to the tracks CLIMADA returns (one per unique (year, tc_number))."""
    out_dir = HAZARD_DIR / "STORM"; out_dir.mkdir(parents=True, exist_ok=True)
    year_groups = {}
    for basin in ('SI', 'SP'):
        for loop_idx in range(n_loops):
            path = TC_DATA / "STORM" / f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_{loop_idx}.txt"
            tracks = TCTracks.from_simulations_storm(str(path))
            if not tracks.data:
                continue
            raw = pd.read_csv(path, header=None)
            years = raw.drop_duplicates(subset=[0, 2], keep="first")[0].astype(int).tolist()
            if len(years) != len(tracks.data):
                raise ValueError(f"{path}: {len(years)} storms in the raw file, {len(tracks.data)} tracks")
            for tr, y in zip(tracks.data, years):
                year_groups.setdefault(loop_idx * years_per_loop + y, []).append(tr)
    for gy, trs in sorted(year_groups.items()):
        write_year(trs, out_dir / f"TC_{reg}_025res_STORM_year{gy:04d}.hdf5")


# ------------------------------------------------------------------------------------------------
# IRIS
# ------------------------------------------------------------------------------------------------
IRIS_COLUMNS = ['tcid', 'basin', 'year', 'tc', 'month', 'timestep', 'lon', 'lat', 'vmax', 'pmin', 'rmw', 'r18']


def iris_years(n_realisations=100):
    """The IRIS year (already absolute, e.g. n3 spans years 300-399) is read from the raw file and matched in file
    order (by unique tcid) to the tracks returned by read_simulations_iris."""
    out_dir = HAZARD_DIR / "IRIS"; out_dir.mkdir(parents=True, exist_ok=True)
    for n in range(n_realisations):
        year_groups = {}
        for basin in ('SI', 'SP'):
            path = TC_DATA / "IRIS" / f"IRIS_{basin}_100Y_n{n}.txt"
            if not path.exists():
                continue
            tracks = read_simulations_iris(path)
            raw = pd.read_csv(path, sep=r"\s+", comment='#', names=IRIS_COLUMNS)
            years = raw.drop_duplicates(subset=['tcid'], keep='first')['year'].astype(int).tolist()
            if len(years) != len(tracks):
                raise ValueError(f"{path}: {len(years)} storms in the raw file, {len(tracks)} tracks")
            for tr, y in zip(tracks, years):
                year_groups.setdefault(y, []).append(tr)
        for gy, trs in sorted(year_groups.items()):
            write_year(trs, out_dir / f"TC_{reg}_025res_IRIS_year{gy:05d}.hdf5", equal_timestep=True)


# ------------------------------------------------------------------------------------------------
# CHAZ
# ------------------------------------------------------------------------------------------------
def chaz_years(n_files=250, n_ensembles=40):
    """Each (track file, ensemble member) is one synthetic year: global_year = (file - 1) * 40 + (member - 1)."""
    out_dir = HAZARD_DIR / "CHAZ"; out_dir.mkdir(parents=True, exist_ok=True)
    for i_file in range(1, n_files + 1):
        fname = TC_DATA / "CHAZ" / f"2017_{i_file:03d}.nc"
        if not fname.exists():
            continue
        for ens in range(1, n_ensembles + 1):
            gy = (i_file - 1) * n_ensembles + (ens - 1)
            try:
                tracks = TCTracks.from_simulations_chaz(fname, ensemble_nums=[ens])
            except Exception as e:                    # some members are empty
                print(f"{fname.name} member {ens}: {e}")
                continue
            tracks = drop_nan_tracks(tracks)
            if tracks.data:
                write_year(tracks.data, out_dir / f"TC_{reg}_025res_CHAZ_year{gy:04d}.hdf5", split=True)


# ------------------------------------------------------------------------------------------------
# IBTrACS seasons and counterfactual seasons
# ------------------------------------------------------------------------------------------------
def ibtracs_au_tracks(first_season=1980, last_season=2020):
    """Storms with at least one point in the Australian sub-basins (WA, EA) of the IBTrACS csv, and their season."""
    ib = pd.read_csv(ROOT / "ibtracs.since1980.list.v04r01.csv", usecols=["SID", "SEASON", "SUBBASIN", "ISO_TIME"],
                     low_memory=False)
    ib["SEASON"] = pd.to_numeric(ib["SEASON"], errors="coerce")
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    ib = ib.dropna(subset=["SEASON", "ISO_TIME"])
    sid_to_season = ib.drop_duplicates("SID").set_index("SID")["SEASON"].astype(int).to_dict()
    au_sids = set(ib.loc[ib["SUBBASIN"].isin(("WA", "EA")), "SID"])

    def season(tr):                                   # SH seasons run July-June
        s = sid_to_season.get(tr.sid)
        if s is not None:
            return s
        t = pd.Timestamp(tr.time.values[0]); return t.year + (1 if t.month >= 7 else 0)

    tracks = TCTracks.from_ibtracs_netcdf(year_range=(first_season - 1, last_season))   # SID year = genesis year
    tracks.data = [tr for tr in tracks.data if tr.sid in au_sids]
    return tracks, season


def ibtracs_seasons(first_season=1980, last_season=2020):
    out_dir = HAZARD_DIR / "IBTrACS_AU"; out_dir.mkdir(parents=True, exist_ok=True)
    tracks, season = ibtracs_au_tracks(first_season, last_season)
    groups = {}
    for tr in tracks.data:
        groups.setdefault(season(tr), []).append(tr)
    for s, trs in sorted(groups.items()):
        if first_season <= s <= last_season:
            batch = TCTracks(); batch.data = trs; batch.equal_timestep(time_step_h=1., land_params=False)
            write_year(batch.data, out_dir / f"TC_{reg}_025res_IBTrACS_season{s}.hdf5")


def counterfactual_seasons(replace_prob=0.7, seasons=range(1980, 2020), seeds=range(1, 201)):
    tracks, season = ibtracs_au_tracks(1980, 2020)
    sid_to_year = {tr.sid: season(tr) for tr in tracks.data}

    # IBTrACS_p: 30 perturbed versions of every storm, generated once
    synth = cp.deepcopy(tracks)
    synth.data = [tr for tr in synth.data if tr.time.size > 1]
    synth.equal_timestep(time_step_h=1.)
    synth.calc_perturbed_trajectories(nb_synth_tracks=30, seed=42)
    ibp = {}
    for tr in synth.data:
        if not tr.attrs.get('orig_event_flag', True):
            ibp.setdefault(tr.sid.rsplit('_gen', 1)[0], []).append(tr)

    # WN-C: the usable Mini WN-C members of every storm (no fill tracks)
    wnc = {}
    for tr in tracks.data:
        path = TC_DATA / "MiniTC" / f"{tr.sid}.csv"
        if path.exists():
            event = pd.Series({"SID": tr.sid, "LAT": float(tr.lat.values[0]), "LON": float(tr.lon.values[0]),
                               "ISO_TIME": pd.Timestamp(tr.time.values[0])})
            m = read_simulations_AI(path=str(path), IBTrACS_event=event, max_dist_deg=8.0)
            if m:
                wnc[tr.sid] = m

    for name, lookup, out in (('IBTrACS_p', ibp, 'IBTrACS_p'), ('Mini-WNC', wnc, 'Mini-TC')):
        out_dir = HAZARD_DIR / "Counterfactuals" / out / f"prob{round(replace_prob * 100):03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in seasons:
            real = [tr for tr in tracks.data if sid_to_year.get(tr.sid) == s]
            for seed in seeds:
                rng = np.random.default_rng(seed)
                cf = []
                for tr in real:
                    cand = lookup.get(tr.sid, [])
                    cf.append((cand[rng.integers(len(cand))] if cand and rng.random() < replace_prob else tr).copy(deep=True))
                batch = TCTracks(); batch.data = cf; batch.equal_timestep(time_step_h=1., land_params=False)
                write_year(batch.data, out_dir / f"TC_{reg}_025res_{name}_season{s}_seed{seed:04d}.hdf5")


if __name__ == '__main__':
    {'STORM': storm_years, 'IRIS': iris_years, 'CHAZ': chaz_years, 'IBTrACS': ibtracs_seasons,
     'counterfactual': counterfactual_seasons}[sys.argv[1]]()
