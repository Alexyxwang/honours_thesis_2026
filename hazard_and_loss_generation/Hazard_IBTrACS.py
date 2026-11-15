import os
import glob
import numpy as np
from pathlib import Path
import pandas as pd
import copy as cp

from climada.hazard import Centroids, TCTracks, TropCyclone

from helper import read_simulations_AI, get_AUS_centroids

#----------------------------------------PATHS / DIRECTORIES-------------------------------------------
IBTRACS_PATH = "../ibtracs.since1980.list.v04r01.csv"
WN_folder = "../TC_datasets/MiniTC"    
WN_QM_folder = "../TC_datasets/MiniTC-QM"   

HAZARD_DIR = Path("../hazard")
IBTRACS_HAZARD_DIR = HAZARD_DIR / "IBTrACS_AU"            # filtered AU-only files, kept separate from old all-SH files
COUNTERFACTUAL_DIR = HAZARD_DIR / "Counterfactuals"
IBTRACS_P_DIR = COUNTERFACTUAL_DIR / "IBTrACS_p"
AI_CF_DIR = COUNTERFACTUAL_DIR / "Mini-TC"
AI_CF_QM_DIR = COUNTERFACTUAL_DIR / "Mini-TC-QM"

for _d in (IBTRACS_HAZARD_DIR, COUNTERFACTUAL_DIR, IBTRACS_P_DIR, AI_CF_DIR, AI_CF_QM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#----------------------------------------SETTINGS-------------------------------------------
reg = 'SH'
cent = get_AUS_centroids()

#----------------------------------------SEASON + SUBBASIN FILTER-------------------------------------------
AU_SUBBASINS = ("WA", "EA")      # Australian region

_ib = pd.read_csv(
    IBTRACS_PATH,
    usecols=["SID", "SEASON", "SUBBASIN", "ISO_TIME"],
    low_memory=False,
)
_ib["SEASON"] = pd.to_numeric(_ib["SEASON"], errors="coerce")      # also drops the units row
_ib["ISO_TIME"] = pd.to_datetime(_ib["ISO_TIME"], errors="coerce")
_ib = _ib.dropna(subset=["SEASON", "ISO_TIME"])

# season lookup
SID_TO_SEASON = (_ib.drop_duplicates("SID").set_index("SID")["SEASON"].astype(int).to_dict())

# keep any storm with at least one track point in WA or EA (whole track is kept)
AU_SIDS = set(_ib.loc[_ib["SUBBASIN"].isin(AU_SUBBASINS), "SID"])
print(f"{len(AU_SIDS)} storms have at least one point in {AU_SUBBASINS}")


def get_season(tr):
    """IBTrACS SEASON for a CLIMADA track (SH seasons run Jul-Jun)."""
    s = SID_TO_SEASON.get(tr.sid)
    if s is not None:
        return s
    t = pd.Timestamp(tr.time.values[0])          # fallback if SID isn't in the CSV
    return t.year + (1 if t.month >= 7 else 0)


#----------------------------------------IBTrACS-------------------------------------------
# year_range filters on the SID prefix (genesis calendar year), so load one year either
# side and drop incomplete seasons after grouping
FIRST_SEASON, LAST_SEASON = 1980, 2020
IBTrACS = TCTracks.from_ibtracs_netcdf(year_range=(FIRST_SEASON - 1, LAST_SEASON))

# keep only storms with at least one point in WA or EA
n_before = len(IBTrACS.data)
IBTrACS.data = [tr for tr in IBTrACS.data if tr.sid in AU_SIDS]
print(f"Filtered tracks: kept {len(IBTrACS.data)} of {n_before}")

year_groups = {}
for tr in IBTrACS.data:
    year_groups.setdefault(get_season(tr), []).append(tr)

year_groups = {s: v for s, v in year_groups.items() if FIRST_SEASON <= s <= LAST_SEASON}

n_written, n_skipped = 0, 0
for year, year_tracks in sorted(year_groups.items()):
    haz_str = f"TC_{reg}_025res_IBTrACS_season{year}.hdf5"
    hazard_path = IBTRACS_HAZARD_DIR.joinpath(haz_str)

    if hazard_path.is_file():
        print(f"Skipping IBTrACS season {year} -- already exists at {hazard_path}")
        n_skipped += 1
        continue

    print(f"Processing IBTrACS season {year}.")
    batch = TCTracks()
    batch.data = year_tracks
    batch.equal_timestep(time_step_h=1., land_params=False)

    tc_hazard = TropCyclone.from_tracks(batch, centroids=cent)
    tc_hazard.check()
    tc_hazard.write_hdf5(hazard_path)
    n_written += 1

print(f"Wrote {n_written} new seasonal hazard files, skipped {n_skipped} existing ones, in {IBTRACS_HAZARD_DIR}")


#----------------------------------------IBTrACS_p-------------------------------------------

def build_ibtracs_p_lookup(tracks_hist, nb_synth_tracks=9, seed=None):
    """
    Generate IBTrACS_p (perturbed) counterfactual tracks for every storm in
    `tracks_hist` ONCE, and return lookups to repeatedly sample from instead
    of regenerating perturbations on every call.

    Returns:
        sid_to_synth: dict {original_sid: [synthetic_track_1, ..., synthetic_track_N]}
        sid_to_year:  dict {original_sid: season}, from the IBTrACS SEASON of that storm
    """
    tracks_synth = cp.deepcopy(tracks_hist)
    tracks_synth.data = [tr for tr in tracks_synth.data if tr.time.size > 1]
    tracks_synth.equal_timestep(time_step_h=1.)
    tracks_synth.calc_perturbed_trajectories(nb_synth_tracks=nb_synth_tracks, seed=seed)

    sid_to_synth = {}
    for tr in tracks_synth.data:
        if tr.attrs.get('orig_event_flag', True):
            continue  # the real historical track, not a synthetic counterfactual
        base_sid = tr.sid.rsplit('_gen', 1)[0] if '_gen' in tr.sid else tr.sid
        sid_to_synth.setdefault(base_sid, []).append(tr)

    sid_to_year = {
        tr.sid: get_season(tr)
        for tr in tracks_hist.data
    }

    print(f"Generated counterfactuals for {len(sid_to_synth)}/{len(tracks_hist.data)} "
          f"historical storms ({nb_synth_tracks} each)")

    return sid_to_synth, sid_to_year


def format_prob(replace_prob):
    """Turn e.g. 0.7 -> 'prob070', 0.05 -> 'prob005', for use in filenames."""
    return f"prob{round(replace_prob * 100):03d}"


def generate_counterfactual_year(year, tracks_hist, sid_to_synth, sid_to_year,
                                  cent, reg='SH', model_name='IBTrACS_p',
                                  replace_prob=0.5, seed=None,
                                  out_dir=COUNTERFACTUAL_DIR, save=True):
    """
    Build one counterfactual version of season `year`'s TC history: start from
    the real historical tracks that occurred in that season, and randomly
    replace each one (independently, with probability `replace_prob`) with one
    of its pre-generated counterfactual siblings (from sid_to_synth -- either
    IBTrACS_p perturbations or AI-ensemble matches), drawn uniformly at
    random.

    Storms with no counterfactual siblings available always keep their
    real historical version, regardless of the coin flip.

    If save=True (default), also computes the windfield hazard for this
    draw and writes it to
    out_dir/{prob_str}/TC_{reg}_025res_{model_name}_season{year}_seed{seed:04d}.hdf5,
    where prob_str is a subdirectory encoding replace_prob (e.g.
    replace_prob=0.7 -> out_dir/prob070/...).

    Returns (tracks, tc_hazard) -- tc_hazard is None if save=False.
    """
    rng = np.random.default_rng(seed)

    year_tracks_real = [tr for tr in tracks_hist.data if sid_to_year.get(tr.sid) == year]
    if not year_tracks_real:
        print(f"No historical storms found for season {year}")
        return TCTracks(), None

    counterfactual_tracks = []
    for tr in year_tracks_real:
        candidates = sid_to_synth.get(tr.sid, [])
        if candidates and rng.random() < replace_prob:
            chosen = candidates[rng.integers(len(candidates))]
            counterfactual_tracks.append(chosen.copy(deep=True))
        else:
            counterfactual_tracks.append(tr.copy(deep=True))

    result = TCTracks()
    result.data = counterfactual_tracks

    tc_hazard = None
    if save:
        result.equal_timestep(time_step_h=1., land_params=False)
        tc_hazard = TropCyclone.from_tracks(result, centroids=cent)
        tc_hazard.check()

        prob_dir = Path(out_dir) / format_prob(replace_prob)
        prob_dir.mkdir(parents=True, exist_ok=True)

        seed_str = f"seed{seed:04d}" if seed is not None else "seedNone"
        haz_str = f"TC_{reg}_025res_{model_name}_season{year}_{seed_str}.hdf5"
        hazard_path = prob_dir.joinpath(haz_str)
        tc_hazard.write_hdf5(hazard_path)
        print(f"Wrote counterfactual hazard for season {year} (prob={replace_prob}, {seed_str}) -> {hazard_path}")

    return result, tc_hazard


# Sweep of replace probabilities to test -- add/remove values as needed.
REPLACE_PROBS = [0.7]

sid_to_synth, sid_to_year = build_ibtracs_p_lookup(IBTrACS, nb_synth_tracks=30, seed=42)

for replace_prob in REPLACE_PROBS:
    for year in range(1980, 2020):
        for seed in range(1, 201):
            generate_counterfactual_year(
                year, IBTrACS, sid_to_synth, sid_to_year, cent,
                reg=reg, replace_prob=replace_prob, seed=seed,
                out_dir=IBTRACS_P_DIR
            )


#-------------------------------AI counterfactuals--------------------------------------------------
def build_ai_lookup(tracks_hist, ai_folder, max_dist_deg=8.0):
    """
    Load WeatherNext (FNv3.2) AI-ensemble counterfactual tracks for every
    storm in `tracks_hist` that has a matching ensemble file on disk, ONCE,
    and return lookups to repeatedly sample from -- same shape as
    build_ibtracs_p_lookup(), just sourced from AI ensemble files instead
    of calc_perturbed_trajectories().

    Only storms with an existing {sid}.csv in ai_folder contribute -- since
    WeatherNext only has ensembles for post-2022 AU storms, this naturally
    restricts things to the right subset without needing to pre-filter
    tracks_hist by basin/year yourself.

    Returns:
        sid_to_synth: dict {sid: [ai_track_1, ..., ai_track_N]}
        sid_to_year:  dict {sid: season}, from the IBTrACS SEASON of that storm
    """
    sid_to_synth = {}
    sid_to_year = {}

    n_checked = 0
    for tr in tracks_hist.data:
        sid = tr.sid
        sid_to_year[sid] = get_season(tr)

        ensemble_filepath = os.path.join(ai_folder, f"{sid}.csv")
        if not os.path.exists(ensemble_filepath):
            continue

        n_checked += 1
        # minimal "event" record read_simulations_AI expects, built
        # straight from the CLIMADA track itself -- no raw ibtracs csv needed
        event = pd.Series({
            "SID": sid,
            "LAT": float(tr.lat.values[0]),
            "LON": float(tr.lon.values[0]),
            "ISO_TIME": pd.Timestamp(tr.time.values[0]),
        })

        matched_tracks = read_simulations_AI(
            path=ensemble_filepath,
            IBTrACS_event=event,
            max_dist_deg=max_dist_deg,
        )

        if matched_tracks:
            sid_to_synth[sid] = matched_tracks

    print(f"Found {n_checked} storms with a WeatherNext ensemble file, "
          f"{len(sid_to_synth)} with at least one matched track "
          f"(of {len(tracks_hist.data)} total historical storms)")

    return sid_to_synth, sid_to_year


sid_to_synth_ai, sid_to_year_ai = build_ai_lookup(IBTrACS, WN_folder, max_dist_deg=8.0)

for replace_prob in REPLACE_PROBS:
    for year in range(1980, 2020):          # seasons 2004-2021
        for seed in range(1, 201):
            generate_counterfactual_year(
                year, IBTrACS, sid_to_synth_ai, sid_to_year_ai, cent,
                reg=reg, model_name='Mini-WNC', replace_prob=replace_prob, seed=seed,
                out_dir=AI_CF_DIR
            )

sid_to_synth_ai, sid_to_year_ai = build_ai_lookup(IBTrACS, WN_QM_folder, max_dist_deg=8.0)

for replace_prob in REPLACE_PROBS:
    for year in range(1980, 2020):          # seasons 2004-2021
        for seed in range(1, 201):
            generate_counterfactual_year(
                year, IBTrACS, sid_to_synth_ai, sid_to_year_ai, cent,
                reg=reg, model_name='Mini-WNC', replace_prob=replace_prob, seed=seed,
                out_dir=AI_CF_QM_DIR
            )

# import os
# import glob
# import numpy as np
# from pathlib import Path
# import pandas as pd
# import copy as cp
# import numpy as np

# # import CLIMADA modules:
# from climada.hazard import Centroids, TCTracks, TropCyclone

# # custom IRIS track-file parser (keep alongside this script, or on PYTHONPATH):
# from helper import read_simulations_AI, get_AUS_centroids

# cent = get_AUS_centroids()

# # new folder so the filtered AU-only files don't mix with old all-SH files
# HAZARD_DIR = Path("../hazard/IBTrACS_AU")
# HAZARD_DIR.mkdir(parents=True, exist_ok=True)
# reg = 'SH'

# #----------------------------------------SEASON + CYCLONE FILTER-------------------------------------------
# IBTRACS_PATH = "../../ibtracs.since1980.list.v04r01.csv"
# MIN_WIND_KT = 34                 # tropical cyclone threshold (Cat 1+), WMO_WIND
# AU_SUBBASINS = ("WA", "EA")      # Australian region, from the storm's first record

# _ib = pd.read_csv(
#     IBTRACS_PATH,
#     usecols=["SID", "SEASON", "BASIN", "SUBBASIN", "ISO_TIME", "WMO_WIND"],
#     low_memory=False,
# )
# _ib["SEASON"] = pd.to_numeric(_ib["SEASON"], errors="coerce")      # also drops the units row
# _ib["WMO_WIND"] = pd.to_numeric(_ib["WMO_WIND"], errors="coerce")
# _ib["ISO_TIME"] = pd.to_datetime(_ib["ISO_TIME"], errors="coerce")
# _ib = _ib.dropna(subset=["SEASON", "ISO_TIME"])

# # season lookup
# SID_TO_SEASON = (_ib.drop_duplicates("SID").set_index("SID")["SEASON"].astype(int).to_dict())

# # first record per storm (SI/SP basins), must be in the Australian subbasins
# _first = (
#     _ib[_ib["BASIN"].isin(["SI", "SP"])]
#     .sort_values("ISO_TIME")
#     .drop_duplicates("SID", keep="first")
# )
# _au_first = set(_first.loc[_first["SUBBASIN"].isin(AU_SUBBASINS), "SID"])

# # peak wind over all rows of each storm, must reach cyclone strength
# _max_wind = _ib.groupby("SID")["WMO_WIND"].max()
# _cyclone = set(_max_wind[_max_wind >= MIN_WIND_KT].index)

# AU_CYCLONE_SIDS = _au_first & _cyclone
# print(f"{len(AU_CYCLONE_SIDS)} Australian-region tropical cyclones pass the filter")


# def get_season(tr):
#     """IBTrACS SEASON for a CLIMADA track (SH seasons run Jul-Jun)."""
#     s = SID_TO_SEASON.get(tr.sid)
#     if s is not None:
#         return s
#     t = pd.Timestamp(tr.time.values[0])          # fallback if SID isn't in the CSV
#     return t.year + (1 if t.month >= 7 else 0)   # check this convention against the CSV


# #----------------------------------------IBTrACS-------------------------------------------
# # year_range filters on the SID prefix (genesis calendar year), so load one year either
# # side and drop incomplete seasons after grouping
# FIRST_SEASON, LAST_SEASON = 1980, 2020
# IBTrACS = TCTracks.from_ibtracs_netcdf(year_range=(FIRST_SEASON - 1, LAST_SEASON))

# # keep only Australian-region tropical cyclones (same filter as the plots)
# n_before = len(IBTrACS.data)
# IBTrACS.data = [tr for tr in IBTrACS.data if tr.sid in AU_CYCLONE_SIDS]
# print(f"Filtered tracks: kept {len(IBTrACS.data)} of {n_before}")

# year_groups = {}
# for tr in IBTrACS.data:
#     year_groups.setdefault(get_season(tr), []).append(tr)

# year_groups = {s: v for s, v in year_groups.items() if FIRST_SEASON <= s <= LAST_SEASON}

# n_written, n_skipped = 0, 0
# for year, year_tracks in sorted(year_groups.items()):
#     haz_str = f"TC_{reg}_025res_IBTrACS_season{year}.hdf5"
#     hazard_path = HAZARD_DIR.joinpath(haz_str)

#     if hazard_path.is_file():
#         print(f"Skipping IBTrACS season {year} -- already exists at {hazard_path}")
#         n_skipped += 1
#         continue

#     print(f"Processing IBTrACS season {year}.")
#     batch = TCTracks()
#     batch.data = year_tracks
#     batch.equal_timestep(time_step_h=1., land_params=False)

#     tc_hazard = TropCyclone.from_tracks(batch, centroids=cent)
#     tc_hazard.check()
#     tc_hazard.write_hdf5(hazard_path)
#     n_written += 1

# print(f"Wrote {n_written} new seasonal hazard files, skipped {n_skipped} existing ones, in {HAZARD_DIR}")


# #----------------------------------------IBTrACS_p-------------------------------------------

# def build_ibtracs_p_lookup(tracks_hist, nb_synth_tracks=9, seed=None):
#     """
#     Generate IBTrACS_p (perturbed) counterfactual tracks for every storm in
#     `tracks_hist` ONCE, and return lookups to repeatedly sample from instead
#     of regenerating perturbations on every call.

#     Returns:
#         sid_to_synth: dict {original_sid: [synthetic_track_1, ..., synthetic_track_N]}
#         sid_to_year:  dict {original_sid: season}, from the IBTrACS SEASON of that storm
#     """
#     tracks_synth = cp.deepcopy(tracks_hist)
#     tracks_synth.data = [tr for tr in tracks_synth.data if tr.time.size > 1]
#     tracks_synth.equal_timestep(time_step_h=1.)
#     tracks_synth.calc_perturbed_trajectories(nb_synth_tracks=nb_synth_tracks, seed=seed)

#     sid_to_synth = {}
#     for tr in tracks_synth.data:
#         if tr.attrs.get('orig_event_flag', True):
#             continue  # the real historical track, not a synthetic counterfactual
#         base_sid = tr.sid.rsplit('_gen', 1)[0] if '_gen' in tr.sid else tr.sid
#         sid_to_synth.setdefault(base_sid, []).append(tr)

#     sid_to_year = {
#         tr.sid: get_season(tr)
#         for tr in tracks_hist.data
#     }

#     print(f"Generated counterfactuals for {len(sid_to_synth)}/{len(tracks_hist.data)} "
#           f"historical storms ({nb_synth_tracks} each)")

#     return sid_to_synth, sid_to_year

# HAZARD_DIR = Path("../hazard") 
# COUNTERFACTUAL_DIR = HAZARD_DIR / "Counterfactuals"
# COUNTERFACTUAL_DIR.mkdir(parents=True, exist_ok=True)


# def format_prob(replace_prob):
#     """Turn e.g. 0.7 -> 'prob070', 0.05 -> 'prob005', for use in filenames."""
#     return f"prob{round(replace_prob * 100):03d}"


# def generate_counterfactual_year(year, tracks_hist, sid_to_synth, sid_to_year,
#                                   cent, reg='SH', model_name='IBTrACS_p',
#                                   replace_prob=0.5, seed=None,
#                                   out_dir=COUNTERFACTUAL_DIR, save=True):
#     """
#     Build one counterfactual version of season `year`'s TC history: start from
#     the real historical tracks that occurred in that season, and randomly
#     replace each one (independently, with probability `replace_prob`) with one
#     of its pre-generated counterfactual siblings (from sid_to_synth -- either
#     IBTrACS_p perturbations or AI-ensemble matches), drawn uniformly at
#     random.

#     Storms with no counterfactual siblings available always keep their
#     real historical version, regardless of the coin flip.

#     If save=True (default), also computes the windfield hazard for this
#     draw and writes it to
#     out_dir/{prob_str}/TC_{reg}_025res_{model_name}_season{year}_seed{seed:04d}.hdf5,
#     where prob_str is a subdirectory encoding replace_prob (e.g.
#     replace_prob=0.7 -> out_dir/prob070/...).

#     Returns (tracks, tc_hazard) -- tc_hazard is None if save=False.
#     """
#     rng = np.random.default_rng(seed)

#     year_tracks_real = [tr for tr in tracks_hist.data if sid_to_year.get(tr.sid) == year]
#     if not year_tracks_real:
#         print(f"No historical storms found for season {year}")
#         return TCTracks(), None

#     counterfactual_tracks = []
#     for tr in year_tracks_real:
#         candidates = sid_to_synth.get(tr.sid, [])
#         if candidates and rng.random() < replace_prob:
#             chosen = candidates[rng.integers(len(candidates))]
#             counterfactual_tracks.append(chosen.copy(deep=True))
#         else:
#             counterfactual_tracks.append(tr.copy(deep=True))

#     result = TCTracks()
#     result.data = counterfactual_tracks

#     tc_hazard = None
#     if save:
#         result.equal_timestep(time_step_h=1., land_params=False)
#         tc_hazard = TropCyclone.from_tracks(result, centroids=cent)
#         tc_hazard.check()

#         prob_dir = Path(out_dir) / format_prob(replace_prob)
#         prob_dir.mkdir(parents=True, exist_ok=True)

#         seed_str = f"seed{seed:04d}" if seed is not None else "seedNone"
#         haz_str = f"TC_{reg}_025res_{model_name}_season{year}_{seed_str}.hdf5"
#         hazard_path = prob_dir.joinpath(haz_str)
#         tc_hazard.write_hdf5(hazard_path)
#         print(f"Wrote counterfactual hazard for season {year} (prob={replace_prob}, {seed_str}) -> {hazard_path}")

#     return result, tc_hazard


# # Sweep of replace probabilities to test -- add/remove values as needed.
# REPLACE_PROBS = [0.7]

# sid_to_synth, sid_to_year = build_ibtracs_p_lookup(IBTrACS, nb_synth_tracks=30, seed=42)

# for replace_prob in REPLACE_PROBS:
#     for year in range(1980, 2020):
#         for seed in range(1, 201):
#             generate_counterfactual_year(
#                 year, IBTrACS, sid_to_synth, sid_to_year, cent,
#                 reg=reg, replace_prob=replace_prob, seed=seed,
#                 out_dir=COUNTERFACTUAL_DIR / "IBTrACS_p"
#             )


# #-------------------------------AI counterfactuals--------------------------------------------------
# def build_ai_lookup(tracks_hist, ai_folder, max_dist_deg=8.0):
#     """
#     Load WeatherNext (FNv3.2) AI-ensemble counterfactual tracks for every
#     storm in `tracks_hist` that has a matching ensemble file on disk, ONCE,
#     and return lookups to repeatedly sample from -- same shape as
#     build_ibtracs_p_lookup(), just sourced from AI ensemble files instead
#     of calc_perturbed_trajectories().

#     Only storms with an existing {sid}.csv in ai_folder contribute -- since
#     WeatherNext only has ensembles for post-2022 AU storms, this naturally
#     restricts things to the right subset without needing to pre-filter
#     tracks_hist by basin/year yourself.

#     Returns:
#         sid_to_synth: dict {sid: [ai_track_1, ..., ai_track_N]}
#         sid_to_year:  dict {sid: season}, from the IBTrACS SEASON of that storm
#     """
#     sid_to_synth = {}
#     sid_to_year = {}

#     n_checked = 0
#     for tr in tracks_hist.data:
#         sid = tr.sid
#         sid_to_year[sid] = get_season(tr)

#         ensemble_filepath = os.path.join(ai_folder, f"{sid}.csv")
#         if not os.path.exists(ensemble_filepath):
#             continue

#         n_checked += 1
#         # minimal "event" record read_simulations_AI expects, built
#         # straight from the CLIMADA track itself -- no raw ibtracs csv needed
#         event = pd.Series({
#             "SID": sid,
#             "LAT": float(tr.lat.values[0]),
#             "LON": float(tr.lon.values[0]),
#             "ISO_TIME": pd.Timestamp(tr.time.values[0]),
#         })

#         matched_tracks = read_simulations_AI(
#             path=ensemble_filepath,
#             IBTrACS_event=event,
#             max_dist_deg=max_dist_deg,
#         )

#         if matched_tracks:
#             sid_to_synth[sid] = matched_tracks

#     print(f"Found {n_checked} storms with a WeatherNext ensemble file, "
#           f"{len(sid_to_synth)} with at least one matched track "
#           f"(of {len(tracks_hist.data)} total historical storms)")

#     return sid_to_synth, sid_to_year


# WN_folder = r"..\..\TC_datasets\MiniTC-test-QM"

# sid_to_synth_ai, sid_to_year_ai = build_ai_lookup(IBTrACS, WN_folder, max_dist_deg=8.0)

# for replace_prob in REPLACE_PROBS:
#     for year in range(1980, 2020):          # seasons 2004-2021
#         for seed in range(1, 201):
#             generate_counterfactual_year(
#                 year, IBTrACS, sid_to_synth_ai, sid_to_year_ai, cent,
#                 reg=reg, model_name='Mini-WNC', replace_prob=replace_prob, seed=seed,
#                 out_dir=COUNTERFACTUAL_DIR / "MiniTC-test-QM"
#             )
