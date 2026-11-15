"""
description: Load TC tracks from the IRIS model (Imperial College Storm Model;
            Sparks & Toumi, 2024, Sci. Data) and calculate the 2D windfield
            after Holland (2008), writing one hazard hdf5 file per synthetic
            year instead of one combined file.

            Each of the 100 realisation files (IRIS_{SI,SP}_100Y_n{0..99}.txt)
            spans 100 years, indicated by column 2 ("year") of the raw txt
            file: #tcid basin year tc month timestep lon lat vmax pmin rmw r18.
            SI and SP realisations with the same index n run in parallel (not
            sequential), so realisation n's year y in SI and realisation n's
            year y in SP are the SAME synthetic year -- their storms get
            combined into one hazard file, not treated as two separate years.

            global_year = realisation_idx * 100 + local_year, giving up to
            100*100 = 10,000 distinct synthetic years total. As with
            CHAZ/STORM, years with zero storms in either basin simply get no
            file written -- treat a missing file as a zero-loss year
            downstream.

            iris_parse.read_simulations_iris() does not preserve IRIS's own
            per-storm year (same issue as STORM) -- the real year only
            exists in the raw txt file, which is read directly here and
            matched (in file order, by unique tcid) to the tracks
            read_simulations_iris() returns.

            Processing one realisation (and within it, one year) at a time
            is what keeps this memory-safe -- at most one realisation file's
            tracks are held at once, and the actual windfield computation
            only ever runs over a single year's handful of storms before
            being written to disk and discarded, rather than building one
            combined hazard over all 10,000 years' worth of tracks at once.
"""

import pandas as pd
from pathlib import Path

from climada.hazard import TCTracks, TropCyclone
from helper import get_AUS_centroids, read_simulations_iris

cent = get_AUS_centroids()

IRIS_DIR = Path("../TC_datasets/IRIS")
HAZARD_DIR = Path("../hazard/IRIS")  # one file per synthetic year lives here
HAZARD_DIR.mkdir(parents=True, exist_ok=True)

reg = 'SH'  # combined South Indian + South Pacific, same convention as STORM/CHAZ
BASIN_S = ['SI', 'SP']
N_REALISATIONS = 100      # IRIS_{basin}_100Y_n0.txt ... n99.txt
YEARS_PER_REALISATION = 100

IRIS_COLUMNS = ['tcid', 'basin', 'year', 'tc', 'month', 'timestep',
                'lon', 'lat', 'vmax', 'pmin', 'rmw', 'r18']


def _get_storm_years(path):
    """Read the raw IRIS txt file directly and return, in file order, the
    `year` (0-99) of each unique storm (identified by `tcid`) -- one entry
    per storm, matching the order read_simulations_iris() returns tracks in.
    """
    raw = pd.read_csv(path, sep=r"\s+", comment='#', names=IRIS_COLUMNS)
    storm_years = (
        raw.drop_duplicates(subset=['tcid'], keep='first')['year']
        .astype(int)
        .tolist()
    )
    return storm_years


def _load_tagged_tracks(path):
    """Load one IRIS realisation file's tracks, tagging each with its
    GLOBAL year (already absolute in the raw file -- e.g. file n3 spans
    years 300-399, not 0-99).
    """
    tracks = read_simulations_iris(path)
    if not tracks:
        return []

    storm_years = _get_storm_years(path)
    if len(storm_years) != len(tracks):
        raise ValueError(
            f"{path}: found {len(storm_years)} unique tcid storms in the raw "
            f"file but read_simulations_iris() returned {len(tracks)} tracks "
            "-- order/column assumptions don't hold for this file, check "
            "before proceeding."
        )

    for tr, global_year in zip(tracks, storm_years):
        tr.attrs["global_year"] = global_year
    return tracks


def init_tc_hazard_per_year(cent, reg, basin=BASIN_S, iris_dir=IRIS_DIR,
                             n_realisations=N_REALISATIONS,
                             years_per_realisation=YEARS_PER_REALISATION,
                             load_haz=False):
    written = {}

    print("--- Starting IRIS Tracks Processing ---")
    for realisation_idx in range(n_realisations):
        expected_range = range(
            realisation_idx * years_per_realisation,
            (realisation_idx + 1) * years_per_realisation,
        )

        year_groups = {}
        for j in basin:
            fname = iris_dir.joinpath(f"IRIS_{j}_100Y_n{realisation_idx}.txt")
            if not fname.exists():
                print(f"Warning: File {fname} not found. Skipping...")
                continue

            tracks = _load_tagged_tracks(fname)
            for tr in tracks:
                gy = tr.attrs["global_year"]
                if gy not in expected_range:
                    print(f"WARNING: {fname} produced global_year={gy}, "
                          f"outside expected range {expected_range.start}-"
                          f"{expected_range.stop - 1} -- check the year "
                          "offset assumption for this file.")
                year_groups.setdefault(gy, []).append(tr)

        print(f"Loaded realisation n{realisation_idx}: "
              f"{sum(len(v) for v in year_groups.values())} tracks across "
              f"{len(year_groups)}/{years_per_realisation} years")

        for global_year, year_tracks in sorted(year_groups.items()):
            print(f"Processing year {global_year}.")
            haz_str = f"TC_{reg}_025res_IRIS_year{global_year:05d}.hdf5"
            hazard_path = HAZARD_DIR.joinpath(haz_str)

            if load_haz and hazard_path.is_file():
                written[global_year] = hazard_path
                continue

            batch = TCTracks()
            batch.data = year_tracks
            batch.equal_timestep(time_step_h=1.)

            tc_hazard = TropCyclone.from_tracks(batch, centroids=cent)
            tc_hazard.check()
            tc_hazard.write_hdf5(hazard_path)
            written[global_year] = hazard_path

        if (realisation_idx + 1) % 10 == 0 or realisation_idx == n_realisations - 1:
            print(f"[{reg}] processed {realisation_idx + 1}/{n_realisations} "
                  f"realisations ({len(written)} year-files written so far)")

    return written


############################ Call functions ##################################
year_files = init_tc_hazard_per_year(cent, reg)
