"""
description: Load TC tracks from the STORM model and calculate the 2D windfield
            after Holland (2008), writing one hazard hdf5 file per synthetic
            year instead of one combined file.

            STORM's 6 loop files each span 1000 independent synthetic years
            (years 0-999 within that loop), so combined across SI+SP and all
            6 loops, there are 6*1000 = 6000 independent synthetic years
            total. Each track is tagged with a GLOBAL year id
            (loop_idx*1000 + local_year) so tracks belonging to the same
            synthetic year -- possibly several separate named storms -- end
            up in the same output file, and every year gets its own
            TC_{reg}_025res_STORM_year{global_year:04d}.hdf5. That lets a
            downstream analysis treat each file as one independent "draw"
            and build an aggregate (not just occurrence) exceedance curve
            by summing losses within a year and comparing across years,
            without needing to fake any dates.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# import CLIMADA modules:
from climada.hazard import Centroids, TCTracks, TropCyclone
from helper import get_AUS_centroids

cent = get_AUS_centroids()

STORM_DIR = Path("../TC_datasets/STORM")
HAZARD_DIR = Path("../hazard/STORM")  # one file per synthetic year lives here
HAZARD_DIR.mkdir(parents=True, exist_ok=True)

reg = 'SH'
BASIN_S = ['SI', 'SP']
YEARS_PER_LOOP = 1000
N_LOOPS = 10


def init_STORM_tracks(basin):
    """ Load all STORM tracks for the basin(s) of interest, tagging each
    track with a GLOBAL year id (0 .. N_LOOPS*YEARS_PER_LOOP - 1).

    CLIMADA's TCTracks.from_simulations_storm() does NOT preserve STORM's
    own per-storm year -- every track it returns is dated to 1980. The
    real synthetic year (0-999 within a loop) only exists in the raw
    STORM txt file itself, as column 0. We read that directly and match
    it, in file order, to the tracks CLIMADA produces (one row per unique
    (year, tc_number) storm -- same order the tracks come out in).
    """
    all_tracks = []
    for j in basin:
        fname = lambda i: f"STORM_DATA_IBTRACS_{j}_1000_YEARS_{i}.txt"
        for loop_idx in range(N_LOOPS):
            print(f"{j}: Initiating STORM {loop_idx}/{N_LOOPS}")
            path = os.path.join(STORM_DIR, fname(loop_idx))
            tracks_loop = TCTracks.from_simulations_storm(path)

            if not tracks_loop.data:
                continue

            # STORM txt columns: YEAR, MONTH, TC_NUM, TIME_STEP, BASIN_ID,
            # LAT, LON, PRESSURE, WIND, RMW, CATEGORY, LANDFALL, DIST2LAND
            raw = pd.read_csv(path, header=None)
            storm_years = (
                raw.drop_duplicates(subset=[0, 2], keep="first")[0]
                .astype(int)
                .tolist()
            )

            if len(storm_years) != len(tracks_loop.data):
                raise ValueError(
                    f"{path}: found {len(storm_years)} unique (year, tc_number) "
                    f"storms in the raw file but CLIMADA returned "
                    f"{len(tracks_loop.data)} tracks -- order/column assumptions "
                    "below don't hold for this file, check before proceeding."
                )

            for tr, local_year in zip(tracks_loop.data, storm_years):
                tr.attrs["global_year"] = loop_idx * YEARS_PER_LOOP + local_year
                all_tracks.append(tr)

    tracks_STORM = TCTracks()
    tracks_STORM.data = all_tracks
    # tracks_STORM.equal_timestep(time_step_h=1.)
    return tracks_STORM


def init_tc_hazard_per_year(tracks, cent, reg, n_loops=N_LOOPS,
                             years_per_loop=YEARS_PER_LOOP, load_haz=False):
    """
    Build one TC hazard file per synthetic year, writing each to
    HAZARD_DIR/TC_{reg}_025res_STORM_year{global_year:04d}.hdf5.

    Years with zero storms get no file written at all -- treat a missing
    file as a zero-loss year downstream. If load_haz=True, years whose
    file already exists on disk are skipped (safe to resume an interrupted
    run).

    Returns a dict {global_year: hazard_path} for every year that has a
    file on disk after this call.
    """
    year_groups = {}
    for tr in tracks.data:
        year_groups.setdefault(tr.attrs["global_year"], []).append(tr)

    all_global_years = range(n_loops * years_per_loop)
    n_total = len(all_global_years)
    n_with_storms = len(year_groups)
    print(f"[{reg}] {n_with_storms}/{n_total} synthetic years contain at least one storm")

    written = {}
    for k, global_year in enumerate(all_global_years, start=1):
        print(f"Processing STORM year {global_year}/{n_total}")
        year_tracks = year_groups.get(global_year)
        if not year_tracks:
            continue  # no storms this synthetic year -- no file written

        haz_str = f"TC_{reg}_025res_STORM_year{global_year:04d}.hdf5"
        hazard_path = HAZARD_DIR.joinpath(haz_str)

        if load_haz and hazard_path.is_file():
            written[global_year] = hazard_path
            continue

        batch = TCTracks()
        batch.data = year_tracks
        tc_hazard = TropCyclone.from_tracks(batch, centroids=cent)
        tc_hazard.frequency = 1/(n_loops * years_per_loop)
        tc_hazard.check()
        tc_hazard.write_hdf5(hazard_path)
        written[global_year] = hazard_path

        if k % 500 == 0 or k == n_total:
            print(f"[{reg}] processed {k}/{n_total} synthetic years ({len(written)} files written)")

    return written


############################ Call functions ##################################
tracks_STORM = init_STORM_tracks(basin=BASIN_S)
year_files = init_tc_hazard_per_year(tracks_STORM, cent, reg, load_haz=False)
