"""
description: Load TC tracks from the CHAZ model and calculate the 2D windfield
            after Holland (2008), writing one hazard hdf5 file per (file,
            ensemble) combination -- i.e. one file per synthetic year --
            instead of one combined file.

            Each CHAZ file/ensemble combination (2017_XXX.nc, ensemble e) is
            already an independent draw on its own -- unlike STORM, there's
            no need to parse a real year out of anything: calling
            TCTracks.from_simulations_chaz(fname, ensemble_nums=[e]) for one
            ensemble from one file *is* one synthetic year. 100 files x 40
            ensembles = 4000 synthetic years total.

            CHAZ's tracks are global, so after computing the windfield, each
            year's hazard is filtered down to the Southern Hemisphere ('SH')
            basin using the same basin_split_haz() region-id approach as
            Hazard_IBTRACS.py, before being written out. (BASIN_BOUNDS,
            reg_id and basin_split_haz live in helper.py.)
"""

from pathlib import Path

# import CLIMADA modules:
from climada.hazard import TCTracks, TropCyclone
from helper import get_AUS_centroids, basin_split_haz
import numpy as np  


def drop_nan_tracks(tc_tracks):
    """Remove any track with missing lat/lon values, in place."""
    tc_tracks.data = [
        tr for tr in tc_tracks.data
        if not (np.isnan(tr.lat.values).any() or np.isnan(tr.lon.values).any())
    ]
    return tc_tracks

cent = get_AUS_centroids()

CHAZ_DIR = Path("../TC_datasets/CHAZ")
HAZARD_DIR = Path("../hazard/CHAZ") 
HAZARD_DIR.mkdir(parents=True, exist_ok=True)

reg = 'SH'
N_FILES = 250        # 2017_001.nc ... 2017_250.nc
N_ENSEMBLES = 40     # ensemble members per file


def init_tc_hazard_per_year(cent, reg, chaz_dir=CHAZ_DIR, n_files=N_FILES,
                             n_ensembles=N_ENSEMBLES, load_haz=False):
    """
    Build one TC hazard file per (file, ensemble) combination -- i.e. per
    synthetic year -- writing each to
    HAZARD_DIR/TC_{reg}_025res_CHAZ_year{global_year:04d}.hdf5.

    global_year = (i_file - 1) * n_ensembles + (ens - 1), so files stay in
    a stable 0 .. n_files*n_ensembles-1 order matching Hazard_STORM.py's
    per-year naming convention.

    Combinations with zero tracks (missing file, or an ensemble with no
    storms) get no file written -- treat a missing file as a zero-loss
    year downstream. If load_haz=True, years whose file already exists on
    disk are skipped (safe to resume an interrupted run).

    Returns a dict {global_year: hazard_path} for every year that has a
    file on disk after this call.
    """
    n_total = n_files * n_ensembles
    written = {}

    print("--- Starting CHAZ Tracks Processing ---")
    for i_file in range(1, n_files + 1):
        print(f"Processing file {i_file:03d}/{n_files}:")
        fname = chaz_dir.joinpath(f"2017_{i_file:03d}.nc")
        if not fname.exists():
            print(f"Warning: File {fname} not found. Skipping...")
            continue

        print(f"Loading tracks from: {fname.name}")
        for ens in range(1, n_ensembles + 1):
            print(f"Processing ensemble {ens:02d}/40:")
            global_year = (i_file - 1) * n_ensembles + (ens - 1)
            haz_str = f"TC_{reg}_025res_CHAZ_year{global_year:04d}.hdf5"
            hazard_path = HAZARD_DIR.joinpath(haz_str)

            if load_haz and hazard_path.is_file():
                written[global_year] = hazard_path
                continue

            try:
                tracks_i = TCTracks.from_simulations_chaz(fname, ensemble_nums=[ens])
            except Exception as e:
                print(f"Error loading {fname.name} ensemble {ens}: {e}")
                continue

            if not tracks_i.data:
                continue 

            tracks_i = drop_nan_tracks(tracks_i)   
            if not tracks_i.data:
                continue

            tc_hazard = TropCyclone.from_tracks(tracks_i, centroids=cent)
            tc_hazard = basin_split_haz(tc_hazard, reg)

            tc_hazard.check()
            tc_hazard.write_hdf5(hazard_path)
            written[global_year] = hazard_path

            if (global_year + 1) % 500 == 0 or global_year == n_total - 1:
                print(f"[{reg}] processed {global_year + 1}/{n_total} synthetic years "
                      f"({len(written)} files written)")

    return written


############################ Call functions ##################################
year_files = init_tc_hazard_per_year(cent, reg, load_haz=False)
