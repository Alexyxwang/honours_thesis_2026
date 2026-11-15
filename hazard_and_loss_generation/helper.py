import os
import re
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

from climada.hazard import Centroids, TropCyclone


def _tc_category(vmax_ms):
    """Saffir-Simpson category (int, -1 = tropical depression) from peak 1-min vmax in m/s."""
    vmax_kn = np.max(vmax_ms) * MS_TO_KN
    return int(np.argmax(vmax_kn < np.array(SAFFIR_SIM_CAT_KN)) - 1)


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------IRIS data parsing-------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

IRIS_COLUMNS = ["tcid", "basin", "year", "tc", "month", "timestep",
                "lon", "lat", "vmax", "pmin", "rmw", "r18"]
IRIS_BASIN_ENV_PRESSURE = {"SI": 1005, "SP": 1004}  # same basin defaults CLIMADA uses for STORM
KM_TO_NMI = 0.5399568   # km -> nautical miles, for radius_max_wind
KN_TO_MS = 0.514444
MS_TO_KN = 1.9438445    # m/s -> knots, only used to classify Saffir-Simpson category
SAFFIR_SIM_CAT_KN = [34, 64, 83, 96, 113, 137, 1000]


def read_simulations_iris(path):
    """Read one IRIS realization txt file into a list of xr.Dataset tracks (CLIMADA TCTracks format).

    Columns: tcid basin year tc month timestep lon lat vmax pmin rmw r18
      - tcid    : storm ID, unique within this file
      - basin   : 2-letter basin code (already 'SI'/'SP' -- no numeric decoding needed, unlike STORM)
      - year    : synthetic sample-year within this 100-year realization (0-99, not a calendar year)
      - month   : genesis month of the storm (constant per track)
      - timestep: row index within the track; IRIS data is sampled at fixed 3h intervals
      - vmax    : 1-minute sustained wind, m/s (already 1-min -- no STORM-style 0.88 conversion)
      - pmin    : central pressure, hPa
      - rmw     : radius of maximum wind, km
      - r18     : radius of 18 m/s (gale-force) wind, km -- kept in the raw dataframe only;
                  STORM's tracks don't carry an equivalent "radius_oci" either, so this is not
                  fed into TropCyclone.from_tracks, keeping IRIS consistent with how STORM is
                  already handled in this notebook.

    Returns
    -------
    list of xr.Dataset, one per storm track.
    """
    df = pd.read_csv(
        path, sep=r"\s+", comment="#", names=IRIS_COLUMNS,
        dtype={"tcid": int, "year": int, "tc": int, "month": int, "timestep": int},
    )
    df["rmw_nmi"] = df["rmw"] * KM_TO_NMI

    fname = os.path.basename(path)
    m = re.search(r"_n(\d+)\.txt$", fname)
    realization = int(m.group(1)) if m else 0

    tracks = []
    for tcid, group in df.groupby("tcid", sort=True):
        basin = group["basin"].iloc[0]
        month = int(group["month"].iloc[0])
        # Synthetic tracks have no real calendar year (like STORM, fixed at 1980); only the
        # genesis month and 3h spacing matter for equal_timestep()/TropCyclone.from_tracks().
        time = [dt.datetime(1980, month, 1) + dt.timedelta(hours=3 * int(ts))
                for ts in group["timestep"]]
        env_pressure = np.full(len(group), IRIS_BASIN_ENV_PRESSURE.get(basin, 1010))
        tracks.append(
            xr.Dataset(
                {
                    "time_step": ("time", np.full(len(group), 3.0)),
                    "max_sustained_wind": ("time", group["vmax"].values),
                    "central_pressure": ("time", group["pmin"].values),
                    "radius_max_wind": ("time", group["rmw_nmi"].values),
                    "environmental_pressure": ("time", env_pressure),
                    "basin": ("time", np.full(len(group), basin, dtype="<U2")),
                },
                coords={
                    "time": ("time", time),
                    "lat": ("time", group["lat"].values),
                    "lon": ("time", group["lon"].values),
                },
                attrs={
                    "max_sustained_wind_unit": "m/s",
                    "central_pressure_unit": "mb",
                    "name": f"{fname}-{tcid}",
                    "sid": f"{fname}-{tcid}",
                    "orig_event_flag": True,
                    "data_provider": "IRIS",
                    "id_no": realization * 100_000 + tcid,
                    "category": _tc_category(group["vmax"].values),
                },
            )
        )
    return tracks



#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------WeatherNext data parsing------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

WN_C_COLUMNS = [
    "init_time",
    "track_id",
    "sample",
    "valid_time",
    "lead_time",
    "lead_time_hours",
    "lat",
    "lon",
    "minimum_sea_level_pressure_hpa",
    "maximum_sustained_wind_speed_knots",
    "radius_of_maximum_winds_km",
    "radius_34_knot_winds_ne_km",
    "radius_34_knot_winds_se_km",
    "radius_34_knot_winds_sw_km",
    "radius_34_knot_winds_nw_km",
    "radius_50_knot_winds_ne_km",
    "radius_50_knot_winds_se_km",
    "radius_50_knot_winds_sw_km",
    "radius_50_knot_winds_nw_km",
    "radius_64_knot_winds_ne_km",
    "radius_64_knot_winds_se_km",
    "radius_64_knot_winds_sw_km",
    "radius_64_knot_winds_nw_km",
]

IRIS_BASIN_ENV_PRESSURE = {"SI": 1005, "SP": 1004}  # same basin defaults CLIMADA uses for STORM

def read_simulations_AI(path, IBTrACS_event, max_dist_deg=8.0):
    """
    Reads WeatherNext/AI TC ensemble tracks, matches each ensemble sample's track 
    to the IBTrACS initial location (handling coordinate wrap-around), and converts 
    matching tracks to CLIMADA-compliant xarray Datasets.
    """
    dtypes = {
        "track_id": "str",
        "sample": "int64",
        "lead_time": "str",
        "lead_time_hours": "int64",
        "lat": "float64",
        "lon": "float64",
        "minimum_sea_level_pressure_hpa": "float64",
        "maximum_sustained_wind_speed_knots": "float64",
        "radius_of_maximum_winds_km": "float64",
    }

    df = pd.read_csv(path, comment="#", dtype=dtypes, parse_dates=["init_time", "valid_time"])
    # raw WN-C reforecast output names the ensemble member column "ensemble_member"
    if "sample" not in df.columns and "ensemble_member" in df.columns:
        df = df.rename(columns={"ensemble_member": "sample"})

    # Ensure lead time in hours exists
    if "lead_time_hours" not in df.columns:
        if "lead_time" in df.columns:
            df["lead_time_hours"] = (pd.to_timedelta(df["lead_time"]).dt.total_seconds() / 3600.0).astype("int64")
        elif "valid_time" in df.columns and "init_time" in df.columns:
            df["lead_time_hours"] = ((df["valid_time"] - df["init_time"]).dt.total_seconds() / 3600.0).astype("int64")
        else:
            raise KeyError(f"File '{path}' missing lead time fields.")

    # Standardize column names
    df = df.rename(columns={
        "track_id": "tcid",
        "minimum_sea_level_pressure_hpa": "pmin",
        "maximum_sustained_wind_speed_knots": "vmax_kn",
        "radius_of_maximum_winds_km": "rmw_km",
        "lead_time_hours": "timestep_h"
    })

    # Standardize longitudes to [-180, 180] range
    df["lon"] = np.where(df["lon"] > 180, df["lon"] - 360, df["lon"])

    # Conversions for CLIMADA
    df["vmax"] = df["vmax_kn"] * KN_TO_MS
    df["rmw_nmi"] = df["rmw_km"] * KM_TO_NMI

    # Target metadata from IBTrACS reference
    ib_lat = float(IBTrACS_event.get("LAT", IBTrACS_event.get("lat")))
    ib_lon = float(IBTrACS_event.get("LON", IBTrACS_event.get("lon")))
    if ib_lon > 180:
        ib_lon -= 360  # Ensure IBTrACS longitude matches [-180, 180] convention

    basin = str(IBTrACS_event.get("BASIN", IBTrACS_event.get("basin", "SI")))
    ref_sid = str(IBTrACS_event.get("SID", "UNKNOWN"))

    tracks = []
    total_samples = df["sample"].nunique()

    # Match closest track per ensemble sample
    for sample_id, sample_group in df.groupby("sample", sort=True):
        # Initial points per track
        earliest_idx = sample_group.groupby("tcid")["timestep_h"].idxmin()
        initial_points = sample_group.loc[earliest_idx].copy()

        # Wrap-aware distance calculation
        dlat = initial_points["lat"] - ib_lat
        dlon = (initial_points["lon"] - ib_lon + 180) % 360 - 180
        initial_points["dist_deg"] = np.sqrt(dlat**2 + dlon**2)

        # Select closest track in sample
        closest_row = initial_points.loc[initial_points["dist_deg"].idxmin()]
        closest_tcid = closest_row["tcid"]
        min_dist = closest_row["dist_deg"]

        if min_dist > max_dist_deg:
            print(f"Sample {sample_id}: Track {closest_tcid[:8]} exceeds max distance ({min_dist:.2f}° > {max_dist_deg}°). Skipped.")
            continue

        # Build track dataset
        tc_track_df = sample_group[sample_group["tcid"] == closest_tcid].sort_values("timestep_h").reset_index(drop=True)

        MIN_TRACK_POINTS = 4  # cubic-spline interpolation in equal_timestep() needs at least this many
        if len(tc_track_df) < MIN_TRACK_POINTS:
            print(f"Sample {sample_id}: Track {closest_tcid[:8]} has only {len(tc_track_df)} "
                f"timestep(s) (< {MIN_TRACK_POINTS}), too short to interpolate. Skipped.")
            continue

        env_p = IRIS_BASIN_ENV_PRESSURE.get(basin, 1013.25)
        env_pressure = np.full(len(tc_track_df), env_p)

        time_steps = tc_track_df["valid_time"].diff().dt.total_seconds().fillna(0) / 3600.0
        time_steps.iloc[0] = time_steps.iloc[1] if len(time_steps) > 1 else 6.0

        track_ds = xr.Dataset(
            {
                "time_step": ("time", time_steps.values),
                "max_sustained_wind": ("time", tc_track_df["vmax"].values),
                "central_pressure": ("time", tc_track_df["pmin"].values),
                "radius_max_wind": ("time", tc_track_df["rmw_nmi"].values),
                "environmental_pressure": ("time", env_pressure),
                "basin": ("time", np.full(len(tc_track_df), basin, dtype="<U2")),
            },
            coords={
                "time": ("time", tc_track_df["valid_time"].values),
                "lat": ("time", tc_track_df["lat"].values),
                "lon": ("time", tc_track_df["lon"].values),
            },
            attrs={
                "max_sustained_wind_unit": "m/s",
                "central_pressure_unit": "mb",
                "name": f"{ref_sid}_S{sample_id}_{closest_tcid[:8]}",
                "sid": f"{ref_sid}_S{sample_id}",
                "orig_event_flag": True,
                "data_provider": "WeatherNext_AI",
                "id_no": int(sample_id),
                "category": _tc_category(tc_track_df["vmax"].values),
            },
        )
        tracks.append(track_ds)

    print(f"Successfully loaded {len(tracks)} matching tracks out of {total_samples} samples.")
    
    return tracks 


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------Basin separation-------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


BASIN_BOUNDS = {
    # North Atlantic/Eastern Pacific Basin
    'AP': [-180.0, 0.0, 0.0, 65.0],

    # Indian Ocean Basin
    'IO': [30.0, 100.0, 0.0, 40.0],

    # Southern Hemisphere Basin
    'SH': [-180.0, 180.0, -60.0, 0.0],

    # Western Pacific Basin
    'WP': [100.0, 180.0, 0.0, 65.0],
}

reg_id = {'AP': 5000, 'IO': 5001, 'SH': 5002, 'WP': 5003}

def basin_split_haz(hazard, basin):
    """ Split CHAZ global hazard up into ocean basins of choice """
    # get basin bounds
    x_min, x_max, y_min, y_max = BASIN_BOUNDS[str(basin)]
    basin_idx = (hazard.centroids.lat > y_min) & (
                 hazard.centroids.lat < y_max) & (
                 hazard.centroids.lon > x_min) & (
                 hazard.centroids.lon < x_max)

    # region_id is a read-only property backed by the centroids'
    # GeoDataFrame -- set the underlying column instead of the property
    if 'region_id' not in hazard.centroids.gdf.columns:
        hazard.centroids.gdf['region_id'] = 0

    hazard.centroids.gdf.loc[basin_idx, 'region_id'] = reg_id[basin]
    tc_haz_split = hazard.select(reg_id=reg_id[basin])
    return tc_haz_split

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------get_centroids-----------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def get_AUS_centroids(aus_bounds=(110, -45, 155, -10), res=0.25):
    return Centroids.from_pnt_bounds(aus_bounds, res=0.25)
