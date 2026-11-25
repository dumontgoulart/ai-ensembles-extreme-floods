#!/usr/bin/env python3
"""
Preprocess all downloaded datasets for precipitation forecast comparison.

Event and spatial parameters are read from event_config.py.
Change ACTIVE_EVENT there to switch between configured events.

Steps:
1. Load GRIB files (IFS, AIFS) and NetCDF files (ERA5, E-OBS)
2. De-accumulate precipitation using .diff("step") for forecast data
3. Convert units from metres to millimetres (×1000)
4. Regrid all datasets to a common 0.25° grid
5. Align valid times (init_date + step for forecasts; timestamps for reanalysis)
6. Save processed files to data/processed/{slug}/
"""

import os
import sys
import glob
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from event_config import cfg, init_pairs

import eccodes
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration (derived from event_config)
# =============================================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(PROJECT_DIR, "data", "raw",       cfg["slug"])
PROC_DIR = os.path.join(PROJECT_DIR, "data", "processed", cfg["slug"])

# Derive (date_compact, time_label) pairs from cfg["init_dates"]
INIT_DATE_TIMES = [
    (d.replace("-", ""), hh)
    for d, hh in init_pairs(cfg)
]

EVENT_START = cfg["ref_start"]
EVENT_END   = cfg["ref_end"]

LAT_MIN, LAT_MAX = cfg["lat_min"], cfg["lat_max"]
LON_MIN, LON_MAX = cfg["lon_min"], cfg["lon_max"]


# =============================================================================
# Helper functions
# =============================================================================
def regrid_to_common(ds, var_name, target_lat, target_lon):
    """Regrid a dataset to the common lat/lon grid using linear interpolation."""
    da = ds[var_name] if isinstance(ds, xr.Dataset) else ds

    lat_name = None
    lon_name = None
    for name in da.dims:
        if name in ("latitude", "lat"):
            lat_name = name
        elif name in ("longitude", "lon"):
            lon_name = name

    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot identify lat/lon dims in {list(da.dims)}")

    da_regridded = da.interp(
        {lat_name: target_lat, lon_name: target_lon},
        method="linear",
    )

    if lat_name != "latitude":
        da_regridded = da_regridded.rename({lat_name: "latitude"})
    if lon_name != "longitude":
        da_regridded = da_regridded.rename({lon_name: "longitude"})

    return da_regridded.to_dataset(name=var_name)


def _grib_to_xarray_cfgrib(path, member_number):
    """Load a single GRIB file via cfgrib and ensure a 'number' dim."""
    ds = xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    if "number" not in ds.dims:
        ds = ds.expand_dims("number").assign_coords(number=[member_number])
    return ds


def _grib_to_xarray_eccodes(path):
    """Load a GRIB file via eccodes.

    Used for AIFS files where cfgrib raises a TypeError on encoding.
    Handles:
    - 0-360 longitude convention -> normalized to -180..180
    - Multiple members (pf) or single member (cf) in one file
    - Accumulated TP steps encoded as '0-6', '0-12', etc.

    Returns
    -------
    xr.Dataset with dims (number, step, latitude, longitude) and 'tp' in original units.
    """
    msgs_by_member_step = {}  # {(member, step_end_hours): 2D np.array}
    lat_vals = None
    lon_vals = None
    Ni = Nj = None

    with open(path, "rb") as f:
        while True:
            msg = eccodes.codes_grib_new_from_file(f)
            if msg is None:
                break
            try:
                step_range = str(eccodes.codes_get(msg, "stepRange"))
                step_end = int(step_range.split("-")[-1])

                try:
                    member = int(eccodes.codes_get(msg, "number"))
                except Exception:
                    member = 0

                if lat_vals is None:
                    Ni = eccodes.codes_get(msg, "Ni")
                    Nj = eccodes.codes_get(msg, "Nj")
                    raw_lats = eccodes.codes_get_array(msg, "latitudes")
                    raw_lons = eccodes.codes_get_array(msg, "longitudes")
                    lat_vals = np.sort(np.unique(raw_lats))[::-1]
                    lons_norm = np.where(raw_lons > 180, raw_lons - 360, raw_lons)
                    lon_vals = np.sort(np.unique(lons_norm))

                values = eccodes.codes_get_array(msg, "values")
                values_2d = values.reshape(Nj, Ni)
                lons_flat = lons_norm[::Nj] if len(lons_norm) == Ni * Nj else np.sort(np.unique(lons_norm))
                lon_order = np.argsort(np.unique(np.where(raw_lons > 180, raw_lons - 360, raw_lons)))
                values_2d = values_2d[:, lon_order]

                msgs_by_member_step[(member, step_end)] = values_2d
            finally:
                eccodes.codes_release(msg)

    members = sorted(set(k[0] for k in msgs_by_member_step))
    steps_h = sorted(set(k[1] for k in msgs_by_member_step))

    data = np.full(
        (len(members), len(steps_h), len(lat_vals), len(lon_vals)),
        np.nan, dtype=np.float32,
    )
    for mi, m in enumerate(members):
        for si, s in enumerate(steps_h):
            if (m, s) in msgs_by_member_step:
                data[mi, si] = msgs_by_member_step[(m, s)]

    step_td = np.array([np.timedelta64(int(s), "h") for s in steps_h])

    return xr.Dataset(
        {"tp": (["number", "step", "latitude", "longitude"], data)},
        coords={
            "number": members,
            "step": step_td,
            "latitude": lat_vals,
            "longitude": lon_vals,
        },
    )


def load_grib_forecast(raw_dir, model_name, init_date_times):
    """Load and merge GRIB forecast files for all init dates and times."""
    use_eccodes = (model_name == "aifs")
    all_data = []

    for date_str, time_label in init_date_times:
        cf_path = os.path.join(raw_dir, f"{model_name}_enfo_cf_tp_{date_str}_{time_label}.grib")
        pf_path = os.path.join(raw_dir, f"{model_name}_enfo_pf_tp_{date_str}_{time_label}.grib")
        # Perturbed members may be split across files: pf (1-10) and pf11-50
        # (complementary members for the full 51-member ensemble).
        pf_paths = sorted(glob.glob(os.path.join(
            raw_dir, f"{model_name}_enfo_pf*_tp_{date_str}_{time_label}.grib")))

        if not os.path.exists(cf_path) or not os.path.exists(pf_path):
            print(f"  [WARNING] Missing files for {model_name} {date_str} {time_label}UTC, skipping")
            continue

        try:
            if use_eccodes:
                ds_cf = _grib_to_xarray_eccodes(cf_path)
                ds_pfs = [_grib_to_xarray_eccodes(p) for p in pf_paths]
            else:
                ds_cf = _grib_to_xarray_cfgrib(cf_path, member_number=0)
                ds_pfs = [_grib_to_xarray_cfgrib(p, member_number=None) for p in pf_paths]
        except Exception as e:
            print(f"  [WARNING] Failed to load {model_name} {date_str} {time_label}UTC: {e}")
            continue

        ds_combined = xr.concat([ds_cf] + ds_pfs, dim="number")
        # Guard against a member appearing in two files.
        _, first = np.unique(ds_combined["number"].values, return_index=True)
        ds_combined = ds_combined.isel(number=np.sort(first)).sortby("number")
        print(f"  Loaded {model_name} {date_str} {time_label}UTC: "
              f"{ds_combined.sizes['number']} members from {1 + len(pf_paths)} files")

        init_hour = int(time_label)
        init_dt = np.datetime64(
            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{init_hour:02d}:00"
        )
        ds_combined = ds_combined.expand_dims("init_time").assign_coords(
            init_time=[init_dt]
        )

        all_data.append(ds_combined)

    if not all_data:
        raise FileNotFoundError(f"No {model_name} forecast files found in {raw_dir}")

    return xr.concat(all_data, dim="init_time")


def deaccumulate_precipitation(ds, var_name="tp"):
    """De-accumulate precipitation along the step dimension using .diff()."""
    print("  De-accumulating precipitation with .diff('step')...")
    da = ds[var_name]
    da_diff = da.diff("step")
    ds_out = da_diff.to_dataset(name=var_name)
    ds_out[var_name] = ds_out[var_name].clip(min=0)
    return ds_out


# =============================================================================
# Main processing
# =============================================================================
def process_forecasts():
    """Process IFS (and optionally AIFS) forecast GRIB files."""
    models_to_process = ["ifs"]
    if cfg["has_aifs"]:
        models_to_process.append("aifs")
    else:
        print(f"\n[INFO] AIFS not available for event '{cfg['slug']}' — skipping AIFS processing.")

    for model_name in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {model_name.upper()} ENS forecasts — {cfg['name']}")
        print(f"{'='*60}")

        model_raw_dir = os.path.join(RAW_DIR, model_name)
        output_path   = os.path.join(PROC_DIR, f"{model_name}_processed.nc")

        if os.path.exists(output_path):
            print(f"  [SKIP] Already exists: {output_path}")
            continue

        try:
            ds = load_grib_forecast(model_raw_dir, model_name, INIT_DATE_TIMES)
        except FileNotFoundError as e:
            # IFS is required; AIFS (or any secondary product) with no/incomplete
            # files is skipped so ERA5/MSWEP/WeatherNext still get processed.
            if model_name == "ifs":
                raise
            print(f"  [WARNING] {model_name.upper()} skipped (no complete files): {e}")
            continue
        print(f"  Loaded: {ds.dims}")

        ds = deaccumulate_precipitation(ds, var_name="tp")

        if model_name == "ifs":
            print("  Converting m -> mm (×1000) [IFS]...")
            ds["tp"] = ds["tp"] * 1000.0
        else:
            print("  AIFS units already in kg/m² (= mm), no conversion needed.")
        ds["tp"].attrs["units"]     = "mm"
        ds["tp"].attrs["long_name"] = "Precipitation per step"

        print(f"  Saving to {output_path}...")
        ds.to_netcdf(output_path)
        print(f"  [DONE] {model_name.upper()}: {output_path}")


def process_era5():
    """Process ERA5 reanalysis NetCDF file."""
    print(f"\n{'='*60}")
    print(f"Processing ERA5 reanalysis — {cfg['name']}")
    print(f"{'='*60}")

    input_path  = os.path.join(RAW_DIR, "era5", f"era5_tp_{cfg['slug']}.nc")
    output_path = os.path.join(PROC_DIR, "era5_processed.nc")

    if os.path.exists(output_path):
        print(f"  [SKIP] Already exists: {output_path}")
        return

    if not os.path.exists(input_path):
        print(f"  [WARNING] ERA5 file not found: {input_path}")
        return

    print(f"  Loading {input_path}...")
    ds = xr.open_dataset(input_path)

    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    tp_var = None
    for candidate in ["tp", "total_precipitation", "var228"]:
        if candidate in ds.data_vars:
            tp_var = candidate
            break

    if tp_var is None:
        print(f"  [ERROR] Cannot find precipitation variable in {list(ds.data_vars)}")
        return

    if tp_var != "tp":
        ds = ds.rename({tp_var: "tp"})

    print("  Converting m -> mm (×1000) [ERA5, keeping 1-h native resolution]...")
    ds["tp"] = ds["tp"] * 1000.0
    ds["tp"].attrs["units"]     = "mm"
    ds["tp"].attrs["long_name"] = "Precipitation per hour"

    ds_out = ds.sel(time=slice(EVENT_START, EVENT_END))

    print(f"  Saving to {output_path}...")
    ds_out.to_netcdf(output_path)
    print(f"  [DONE] ERA5: {output_path}")


def process_mswep():
    """Process MSWEP V3.16 raw extraction into standard processed format."""
    print(f"\n{'='*60}")
    print(f"Processing MSWEP V3.16 — {cfg['name']}")
    print(f"{'='*60}")

    input_path  = os.path.join(RAW_DIR, "mswep", f"mswep_tp_{cfg['slug']}.nc")
    output_path = os.path.join(PROC_DIR, "mswep_processed.nc")

    if os.path.exists(output_path):
        print(f"  [SKIP] Already exists: {output_path}")
        return

    if not os.path.exists(input_path):
        print("  [WARNING] MSWEP raw file not found. Run scripts/extract_mswep.py first.")
        return

    print(f"  Loading {input_path}...")
    ds = xr.open_dataset(input_path)

    # Detect temporal resolution from time step
    times = ds["time"].values
    if len(times) > 1:
        dt_h = int((times[1] - times[0]).astype("timedelta64[h]").astype(int))
    else:
        dt_h = 3  # fallback

    # MSWEP timestamps mark the START of each accumulation window;
    # ERA5 timestamps mark the END. Shift forward by one interval to align.
    ds["time"] = ds["time"] + np.timedelta64(dt_h, "h")

    # Convert mm/interval → mm/h for consistency with ERA5. Go by the units the
    # file carries, not its time spacing: hourly-folder files sampled every 3 h
    # are already mm/h, and dividing them by 3 cut MSWEP to ~1/3 (2026-04..10).
    units = ds["precipitation"].attrs.get("units", "")
    if units == f"mm/{dt_h}h":
        ds["precipitation"] = ds["precipitation"] / float(dt_h)
    elif not (units == "mm/h" and dt_h == 1):
        raise ValueError(f"MSWEP: units '{units}' at {dt_h}-h spacing -- expected the 3hourly "
                         "product (mm/3h); re-run scripts/extract_mswep.py")
    ds["precipitation"].attrs["units"]     = "mm/h"
    ds["precipitation"].attrs["long_name"] = "Precipitation rate"
    print(f"  Detected {dt_h}-hourly MSWEP (time shift +{dt_h}h, rate /{dt_h if dt_h != 1 else '—'})")

    ds = ds.rename({"precipitation": "tp", "lat": "latitude", "lon": "longitude"})

    print(f"  Saving to {output_path}...")
    ds.to_netcdf(output_path)
    print(f"  [DONE] MSWEP: {output_path}")


def process_weathernext():
    """Stack per-init WeatherNext NetCDF files into standard processed format."""
    print(f"\n{'='*60}")
    print(f"Processing WeatherNext 2.0.0 — {cfg['name']}")
    print(f"{'='*60}")

    raw_dir     = os.path.join(RAW_DIR, "weathernext")
    output_path = os.path.join(PROC_DIR, "weathernext_processed.nc")

    if os.path.exists(output_path):
        print(f"  [SKIP] Already exists: {output_path}")
        return

    files = sorted(glob.glob(os.path.join(raw_dir, "weathernext_tp_*.nc")))
    if not files:
        print("  [WARNING] No WeatherNext raw files found. Run scripts/download_weathernext.py first.")
        return

    # Group member files per init: weathernext_tp_{init}.nc (samples 0-9) and
    # weathernext_tp_{init}_m10-63.nc (complementary samples). Only inits that
    # belong to this event are kept -- a stray file from another event (e.g. a
    # test download) must not leak into the processed dataset.
    wanted = {d.replace("-", "") + "_" + hh for d, hh in init_pairs(cfg)}
    groups = {}
    for f in files:
        key = os.path.basename(f)[len("weathernext_tp_"):][:11]
        if key not in wanted:
            print(f"  [SKIP] {os.path.basename(f)}: init not in this event's init_dates")
            continue
        groups.setdefault(key, []).append(f)

    print(f"  Found {len(groups)} inits ({sum(len(v) for v in groups.values())} files)...")
    datasets = []
    for key in sorted(groups):
        parts = [xr.open_dataset(f) for f in sorted(groups[key])]
        if any("tp" not in p.data_vars for p in parts):
            print(f"  [WARNING] Skipping malformed init: {key}")
            continue
        # Older downloads (kristin_2026, March) name the grid lat/lon, newer ones
        # latitude/longitude; unify per file or concat broadcasts the two grids
        # against each other. The grids must be the same points to stack members.
        parts = [p.rename({"lat": "latitude", "lon": "longitude"}) if "lat" in p.dims else p
                 for p in parts]
        for p in parts[1:]:
            for c in ("latitude", "longitude"):
                if not np.allclose(p[c].values, parts[0][c].values, atol=1e-4):
                    raise ValueError(f"WeatherNext {key}: {c} differs between member files")
        ds = parts[0] if len(parts) == 1 else xr.concat(parts, dim="number", coords="minimal",
                                                          compat="override", join="override")
        if "number" in ds.dims:
            _, first = np.unique(ds["number"].values, return_index=True)
            ds = ds.isel(number=np.sort(first)).sortby("number")
        print(f"    {key}: {ds.sizes.get('number', '?')} members")
        if "lat" in ds.dims:
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        # Normalise to common dims: (init_time, number, step, latitude, longitude)
        if "init_time" in ds.dims:
            # Already fully stacked (shouldn't happen for single-init files, but handle gracefully)
            if "time" in ds.dims:
                ds = ds.rename({"time": "step"})
        elif "step" in ds.dims:
            # Monolithic format: step dim exists, time is a scalar coord = init datetime
            init_dt = ds["time"].values
            ds = ds.drop_vars("time", errors="ignore")
            ds = ds.assign_coords(init_time=init_dt)
            ds = ds.expand_dims("init_time")
        elif "time" in ds.dims:
            # Per-init format: time dim = lead steps, init_time is scalar coord
            init_dt = ds["init_time"].values if "init_time" in ds.coords else None
            ds = ds.rename({"time": "step"})
            ds = ds.drop_vars("init_time", errors="ignore")
            if init_dt is not None:
                ds = ds.assign_coords(init_time=init_dt)
                ds = ds.expand_dims("init_time")
        else:
            print(f"  [WARNING] Unrecognised dim layout {dict(ds.dims)} for init {key}, skipping")
            continue
        datasets.append(ds)

    combined = xr.concat(datasets, dim="init_time")
    for var in ["datetime", "lead_time_secs"]:
        if var in combined.coords:
            combined = combined.drop_vars(var)

    print(f"  Saving to {output_path}...")
    combined.to_netcdf(output_path)
    print(f"  [DONE] WeatherNext: {output_path}")


def process_weathernext3():
    """Stack per-init WeatherNext 3 NetCDF files into standard processed format.

    WN3 comes from the statistics product, so `number` has length 1 and holds the
    64-member ensemble mean rather than a member. Downstream code that takes
    .mean("number") therefore still gets the right field, but WN3 must be kept
    out of any member-level analysis (ensemble_scan.py).
    """
    print(f"\n{'='*60}")
    print(f"Processing WeatherNext 3.0.0 (statistics) — {cfg['name']}")
    print(f"{'='*60}")

    raw_dir     = os.path.join(RAW_DIR, "weathernext3")
    output_path = os.path.join(PROC_DIR, "weathernext3_processed.nc")

    if os.path.exists(output_path):
        print(f"  [SKIP] Already exists: {output_path}")
        return

    files = sorted(glob.glob(os.path.join(raw_dir, "weathernext3_tp_*.nc")))
    if not files:
        print("  [WARNING] No WeatherNext 3 raw files. Run scripts/download_weathernext3.py first.")
        return

    print(f"  Found {len(files)} init files...")
    datasets = []
    for f in files:
        ds = xr.open_dataset(f)
        if "tp" not in ds.data_vars:
            print(f"  [WARNING] Skipping malformed file: {os.path.basename(f)}")
            continue
        # init_time is taken from the filename: it is the one piece of metadata
        # guaranteed present regardless of how the zarr coords survived the round trip.
        stem = os.path.basename(f).replace("weathernext3_tp_", "").replace(".nc", "")
        init_dt = np.datetime64(
            f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}T{stem[9:11]}:00:00")
        ds = ds.drop_vars(["init_time", "datetime"], errors="ignore")
        ds = ds.assign_coords(init_time=init_dt).expand_dims("init_time")
        datasets.append(ds)

    if not datasets:
        print("  [WARNING] No usable WeatherNext 3 files.")
        return

    combined = xr.concat(datasets, dim="init_time")
    print(f"  Saving to {output_path}...")
    combined.to_netcdf(output_path)
    print(f"  [DONE] WeatherNext 3: {output_path}  dims={dict(combined.sizes)}")


def process_eobs():
    """Process E-OBS gridded observations NetCDF file."""
    print(f"\n{'='*60}")
    print(f"Processing E-OBS observations — {cfg['name']}")
    print(f"{'='*60}")

    input_path  = os.path.join(RAW_DIR, "eobs", f"eobs_rr_{cfg['slug']}.nc")
    output_path = os.path.join(PROC_DIR, "eobs_processed.nc")

    if os.path.exists(output_path):
        print(f"  [SKIP] Already exists: {output_path}")
        return

    if not os.path.exists(input_path):
        print("  [WARNING] E-OBS file not found. Skipping.")
        print("  Analysis will use ERA5 as the only reference dataset.")
        return

    print(f"  Loading {input_path}...")
    ds = xr.open_dataset(input_path)

    rr_var = None
    for candidate in ["rr", "precipitation_amount"]:
        if candidate in ds.data_vars:
            rr_var = candidate
            break

    if rr_var is None:
        print(f"  [ERROR] Cannot find precipitation variable in {list(ds.data_vars)}")
        return

    if rr_var != "rr":
        ds = ds.rename({rr_var: "rr"})

    try:
        time_dim = "time" if "time" in ds.dims else list(ds.dims)[0]
        ds = ds.sel({time_dim: slice(EVENT_START, EVENT_END)})
        if len(ds[time_dim]) == 0:
            print(f"  [WARNING] No E-OBS data for event window {EVENT_START}–{EVENT_END}. Skipping.")
            return
    except Exception as e:
        print(f"  [WARNING] Failed to subset E-OBS: {e}")
        return

    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    ds = ds.sel({lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)})

    ds["rr"].attrs["units"]     = "mm/day"
    ds["rr"].attrs["long_name"] = "Daily precipitation sum"

    print(f"  Saving to {output_path}...")
    ds.to_netcdf(output_path)
    print(f"  [DONE] E-OBS: {output_path}")


def main():
    """Run all preprocessing steps."""
    print("=" * 60)
    print(f"PREPROCESSING: {cfg['name']} Forecast Comparison")
    print(f"  Event slug: {cfg['slug']}")
    print(f"  Output dir: {PROC_DIR}")
    print("=" * 60)

    os.makedirs(PROC_DIR, exist_ok=True)

    process_forecasts()
    process_era5()
    process_mswep()
    process_weathernext()
    process_weathernext3()
    process_eobs()

    print(f"\n{'='*60}")
    print("All preprocessing complete.")
    print(f"Processed files in: {PROC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
