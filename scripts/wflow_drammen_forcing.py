#!/usr/bin/env python3
"""
PARALLEL Drammen-specific wflow forcing + setup builder (Storm Hans 2023).

The Drammen wflow model (data/basins/wflow_drammen) is on a PROJECTED grid
(ETRS89 / UTM zone 33N, EPSG:25833; dims y/x in metres) and uses a warm-start
flag, a nested staticmaps path, and a "Q"/gauges_drammen CSV output. That is
incompatible with the generic lat/lon AIFS pipeline (wflow_setup.py /
wflow_prep_forcing.py / wflow_build_meteo.py), so this module is intentionally
SEPARATE and does not touch those shared scripts.

It reprojects by sampling: each model cell centre (x,y) is transformed to
lat/lon with pyproj, and the lat/lon forcing sources (MSWEP 3h, ERA5-Land daily
temp/PET, IFS/WN2/ERA5 forecast precip) are bilinearly sampled at those points,
producing forcing on the native (y, x) grid that wflow expects.

Per case it: (1) copies the model once, (2) builds 3-hourly forcing into
inmaps_{case}/, (3) writes a patched TOML wflow_sbm_hans_2023_{case}.toml
(cold start forced on, event time bounds, 3h step, path_forcing).

Run with the generic runner afterwards:  python scripts/wflow_run.py hans_2023 <case>

Usage:
    python scripts/wflow_drammen_forcing.py hist_3h
    python scripts/wflow_drammen_forcing.py ifs_jul31
"""

import os
import re
import sys
import glob
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pyproj

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
# Reuse the (grid-agnostic) MSWEP reader and the DestinE URL helper only.
from wflow_prep_forcing import read_mswep_period, MSWEP_PAST, BBOX_BUFFER_DEG
from wflow_build_meteo import _edh_url, EDH_URL  # noqa: F401

warnings.filterwarnings("ignore")

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_MODEL = os.path.join(REPO_ROOT, "data", "basins", "wflow_drammen")
STATICMAPS_REL = "staticmaps/staticmaps_v5_updated_reservoirs.nc"
MODEL_DIR   = os.path.join(REPO_ROOT, "data", "wflow", "hans_2023", "model")
MODEL_EPSG  = 25833

# ── event / case configuration (parallel to the generic EVENT_CONFIG) ──────────
SIM_START = "2023-04-01 00:00"
SIM_END   = "2023-08-15 00:00"   # ifs_jul31/wn2_jul31 horizon (360 h)

CASES = {
    "hist_3h": {
        "precip_mode": "mswep",
        "basename":    "hans_mswep_era5land_3h",
        "label":       "MSWEP 3h + ERA5-Land (disaggregated)",
        "gridded_output": True,
    },
    "era5_aug06": {
        "precip_mode": "era5_blend",
        "era5_src":    "data/processed/hans_2023/era5_processed.nc",
        "era5_start":  "2023-08-06",
        "basename":    "hans_era5_aug06_3h",
        "label":       "ERA5 reanalysis (event Aug 6+) + MSWEP spinup",
    },
    # IFS ENS — long lead (init Jul 31, ~7 days) and near lead (Aug 5, ~2 days)
    "ifs_jul31": {
        "precip_mode": "blend", "forecast_src": "data/processed/hans_2023/ifs_processed.nc",
        "init_date": "2023-07-31", "basename": "hans_ifs_jul31_3h",
        "label": "IFS ENS (init 2023-07-31 00Z, 7-day lead) + MSWEP spinup",
    },
    "ifs_aug05": {
        "precip_mode": "blend", "forecast_src": "data/processed/hans_2023/ifs_processed.nc",
        "init_date": "2023-08-05", "basename": "hans_ifs_aug05_3h",
        "label": "IFS ENS (init 2023-08-05 00Z, 2-day lead) + MSWEP spinup",
    },
    # WeatherNext 2 — long lead (Jul 31) and near lead (Aug 5)
    "wn2_jul31": {
        "precip_mode": "blend", "forecast_src": "data/processed/hans_2023/weathernext_processed.nc",
        "init_date": "2023-07-31", "basename": "hans_wn2_jul31_3h",
        "label": "WeatherNext 2 (init 2023-07-31 00Z, 7-day lead) + MSWEP spinup",
    },
    "wn2_aug05": {
        "precip_mode": "blend", "forecast_src": "data/processed/hans_2023/weathernext_processed.nc",
        "init_date": "2023-08-05", "basename": "hans_wn2_aug05_3h",
        "label": "WeatherNext 2 (init 2023-08-05 00Z, 2-day lead) + MSWEP spinup",
    },
    # -1 d start (2026-10-15): the lead-selection check shows IFS ahead of WN2 at
    # 1-1.5 d for Hans, which the -7/-2 d starts miss
    "ifs_aug06": {
        "precip_mode": "blend", "forecast_src": "data/processed/hans_2023/ifs_processed.nc",
        "init_date": "2023-08-06", "basename": "hans_ifs_aug06_3h",
        "label": "IFS ENS (init 2023-08-06 00Z, 1-day lead) + MSWEP spinup",
    },
    "wn2_aug06": {
        "precip_mode": "blend", "forecast_src": "data/processed/hans_2023/weathernext_processed.nc",
        "init_date": "2023-08-06", "basename": "hans_wn2_aug06_3h",
        "label": "WeatherNext 2 (init 2023-08-06 00Z, 1-day lead) + MSWEP spinup",
    },
}


# =============================================================================
# Model grid → lat/lon sampling target
# =============================================================================
def load_model_grid():
    """Return (x, y, lat2d, lon2d, spatial_ref) for the Drammen model grid."""
    sm = xr.open_dataset(os.path.join(SOURCE_MODEL, STATICMAPS_REL))
    x = sm["x"].values
    y = sm["y"].values
    spatial_ref = sm["spatial_ref"] if "spatial_ref" in sm else None
    sm.close()
    tr = pyproj.Transformer.from_crs(MODEL_EPSG, 4326, always_xy=True)
    X, Y = np.meshgrid(x, y)          # (ny, nx)
    lon2d, lat2d = tr.transform(X, Y)  # degrees
    return x, y, lat2d, lon2d, spatial_ref


def latlon_bbox(lat2d, lon2d, buffer=BBOX_BUFFER_DEG):
    return {
        "lat_min": float(lat2d.min()) - buffer,
        "lat_max": float(lat2d.max()) + buffer,
        "lon_min": float(lon2d.min()) - buffer,
        "lon_max": float(lon2d.max()) + buffer,
    }


def _fill_nan_nearest(da: xr.DataArray) -> xr.DataArray:
    """Fill NaN cells (e.g. ERA5-Land sea gaps) with the nearest valid value,
    per 2-D (latitude, longitude) slice. Uses a distance transform (no bottleneck)."""
    from scipy import ndimage
    vals = np.array(da.values, dtype="float64")
    flat = vals.reshape(-1, vals.shape[-2], vals.shape[-1]) if vals.ndim == 3 else vals[None]
    for i in range(flat.shape[0]):
        sl = flat[i]
        m = np.isnan(sl)
        if m.any() and (~m).any():
            idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
            flat[i] = sl[tuple(idx)]
    return da.copy(data=flat.reshape(vals.shape).astype(da.dtype))


def sample_to_model(src: xr.DataArray, y, x, lat2d, lon2d) -> xr.DataArray:
    """Bilinearly sample a lat/lon source DataArray at the model cell centres.

    Returns a DataArray with dims (..., y, x) on the model grid. `src` must have
    latitude/longitude dims (any monotonicity); other dims (e.g. time) are kept.
    """
    lat_dim = next(d for d in src.dims if d.lower() in ("lat", "latitude"))
    lon_dim = next(d for d in src.dims if d.lower() in ("lon", "longitude"))
    src = src.rename({lat_dim: "latitude", lon_dim: "longitude"})
    # xr.interp needs increasing coords
    src = src.sortby("latitude").sortby("longitude")

    lat_da = xr.DataArray(lat2d, dims=("y", "x"), coords={"y": y, "x": x})
    lon_da = xr.DataArray(lon2d, dims=("y", "x"), coords={"y": y, "x": x})
    out = src.interp(latitude=lat_da, longitude=lon_da,
                     kwargs={"fill_value": None})  # nearest-edge extrapolation off
    # Drop helper coords, keep (…, y, x)
    return out.reset_coords(["latitude", "longitude"], drop=True)


# =============================================================================
# ERA5-Land daily temp/PET → sampled to model, disaggregated to 3h
# =============================================================================
def build_temp_pet_3h(times_3h, y, x, lat2d, lon2d, bbox) -> xr.Dataset:
    """Daily ERA5-Land t2m/pev → model grid → 3-hourly temp (°C) and pet (mm/3h)."""
    start = pd.Timestamp(times_3h[0]).strftime("%Y-%m-%d")
    end   = (pd.Timestamp(times_3h[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lon0 = (bbox["lon_min"]) % 360
    lon1 = (bbox["lon_max"]) % 360
    z = xr.open_zarr(_edh_url(), chunks={})
    sub = z.sel(valid_time=slice(start, end),
                latitude=slice(bbox["lat_max"], bbox["lat_min"]),
                longitude=slice(lon0, lon1))[["t2m", "pev"]].load()
    sub = sub.assign_coords(longitude=((sub.longitude + 180) % 360) - 180).sortby("longitude")
    temp_d = (sub["t2m"] - 273.15).rename({"valid_time": "time"})
    pet_d  = (-sub["pev"] * 1000.0).clip(min=0.0).rename({"valid_time": "time"})

    # ERA5-Land is land-only → NaN over sea/large water; fill spatial gaps by
    # nearest valid so coastal model cells never sample a NaN.
    temp_d = _fill_nan_nearest(temp_d)
    pet_d  = _fill_nan_nearest(pet_d)

    temp_dm = sample_to_model(temp_d, y, x, lat2d, lon2d)   # (time_daily, y, x)
    pet_dm  = sample_to_model(pet_d,  y, x, lat2d, lon2d)

    # temp: linear interpolation to 3h; pet: that day's value / 8 (uniform)
    temp_3h = temp_dm.interp(time=times_3h, kwargs={"fill_value": "extrapolate"}).astype("float32")
    day_idx = pd.DatetimeIndex(times_3h).normalize()
    pet_sel = pet_dm.sel(time=xr.DataArray(day_idx, dims="time"), method="nearest")
    pet_3h  = (pet_sel / 8.0).clip(min=0.0).assign_coords(time=times_3h).astype("float32")
    return xr.Dataset({"temp": temp_3h, "pet": pet_3h})


# =============================================================================
# Forecast / ERA5 event precip → 3h on lat/lon (then sampled to model)
# =============================================================================
def _resample_forecast_3h_latlon(case_cfg, member=None) -> xr.DataArray:
    """Forecast precip on 3 h slots (mm/3h), native lat/lon grid, stamped at the
    end of each interval. Same mass-conserving re-binning as the generic
    pipeline (wflow_prep_forcing.accum_to_slots); `member` selects one member,
    None gives the ensemble mean.
    """
    from wflow_prep_forcing import forecast_slots_latlon
    return forecast_slots_latlon({"member_dim": "number", "forecast_var": "tp", **case_cfg}, member)


def _era5_3h_latlon(case_cfg) -> xr.DataArray:
    """ERA5 reanalysis hourly precip → 3h sums (mm/3h), on the native grid."""
    ds = xr.open_dataset(os.path.join(REPO_ROOT, case_cfg["era5_src"]))
    # (t-1h, t] hourly values -> (T-3h, T] stamped T, as in the generic pipeline
    tp_3h = ds["tp"].resample(time="3h", closed="right", label="right").sum(skipna=False).clip(min=0.0)
    # keep only complete (3-hour) bins: a partial bin at either end of the ERA5
    # record would under-count that slot
    n_h = pd.Series(1, index=pd.DatetimeIndex(ds["tp"]["time"].values)).resample(
        "3h", closed="right", label="right").sum()
    tp_3h = tp_3h.sel(time=n_h.index[n_h.values == 3])
    ds.close()
    return tp_3h


# =============================================================================
# TOML patch (Drammen-specific: force cold start, keep nested static path)
# =============================================================================
def write_toml(case: str, case_cfg: dict):
    src_toml = os.path.join(SOURCE_MODEL, "wflow_sbm.toml")
    content = open(src_toml).read()
    patches = {
        r"^(dir_output\s*=\s*).*$":          f'\\g<1>"run_hans_2023_{case}"',
        r"^\s*#?\s*(starttime\s*=\s*).*$":    f'\\g<1>{pd.Timestamp(SIM_START).strftime("%Y-%m-%dT%H:%M:%S")}',
        r"^\s*#?\s*(endtime\s*=\s*).*$":      f'\\g<1>{pd.Timestamp(SIM_END).strftime("%Y-%m-%dT%H:%M:%S")}',
        r"^\s*#?\s*(timestepsecs\s*=\s*).*$": r"\g<1>10800",
        r"^(cold_start__flag\s*=\s*).*$":     r"\g<1>true",
        # the operational Drammen TOML has fews_run__flag = true, which starts
        # Wflow one step after `starttime`; off, like every other model
        r"^(fews_run__flag\s*=\s*).*$":       r"\g<1>false",
        r"^(path_forcing\s*=\s*).*$":         f'\\g<1>"inmaps_{case}/*.nc"',
    }
    for pat, rep in patches.items():
        content = re.sub(pat, rep, content, flags=re.MULTILINE)

    # Strip gridded output.nc (~large) unless this case explicitly wants it.
    if not case_cfg.get("gridded_output", False):
        out, skip = [], False
        for line in content.splitlines(keepends=True):
            s = line.strip()
            if s.startswith("["):
                skip = s.strip("[]").startswith("output.netcdf_grid")
            if not skip:
                out.append(line)
        content = "".join(out)

    dst_toml = os.path.join(MODEL_DIR, f"wflow_sbm_hans_2023_{case}.toml")
    open(dst_toml, "w").write(content)
    print(f"  TOML: {os.path.basename(dst_toml)}")


# =============================================================================
# Model copy (once) — exclude bulky/irrelevant dirs
# =============================================================================
def ensure_model_copied():
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, STATICMAPS_REL)):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    skip = {"run_default", "dump", "instate", "outstate"}
    for item in sorted(os.listdir(SOURCE_MODEL)):
        if item in skip or item.startswith("run_"):
            continue
        s = os.path.join(SOURCE_MODEL, item)
        d = os.path.join(MODEL_DIR, item)
        if os.path.exists(d):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    os.makedirs(os.path.join(MODEL_DIR, "outstate"), exist_ok=True)
    print(f"  Copied Drammen model → {MODEL_DIR}")


# =============================================================================
# Build forcing for one case (monthly files on the model grid)
# =============================================================================
_ENC = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in ("precip", "temp", "pet")}


def _build_base_month(base_path, times_3h, y, x, lat2d, lon2d, bbox, spatial_ref):
    """Shared per-month base (MSWEP precip + ERA5-Land temp/pet), built once.

    MSWEP stamps the START of its 3 h interval; slot T (wflow: (T-3h, T]) reads
    the file stamped T-3h (see wflow_prep_forcing.mswep_slots). A base month
    written before that fix gets its precip replaced and keeps temp/pet.
    """
    from wflow_prep_forcing import base_is_current
    if base_is_current(base_path):
        return
    old_ds = None
    if os.path.exists(base_path):
        with xr.open_dataset(base_path) as o:
            old_ds = o.load()
        times_3h = pd.DatetimeIndex(old_ds["time"].values)   # refresh on the file's own times
    mswep_ll = read_mswep_period(times_3h[0] - pd.Timedelta("3h"), times_3h[-1] - pd.Timedelta("3h"),
                                 bbox, MSWEP_PAST)
    if mswep_ll.sizes["time"] != len(times_3h):
        raise FileNotFoundError(f"MSWEP incomplete for {times_3h[0]} - {times_3h[-1]}")
    precip = sample_to_model(mswep_ll, y, x, lat2d, lon2d).clip(min=0.0)
    precip = precip.assign_coords(time=times_3h).astype("float32")
    precip.attrs.update(units="mm/3h", time_convention="end")
    if old_ds is not None:
        out = old_ds
        out["precip"] = precip
    else:
        tp = build_temp_pet_3h(times_3h, y, x, lat2d, lon2d, bbox)
        out = xr.Dataset({"precip": precip, "temp": tp["temp"], "pet": tp["pet"]}).assign_coords(y=y, x=x)
        if spatial_ref is not None:
            out["spatial_ref"] = spatial_ref
    tmp = f"{base_path}.tmp.{os.getpid()}"
    out.to_netcdf(tmp, encoding=_ENC)
    os.replace(tmp, base_path)


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_case(case: str):
    case_cfg = CASES[case]
    ensure_model_copied()

    x, y, lat2d, lon2d, spatial_ref = load_model_grid()
    bbox = latlon_bbox(lat2d, lon2d)
    sim_start, sim_end = pd.Timestamp(SIM_START), pd.Timestamp(SIM_END)

    # Event precip source (blend / era5_blend only), sampled to model grid.
    fcast = None
    fstart = None
    if case_cfg["precip_mode"] == "blend":
        print("  Extracting forecast precip (ens mean, 3h)…")
        fcast = sample_to_model(_resample_forecast_3h_latlon(case_cfg), y, x, lat2d, lon2d).clip(min=0.0)
        from wflow_prep_forcing import forecast_init
        fstart = forecast_init(case_cfg)
    elif case_cfg["precip_mode"] == "era5_blend":
        print("  Extracting ERA5 reanalysis precip (3h)…")
        fcast = sample_to_model(_era5_3h_latlon(case_cfg), y, x, lat2d, lon2d).clip(min=0.0)
        fstart = pd.Timestamp(case_cfg["era5_start"])

    out_dir  = os.path.join(MODEL_DIR, f"inmaps_{case}")
    base_dir = os.path.join(MODEL_DIR, "inmaps_base")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    for period in pd.period_range(sim_start, sim_end, freq="M"):
        mon_start = max(pd.Timestamp(period.start_time).floor("3h"), sim_start)
        mon_end   = min(pd.Timestamp(period.end_time).floor("3h"),   sim_end)
        times_3h  = pd.date_range(mon_start, mon_end, freq="3h")
        ym = f"{period.year}{period.month:02d}"
        base_path = os.path.join(base_dir, f"base_{ym}.nc")
        out_path  = os.path.join(out_dir, f"{case_cfg['basename']}_{ym}.nc")

        from wflow_prep_forcing import base_is_current
        if not base_is_current(base_path):
            print(f"  month {ym}: building base (MSWEP + temp/pet)…")
            _build_base_month(base_path, times_3h, y, x, lat2d, lon2d, bbox, spatial_ref)
        if os.path.exists(out_path):
            print(f"  [skip] {os.path.basename(out_path)}")
            continue

        # Pure spinup / all-MSWEP month → identical to base: hardlink.
        if fcast is None or mon_end <= fstart:
            _link_or_copy(base_path, out_path)
            print(f"  reuse base → {os.path.basename(out_path)}")
            continue

        # Event month: keep base temp/pet, overlay forecast precip from fstart.
        base_ds = xr.open_dataset(base_path)
        # slot stamped fstart = the 3 h before the init -> stays MSWEP
        ev = times_3h[times_3h >  fstart]
        sp = times_3h[times_3h <= fstart]
        parts = []
        if len(sp):
            parts.append(base_ds["precip"].sel(time=sp))
        if len(ev):
            # Split at the forecast horizon: `sel(method="nearest")` has no upper
            # bound, so without this the last forecast step would be repeated for
            # every slot past the end of the forecast, fabricating rain inside the
            # scored window. Zero-fill instead — see wflow_prep_forcing.py.
            horizon = pd.DatetimeIndex(fcast["time"].values).max()
            ev_in, ev_out = ev[ev <= horizon], ev[ev > horizon]
            if len(ev_in):
                parts.append(fcast.sel(time=ev_in))   # exact 3 h slots
            if len(ev_out):
                parts.append(xr.zeros_like(fcast.isel(time=0, drop=True))
                               .expand_dims(time=ev_out))
        precip = (xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]).reindex(time=times_3h).astype("float32")
        out = xr.Dataset({"precip": precip, "temp": base_ds["temp"], "pet": base_ds["pet"]})
        if "spatial_ref" in base_ds:
            out["spatial_ref"] = base_ds["spatial_ref"]
        tmp = f"{out_path}.tmp.{os.getpid()}"
        out.to_netcdf(tmp, encoding=_ENC)
        os.replace(tmp, out_path)
        base_ds.close()
        print(f"  overlay forecast ({len(ev)} slots) → {os.path.basename(out_path)}")

    write_toml(case, case_cfg)
    print(f"Done: {case}  → {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Build Drammen (Hans) wflow forcing on the UTM grid")
    ap.add_argument("case", help=f"one of {list(CASES)}")
    args = ap.parse_args()
    if args.case not in CASES:
        sys.exit(f"Unknown case '{args.case}'. Options: {list(CASES)}")
    print(f"=== Drammen forcing (parallel): hans_2023 / {args.case} ===")
    print(f"Label: {CASES[args.case]['label']}")
    build_case(args.case)


if __name__ == "__main__":
    main()
