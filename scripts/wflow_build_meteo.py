#!/usr/bin/env python3
"""
Build daily temp/PET inmaps for a wflow model from ERA5-Land (DestinE Earth
Data Hub), regridded to the model grid. Produces the daily files that
wflow_prep_forcing.py expects (it reads the grid + temp/PET from them; precip
comes from MSWEP/forecasts).

Used for events whose model did not ship with pre-built daily inmaps
(e.g. texas_2025). ERA5-Land provides 2 m temperature (t2m) and potential
evaporation (pev) directly, at 0.1° — finer than the 0.25° ERA5 used for the
Iberian events (those used hydromt debruin PET; here we use ERA5-Land pev).

Auth: needs a DestinE Earth Data Hub API key in ~/.netrc under
`machine api.earthdatahub.destine.eu`.

Usage:
    python scripts/wflow_build_meteo.py \
        --model data/wflow/texas_2025/model \
        --start 2025-01-01 --end 2025-07-12 \
        --basename era5land_temp_pet_daily
"""

import argparse
import os
import netrc
import numpy as np
import pandas as pd
import xarray as xr

EDH_HOST = "api.earthdatahub.destine.eu"     # ~/.netrc entry holding the key
EDH_DATA = "data.earthdatahub.destine.eu"
# Global ERA5 single-levels (hourly, ocean-covered) — replaces ERA5-Land so that
# coastal basins (e.g. Toamasina) get valid temp/PET on every cell (no land-mask
# NaN). Aggregated hourly→daily below. (t2m, pev, tp are all present.)
# Zarr v3 store (needs zarr>=3), updated monthly: reached 2026-09-15 on 2026-10-06.
# The older reanalysis-era5-single-levels-v0.zarr stopped at 2026-05-31; values
# are identical over the overlap (max |dtp| 0.003 mm/h).
EDH_URL  = "era5/era5-single-levels-atmosphere-v0.zarr"
BUFFER_DEG = 0.5


def apply_lapse_correction(temp, model_dir, lapse_rate):
    """Correct interpolated ERA5 temperature for the model's real elevation.

    ERA5's 0.25° orography is heavily smoothed: over steep terrain its effective
    surface can sit 1,000–2,000 m below the real ridges, so bilinearly
    interpolating t2m onto a fine grid gives a near-uniform, far-too-warm field
    across a basin that actually spans thousands of metres. For a snow/ice basin
    that is a first-order error — it puts the rain/snow line in the wrong place
    and melts the pack at the wrong time, which matters more than any difference
    between the forecasts being compared.

    The correction needs ERA5's *effective* elevation, which is not in the file.
    We approximate it the way hydromt's own temp_correction does: coarse-grain
    the model DEM to roughly the forcing resolution, treat that as the elevation
    ERA5 "sees", and shift by the lapse rate times the difference. Basin-mean
    temperature is therefore nearly unchanged while the vertical structure is
    restored.
    """
    with xr.open_dataset(os.path.join(model_dir, "staticmaps.nc")) as sm:
        dem_var = next((v for v in ("land_elevation", "meta_subgrid_elevation",
                                    "wflow_dem", "dem") if v in sm), None)
        if dem_var is None:
            print("  [WARN] no elevation map in staticmaps — skipping lapse correction")
            return temp
        dem = sm[dem_var].load()

    # ERA5 single-levels is 0.25°; the model grid is ~1/120°, so ~30 cells per
    # ERA5 cell in each direction.
    #
    # Use a CENTRED ROLLING mean rather than coarsen()+interp(). Coarsening a
    # small basin (Vispa is 46 rows) by 30 leaves ~2 coarse cells, whose centres
    # span far less than the model grid — interpolating back then leaves NaN over
    # most of the domain, and wflow rejects any forcing with missing values on
    # active cells. min_periods=1 keeps every cell defined, including the edges.
    # For a basin smaller than one ERA5 cell this correctly degenerates to
    # "ERA5 sees roughly one elevation for the whole basin".
    res = float(abs(dem["latitude"][1] - dem["latitude"][0]))
    win = max(1, int(round(0.25 / res)))
    dem_coarse = dem.rolling(latitude=win, longitude=win,
                             center=True, min_periods=1).mean()
    delta = (dem - dem_coarse)
    corr = (lapse_rate * delta).fillna(0.0)   # never inject NaN into the forcing
    # Report the NaN count explicitly: min/max silently skip NaN, so a field that
    # is mostly missing still prints a healthy-looking range (this hid a bug once).
    n_nan = int(corr.isnull().sum())
    print(f"  lapse correction: DEM {float(dem.min()):.0f}–{float(dem.max()):.0f} m, "
          f"rolling {win}x{win} (~0.25°), ΔT {float(corr.min()):+.1f} to "
          f"{float(corr.max()):+.1f} °C (mean {float(corr.mean()):+.2f}), "
          f"NaN cells {n_nan}")
    if n_nan:
        raise ValueError(f"lapse correction produced {n_nan} NaN cells — wflow "
                         f"rejects forcing with missing values on active cells")
    return temp + corr


def _edh_url() -> str:
    key = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators(EDH_HOST)[2]
    return f"https://edh:{key}@{EDH_DATA}/{EDH_URL}"


def cds_tail(t0, t1, lat_n, lat_s, lon_w, lon_e, model_dir):
    """Hourly t2m/pev/tp from the CDS (ERA5T) for t0..t1, on EDH's 0..360 grid.

    The download is kept next to the inmaps (meta_cds_tail_*.nc) so a rerun
    does not fetch it again. Needs ~/.cdsapirc.
    """
    import cdsapi
    out = os.path.join(model_dir, "inmaps", f"meta_cds_tail_{t0:%Y%m%d%H}_{t1:%Y%m%d%H}.nc")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        days = pd.date_range(t0.normalize(), t1.normalize(), freq="D")
        cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature", "potential_evaporation", "total_precipitation"],
            "year": sorted({f"{d:%Y}" for d in days}),
            "month": sorted({f"{d:%m}" for d in days}),
            "day": sorted({f"{d:%d}" for d in days}),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [float(np.ceil(lat_n * 4) / 4), float(np.floor(lon_w * 4) / 4),
                     float(np.floor(lat_s * 4) / 4), float(np.ceil(lon_e * 4) / 4)],
            "data_format": "netcdf", "download_format": "unarchived",
        }, out + ".tmp")
        os.replace(out + ".tmp", out)
    import zipfile
    if zipfile.is_zipfile(out):
        # instant (t2m) and accumulated (pev, tp) fields come as two files in a zip
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(out) as z:
                z.extractall(tmp)
            parts = [xr.open_dataset(os.path.join(tmp, f)).load() for f in sorted(os.listdir(tmp))]
        merged = xr.merge([p.drop_vars("expver", errors="ignore") for p in parts], compat="override")
        merged.to_netcdf(out + ".tmp")
        os.replace(out + ".tmp", out)
    ds = xr.open_dataset(out)
    if "time" in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    for _d in ("number", "expver"):
        if _d in ds.coords and _d not in ds.dims:
            ds = ds.drop_vars(_d)
    ds = ds[["t2m", "pev", "tp"]].sel(valid_time=slice(t0, t1)).load()
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude").sortby("latitude", ascending=False)
    print(f"  CDS tail: {ds.sizes['valid_time']} hours")
    return ds


def main():
    ap = argparse.ArgumentParser(description="Build daily temp/PET inmaps from global ERA5")
    ap.add_argument("--model", required=True, help="model dir containing staticmaps.nc")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (spinup start)")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD (sim end)")
    ap.add_argument("--basename", default="era5land_temp_pet_daily")
    ap.add_argument("--temp-lapse-correct", action="store_true",
                    help="Correct interpolated ERA5 temperature for the model's real "
                         "elevation. OFF by default so existing events are unchanged; "
                         "essential for steep basins (e.g. Vispa spans 650-4,600 m).")
    ap.add_argument("--lapse-rate", type=float, default=-0.0065,
                    help="°C per m (default -0.0065, the standard environmental rate)")
    args = ap.parse_args()

    # Target grid from the model staticmaps
    sm = xr.open_dataset(os.path.join(args.model, "staticmaps.nc"))
    tlat = sm["latitude"].values
    tlon = sm["longitude"].values
    spatial_ref = sm["spatial_ref"] if "spatial_ref" in sm else None
    sm.close()
    print(f"Target grid: {len(tlat)} x {len(tlon)}  "
          f"lat {tlat.min():.3f}..{tlat.max():.3f}  lon {tlon.min():.3f}..{tlon.max():.3f}")

    # ERA5-Land uses 0..360 longitude; build a selection window (+buffer)
    lon0 = (tlon.min() - BUFFER_DEG) % 360
    lon1 = (tlon.max() + BUFFER_DEG) % 360
    lat0 = tlat.max() + BUFFER_DEG
    lat1 = tlat.min() - BUFFER_DEG

    print("Opening global ERA5 single-levels (DestinE Earth Data Hub)…")
    z = xr.open_zarr(_edh_url(), chunks={})
    sub = z[["t2m", "pev", "tp"]].sel(valid_time=slice(args.start, args.end),
                                      latitude=slice(lat0, lat1),
                                      longitude=slice(lon0, lon1))
    # Drop singleton reanalysis dims (number / surface) if present.
    for _d in ("number", "surface"):
        if _d in sub.dims and sub.sizes[_d] == 1:
            sub = sub.isel({_d: 0}, drop=True)

    # The EDH store is updated monthly; days past its end come from CDS (ERA5T).
    last = pd.Timestamp(sub["valid_time"].values[-1])
    want = pd.Timestamp(args.end) + pd.Timedelta("23h")
    if last < want:
        print(f"  EDH ends {last}; fetching {last + pd.Timedelta('1h')} .. {want} from CDS")
        tail = cds_tail(last + pd.Timedelta("1h"), want, lat0, lat1,
                        tlon.min() - BUFFER_DEG, tlon.max() + BUFFER_DEG, args.model)
        tail = tail.reindex(latitude=sub["latitude"].values, longitude=sub["longitude"].values,
                            method="nearest", tolerance=1e-3)
        sub = xr.concat([sub.load(), tail], dim="valid_time")

    # Hourly → daily (UTC): temp = daily mean; PET/precip = daily accumulation.
    # pev is a per-hour negative (upward) flux → daily sum, sign-flip → mm/day.
    temp   = (sub["t2m"] - 273.15).resample(valid_time="1D").mean()
    pet    = ((-sub["pev"]).resample(valid_time="1D").sum() * 1000.0).clip(min=0.0)
    precip = (sub["tp"].resample(valid_time="1D").sum() * 1000.0).clip(min=0.0)
    ds = xr.Dataset({"precip": precip, "temp": temp, "pet": pet}) \
        .rename({"valid_time": "time"}).load()
    print(f"  ERA5 daily subset: {dict(ds.sizes)}")

    # Convert longitude 0..360 → -180..180 to match the model grid
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")

    # Regrid to model grid (bilinear). Global ERA5 covers ocean, so coastal
    # basins get valid temp/PET everywhere — no land-mask NaN, no fill needed.
    ds = ds.interp(latitude=tlat, longitude=tlon, method="linear",
                   kwargs={"fill_value": None})

    if args.temp_lapse_correct:
        ds["temp"] = apply_lapse_correction(ds["temp"], args.model, args.lapse_rate)

    temp.attrs = {"unit": "degree C.", "source": "era5 t2m"}
    pet.attrs  = {"unit": "mm", "source": "era5 pev", "pet_method": "era5_pev"}
    ds["temp"].attrs = temp.attrs
    ds["pet"].attrs  = pet.attrs
    if spatial_ref is not None:
        ds["spatial_ref"] = spatial_ref

    # Write monthly files matching the daily-inmaps naming convention
    out_dir = os.path.join(args.model, "inmaps")
    os.makedirs(out_dir, exist_ok=True)
    months = pd.PeriodIndex(pd.DatetimeIndex(ds["time"].values).to_period("M").unique(), freq="M")
    for p in months:
        mds = ds.sel(time=str(p))
        out = os.path.join(out_dir, f"{args.basename}_{p.year}{p.month:02d}.nc")
        enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
               for v in ("precip", "temp", "pet")}
        mds.to_netcdf(out, encoding=enc)
        print(f"  wrote {os.path.basename(out)}  "
              f"temp {float(mds.temp.mean()):.1f}°C  pet {float(mds.pet.mean()):.2f} mm/d")
    print(f"Done: {len(months)} monthly files in {out_dir}")


if __name__ == "__main__":
    main()
