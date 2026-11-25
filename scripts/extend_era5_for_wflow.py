#!/usr/bin/env python3
"""
ERA5 hourly precipitation for the wflow ERA5 reference cases that start before
the event's ERA5 download window (ref_start).

bc_ar_2024 era5_oct16 and kristin_2026 era5_jan22 (also its sub-basins) start
at the long-lead forecast init (16 Oct / 22 Jan), but era5_processed.nc covers
ref_start..ref_end only (17 Oct / 26 Jan onward). The old forcing filled the
gap by repeating the first ERA5 value (`sel(method="nearest")`), i.e. 24 h
(BC) and 4 days (Kristin) of invented rain.

This reads only the missing days from the DestinE Earth Data Hub ERA5 (same
product and grid as the CDS file; the first overlapping day is compared),
joins them in front of era5_processed.nc and writes
data/processed/<slug>/era5_wflow.nc (mm per hour, ERA5 stamps = end of hour).
The meteo-analysis file era5_processed.nc is left untouched.

Usage (AIFS_EVENT selects the event, as for every downloader):
    AIFS_EVENT=bc_ar_2024 python scripts/extend_era5_for_wflow.py
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_config import cfg  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# earliest ERA5 wflow case start per event (wflow_prep_forcing EVENT_CONFIG)
CASE_START = {"bc_ar_2024": "2024-10-16", "kristin_2026": "2026-01-22"}


def main():
    slug = cfg["slug"]
    start = pd.Timestamp(CASE_START[slug])
    proc = os.path.join(REPO, "data", "processed", slug)
    base = xr.open_dataset(os.path.join(proc, "era5_processed.nc"))["tp"]
    t0 = pd.Timestamp(base["time"].values[0])
    days = pd.date_range(start, t0 - pd.Timedelta("1h"), freq="D")
    print(f"{slug}: era5_processed starts {t0}; downloading {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}")

    # ERA5 single levels from the DestinE Earth Data Hub (same hourly product
    # and 0.25 deg grid as the CDS download; key in ~/.netrc, see
    # wflow_build_meteo.py). A few hundred kB, so this is a seconds-long read.
    from wflow_build_meteo import _edh_url
    z = xr.open_zarr(_edh_url(), chunks={})["tp"]
    lat, lon = base["latitude"].values, base["longitude"].values
    edh = (z.sel(latitude=lat, longitude=lon % 360, method="nearest", tolerance=1e-3)
             .sel(valid_time=slice(start, t0 + pd.Timedelta("23h")))
             .rename({"valid_time": "time"}).load() * 1000.0)
    edh = edh.assign_coords(latitude=lat, longitude=lon)
    edh = edh.drop_vars([c for c in edh.coords if c not in ("time", "latitude", "longitude")])

    # consistency: the first 24 h of era5_processed (CDS) must match the hub
    ov = edh.sel(time=slice(t0, t0 + pd.Timedelta("23h")))
    diff = float(np.abs(ov.values - base.sel(time=ov["time"]).values).max())
    print(f"  overlap {len(ov.time)} h vs era5_processed (CDS): max |diff| {diff:.4f} mm/h "
          f"(max value {float(base.sel(time=ov['time']).max()):.2f})")
    if diff > 0.05:
        raise ValueError("Earth Data Hub and CDS ERA5 disagree on the overlap")
    ext = edh.sel(time=slice(start, t0 - pd.Timedelta("1h")))

    out = xr.concat([ext, base.drop_vars([c for c in ("number", "expver") if c in base.coords])],
                    dim="time").to_dataset(name="tp")
    t = pd.DatetimeIndex(out["time"].values)
    if not (t.is_monotonic_increasing and ((t[1:] - t[:-1]) == pd.Timedelta("1h")).all()):
        raise ValueError("merged ERA5 series is not continuous hourly")
    if t[0] > start + pd.Timedelta("1h"):
        raise ValueError(f"merged series starts {t[0]}, need {start} + 1 h")
    out["tp"].attrs.update(units="mm", long_name="Precipitation per hour",
                           note="ERA5 hourly tp, stamp = end of hour; extended before ref_start for wflow")
    dst = os.path.join(proc, "era5_wflow.nc")
    out.to_netcdf(dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print(f"[DONE] {dst}: {t[0]} -> {t[-1]} ({len(t)} h, no gaps)")


if __name__ == "__main__":
    main()
