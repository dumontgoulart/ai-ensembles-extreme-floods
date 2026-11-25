#!/usr/bin/env python3
"""
Kristin: ERA5 rain scaled to the 11 IPMA gauges, in the format wflow_prep_forcing reads.

kristin_reference_rain.py builds the station-scaled ERA5 (one multiplicative field from
the gauge event totals, applied to every hour) on a 0.1 deg box for 21 Jan - 4 Feb. The
wflow ERA5 cases read data/processed/kristin_2026/era5_wflow.nc (hourly, end-stamped,
0.25 deg, 22 Jan - 7 Feb). Here the scaling field (scaled / raw ERA5 event total on the
0.1 deg grid, clipped to the 0.4-3 range of the build) is applied to every hour of
era5_wflow.nc interpolated to 0.1 deg, so the file covers the whole simulation and
equals the scaled reference inside its period.

    python scripts/kristin_era5_reference.py

Output: data/processed/kristin_2026/era5_wflow_stationscaled.nc (tp, mm per hour)
"""
import os

import numpy as np
import xarray as xr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "processed", "kristin_2026", "era5_wflow.nc")
SCALED = os.path.join(REPO, "data", "raw", "kristin_2026", "obs", "ref_rain_era5_rescaled_hourly.nc")
BOX = os.path.join(REPO, "data", "raw", "kristin_2026", "imerg", "era5_tp_1h_box.nc")
OUT = os.path.join(REPO, "data", "processed", "kristin_2026", "era5_wflow_stationscaled.nc")


def main():
    raw = xr.open_dataset(RAW)["tp"].load().sortby("latitude")
    sc = xr.open_dataset(SCALED)["precip"].load()
    # the scaling field, from the ERA5 the build used (same values as era5_wflow.nc)
    box = xr.open_dataset(BOX)["tp"].load().rename(valid_time="time").sortby("latitude")
    box01 = box.interp(latitude=sc.latitude, longitude=sc.longitude, method="linear")
    t = np.intersect1d(box01.time.values, sc.time.values)
    b_tot, s_tot = box01.sel(time=t).sum("time"), sc.sel(time=t).sum("time")
    k = (s_tot / b_tot).where(b_tot > 0.5, 1.0).clip(0.4, 3.0)
    # era5_wflow.nc starts at 37.75 N, 9.5 W: keep the 0.1 deg cells it covers (both sub-basins are well inside)
    sc = sc.sel(latitude=slice(float(raw.latitude.min()), None), longitude=slice(float(raw.longitude.min()), None))
    k, s_tot = k.sel(latitude=sc.latitude, longitude=sc.longitude), s_tot.sel(latitude=sc.latitude, longitude=sc.longitude)
    raw01 = raw.interp(latitude=sc.latitude, longitude=sc.longitude, method="linear")
    out = (raw01 * k).astype("float32").transpose("time", "latitude", "longitude").sortby("latitude", ascending=False)   # north-up like era5_wflow.nc
    t = np.intersect1d(t, out.time.values)       # era5_wflow.nc starts on 22 Jan
    diff = float(abs(out.sel(time=t).sum("time") - sc.sel(time=t).sum("time")).max())
    out.attrs = {"units": "mm", "long_name": "ERA5 hourly precipitation scaled to 11 IPMA gauges",
                 "note": "stamp = end of hour; scripts/kristin_era5_reference.py"}
    out.to_dataset(name="tp").to_netcdf(OUT, encoding={"tp": {"zlib": True, "complevel": 1}})
    print(f"scaling field {float(k.min()):.2f}-{float(k.max()):.2f} (median {float(k.median()):.2f}); "
          f"max |event total - scaled reference| {diff:.2f} mm over {len(t)} common hours")
    print(OUT)


if __name__ == "__main__":
    main()
