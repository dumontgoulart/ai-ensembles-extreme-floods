#!/usr/bin/env python3
"""
Zermatt 2024: rescale the MeteoSwiss gauge grid (scripts/gauge_grid.py) to RhiresD.

The station grid has the right timing (daily r 0.91 against RhiresD) but is 40 %
too dry over the Vispa basin, 53 % on the Saas side (scripts/rhiresd_check.py),
because the hourly SwissMetNet stations sit in the valleys. RhiresD (MeteoSwiss,
1 km, daily 06-06 UTC, built from the full climatological network) has the ridge
rain. Here each 06-06 UTC day is set to the RhiresD total in every cell, and the
hourly distribution within the day comes from the station grid:

    new(h, cell) = RhiresD(day, cell) * f(h, cell)
    f = grid hourly fraction of the cell's day total (cell total >= 0.2 mm);
        otherwise the fraction of the grid's basin-mean hourly total that day
        (a day with no station rain over a cell still gets RhiresD rain, timed
        like the rest of the basin)
Cells without RhiresD (outside Switzerland) keep the original grid.

Caveat: RhiresD is itself interpolated and has gauge undercatch at height, so
it is a better, not a perfect, reference; the result is a lower bound on the
ridge rain.

Output: data/raw/zermatt_2024/obs/gauge_grid_hourly_rhiresd.nc (same grid and
end-stamped hourly mm as gauge_grid_hourly.nc), figures/zermatt_2024/rhiresd_check/
    rescale_check.csv, rescale_maps.png, rescale_hourly.png

    python scripts/rhiresd_rescale.py
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = os.path.join(REPO, "data/raw/zermatt_2024/obs")
OUT = os.path.join(REPO, "figures/zermatt_2024/rhiresd_check")
MIN_DAY_MM = 0.2


def main():
    os.makedirs(OUT, exist_ok=True)
    gg = xr.open_dataset(os.path.join(OBS, "gauge_grid_hourly.nc"))["precip"].load()
    lon, lat = gg.lon.values, gg.lat.values
    LON, LAT = np.meshgrid(lon, lat)

    # RhiresD onto the grid's lon/lat: grid points -> LV95, bilinear in (N, E)
    rh = xr.open_dataset(os.path.join(OBS, "rhiresd_2024_jun.nc"))["precip"].load()
    pts = gpd.GeoSeries(gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs=4326).to_crs(2056)
    pq = np.c_[pts.y.values, pts.x.values]

    # 06-06 UTC day label (start) of every end-stamped hourly slot
    t = pd.DatetimeIndex(gg.time.values)
    day = (t - pd.Timedelta("1s") - pd.Timedelta("6h")).floor("D")
    days = pd.DatetimeIndex(sorted(set(day)))
    new = gg.copy(deep=True).values
    log = []
    for d in days:
        if d not in pd.DatetimeIndex(rh.time.values):
            continue
        idx = np.where(day == d)[0]
        if len(idx) < 24:                        # incomplete day at the edge of the grid period: keep
            continue
        r = rh.sel(time=d).values                # (N, E)
        f = RegularGridInterpolator((rh.N.values, rh.E.values), np.nan_to_num(r, nan=-1.0),
                                    bounds_error=False, fill_value=-1.0)
        Rd = f(pq).reshape(LON.shape)
        valid = Rd >= 0
        h = gg.values[idx]                        # (24, y, x)
        tot = h.sum(0)
        basin_pat = h.mean((1, 2)); bp_sum = basin_pat.sum()
        pat_basin = (basin_pat / bp_sum) if bp_sum > 0 else np.full(len(idx), 1 / len(idx))
        frac = np.where(tot[None] >= MIN_DAY_MM, h / np.where(tot > 0, tot, 1)[None], pat_basin[:, None, None])
        sc = np.where(valid[None], np.maximum(Rd, 0)[None] * frac, h)
        new[idx] = sc
        log.append({"day": d, "grid_mean_mm": float(tot[valid].mean()), "rhiresd_mean_mm": float(Rd[valid].mean()),
                    "new_mean_mm": float(sc.sum(0)[valid].mean()), "cells_pattern_fallback": int(((tot < MIN_DAY_MM) & valid & (Rd > MIN_DAY_MM)).sum())})
    out = gg.copy(data=new.astype("float32"))
    out.attrs.update(gg.attrs)
    out.attrs["history"] = "scaled to MeteoSwiss RhiresD daily totals (06-06 UTC) with the station grid's hourly pattern; scripts/rhiresd_rescale.py"
    path = os.path.join(OBS, "gauge_grid_hourly_rhiresd.nc")
    enc = {"precip": {"zlib": True, "complevel": 1, "dtype": "float32"}}
    out.to_dataset(name="precip").to_netcdf(path, encoding=enc)
    lg = pd.DataFrame(log)
    lg.to_csv(os.path.join(OUT, "rescale_check.csv"), index=False, float_format="%.2f")
    print(lg.round(1).to_string(index=False))
    print(f"event totals over the grid (mm, mean of cells): original {gg.sum('time').mean().item():.0f}, rescaled {out.sum('time').mean().item():.0f}")

    # maps and a hydrologically relevant time series (basin-mean hourly)
    basin = gpd.read_file(os.path.join(REPO, "data/wflow/zermatt_2024/model/staticgeoms/basins.geojson")).to_crs(4326)
    inside = gpd.GeoSeries(gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs=4326).within(basin.union_all()).values.reshape(LON.shape)
    fig, axs = plt.subplots(1, 3, figsize=(18 / 2.54, 6.5 / 2.54), constrained_layout=True)
    a, b = gg.sum("time").values, out.sum("time").values
    vmax = np.nanpercentile(b, 99)
    for ax, z, ttl in zip(axs[:2], (a, b), ("Station grid (original)", "Rescaled to RhiresD")):
        pc = ax.pcolormesh(lon, lat, z, vmin=0, vmax=vmax, cmap="Blues", shading="nearest")
        basin.boundary.plot(ax=ax, color="k", lw=0.6); ax.set_title(ttl, fontsize=7); ax.tick_params(labelsize=5.5)
    fig.colorbar(pc, ax=axs[:2], shrink=0.8, label="total 10 Jun-11 Jul (mm)")
    pc2 = axs[2].pcolormesh(lon, lat, (b - a), cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    basin.boundary.plot(ax=axs[2], color="k", lw=0.6); axs[2].set_title("Change (mm)", fontsize=7); axs[2].tick_params(labelsize=5.5)
    fig.colorbar(pc2, ax=axs[2], shrink=0.8)
    fig.savefig(os.path.join(OUT, "rescale_maps.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18 / 2.54, 5 / 2.54), constrained_layout=True)
    sl = slice("2024-06-20", "2024-07-02")
    for da, lab, col in ((gg, "station grid", "#999999"), (out, "rescaled to RhiresD", "#1f77b4")):
        s = pd.Series(da.values[:, inside].mean(1), index=da.time.values).loc[sl]
        ax.plot(s.index, s.values, label=lab, color=col, lw=0.9)
    ax.set_ylabel("basin-mean rain (mm/h)", fontsize=6); ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=5.5)
    fig.savefig(os.path.join(OUT, "rescale_hourly.png"), dpi=200)
    print(path)


if __name__ == "__main__":
    main()
