#!/usr/bin/env python3
"""
Gauge-based hourly precipitation grid for Catalonia (Oct 2026) from the SMC
XEMA 30-min network, to be checked BEFORE it forces wflow as the reference.

Method (first version, deliberately simple and transparent):
  * every 30-min step, inverse-distance weighting (power 2) of the stations
    that reported at that step, nearest K=8 within 25 km; distances on a local
    equirectangular projection (km)
  * all stations in the download box are used, so the domain edges are not
    starved; output on a 1/120 deg grid (the wflow/MERIT resolution) over the
    coastal Besos..Tordera area
  * summed to hourly, end-stamped (XEMA timestamps are UTC; whether they mark
    the start or the end of the 30 min is not documented -- at most a 30 min
    timing ambiguity, assumed here to be the end)

Quality check: leave-one-out cross-validation at every station inside the
grid area -- each station is re-predicted from the others, for the event total
and the hourly series -- and the same scores for ERA5 at the same stations.

Outputs:
  data/raw/catalonia_2026/obs/xema_grid_hourly.nc        precip (mm/h)
  figures/catalonia_2026/forcing_validation/
      xema_grid_event_total.png       grid vs ERA5 totals, stations overlaid
      xema_grid_loo.png               LOO scatter (totals) + hourly scores
      xema_grid_loo.csv               per-station LOO and ERA5 scores

    python scripts/xema_grid.py
"""
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "raw", "catalonia_2026")
OUTF = os.path.join(REPO, "figures", "catalonia_2026", "forcing_validation")
T0, T1 = pd.Timestamp("2026-09-15"), pd.Timestamp("2026-10-10")       # grid period: wflow reference from Sep 15 (MSWEP NRT ends Sep 30)
E0, E1 = pd.Timestamp("2026-10-03"), pd.Timestamp("2026-10-06")       # event-total window
BOX = (1.90, 41.33, 2.95, 42.00)                                      # lon0, lat0, lon1, lat1
RES = 1.0 / 120.0
K, RMAX, POW = 8, 25.0, 2.0
LAT0 = 41.65


def xy(lon, lat):
    """Local equirectangular km."""
    return np.c_[(np.asarray(lon) - 2.4) * 111.32 * np.cos(np.radians(LAT0)), (np.asarray(lat) - LAT0) * 110.57]


def idw(src_xy, src_val, dst_xy):
    """IDW of one time step; src_val may hold NaN (station did not report)."""
    ok = np.isfinite(src_val)
    if ok.sum() == 0:
        return np.full(len(dst_xy), np.nan)
    tree = cKDTree(src_xy[ok])
    k = min(K, ok.sum())
    d, i = tree.query(dst_xy, k=k, distance_upper_bound=RMAX)
    d, i = np.atleast_2d(d.T).T, np.atleast_2d(i.T).T
    v = np.where(np.isfinite(d), src_val[ok][np.minimum(i, ok.sum() - 1)], np.nan)
    w = np.where(np.isfinite(d), 1.0 / np.maximum(d, 0.05) ** POW, 0.0)
    ws = w.sum(1)
    return np.where(ws > 0, np.nansum(w * np.nan_to_num(v), 1) / np.where(ws > 0, ws, 1), np.nan)


def load_xema():
    d = pd.read_csv(os.path.join(RAW, "xema", "xema_precip_30min_2026-09-14_2026-10-10.csv"), parse_dates=["time_utc"])
    s = pd.read_csv(os.path.join(RAW, "xema", "xema_stations.csv")).rename(
        columns={"codi_estacio": "station", "nom_estacio": "name", "latitud": "lat", "longitud": "lon", "altitud": "elev"})
    d = d[(d.time_utc >= T0 - pd.Timedelta("30min")) & (d.time_utc < T1)]
    p = d.pivot_table(index="time_utc", columns="station", values="precip_mm", aggfunc="first")
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="30min"))
    s = s[s.station.isin(p.columns)].set_index("station")
    return p[s.index], s


def era5_at(s, t0, t1):
    ds = xr.open_dataset(os.path.join(RAW, "era5", "era5_tp_catalonia_2026.nc"))
    tn = "valid_time" if "valid_time" in ds.dims else "time"
    v = ds["tp"]
    v = v.rename({tn: "time"})
    if v.attrs.get("units", "m") in ("m", "metre", "meter") or float(v.max()) < 1:
        v = v * 1000.0
    lat = "latitude" if "latitude" in v.dims else "lat"
    lon = "longitude" if "longitude" in v.dims else "lon"
    tot = v.sel(time=slice(t0 + pd.Timedelta("1h"), t1)).sum("time")
    pts = tot.interp({lat: xr.DataArray(s.lat.values, dims="pt"), lon: xr.DataArray(s.lon.values, dims="pt")}, method="nearest")
    hourly = v.sel(time=slice(t0 + pd.Timedelta("1h"), t1)).interp(
        {lat: xr.DataArray(s.lat.values, dims="pt"), lon: xr.DataArray(s.lon.values, dims="pt")}, method="nearest")
    return pts.values, pd.DataFrame(hourly.values, index=pd.DatetimeIndex(hourly.time.values), columns=s.index), tot


def scores(o, f):
    m = np.isfinite(o) & np.isfinite(f)
    o, f = o[m], f[m]
    if len(o) < 3:
        return dict(bias_pct=np.nan, r=np.nan, rmse=np.nan)
    return dict(bias_pct=100 * (f.sum() - o.sum()) / max(o.sum(), 1e-9), r=np.corrcoef(o, f)[0, 1],
                rmse=float(np.sqrt(np.mean((f - o) ** 2))))


def main():
    os.makedirs(OUTF, exist_ok=True)
    p, s = load_xema()
    sxy = xy(s.lon, s.lat)
    lons = np.arange(BOX[0] + RES / 2, BOX[2], RES)
    lats = np.arange(BOX[1] + RES / 2, BOX[3], RES)
    LO, LA = np.meshgrid(lons, lats)
    gxy = xy(LO.ravel(), LA.ravel())

    # 30-min grids -> hourly (end-stamped: the hour ending at t holds t-30min and t)
    vals = p.values
    g30 = np.stack([idw(sxy, vals[t], gxy) for t in range(len(p))]).reshape(len(p), len(lats), len(lons))
    da30 = xr.DataArray(g30, dims=("time", "lat", "lon"), coords={"time": p.index, "lat": lats, "lon": lons})
    hourly = da30.resample(time="1h", label="right", closed="right").sum(min_count=2)
    hourly = hourly.sel(time=slice(T0 + pd.Timedelta("1h"), T1))
    hourly.attrs = {"units": "mm", "long_name": "XEMA IDW precipitation in the hour ending at time",
                    "method": f"IDW power {POW}, nearest {K} within {RMAX} km, per 30-min step"}
    out = os.path.join(RAW, "obs", "xema_grid_hourly.nc")
    hourly.to_dataset(name="precip").to_netcdf(out, encoding={"precip": {"zlib": True}})
    print(out, dict(hourly.sizes))

    # leave-one-out at stations inside the grid area
    inside = s[(s.lon.between(BOX[0], BOX[2])) & (s.lat.between(BOX[1], BOX[3]))].index
    ev = (p.index > E0) & (p.index <= E1)
    obs_tot = p.loc[ev].sum(min_count=int(0.9 * ev.sum()))
    e5_tot, e5_h, e5_grid = era5_at(s.loc[inside], E0, E1)
    ph = p.resample("1h", label="right", closed="right").sum(min_count=2)
    lo = np.where((p.index > E0 - pd.Timedelta("1h")) & (p.index <= E1))[0]   # LOO only over the event window
    rows = []
    for j, st in enumerate(inside):
        k = list(s.index).index(st)
        keep = np.ones(len(s), bool); keep[k] = False
        pred30 = np.array([idw(sxy[keep], vals[t, keep], sxy[k:k + 1])[0] for t in lo])
        pr = pd.Series(pred30, index=p.index[lo])
        prh = pr.resample("1h", label="right", closed="right").sum(min_count=2)
        o_h = ph[st].loc[(ph.index > E0) & (ph.index <= E1)]
        f_h = prh.loc[o_h.index]
        e_h = e5_h[st].reindex(o_h.index)
        r = {"station": st, "name": s.loc[st, "name"], "lon": s.loc[st, "lon"], "lat": s.loc[st, "lat"],
             "obs_total": obs_tot.get(st, np.nan), "loo_total": pr.loc[(pr.index > E0) & (pr.index <= E1)].sum(), "era5_total": e5_tot[j]}
        r.update({f"loo_h_{k_}": v for k_, v in scores(o_h.values, f_h.values).items()})
        r.update({f"era5_h_{k_}": v for k_, v in scores(o_h.values, e_h.values).items()})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUTF, "xema_grid_loo.csv"), index=False)
    ok = t.obs_total.notna()
    summ = {
        "LOO totals": scores(t.obs_total[ok].values, t.loo_total[ok].values),
        "ERA5 totals": scores(t.obs_total[ok].values, t.era5_total[ok].values),
        "LOO hourly (median over stations)": t[[c for c in t if c.startswith("loo_h_")]].median().round(2).to_dict(),
        "ERA5 hourly (median over stations)": t[[c for c in t if c.startswith("era5_h_")]].median().round(2).to_dict(),
    }
    for k_, v in summ.items():
        print(f"{k_:36s}", {a: round(b, 2) for a, b in v.items()})

    # figures
    tot = hourly.sel(time=slice(E0 + pd.Timedelta("1h"), E1)).sum("time")
    vmax = float(np.nanpercentile(t.obs_total, 99)) * 1.05
    fig, ax = plt.subplots(1, 2, figsize=(18 / 2.54, 8 / 2.54), sharey=True)
    for a, field, title in ((ax[0], tot, "XEMA IDW grid (1 km)"), (ax[1], e5_grid, "ERA5 (0.25°)")):
        lat = "lat" if "lat" in field.dims else "latitude"
        lon = "lon" if "lon" in field.dims else "longitude"
        im = a.pcolormesh(field[lon], field[lat], field, cmap="YlGnBu", vmin=0, vmax=vmax, shading="auto")
        a.scatter(s.loc[inside, "lon"], s.loc[inside, "lat"], c=obs_tot.reindex(inside), cmap="YlGnBu", vmin=0, vmax=vmax,
                  edgecolor="k", lw=0.4, s=16, zorder=3)
        a.set_xlim(BOX[0], BOX[2]); a.set_ylim(BOX[1], BOX[3]); a.set_title(title, fontsize=7, loc="left")
        a.tick_params(labelsize=6)
    bfile = os.path.join(REPO, "data/wflow/catalonia_2026/model/staticgeoms/basins.geojson")
    if os.path.exists(bfile):
        import geopandas as gpd
        b = gpd.read_file(bfile)
        for a in ax:
            b.boundary.plot(ax=a, color="k", lw=0.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("event total 3–5 Oct (mm)", fontsize=6); cb.ax.tick_params(labelsize=6)
    fig.suptitle("Catalonia 3–5 Oct 2026: gauge-based grid vs ERA5 (dots = XEMA gauges, same colour scale)", fontsize=7.5)
    fig.savefig(os.path.join(OUTF, "xema_grid_event_total.png"), dpi=220, bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(16 / 2.54, 7.5 / 2.54))
    m = max(t.obs_total.max(), t.loo_total.max(), t.era5_total.max()) * 1.05
    ax[0].scatter(t.obs_total, t.loo_total, s=12, label="XEMA grid (leave-one-out)", color="#1f6f8b")
    ax[0].scatter(t.obs_total, t.era5_total, s=12, label="ERA5", color="#888", marker="s")
    ax[0].plot([0, m], [0, m], "k-", lw=0.6)
    ax[0].set_xlim(0, m); ax[0].set_ylim(0, m)
    ax[0].set_xlabel("observed event total (mm)", fontsize=6.5); ax[0].set_ylabel("estimate at the gauge (mm)", fontsize=6.5)
    ax[0].legend(fontsize=5.5, frameon=False); ax[0].tick_params(labelsize=6)
    ax[0].set_title("(a) event totals at withheld gauges", fontsize=7, loc="left")
    d = pd.DataFrame({"XEMA grid\n(leave-one-out)": t.loo_h_r, "ERA5": t.era5_h_r})
    ax[1].boxplot([d[c].dropna() for c in d], tick_labels=d.columns, widths=0.5)
    ax[1].set_ylabel("hourly correlation with the gauge", fontsize=6.5); ax[1].tick_params(labelsize=6)
    ax[1].set_title("(b) hourly series, 3–5 Oct", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTF, "xema_grid_loo.png"), dpi=220, bbox_inches="tight")
    print(OUTF)


if __name__ == "__main__":
    main()
