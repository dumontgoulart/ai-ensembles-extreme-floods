#!/usr/bin/env python3
"""
Gauge-based hourly precipitation grid for an event from its hourly station
data, with a leave-one-out check, to be reviewed BEFORE it forces wflow as the
reference (step 3 of the extreme-gauge framework). Generalises xema_grid.py
(Catalonia, 30-min XEMA) to the events with hourly station networks.

Method (as xema_grid.py): every hour, inverse-distance weighting (power 2) of
the stations that reported, nearest K within RMAX km, on a 1/120 deg grid over
the model basins (+ margin); values end-stamped (hour ending at time). No
elevation dependence: in mountain basins IDW from valley stations is a lower
bound for the ridges -- the leave-one-out scores and the map show how much.

Checks:
  * leave-one-out at every station inside the grid box: event total and hourly
    correlation, the same at ERA5 (nearest cell)
  * timing: correlation of the station-mean hourly series with ERA5 at the
    stations for lags -2..+2 h (a non-zero best lag means the station stamps
    are not end-of-hour)

Outputs:
  data/raw/<slug>/obs/gauge_grid_hourly.nc
  figures/<slug>/forcing_validation/gauge_grid_event_total.png, gauge_grid_loo.png,
                                    gauge_grid_loo.csv

    python scripts/gauge_grid.py boris_sudetes
"""
import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_config import EVENTS                   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, POW, MARGIN = 1.0 / 120.0, 2.0, 0.15

# per event: station table (columns renamed to station/lat/lon), hourly file,
# IDW neighbours and radius (km), ERA5 file, basin outline, label
CFG = {
    "boris_sudetes": dict(stations="precip_stations.csv", hourly="precip_stations_hourly.csv",
                          K=8, RMAX=30.0, era5="data/processed/boris_2024/era5_processed.nc",
                          basins="data/wflow/boris_sudetes/model/staticgeoms/basins.geojson",
                          label="CHMI + IMGW gauge grid"),
    "zermatt_2024": dict(stations="precip_stations.csv", hourly="precip_stations_hourly.csv",
                         K=6, RMAX=60.0, era5="data/processed/zermatt_2024/era5_processed.nc",
                         basins="data/wflow/zermatt_2024/model/staticgeoms/basins.geojson",
                         label="MeteoSwiss gauge grid"),
    "hans_2023": dict(stations="frost_stations.csv", hourly="frost_precip_hourly.csv",
                      K=8, RMAX=50.0, era5="data/processed/hans_2023/era5_processed.nc",
                      basins="data/wflow/hans_2023/model/staticgeoms/basins_highres.geojson",
                      label="MET Norway (Frost) gauge grid", rename={"id": "station"}),
}


def to_km(lon, lat, lon0, lat0):
    return np.c_[(np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 110.57]


def idw(src_xy, src_val, dst_xy, K, RMAX):
    ok = np.isfinite(src_val)
    if ok.sum() == 0:
        return np.full(len(dst_xy), np.nan)
    tree = cKDTree(src_xy[ok])
    k = min(K, int(ok.sum()))
    d, i = tree.query(dst_xy, k=k, distance_upper_bound=RMAX)
    d, i = np.atleast_2d(d.T).T, np.atleast_2d(i.T).T
    v = np.where(np.isfinite(d), src_val[ok][np.minimum(i, ok.sum() - 1)], np.nan)
    w = np.where(np.isfinite(d), 1.0 / np.maximum(d, 0.05) ** POW, 0.0)
    ws = w.sum(1)
    return np.where(ws > 0, np.nansum(w * np.nan_to_num(v), 1) / np.where(ws > 0, ws, 1), np.nan)


def scores(o, f):
    m = np.isfinite(o) & np.isfinite(f)
    o, f = o[m], f[m]
    if len(o) < 3 or o.std() == 0 or f.std() == 0:
        return dict(bias_pct=np.nan, r=np.nan, rmse=np.nan)
    return dict(bias_pct=100 * (f.sum() - o.sum()) / max(o.sum(), 1e-9), r=np.corrcoef(o, f)[0, 1],
                rmse=float(np.sqrt(np.mean((f - o) ** 2))))


def load(slug, c):
    obs = os.path.join(REPO, "data", "raw", slug, "obs")
    s = pd.read_csv(os.path.join(obs, c["stations"])).rename(columns=c.get("rename", {}))
    s = s.drop_duplicates("station").set_index("station")
    d = pd.read_csv(os.path.join(obs, c["hourly"]))
    d["time"] = pd.to_datetime(d["time_utc"], utc=True).dt.tz_localize(None)
    d.loc[(d.precip_mm < 0) | (d.precip_mm > 250), "precip_mm"] = np.nan        # physically impossible hourly values
    p = d.pivot_table(index="time", columns="station", values="precip_mm", aggfunc="first")
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="1h"))
    s = s[s.index.isin(p.columns)]
    return p[s.index], s


def era5_hourly(path, lats, lons):
    ds = xr.open_dataset(os.path.join(REPO, path))
    v = ds["tp"]
    tn = "valid_time" if "valid_time" in v.dims else "time"
    v = v.rename({tn: "time"})
    if v.attrs.get("units", "mm") in ("m", "metre", "meter"):
        v = v * 1000.0
    lat = "latitude" if "latitude" in v.dims else "lat"
    lon = "longitude" if "longitude" in v.dims else "lon"
    pts = v.interp({lat: xr.DataArray(lats, dims="pt"), lon: xr.DataArray(lons, dims="pt")}, method="nearest")
    return pd.DataFrame(pts.transpose("time", "pt").values, index=pd.DatetimeIndex(pts.time.values)), v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slug")
    slug = ap.parse_args().slug
    c, ev = CFG[slug], EVENTS[slug]
    outf = os.path.join(REPO, "figures", slug, "forcing_validation")
    os.makedirs(outf, exist_ok=True)
    p, s = load(slug, c)
    basins = gpd.read_file(os.path.join(REPO, c["basins"])).to_crs(4326)
    b0, b1, b2, b3 = basins.total_bounds
    box = (b0 - MARGIN, b1 - MARGIN, b2 + MARGIN, b3 + MARGIN)
    lon0, lat0 = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    sxy = to_km(s.lon, s.lat, lon0, lat0)
    lons = np.arange(np.floor(box[0] * 120) / 120 + RES / 2, box[2], RES)
    lats = np.arange(np.floor(box[1] * 120) / 120 + RES / 2, box[3], RES)
    LO, LA = np.meshgrid(lons, lats)
    gxy = to_km(LO.ravel(), LA.ravel(), lon0, lat0)
    K, RMAX = c["K"], c["RMAX"]

    vals = p.values
    g = np.stack([idw(sxy, vals[t], gxy, K, RMAX) for t in range(len(p))]).reshape(len(p), len(lats), len(lons))
    grid = xr.DataArray(g.astype("float32"), dims=("time", "lat", "lon"),
                        coords={"time": p.index, "lat": lats, "lon": lons})
    grid.attrs = {"units": "mm", "long_name": f"{c['label']}: IDW precipitation in the hour ending at time",
                  "method": f"IDW power {POW}, nearest {K} within {RMAX} km, hourly",
                  "stations": int(len(s))}
    out = os.path.join(REPO, "data", "raw", slug, "obs", "gauge_grid_hourly.nc")
    grid.to_dataset(name="precip").to_netcdf(out, encoding={"precip": {"zlib": True}})
    empty = int(grid.isnull().all(["lat", "lon"]).sum())
    print(f"{out}: {dict(grid.sizes)}, {p.index[0]} -> {p.index[-1]}, {len(s)} stations, empty hours {empty}")

    # event window and stations inside the box
    E0, E1 = pd.Timestamp(ev["event_start"]), pd.Timestamp(ev["event_end"]).ceil("1h")
    ev_idx = (p.index > E0) & (p.index <= E1)
    inside = s[s.lon.between(box[0], box[2]) & s.lat.between(box[1], box[3])].index
    obs_tot = p.loc[ev_idx].sum(min_count=int(0.9 * ev_idx.sum()))
    e5, e5_field = era5_hourly(c["era5"], s.loc[inside, "lat"].values, s.loc[inside, "lon"].values)
    e5.columns = inside

    # timing: station-mean vs ERA5-at-stations, lags -2..+2 h (positive = stations late)
    om = p.loc[p.index.isin(e5.index), inside].mean(axis=1)
    em = e5.reindex(om.index).mean(axis=1)
    lag_r = {L: om.shift(-L).corr(em) for L in range(-2, 3)}
    best = max(lag_r, key=lag_r.get)
    print("timing check, r by lag (h, + = stations late vs ERA5):", {k: round(v, 3) for k, v in lag_r.items()},
          "-> best", best)

    lo = np.where(ev_idx)[0]
    rows = []
    for st in inside:
        k = list(s.index).index(st)
        keep = np.ones(len(s), bool); keep[k] = False
        pred = np.array([idw(sxy[keep], vals[t, keep], sxy[k:k + 1], K, RMAX)[0] for t in lo])
        o_h = p[st].values[lo]
        e_h = e5[st].reindex(p.index[lo]).values
        r = {"station": st, "name": s.loc[st].get("name", ""), "lon": s.loc[st, "lon"], "lat": s.loc[st, "lat"],
             "obs_total": obs_tot.get(st, np.nan), "loo_total": np.nansum(pred), "era5_total": np.nansum(e_h)}
        r.update({f"loo_h_{a}": v for a, v in scores(o_h, pred).items()})
        r.update({f"era5_h_{a}": v for a, v in scores(o_h, e_h).items()})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(outf, "gauge_grid_loo.csv"), index=False)
    ok = t.obs_total.notna()
    print(f"leave-one-out, {ok.sum()} stations with a complete event window ({E0} -> {E1}):")
    for name, col in (("grid (LOO) totals", "loo_total"), ("ERA5 totals", "era5_total")):
        print(f"  {name:20s}", {a: round(v, 2) for a, v in scores(t.obs_total[ok].values, t[col][ok].values).items()})
    print("  hourly r, median over stations: grid (LOO)", round(t.loo_h_r.median(), 2), " ERA5", round(t.era5_h_r.median(), 2))

    # figures
    tot = grid.sel(time=slice(E0 + pd.Timedelta("1h"), E1)).sum("time", min_count=1)
    vmax = float(np.nanpercentile(t.obs_total, 99)) * 1.05 if ok.any() else float(tot.max())
    e5tot = e5_field.sel(time=slice(E0 + pd.Timedelta("1h"), E1)).sum("time")
    fig, ax = plt.subplots(1, 2, figsize=(18 / 2.54, 8 / 2.54), sharey=True)
    for a, field, title in ((ax[0], tot, c["label"] + " (1 km)"), (ax[1], e5tot, "ERA5 (0.25°)")):
        la = "lat" if "lat" in field.dims else "latitude"
        lo_ = "lon" if "lon" in field.dims else "longitude"
        im = a.pcolormesh(field[lo_], field[la], field, cmap="YlGnBu", vmin=0, vmax=vmax, shading="auto")
        a.scatter(s.loc[inside, "lon"], s.loc[inside, "lat"], c=obs_tot.reindex(inside), cmap="YlGnBu",
                  vmin=0, vmax=vmax, edgecolor="k", lw=0.4, s=16, zorder=3)
        basins.boundary.plot(ax=a, color="k", lw=0.6)
        a.set_xlim(box[0], box[2]); a.set_ylim(box[1], box[3]); a.set_title(title, fontsize=7, loc="left")
        a.set_aspect(1 / np.cos(np.radians(lat0)))
        a.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(f"event total {E0:%d %b}–{E1:%d %b} (mm)", fontsize=6); cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"{ev['name']}: gauge grid vs ERA5 (dots = gauges, same colour scale)", fontsize=7.5)
    fig.savefig(os.path.join(outf, "gauge_grid_event_total.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(16 / 2.54, 7.5 / 2.54))
    m = np.nanmax([t.obs_total.max(), t.loo_total.max(), t.era5_total.max()]) * 1.05
    ax[0].scatter(t.obs_total, t.loo_total, s=12, label="gauge grid (leave-one-out)", color="#1f6f8b")
    ax[0].scatter(t.obs_total, t.era5_total, s=12, label="ERA5", color="#888", marker="s")
    ax[0].plot([0, m], [0, m], "k-", lw=0.6); ax[0].set_xlim(0, m); ax[0].set_ylim(0, m)
    ax[0].set_xlabel("observed event total (mm)", fontsize=6.5); ax[0].set_ylabel("estimate at the gauge (mm)", fontsize=6.5)
    ax[0].legend(fontsize=5.5, frameon=False); ax[0].tick_params(labelsize=6)
    ax[0].set_title("(a) event totals at withheld gauges", fontsize=7, loc="left")
    d = pd.DataFrame({"gauge grid\n(leave-one-out)": t.loo_h_r, "ERA5": t.era5_h_r})
    ax[1].boxplot([d[k].dropna() for k in d], tick_labels=d.columns, widths=0.5)
    ax[1].set_ylabel("hourly correlation with the gauge", fontsize=6.5); ax[1].tick_params(labelsize=6)
    ax[1].set_title(f"(b) hourly series, {E0:%d}–{E1:%d %b}", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(outf, "gauge_grid_loo.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(outf)


if __name__ == "__main__":
    main()
