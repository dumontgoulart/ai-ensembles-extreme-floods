#!/usr/bin/env python3
"""
MRMS MultiSensor QPE Pass2 checked against, and scaled to, independent daily gauges (US events).

Same method as california_mrms_gaugescaled.py: CoCoRaHS and COOP daily gauges (GHCN-D; not
ingested by MRMS Pass2) give gauge/MRMS ratios of the period total; one multiplicative field per
flood period (inverse-distance weighted log ratio, nearest 8, damped towards 1 with distance to
the nearest gauge, capped at 0.33-3) is applied to every MRMS hour of that period. Hours outside
the periods are left unscaled. Gauge day = the 24 h ending at OBS_HOUR UTC (07 local) on DATE;
only stations complete over a period; values with a quality flag are dropped. Leave-one-out at
every gauge says how well the scaled product does where there is no gauge.

    python scripts/mrms_gaugescaled.py fetch <slug>   # GHCN-D stations + daily PRCP (AWS mirror)
    python scripts/mrms_gaugescaled.py check <slug>   # MRMS vs gauges, per scored gauge area
    python scripts/mrms_gaugescaled.py build <slug>   # scaled product + leave-one-out

Outputs: data/raw/<slug>/obs/ghcnd_{stations,prcp}.csv, mrms_qpe_1h_gaugescaled.nc;
figures/<slug>/reference_rain/ (mrms_gauge_check.csv, mrms_gauge_loo.csv, mrms_gauge_summary.csv)
"""
import gzip
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = "https://noaa-ghcn-pds.s3.amazonaws.com"
CACHE = os.path.join(REPO, "data", "raw", "_return_periods")
# gauge DATEs (inclusive) per flood period; OBS_HOUR = 07 local in UTC
EVENTS = {"texas_2025":       {"periods": {"P1": ("2025-07-03", "2025-07-07")}, "obs_hour": 12},
          "texas_2026":       {"periods": {"P1": ("2026-07-14", "2026-07-18")}, "obs_hour": 12},
          "mississippi_2026": {"periods": {"P1": ("2026-06-16", "2026-06-21")}, "obs_hour": 12},
          "helene_2024":      {"periods": {"P1": ("2024-09-25", "2024-09-28")}, "obs_hour": 11}}
K, POW, DAMP_KM, CAP = 8, 2.0, 40.0, (0.33, 3.0)
MARGIN = 0.3   # deg around the MRMS box for stations


def paths(slug):
    obs = os.path.join(REPO, "data", "raw", slug, "obs")
    fig = os.path.join(REPO, "figures", slug, "reference_rain")
    return obs, fig


def km(lat, lon, lat0, lon0):
    return np.c_[(np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 110.57]


def field(xy, logr, dst):
    tree = cKDTree(xy)
    d, i = tree.query(dst, k=min(K, len(xy)))
    d, i = d.reshape(len(dst), -1), i.reshape(len(dst), -1)
    w = 1.0 / np.maximum(d, 0.05) ** POW
    f = (w * logr[i]).sum(1) / w.sum(1) * np.exp(-(d[:, 0] / DAMP_KM) ** 2)
    return np.clip(np.exp(f), *CAP)


def _get(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def fetch(slug):
    obs, _ = paths(slug)
    mr = xr.open_dataset(os.path.join(obs, "mrms_qpe_1h.nc"))
    la0, la1 = float(mr.latitude.min()) - MARGIN, float(mr.latitude.max()) + MARGIN
    lo0, lo1 = float(mr.longitude.min()) - MARGIN, float(mr.longitude.max()) + MARGIN
    f = os.path.join(CACHE, "ghcnd-stations.txt")
    if not os.path.exists(f):
        open(f, "wb").write(_get(f"{MIRROR}/ghcnd-stations.txt"))
    st = pd.read_fwf(f, colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)],
                     names=["id", "lat", "lon", "el", "state", "name"])
    st = st[st.lat.between(la0, la1) & st.lon.between(lo0, lo1) & st.id.str.startswith("US")]
    ids = set(st.id)
    rows = []
    for year in sorted({d[:4] for p in EVENTS[slug]["periods"].values() for d in p}):
        print(f"streaming GHCN-D {year} ({len(ids)} stations in the box)")
        req = urllib.request.urlopen(f"{MIRROR}/csv.gz/by_year/{year}.csv.gz", timeout=1800)
        with gzip.open(req, "rt") as fh:
            for line in fh:
                if line[:11] in ids and line[21:25] == "PRCP":
                    rows.append(line)
    pr = pd.read_csv(io.StringIO("".join(rows)), header=None,
                     names=["STATION", "DATE", "ELEMENT", "PRCP", "MFLAG", "QFLAG", "SFLAG", "OBS_TIME"])
    pr["DATE"] = pd.to_datetime(pr.DATE.astype(str))
    pr["PRCP"] = pr.PRCP / 10.0                                                     # tenths of mm -> mm
    lo = min(pd.Timestamp(p[0]) for p in EVENTS[slug]["periods"].values()) - pd.Timedelta("5D")
    hi = max(pd.Timestamp(p[1]) for p in EVENTS[slug]["periods"].values()) + pd.Timedelta("5D")
    pr = pr[pr.DATE.between(lo, hi)].drop(columns="ELEMENT")
    st = st[st.id.isin(pr.STATION)]
    st.assign(net=st.id.str[2]).to_csv(os.path.join(obs, "ghcnd_stations.csv"), index=False)
    pr.to_csv(os.path.join(obs, "ghcnd_prcp.csv"), index=False)
    print(f"{len(st)} stations with PRCP, {len(pr)} station-days -> {obs}/ghcnd_*.csv")


def period_totals(slug, mr):
    """Per period: complete, unflagged gauges with their total and MRMS at the gauge cell."""
    obs, _ = paths(slug)
    st = pd.read_csv(os.path.join(obs, "ghcnd_stations.csv"))
    lat, lon = mr.latitude.values, mr.longitude.values
    st = st[st.lat.between(lat.min(), lat.max()) & st.lon.between(lon.min(), lon.max())].set_index("id")
    pr = pd.read_csv(os.path.join(obs, "ghcnd_prcp.csv"), parse_dates=["DATE"])
    pr = pr[pr.QFLAG.isna() & pr.STATION.isin(st.index)]
    t = pd.DatetimeIndex(mr.time.values)
    out = {}
    for p, (d0, d1) in EVENTS[slug]["periods"].items():
        days = pd.date_range(d0, d1, freq="D")
        g = pr[pr.DATE.isin(days)].groupby("STATION").PRCP.agg(["sum", "count"])
        g = g[g["count"] == len(days)]["sum"]
        a = pd.Timestamp(d0) - pd.Timedelta("1D") + pd.Timedelta(hours=EVENTS[slug]["obs_hour"])
        b = pd.Timestamp(d1) + pd.Timedelta(hours=EVENTS[slug]["obs_hour"])
        hrs = np.where((t > a) & (t <= b))[0]
        tot = mr.isel(time=hrs).sum("time").values
        s = st.loc[g.index]
        ii = np.abs(lat[:, None] - s.lat.values[None]).argmin(0)
        jj = np.abs(lon[:, None] - s.lon.values[None]).argmin(0)
        m = tot[ii, jj]
        ok = (g.values > 1) & (m > 1)
        out[p] = {"hrs": hrs, "tot": tot, "stations": s[ok], "gauge": g.values[ok], "mrms": m[ok]}
    return out


def area_distance(slug, s):
    """Distance (km) of each station to every scored gauge area (0 inside)."""
    m = xr.open_dataset(os.path.join(REPO, "data", "wflow", slug, "gauge_upstream.nc"))["mask"].load()
    gd = [d for d in m.dims if d not in ("latitude", "longitude")][0]
    lat0, lon0 = float(m.latitude.mean()), float(m.longitude.mean())
    LON, LAT = np.meshgrid(m.longitude.values, m.latitude.values)
    xy = km(s.lat.values, s.lon.values, lat0, lon0)
    d = {}
    for g in m[gd].values:
        msk = m.sel({gd: g}).values.astype(bool)
        tree = cKDTree(km(LAT[msk], LON[msk], lat0, lon0))
        d[f"d_{g}"] = tree.query(xy)[0]
    return pd.DataFrame(d, index=s.index)


def check(slug):
    obs, fig = paths(slug)
    os.makedirs(fig, exist_ok=True)
    mr = xr.open_dataset(os.path.join(obs, "mrms_qpe_1h.nc"))["precip"].load()
    rows = []
    for p, v in period_totals(slug, mr).items():
        d = area_distance(slug, v["stations"])
        df = pd.DataFrame({"period": p, "name": v["stations"].name, "net": v["stations"].index.str[2],
                           "lat": v["stations"].lat, "lon": v["stations"].lon,
                           "gauge_mm": v["gauge"], "mrms_mm": v["mrms"]}).join(d)
        rows.append(df)
        print(f"\n{slug} {p}: {len(df)} complete gauges; all: MRMS/gauge ratio of sums "
              f"{df.mrms_mm.sum() / df.gauge_mm.sum():.2f}, median ratio {np.median(df.mrms_mm / df.gauge_mm):.2f}")
        heavy = df[df.gauge_mm >= df.gauge_mm.quantile(0.9)]
        print(f"  wettest 10 % of gauges (>= {heavy.gauge_mm.min():.0f} mm, n {len(heavy)}): "
              f"ratio of sums {heavy.mrms_mm.sum() / heavy.gauge_mm.sum():.2f}")
        for c in d.columns:
            near = df[df[c] <= 10]
            if len(near):
                print(f"  area {c[2:]:>10s}: {len(near):3d} gauges within 10 km, gauge mean {near.gauge_mm.mean():6.1f} mm, "
                      f"ratio of sums {near.mrms_mm.sum() / near.gauge_mm.sum():.2f}, "
                      f"range {(near.mrms_mm / near.gauge_mm).min():.2f}-{(near.mrms_mm / near.gauge_mm).max():.2f}")
    pd.concat(rows).to_csv(os.path.join(fig, "mrms_gauge_check.csv"), float_format="%.2f")


def build(slug):
    obs, fig = paths(slug)
    os.makedirs(fig, exist_ok=True)
    mr = xr.open_dataset(os.path.join(obs, "mrms_qpe_1h.nc"))["precip"].load()
    lat, lon = mr.latitude.values, mr.longitude.values
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    LON, LAT = np.meshgrid(lon, lat)
    dst = km(LAT.ravel(), LON.ravel(), lat0, lon0)
    out = mr.copy()
    rows, summ = [], []
    for p, v in period_totals(slug, mr).items():
        s = v["stations"]
        xy = km(s.lat.values, s.lon.values, lat0, lon0)
        logr = np.log(v["gauge"] / v["mrms"])
        f = field(xy, logr, dst).reshape(LAT.shape)
        out[v["hrs"]] = (mr.isel(time=v["hrs"]) * f[None]).astype("float32")
        d = area_distance(slug, s)
        for n in range(len(s)):
            keep = np.ones(len(s), bool)
            keep[n] = False
            fl = field(xy[keep], logr[keep], xy[n:n + 1])[0]
            rows.append({"period": p, "station": s.index[n], "name": s.name.values[n], "lat": s.lat.values[n],
                         "lon": s.lon.values[n], "gauge_mm": v["gauge"][n], "mrms_mm": v["mrms"][n],
                         "mrms_scaled_loo_mm": v["mrms"][n] * fl, **d.iloc[n].to_dict()})
        summ.append({"period": p, "gauges": len(s), "field_min": float(f.min()), "field_max": float(f.max()),
                     "hours": len(v["hrs"])})
    loo = pd.DataFrame(rows)
    loo.to_csv(os.path.join(fig, "mrms_gauge_loo.csv"), index=False, float_format="%.2f")
    res = []
    subsets = [(p, "all", loo[loo.period == p]) for p in EVENTS[slug]["periods"]]
    subsets += [(p, "wettest 10 %", loo[(loo.period == p) & (loo.gauge_mm >= loo[loo.period == p].gauge_mm.quantile(0.9))])
                for p in EVENTS[slug]["periods"]]
    subsets += [(p, c[2:], loo[(loo.period == p) & (loo[c] <= 20)]) for p in EVENTS[slug]["periods"]
                for c in loo.columns if c.startswith("d_")]
    for p, name, sub in subsets:
        if len(sub) < 2:
            continue
        for col in ("mrms_mm", "mrms_scaled_loo_mm"):
            o, e = sub.gauge_mm.values, sub[col].values
            res.append({"period": p, "subset": name, "n": len(sub), "product": col.replace("_mm", ""),
                        "ratio_of_sums": e.sum() / o.sum(), "mae_mm": np.mean(np.abs(e - o)), "r": np.corrcoef(o, e)[0, 1]})
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(fig, "mrms_gauge_summary.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summ).to_string(index=False))
    print(r.round(2).to_string(index=False))
    out.attrs = dict(mr.attrs, note="MRMS Pass2 scaled to GHCN-D daily gauges per flood period; scripts/mrms_gaugescaled.py")
    f = os.path.join(obs, "mrms_qpe_1h_gaugescaled.nc")
    out.to_dataset(name="precip").to_netcdf(f, encoding={"precip": {"zlib": True, "complevel": 4}})
    print(f)


if __name__ == "__main__":
    {"fetch": fetch, "check": check, "build": build}[sys.argv[1]](sys.argv[2])
