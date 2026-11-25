#!/usr/bin/env python3
"""
California 2022-23: MRMS MultiSensor QPE Pass2 scaled to independent daily gauges (sensitivity).

MRMS is 17-30 % below the GHCN-D gauges (CoCoRaHS + COOP, daily, not ingested by MRMS Pass2)
in the coastal ranges (obs-scout, 21 Oct 2026; data/raw/_return_periods/candidates_mrms_check/).
Here one multiplicative field per flood period is built from the gauge/MRMS ratios of the period
totals (inverse-distance weighted log ratio, nearest 8 within the domain, damped towards 1 with
distance to the nearest gauge, capped at 0.33-3) and applied to every MRMS hour of that period:
    P1  26 Dec 15Z - 2 Jan 15Z  (Cosumnes, Soquel flood)
    P2   3 Jan 15Z - 11 Jan 15Z (Arroyo Seco, San Lorenzo flood)
Gauge day = the 24 h ending at 15 UTC (07 PST) on DATE; only stations complete over a period;
values with a quality flag are dropped. Leave-one-out at every gauge says how well the scaled
product does where there is no gauge.

    python scripts/california_mrms_gaugescaled.py

Outputs: data/raw/california_2023/obs/mrms_qpe_1h_gaugescaled.nc (precip, mm per hour, end-stamped);
figures/california_2023/reference_rain/ (mrms_gauge_loo.csv, mrms_gauge_summary.csv)
"""
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(REPO, "data", "raw", "_return_periods", "candidates_mrms_check")
OBS = os.path.join(REPO, "data", "raw", "california_2023", "obs")
FIG = os.path.join(REPO, "figures", "california_2023", "reference_rain")
PERIODS = {"P1": ("2022-12-27", "2023-01-02"), "P2": ("2023-01-04", "2023-01-11")}   # gauge DATEs
OBS_HOUR = 15                                                                          # UTC
K, POW, DAMP_KM, CAP = 8, 2.0, 40.0, (0.33, 3.0)


def km(lat, lon, lat0=37.4, lon0=-121.3):
    return np.c_[(np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 110.57]


def field(xy, logr, dst):
    tree = cKDTree(xy)
    d, i = tree.query(dst, k=min(K, len(xy)))
    w = 1.0 / np.maximum(d, 0.05) ** POW
    f = (w * logr[i]).sum(1) / w.sum(1) * np.exp(-(d[:, 0] / DAMP_KM) ** 2)
    return np.clip(np.exp(f), *CAP)


def main():
    os.makedirs(FIG, exist_ok=True)
    mr = xr.open_dataset(os.path.join(OBS, "mrms_qpe_1h.nc"))["precip"].load()
    lat, lon = mr.latitude.values, mr.longitude.values
    st = pd.read_csv(os.path.join(CAND, "california_ghcnd_stations_box_with_PRCP_winter2223.csv"))
    st = st[st.lat.between(lat.min(), lat.max()) & st.lon.between(lon.min(), lon.max())]
    pr = pd.read_csv(os.path.join(CAND, "california_ghcnd_prcp_20221220_20230120.csv"), parse_dates=["DATE"])
    q = pr.PRCP_ATTRIBUTES.fillna("").str.split(",").str[1].fillna("")
    pr = pr[(q == "") & pr.STATION.isin(st.id)]
    ii = np.abs(lat[:, None] - st.lat.values[None]).argmin(0)
    jj = np.abs(lon[:, None] - st.lon.values[None]).argmin(0)
    LON, LAT = np.meshgrid(lon, lat)
    dst = km(LAT.ravel(), LON.ravel())
    out = mr.copy()
    rows, summ = [], []
    t = pd.DatetimeIndex(mr.time.values)
    for p, (d0, d1) in PERIODS.items():
        days = pd.date_range(d0, d1, freq="D")
        g = pr[pr.DATE.isin(days)].groupby("STATION").PRCP.agg(["sum", "count"])
        g = g[g["count"] == len(days)]["sum"]
        a = pd.Timestamp(d0) - pd.Timedelta("1D") + pd.Timedelta(hours=OBS_HOUR)
        b = pd.Timestamp(d1) + pd.Timedelta(hours=OBS_HOUR)
        hrs = (t > a) & (t <= b)
        tot = mr.isel(time=np.where(hrs)[0]).sum("time").values
        s = st.set_index("id").loc[g.index]
        k = [list(st.id).index(x) for x in g.index]
        m = tot[ii[k], jj[k]]
        ok = (g.values > 1) & (m > 1)
        xy = km(s.lat.values[ok], s.lon.values[ok])
        logr = np.log(g.values[ok] / m[ok])
        f = field(xy, logr, dst).reshape(LAT.shape)
        out[np.where(hrs)[0]] = (mr.isel(time=np.where(hrs)[0]) * f[None]).astype("float32")
        # leave-one-out at every gauge
        idx = np.where(ok)[0]
        for n, q_ in enumerate(idx):
            keep = np.ones(len(idx), bool); keep[n] = False
            fl = field(xy[keep], logr[keep], xy[n:n + 1])[0]
            rows.append({"period": p, "station": g.index[q_], "name": s.name.values[q_], "lat": s.lat.values[q_], "lon": s.lon.values[q_],
                         "gauge_mm": g.values[q_], "mrms_mm": m[q_], "mrms_scaled_loo_mm": m[q_] * fl,
                         **{c: s[c].values[q_] for c in s.columns if c.startswith("d_")}})
        summ.append({"period": p, "gauges": int(ok.sum()), "field_min": float(f.min()), "field_max": float(f.max()),
                     "hours": int(hrs.sum())})
    loo = pd.DataFrame(rows)
    loo.to_csv(os.path.join(FIG, "mrms_gauge_loo.csv"), index=False, float_format="%.2f")
    res = []
    for (p, sub_name), sub in [((p, "all"), loo[loo.period == p]) for p in PERIODS] + \
            [((p, c[2:]), loo[(loo.period == p) & (loo[c] <= 20)]) for p in PERIODS for c in loo.columns if c.startswith("d_")]:
        if len(sub) < 2:
            continue
        for col in ("mrms_mm", "mrms_scaled_loo_mm"):
            o, e = sub.gauge_mm.values, sub[col].values
            res.append({"period": p, "subset": sub_name, "n": len(sub), "product": col.replace("_mm", ""),
                        "ratio_of_sums": e.sum() / o.sum(), "mae_mm": np.mean(np.abs(e - o)), "r": np.corrcoef(o, e)[0, 1]})
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(FIG, "mrms_gauge_summary.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summ).to_string(index=False))
    print(r.round(2).to_string(index=False))
    out.attrs = dict(mr.attrs, note="MRMS Pass2 scaled to GHCN-D daily gauges per flood period; scripts/california_mrms_gaugescaled.py")
    out.to_dataset(name="precip").to_netcdf(os.path.join(OBS, "mrms_qpe_1h_gaugescaled.nc"),
                                            encoding={"precip": {"zlib": True, "complevel": 4}})
    print(os.path.join(OBS, "mrms_qpe_1h_gaugescaled.nc"))


if __name__ == "__main__":
    main()
