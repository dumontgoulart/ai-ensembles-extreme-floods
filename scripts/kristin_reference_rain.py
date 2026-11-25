#!/usr/bin/env python3
"""
Storm Kristin (Portugal, 27-31 Jan 2026): a validated reference rain from what is open.

There is no gauge network we can use (SNIRH is blocked, IPMA has no archive). What exists:
  * 11 IPMA stations, hourly rain, 20 Jan - 5 Feb 2026 (Meteostat bulk, complete);
  * IMERG Late V07B (satellite, 0.1 deg, half-hourly; NASA GES DISC, Earthdata login);
  * ERA5 hourly (DestinE Earth Data Hub), MSWEP NRT 3-hourly (the current forcing).
Candidates: MSWEP, ERA5 and IMERG raw, and ERA5 and IMERG rescaled to the gauges. The rescaling
is one multiplicative field from the event totals at the gauges (inverse-distance weighted log
ratio, damped towards 1 away from the gauges, capped at 0.4-3), applied to every hour. Each is
validated leave-one-out: the field is built from 10 gauges and scored at the 11th (event total
bias, 3 h correlation), so the numbers say how well the product does where there is no gauge.

    python scripts/kristin_reference_rain.py download
    python scripts/kristin_reference_rain.py build

Outputs: data/raw/kristin_2026/imerg/imerg_late_30min_box.nc, data/raw/kristin_2026/obs/
ref_rain_<product>_hourly.nc (mm per hour, end-stamped, 0.1 deg), figures/kristin_2026/reference_rain/
"""
import glob
import netrc
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

H5LOCK = threading.Lock()      # the HDF5 library is not thread-safe: download in parallel, read one at a time

import numpy as np
import pandas as pd
import requests
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "raw", "kristin_2026")
FIG = os.path.join(REPO, "figures", "kristin_2026", "reference_rain")
GAUGES = os.path.join(REPO, "data", "raw", "_return_periods", "candidates_kristin_pt",
                      "meteostat_hourly_prcp_pt_20260120_20260205.csv")
BOX = (36.8, 42.3, -9.8, -6.0)                      # lat_min, lat_max, lon_min, lon_max
PERIOD = ("2026-01-21 00:00", "2026-02-04 00:00")    # hours ending in (start, end]
EVENT = ("2026-01-26 00:00", "2026-02-01 00:00")    # totals and scores
RES = 0.1


# ── downloads ───────────────────────────────────────────────────────────────
def imerg_url(t):
    e = t + pd.Timedelta("29min59s")
    return ("https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.07/"
            f"{t:%Y}/{t.dayofyear:03d}/3B-HHR-L.MS.MRG.3IMERG.{t:%Y%m%d}-S{t:%H%M%S}-E{e:%H%M%S}."
            f"{t.hour * 60 + t.minute:04d}.V07B.HDF5")


def fetch_imerg(t, auth):
    s = requests.Session()
    s.auth = auth
    for k_ in range(4):
        try:
            time.sleep(0.5 * (k_ + 1))
            r = s.get(imerg_url(t), timeout=180)
            r.raise_for_status()
            with H5LOCK, tempfile.NamedTemporaryFile(suffix=".h5", dir=os.environ.get("TMPDIR", "/tmp")) as f:
                f.write(r.content); f.flush()
                with xr.open_dataset(f.name, group="Grid", engine="netcdf4", decode_times=False) as d:
                    p = d["precipitation"].isel(time=0).sel(lat=slice(BOX[0], BOX[1]), lon=slice(BOX[2], BOX[3])).load()
            return t, p.transpose("lat", "lon").values.astype("float32"), p["lat"].values, p["lon"].values
        except Exception:
            continue
    return t, None, None, None


def download():
    os.makedirs(os.path.join(RAW, "imerg"), exist_ok=True)
    a = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("urs.earthdata.nasa.gov")
    times = pd.date_range(PERIOD[0], PERIOD[1], freq="30min", inclusive="left")
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(lambda t: fetch_imerg(t, (a[0], a[2])), times))
    ok = [r for r in res if r[1] is not None]
    print(f"IMERG Late: {len(ok)}/{len(times)} half-hours", flush=True)
    lat, lon = ok[0][2], ok[0][3]
    da = xr.DataArray(np.stack([r[1] for r in ok]), dims=("time", "latitude", "longitude"),
                      coords={"time": [r[0] for r in ok], "latitude": lat, "longitude": lon}, name="rate_mm_h")
    da = da.where(da >= 0)
    da.to_netcdf(os.path.join(RAW, "imerg", "imerg_late_30min_box.nc"), encoding={"rate_mm_h": {"zlib": True, "complevel": 4}})
    # ERA5 hourly precipitation (hub; end-stamped, metres)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wflow_build_meteo as m
    z = xr.open_zarr(m._edh_url(), chunks={})
    sub = z["tp"].sel(valid_time=slice(pd.Timestamp(PERIOD[0]) + pd.Timedelta("1h"), PERIOD[1]),
                      latitude=slice(BOX[1] + 0.3, BOX[0] - 0.3), longitude=slice((BOX[2] - 0.3) % 360, (BOX[3] + 0.3) % 360)).load()
    for d in ("number", "surface"):
        if d in sub.dims and sub.sizes[d] == 1:
            sub = sub.isel({d: 0}, drop=True)
    sub = sub.assign_coords(longitude=((sub["longitude"] + 180) % 360) - 180).sortby("longitude").sortby("latitude")
    (sub * 1000.0).rename("tp").to_dataset().to_netcdf(os.path.join(RAW, "imerg", "era5_tp_1h_box.nc"))
    print("ERA5 hours:", sub.sizes["valid_time"], flush=True)


# ── products on the 0.1 deg grid, hourly end-stamped mm ─────────────────────
def hourly_products():
    im = xr.open_dataarray(os.path.join(RAW, "imerg", "imerg_late_30min_box.nc"))
    half = (im * 0.5).fillna(0.0)                                       # mm per half hour, stamped at its start
    hr = half.resample(time="1h").sum()                                 # hour starting at t
    hr = hr.assign_coords(time=hr["time"] + pd.Timedelta("1h"))          # end-stamp
    lat, lon = im["latitude"].values, im["longitude"].values
    e5 = xr.open_dataset(os.path.join(RAW, "imerg", "era5_tp_1h_box.nc"))["tp"].rename(valid_time="time")
    e5 = e5.interp(latitude=lat, longitude=lon, method="linear").clip(min=0)
    # MSWEP NRT 3 hourly (the current forcing): file label = start of its 3 h, spread uniformly
    # over the three end-stamped hours (wflow_prep_forcing shifts it to end-stamps the same way)
    prods = {"IMERG Late": hr, "ERA5": e5}
    ms = []
    for f in sorted(glob.glob(os.path.join(os.environ.get("MSWEP_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "MSWEP_V316")), "NRT", "3hourly", "2026*.nc"))):
        if 20 <= int(os.path.basename(f)[4:7]) <= 35:
            with xr.open_dataset(f) as d:
                ms.append(d["precipitation"].sel(lat=slice(BOX[1] + .2, BOX[0] - .2), lon=slice(BOX[2] - .2, BOX[3] + .2)).load())
    ms = xr.concat(ms, dim="time").rename(lat="latitude", lon="longitude").sortby("latitude")
    ms = ms.interp(latitude=lat, longitude=lon, method="linear")
    ms_end = ms.assign_coords(time=ms["time"] + pd.Timedelta("3h"))
    hours = pd.date_range(ms_end["time"].values[0] - pd.Timedelta("2h"), ms_end["time"].values[-1], freq="h")
    prods["MSWEP"] = ms_end.reindex(time=hours, method="bfill") / 3.0
    return prods, lat, lon


def load_gauges():
    d = pd.read_csv(GAUGES, dtype={"station": str})
    d["time"] = pd.to_datetime(d["date_utc"] + " " + d["hour_utc"].astype(str).str.zfill(2) + ":00")
    return d


def sample(da, lat, lon):
    return da.sel(latitude=lat, longitude=lon, method="nearest")


def event_sum(da):
    return da.sel(time=slice(pd.Timestamp(EVENT[0]) + pd.Timedelta("1h"), EVENT[1])).sum("time")


def field_from(ratios, glat, glon, lat2d, lon2d, damp_km=120.0, cap=(0.4, 3.0)):
    """Multiplicative correction field: IDW of log ratios at the gauges, damped towards 1 with distance."""
    lr = np.log(np.clip(ratios, *cap))
    num = np.zeros_like(lat2d); den = np.zeros_like(lat2d); dmin = np.full_like(lat2d, np.inf)
    for r, la, lo in zip(lr, glat, glon):
        d = np.hypot((lat2d - la) * 111.0, (lon2d - lo) * 111.0 * np.cos(np.radians(la))) + 1e-3
        w = 1.0 / d ** 2
        num += w * r; den += w; dmin = np.minimum(dmin, d)
    f = np.exp(num / den * np.exp(-dmin / damp_km))
    return np.clip(f, *cap)


def build():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(os.path.join(RAW, "obs"), exist_ok=True)
    prods, lat, lon = hourly_products()
    G = load_gauges()
    st = G.drop_duplicates("station")[["station", "name", "lat", "lon"]].reset_index(drop=True)
    obs = G.pivot(index="time", columns="station", values="prcp_mm").sort_index()
    ev = obs.loc[pd.Timestamp(EVENT[0]) + pd.Timedelta("1h"):EVENT[1]]
    tot_obs = ev.sum()
    # timing of the gauge hour stamp: lag with the highest correlation to IMERG/ERA5 (hours)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    rows, scaled = [], {}
    for name, da in prods.items():
        cell = {s: sample(da, r.lat, r.lon).sel(time=slice(pd.Timestamp(EVENT[0]) + pd.Timedelta("1h"), EVENT[1])).to_series() for s, r in st.set_index("station").iterrows()}
        tot_p = pd.Series({s: float(v.sum()) for s, v in cell.items()})
        ratio = (tot_obs / tot_p.clip(lower=1.0)).reindex(st["station"])
        for variant in ("raw", "rescaled"):
            for k, s in enumerate(st["station"]):
                if variant == "raw":
                    f = 1.0
                else:
                    keep = [i for i in range(len(st)) if i != k]
                    F = field_from(ratio.values[keep], st["lat"].values[keep], st["lon"].values[keep], lat2d, lon2d)
                    f = float(F[np.abs(lat - st["lat"][k]).argmin(), np.abs(lon - st["lon"][k]).argmin()])
                x = cell[s] * f
                y = ev[s].reindex(x.index)
                x3, y3 = x.resample("3h", closed="right", label="right").sum(), y.resample("3h", closed="right", label="right").sum()
                rows.append({"product": name, "variant": variant, "station": s, "name": st["name"][k], "gauge_mm": tot_obs[s],
                             "product_mm": x.sum(), "bias_pct": 100 * (x.sum() / tot_obs[s] - 1),
                             "r_3h": float(np.corrcoef(x3, y3)[0, 1]), "r_1h": float(np.corrcoef(x, y)[0, 1])})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(FIG, "kristin_reference_loo.csv"), index=False, float_format="%.2f")
    s = t.groupby(["product", "variant"]).agg(median_bias_pct=("bias_pct", "median"), mean_abs_bias_pct=("bias_pct", lambda x: x.abs().mean()),
                                              rmse_mm=("bias_pct", lambda x: 0.0), r_3h_mean=("r_3h", "mean"), r_1h_mean=("r_1h", "mean")).drop(columns="rmse_mm").round(2)
    s["abs_err_mm"] = t.assign(e=(t.product_mm - t.gauge_mm).abs()).groupby(["product", "variant"])["e"].mean().round(1)
    print("\nLeave-one-out against the 11 IPMA gauges (event 26 Jan - 1 Feb; rescaled = built without the station):")
    print(s.to_string())
    print("\nGauge event totals (mm):", {r["name"][:14]: round(tot_obs[r.station]) for _, r in st.iterrows()})
    s.to_csv(os.path.join(FIG, "kristin_reference_summary.csv"))

    # best rescaled product: lowest mean absolute total error, then 3 h correlation
    resc = s.xs("rescaled", level="variant")
    best = resc.sort_values(["abs_err_mm", "r_3h_mean"], ascending=[True, False]).index[0]
    print("\nBest rescaled product:", best)
    for name in ("IMERG Late", "ERA5"):
        da = prods[name]
        tot_p = pd.Series({s_: float(sample(da, r.lat, r.lon).sel(time=slice(pd.Timestamp(EVENT[0]) + pd.Timedelta("1h"), EVENT[1])).sum()) for s_, r in st.set_index("station").iterrows()})
        ratio = (tot_obs / tot_p.clip(lower=1.0)).reindex(st["station"]).values
        F = field_from(ratio, st["lat"].values, st["lon"].values, lat2d, lon2d)
        out = (da * xr.DataArray(F, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})).astype("float32")
        out = out.sel(time=slice(pd.Timestamp(PERIOD[0]) + pd.Timedelta("1h"), PERIOD[1]))
        tag = name.split()[0].lower()
        out.rename("precip").to_dataset().to_netcdf(os.path.join(RAW, "obs", f"ref_rain_{tag}_rescaled_hourly.nc"),
                                                    encoding={"precip": {"zlib": True, "complevel": 4}})
        scaled[name] = (F, out)
    # figure: event totals of the raw and rescaled candidates, with the gauges
    fig, axs = plt.subplots(1, 5, figsize=(22 / 2.54, 5.5 / 2.54), constrained_layout=True)
    panels = [("MSWEP (raw)", prods.get("MSWEP")), ("ERA5 (raw)", prods["ERA5"]), ("IMERG Late (raw)", prods["IMERG Late"]),
              ("ERA5 rescaled", scaled["ERA5"][1]), ("IMERG Late rescaled", scaled["IMERG Late"][1])]
    for ax, (ttl, da) in zip(axs, panels):
        if da is None:
            ax.axis("off"); continue
        z = event_sum(da)
        pc = ax.pcolormesh(lon, lat, z.values, vmin=0, vmax=300, cmap="YlGnBu", shading="nearest")
        ax.scatter(st["lon"], st["lat"], c=tot_obs[st["station"]].values, s=26, edgecolor="k", lw=0.5, cmap="YlGnBu", vmin=0, vmax=300)
        ax.set_title(ttl, fontsize=6); ax.tick_params(labelsize=5)
    fig.colorbar(pc, ax=axs, shrink=0.8, label="26 Jan - 1 Feb total (mm); dots = gauges")
    fig.savefig(os.path.join(FIG, "kristin_reference_totals.png"), dpi=200)
    plt.close(fig)
    print(FIG)


if __name__ == "__main__":
    {"download": download, "build": build}[sys.argv[1]]()
