#!/usr/bin/env python3
"""
DANA (Valencia, 28-30 Oct 2024): a daily gauge grid from AEMET + CHJ-SAIH stations, and
which gridded product matches the gauges best (to give the grid its sub-daily timing).

Gauges (data/raw/dana_2024/obs/):
  * AEMET OpenData daily, 07-07 local (06-06 UTC in CET), 97 stations (download_aemet.py)
  * CHJ SAIH daily reports, windows ending 07 UTC, ~160 stations with coordinates
    (parse_saih_rain_reports.py); SAIH stations within 2 km of an AEMET station are dropped
Event: the three days ending 31 Oct (AEMET 28 Oct 06Z - 31 Oct 06Z, SAIH 28 Oct 07Z - 31 Oct 07Z).

Products, each summed over every gauge's own windows, bilinear at the gauge:
  ERA5 hourly, MSWEP V3.16 Past 3-hourly (MSWEP_V316_test), IMERG Final and
  IMERG Late V07 half-hourly (GES DISC, Earthdata login in ~/.netrc).
Gauge grid: daily IDW (power 2, nearest 8 within 30 km) on the wflow grid; leave-one-out at
every gauge. Gauge-scaled products: product x one field of IDW-weighted log gauge/product
ratios (event totals, damped towards 1 away from the gauges), also leave-one-out.

    python scripts/dana_reference_rain.py download     # IMERG Final + Late (cluster)
    python scripts/dana_reference_rain.py build
    python scripts/dana_reference_rain.py hourly       # v1 hourly reference: daily grid x IMERG Final timing
    python scripts/dana_reference_rain.py partner      # v2: parse the partner files (SAIH 5-min, hourly, Turís 10-min)
    python scripts/dana_reference_rain.py build2       # v2: + partner stations, gauge timing (gauge_grid_hourly_gauges.nc)

Outputs: data/raw/dana_2024/obs/gauge_grid_daily.nc (precip, mm per day, day = 07 local start),
data/raw/dana_2024/obs/dana_gauges_daily.csv; figures/dana_2024/reference_rain/
(products_vs_gauges.csv, loo.csv, summary.csv, totals_map.png, scatter.png)
"""
import netrc
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "raw", "dana_2024")
OBS = os.path.join(RAW, "obs")
FIG = os.path.join(REPO, "figures", "dana_2024", "reference_rain")
BOX = (38.0, 41.3, -2.7, 1.6)                                     # lat_min, lat_max, lon_min, lon_max
PERIOD = ("2024-10-27 00:00", "2024-11-02 12:00")
DAYS = pd.date_range("2024-10-27", "2024-11-01", freq="D")         # day labels (start at 07 local)
EVENT_DAYS = pd.date_range("2024-10-28", "2024-10-30", freq="D")
K, RMAX, POW = 8, 30.0, 2.0
H5LOCK = threading.Lock()


# ── downloads ───────────────────────────────────────────────────────────────
def imerg_url(t, final):
    e = t + pd.Timedelta("29min59s")
    coll, tag = ("GPM_3IMERGHH.07", "3B-HHR") if final else ("GPM_3IMERGHHL.07", "3B-HHR-L")
    return (f"https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/{coll}/{t:%Y}/{t.dayofyear:03d}/"
            f"{tag}.MS.MRG.3IMERG.{t:%Y%m%d}-S{t:%H%M%S}-E{e:%H%M%S}.{t.hour * 60 + t.minute:04d}.V07B.HDF5")


def fetch(t, auth, final):
    s = requests.Session()
    s.auth = auth
    for k in range(5):
        try:
            time.sleep(0.5 * (k + 1))
            r = s.get(imerg_url(t, final), timeout=180)
            r.raise_for_status()
            if r.content[:4] != b"\x89HDF":
                raise ValueError("not HDF5")
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
    for final, name in ((True, "final"), (False, "late")):
        out = os.path.join(RAW, "imerg", f"imerg_{name}_30min_box.nc")
        if os.path.exists(out):
            continue
        with ThreadPoolExecutor(3) as ex:
            res = list(ex.map(lambda t: fetch(t, (a[0], a[2]), final), times))
        ok = [r for r in res if r[1] is not None]
        print(f"IMERG {name}: {len(ok)}/{len(times)} half-hours", flush=True)
        da = xr.DataArray(np.stack([r[1] for r in ok]), dims=("time", "latitude", "longitude"),
                          coords={"time": [r[0] for r in ok], "latitude": ok[0][2], "longitude": ok[0][3]}, name="rate_mm_h")
        da.where(da >= 0).to_netcdf(out, encoding={"rate_mm_h": {"zlib": True, "complevel": 4}})


# ── gauges ──────────────────────────────────────────────────────────────────
def km(lat, lon, lat0=39.5, lon0=-0.7):
    return np.c_[(np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 110.57]


def gauges():
    a = pd.read_csv(os.path.join(OBS, "aemet_daily.csv"), parse_dates=["day", "start_utc", "end_utc"])
    a = a.assign(source="AEMET", station=a.indicativo, name=a.nombre)
    s = pd.read_csv(os.path.join(OBS, "saih_rain_daily.csv"), parse_dates=["window_end_utc"]).dropna(subset=["lat", "lon"])
    s = s.assign(source="SAIH", end_utc=s.window_end_utc, start_utc=s.window_end_utc - pd.Timedelta("1D"),
                 day=(s.window_end_utc - pd.Timedelta("1D")).dt.normalize(), name=s.station,
                 station="SAIH_" + s.code.fillna(s.station).astype(str))
    s = s.drop_duplicates(["station", "day"])
    ai = a.drop_duplicates("station")
    d, _ = cKDTree(km(ai.lat, ai.lon)).query(km(s.lat, s.lon), k=1)
    s = s[d > 2.0]
    g = pd.concat([a, s], ignore_index=True)[["source", "station", "name", "lat", "lon", "day", "start_utc", "end_utc", "precip_mm"]]
    g = g[g.day.isin(DAYS)]
    g.to_csv(os.path.join(OBS, "dana_gauges_daily.csv"), index=False)
    return g


# ── products: hourly end-stamped mm at the gauges ───────────────────────────
def hourly(name):
    if name == "ERA5":
        v = xr.open_dataset(os.path.join(REPO, "data", "processed", "dana_2024", "era5_processed.nc"))["tp"]
        return v.sortby("latitude")                                                  # hourly, end-stamped, mm
    if name == "MSWEP":
        v = xr.open_dataset(os.path.join(RAW, "mswep", "mswep_tp_dana_2024.nc"))["precipitation"]
        v = v.rename(lat="latitude", lon="longitude").sortby("latitude")              # mm per 3 h, label = start
        v = v.assign_coords(time=v.time + pd.Timedelta("3h"))                         # end of its 3 h
        hrs = pd.date_range(pd.Timestamp(v.time.values[0]) - pd.Timedelta("2h"), pd.Timestamp(v.time.values[-1]), freq="h")
        return v.reindex(time=hrs, method="bfill") / 3.0                               # uniform within the 3 h
    im = xr.open_dataarray(os.path.join(RAW, "imerg", f"imerg_{name.split()[1].lower()}_30min_box.nc"))
    hr = (im * 0.5).fillna(0.0).resample(time="1h").sum()                           # hour starting at t
    return hr.assign_coords(time=hr.time + pd.Timedelta("1h")).sortby("latitude")


def at_gauges(v, g):
    """Product sum over each gauge-day window (start, end]."""
    st = g.drop_duplicates("station")
    pts = v.interp(latitude=xr.DataArray(st.lat.values, dims="pt"), longitude=xr.DataArray(st.lon.values, dims="pt"),
                   method="linear").transpose("time", "pt")
    df = pd.DataFrame(pts.values, index=pd.DatetimeIndex(pts.time.values), columns=st.station.values)
    t = df.index
    out = np.full(len(g), np.nan)
    for k, (s_, a, b) in enumerate(zip(g.station, g.start_utc, g.end_utc)):
        m = (t > a) & (t <= b)
        if m.sum() >= 23:
            out[k] = df.loc[m, s_].sum()
    return out


# ── IDW ─────────────────────────────────────────────────────────────────────
def idw(xy, val, dst):
    ok = np.isfinite(val)
    tree = cKDTree(xy[ok])
    d, i = tree.query(dst, k=min(K, ok.sum()), distance_upper_bound=RMAX)
    v = np.where(np.isfinite(d), val[ok][np.minimum(i, ok.sum() - 1)], np.nan)
    w = np.where(np.isfinite(d), 1.0 / np.maximum(d, 0.05) ** POW, 0.0)
    ws = w.sum(1)
    return np.where(ws > 0, np.nansum(w * np.nan_to_num(v), 1) / np.where(ws > 0, ws, 1), np.nan)


def ratio_field(xy, logr, dst, damp_km=40.0, cap=(0.33, 4.0)):
    """IDW of log(gauge/product), damped towards 0 (ratio 1) with distance to the nearest gauge."""
    tree = cKDTree(xy)
    d, i = tree.query(dst, k=min(K, len(xy)))
    w = 1.0 / np.maximum(d, 0.05) ** POW
    f = (w * logr[i]).sum(1) / w.sum(1) * np.exp(-(d[:, 0] / damp_km) ** 2)
    return np.clip(np.exp(f), *cap)


def build():
    os.makedirs(FIG, exist_ok=True)
    g = gauges()
    ev = g[g.day.isin(EVENT_DAYS)]
    complete = ev.groupby("station").precip_mm.count() == len(EVENT_DAYS)
    ev = ev[ev.station.isin(complete[complete].index)].copy()
    prods = ["ERA5", "MSWEP", "IMERG Final", "IMERG Late"]
    for p in prods:
        ev[p] = at_gauges(hourly(p), ev)
    ev.to_csv(os.path.join(FIG, "products_vs_gauges_daily.csv"), index=False)
    tot = ev.groupby(["source", "station", "name", "lat", "lon"])[["precip_mm"] + prods].sum(min_count=3).reset_index()
    print(f"{len(tot)} gauges with the three event days ({(tot.source == 'AEMET').sum()} AEMET, {(tot.source == 'SAIH').sum()} SAIH)")

    # leave-one-out: gauge grid (IDW of the gauge totals, per day) and gauge-scaled products
    xy = km(tot.lat, tot.lon)
    rows = []
    for j in range(len(tot)):
        m = np.ones(len(tot), bool); m[j] = False
        r = {"station": tot.station[j], "gauge": tot.precip_mm[j]}
        daily = [idw(km(*ev[(ev.day == d) & ev.station.isin(tot.station[m])][["lat", "lon"]].values.T),
                     ev[(ev.day == d) & ev.station.isin(tot.station[m])].precip_mm.values, xy[j:j + 1])[0] for d in EVENT_DAYS]
        r["gauge grid"] = np.sum(daily)
        for p in prods:
            ok = m & (tot[p] > 0.5) & (tot.precip_mm > 0.5)
            f = ratio_field(xy[ok], np.log(tot.precip_mm[ok] / tot[p][ok]).values, xy[j:j + 1])[0]
            r[f"{p} gauge-scaled"] = tot[p][j] * f
        rows.append(r)
    loo = pd.DataFrame(rows).merge(tot[["station", "source", "name", "lat", "lon"] + prods], on="station")
    loo.to_csv(os.path.join(FIG, "loo.csv"), index=False, float_format="%.2f")

    cands = prods + ["gauge grid"] + [f"{p} gauge-scaled" for p in prods]
    out = []
    for c in cands:
        for sub, msk in (("all gauges", np.ones(len(loo), bool)), ("core (gauge >= 150 mm)", loo.gauge >= 150)):
            o, f = loo.gauge[msk].values, loo[c][msk].values
            ok = np.isfinite(f) & np.isfinite(o)
            wet = ok & (o >= 20)                                   # median bias only where it rained
            out.append({"product": c, "subset": sub, "n": int(ok.sum()),
                        "bias_total_pct": 100 * (f[ok].sum() - o[ok].sum()) / o[ok].sum(),
                        "median_bias_pct_ge20mm": float(np.median(100 * (f[wet] - o[wet]) / o[wet])),
                        "mae_mm": float(np.mean(np.abs(f[ok] - o[ok]))), "r_totals": float(np.corrcoef(o[ok], f[ok])[0, 1])})
    # daily correlation (station-days), raw products only (no LOO needed)
    for p in prods:
        k = ev[["precip_mm", p]].dropna()
        for o in out:
            if o["product"] == p and o["subset"] == "all gauges":
                o["r_daily"] = float(np.corrcoef(k.precip_mm, k[p])[0, 1])
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(FIG, "summary.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 200)
    print(s.round(2).to_string(index=False))
    print("\nwettest gauges:")
    print(loo.sort_values("gauge", ascending=False).head(10)[["name", "source", "gauge", "gauge grid"] + prods].round(0).to_string(index=False))

    # the grid itself, per day, on the wflow grid
    sm = xr.open_dataset(os.path.join(REPO, "data", "wflow", "dana_2024", "model", "staticmaps.nc"))
    lat, lon = sm.latitude.values, sm.longitude.values
    LON, LAT = np.meshgrid(lon, lat)
    dst = km(LAT.ravel(), LON.ravel())
    grid = np.full((len(DAYS),) + LAT.shape, np.nan, "float32")
    for i, d in enumerate(DAYS):
        x = g[(g.day == d)].dropna(subset=["precip_mm"])
        grid[i] = idw(km(x.lat, x.lon), x.precip_mm.values, dst).reshape(LAT.shape)
    da = xr.DataArray(grid, dims=("day", "latitude", "longitude"), coords={"day": DAYS, "latitude": lat, "longitude": lon},
                      name="precip", attrs={"units": "mm per day", "note": "day = 07 local start (AEMET 06 UTC, SAIH 07 UTC in CET); IDW of AEMET + SAIH daily gauges; scripts/dana_reference_rain.py"})
    da.to_netcdf(os.path.join(OBS, "gauge_grid_daily.nc"), encoding={"precip": {"zlib": True, "complevel": 4}})

    # figures
    fig, axs = plt.subplots(1, 3, figsize=(18 / 2.54, 6.5 / 2.54), constrained_layout=True)
    evg = da.sel(day=EVENT_DAYS).sum("day")
    vmax = 500
    pc = axs[0].pcolormesh(lon, lat, evg, cmap="Blues", vmin=0, vmax=vmax, shading="nearest")
    axs[0].scatter(tot.lon, tot.lat, c=tot.precip_mm, cmap="Blues", vmin=0, vmax=vmax, s=6, edgecolor="k", linewidth=0.3)
    axs[0].set_title("a  gauge grid, 28-30 Oct (mm)", fontsize=7, loc="left")
    fig.colorbar(pc, ax=axs[0], shrink=0.7).ax.tick_params(labelsize=6)
    for ax, (c, ttl) in zip(axs[1:], ((["gauge grid"] + prods[:2], "b  leave-one-out / products"), ([f"{p} gauge-scaled" for p in prods], "c  gauge-scaled (leave-one-out)"))):
        for cc, col in zip(c, ("k", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")):
            ax.scatter(loo.gauge, loo[cc], s=5, alpha=0.6, label=cc, color=col)
        ax.plot([0, 800], [0, 800], "k--", lw=0.6)
        ax.set_xlabel("gauge total (mm)", fontsize=6); ax.set_ylabel("estimate (mm)", fontsize=6)
        ax.set_title(ttl, fontsize=7, loc="left"); ax.legend(fontsize=5, frameon=False); ax.tick_params(labelsize=6)
    for ax in axs:
        ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(FIG, "totals_and_loo.png"), dpi=200)
    plt.close(fig)
    print(FIG)


# ── hourly reference: daily gauge grid x IMERG Final timing ─────────────────
def hourly_reference():
    """new(h, cell) = grid(day, cell) * f(h, cell); f = IMERG Final hourly fraction of the
    cell's day total (day = 06-06 UTC, AEMET's 07 local in CET), or the domain-mean IMERG
    fraction when the cell's IMERG day total is < 0.2 mm. After the last gauge day the file
    continues with MSWEP Past (uniform within its 3 h) to the end of the simulation."""
    gd = xr.open_dataset(os.path.join(OBS, "gauge_grid_daily.nc"))["precip"].load()
    lat, lon = gd.latitude.values, gd.longitude.values
    im = hourly("IMERG Final").interp(latitude=lat, longitude=lon, method="linear").fillna(0.0).clip(min=0)
    parts = []
    for d in gd.day.values:
        t0 = pd.Timestamp(d) + pd.Timedelta("6h")
        hrs = pd.date_range(t0 + pd.Timedelta("1h"), t0 + pd.Timedelta("24h"), freq="h")   # end-stamped
        h = im.sel(time=hrs).values
        tot = h.sum(0)
        dom = h.mean((1, 2)); dom = dom / dom.sum() if dom.sum() > 0 else np.full(24, 1 / 24)
        frac = np.where(tot[None] >= 0.2, h / np.where(tot > 0, tot, 1)[None], dom[:, None, None])
        g = np.nan_to_num(gd.sel(day=d).values)
        parts.append(xr.DataArray((g[None] * frac).astype("float32"), dims=("time", "latitude", "longitude"),
                                  coords={"time": hrs, "latitude": lat, "longitude": lon}))
    ref = xr.concat(parts, dim="time")
    end_grid = pd.Timestamp(ref.time.values[-1])
    ms = hourly("MSWEP").interp(latitude=lat, longitude=lon, method="linear").fillna(0.0)
    ms = ms.sel(time=ms.time > np.datetime64(end_grid)).astype("float32")
    out = xr.concat([ref, ms.drop_vars([c for c in ms.coords if c not in ("time", "latitude", "longitude")])], dim="time")
    out.name = "precip"
    out.attrs = {"units": "mm per hour", "time_convention": "end of hour",
                 "note": f"AEMET + SAIH daily gauge grid with IMERG Final hourly timing to {end_grid}, then MSWEP Past; scripts/dana_reference_rain.py"}
    path = os.path.join(OBS, "gauge_grid_hourly_imergfinal.nc")
    out.to_dataset().to_netcdf(path, encoding={"precip": {"zlib": True, "complevel": 4}})
    chk = float(abs(ref.groupby(((ref.time - pd.Timedelta("1h") - pd.Timedelta("6h")).dt.floor("D"))).sum().values - np.nan_to_num(gd.values)).max())
    print(f"hourly reference {pd.Timestamp(out.time.values[0])} -> {pd.Timestamp(out.time.values[-1])}; "
          f"daily totals reproduced to {chk:.3f} mm; {path}")

    # check figure: domain totals per product and basin-mean hourly series
    basin = os.path.join(REPO, "data", "wflow", "dana_2024", "model", "staticgeoms", "basins.geojson")
    import geopandas as gpd
    b = gpd.read_file(basin).to_crs(4326).union_all()
    LON, LAT = np.meshgrid(lon, lat)
    inside = gpd.GeoSeries(gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs=4326).within(b).values.reshape(LON.shape)
    fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 6 / 2.54), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1, 1.6]})
    ev = ref.sel(time=slice("2024-10-28T07", "2024-10-31T06")).sum("time")
    pc = axs[0].pcolormesh(lon, lat, ev, cmap="Blues", vmin=0, vmax=500, shading="nearest")
    fig.colorbar(pc, ax=axs[0], shrink=0.7, label="28-30 Oct (mm)").ax.tick_params(labelsize=6)
    axs[0].set_title("a  hourly reference, event total", fontsize=7, loc="left")
    sl = slice("2024-10-28T00", "2024-11-01T00")
    for name, da, col in (("reference (gauges x IMERG Final)", out, "k"), ("IMERG Final", im, "#2ca02c"),
                          ("MSWEP Past", hourly("MSWEP").interp(latitude=lat, longitude=lon, method="linear"), "#ff7f0e"),
                          ("ERA5", hourly("ERA5").interp(latitude=lat, longitude=lon, method="linear"), "#1f77b4")):
        x = da.sel(time=sl)
        ser = pd.Series(np.nanmean(x.values[:, inside], axis=1), index=pd.DatetimeIndex(x.time.values))
        axs[1].plot(ser.index, ser.values, color=col, lw=1, label=f"{name} ({ser.sum():.0f} mm)")
    axs[1].set_ylabel("basin-mean rain (mm/h)", fontsize=6)
    axs[1].set_title("b  basin mean (whole DANA model)", fontsize=7, loc="left")
    axs[1].legend(fontsize=5.5, frameon=False)
    for ax in axs:
        ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(FIG, "hourly_reference.png"), dpi=200)
    plt.close(fig)


# ── v2 (23 Oct): partner data -> gauge totals with gauge timing ─────────────
# Partner files (data/raw/dana_2024/dana_obs/, all on local CET = UTC+1, whatever the headers say):
#   * CHJ SAIH 5-minute rain, 184 stations, 28 Oct - 4 Nov; red-filled cells are sensor failures
#     (the sheet says "en rojo datos en FALLO") and are dropped. The 5-min sums match our SAIH
#     daily reports (windows ending 07 UTC) when the file clock is shifted by -1 h.
#   * 29 Oct hourly accumulations at 228 stations (AVAMET, SISRITEL, AEMET, Torrent); V_k is the
#     total to k:00 local (lag check against the nearest SAIH stations: r 0.88 at this offset).
#   * AEMET Turís 10-minute rain, header "UTC" but local: only the -1 h shift reproduces AEMET's
#     daily totals (61.0 mm to 29 Oct 06 UTC, 710.8 mm to 30 Oct 06 UTC).
PARTNER = os.path.join(RAW, "dana_obs")
CET = pd.Timedelta("1h")


def partner():
    """Parse the partner files into obs/saih5_rain_5min.csv.gz, saih5_stations.csv, partner_hourly.csv."""
    import openpyxl
    import pyproj
    f = os.path.join(PARTNER, "Copia de 5-minutal_SAIH_Evento_10.2024.xlsx")
    ws = openpyxl.load_workbook(f, data_only=True).active
    names = [ws.cell(1, j).value for j in range(4, ws.max_column + 1)]
    ux = np.array([ws.cell(2, j).value for j in range(4, ws.max_column + 1)], float)
    uy = np.array([ws.cell(3, j).value for j in range(4, ws.max_column + 1)], float)
    times, vals = [], []
    for row in ws.iter_rows(min_row=15, max_row=ws.max_row, min_col=3, max_col=ws.max_column):
        t = row[0].value
        if not isinstance(t, (pd.Timestamp,)) and not hasattr(t, "year"):
            continue
        times.append(pd.Timestamp(t).round("min") - CET)                             # end of the 5 min, UTC
        vals.append([np.nan if (c.fill is not None and c.fill.fill_type and str(c.fill.fgColor.rgb).upper().endswith("FF0000"))
                     or not isinstance(c.value, (int, float)) else float(c.value) for c in row[1:]])
    lon, lat = pyproj.Transformer.from_crs(25830, 4326, always_xy=True).transform(ux, uy)
    ids = [f"SAIH5_{k:03d}" for k in range(len(names))]
    df = pd.DataFrame(vals, index=pd.DatetimeIndex(times, name="time_utc"), columns=ids)
    df.to_csv(os.path.join(OBS, "saih5_rain_5min.csv.gz"), float_format="%.2f")
    st = pd.DataFrame({"station": ids, "name": names, "lat": lat, "lon": lon,
                       "failed_frac": df.isna().mean().values, "total_mm": df.sum().values})
    st.to_csv(os.path.join(OBS, "saih5_stations.csv"), index=False, float_format="%.5f")
    print(f"SAIH 5-min: {len(ids)} stations, {df.index.min()} -> {df.index.max()} UTC; "
          f"{(st.failed_frac == 1).sum()} entirely failed, {((st.failed_frac > 0) & (st.failed_frac < 1)).sum()} partly")

    c = pd.read_csv(os.path.join(PARTNER, "valencia_hourly_accumulated.csv"))
    c = c[c.source != "AEMET"]                                                       # AEMET: daily already, Turís 10-min below
    vc = [f"V{i}" for i in range(1, 25)]
    acc = c[vc].values
    t_end = pd.date_range("2024-10-29 01:00", periods=24, freq="h") - CET            # V_k ends at k:00 local
    rows = []
    for k, r in enumerate(c.itertuples()):
        a = acc[k]
        inc = np.diff(np.r_[0.0, a])
        for h in range(24):
            rows.append((f"{r.source}_{r.id}".replace(" ", "_"), r.source, r.name, r.lat, r.lon, t_end[h], a[h], inc[h]))
    p = pd.DataFrame(rows, columns=["station", "source", "name", "lat", "lon", "time_utc", "accum_mm", "precip_mm"])
    t = pd.read_excel(os.path.join(PARTNER, "turis_precipitation_2024-10-29_10min.xlsx"))
    tt = pd.to_datetime(t.iloc[:, 0].astype(str).str[:10] + " " + t.iloc[:, 1].astype(str)) - CET
    tu = pd.Series(t.iloc[:, 2].astype(float).values, index=tt).resample("1h", label="right", closed="right").sum()
    p = pd.concat([p, pd.DataFrame({"station": "AEMET10_8337X", "source": "AEMET10", "name": "TURIS (AEMET 10-min)",
                                    "lat": 39.39222, "lon": -0.62472, "time_utc": tu.index, "accum_mm": tu.cumsum().values,
                                    "precip_mm": tu.values})], ignore_index=True)
    p.to_csv(os.path.join(OBS, "partner_hourly.csv"), index=False, float_format="%.2f")
    print(f"partner hourly: {p.station.nunique()} stations ({c.source.value_counts().to_dict()} + Turís 10-min)")


def _saih5_hourly():
    s5 = pd.read_csv(os.path.join(OBS, "saih5_rain_5min.csv.gz"), index_col=0, parse_dates=True)
    st = pd.read_csv(os.path.join(OBS, "saih5_stations.csv"))
    h = s5.resample("1h", label="right", closed="right").sum(min_count=12)          # an hour needs all 12 values
    return h, st


def gauges_v2():
    """Daily (06-06 UTC) station table: AEMET; SAIH from the 5-min data where a day is complete,
    else the SAIH daily reports; new partner stations (29 Oct only) more than 1 km from those."""
    a = pd.read_csv(os.path.join(OBS, "aemet_daily.csv"), parse_dates=["day", "start_utc", "end_utc"])
    a = a.assign(source="AEMET", station=a.indicativo, name=a.nombre)[["source", "station", "name", "lat", "lon", "day", "precip_mm"]]
    h5, st5 = _saih5_hourly()
    rows = []
    for d in DAYS:
        w = h5[(h5.index > d + pd.Timedelta("6h")) & (h5.index <= d + pd.Timedelta("30h"))]
        if len(w) < 24:
            continue
        tot = w.sum(min_count=24)
        for sid, v in tot.dropna().items():
            r = st5[st5.station == sid].iloc[0]
            rows.append(("SAIH5", sid, r["name"], r.lat, r.lon, d, v))
    s5 = pd.DataFrame(rows, columns=["source", "station", "name", "lat", "lon", "day", "precip_mm"])
    old = gauges()                                                                   # AEMET + SAIH daily reports (v1 table)
    sd = old[old.source == "SAIH"][["source", "station", "name", "lat", "lon", "day", "precip_mm"]]
    if len(s5):                                                                      # a SAIH report day is replaced where the 5-min day exists
        d5, i5 = cKDTree(km(st5.lat, st5.lon)).query(km(sd.lat, sd.lon), k=1)
        key5 = set(zip(s5.station, s5.day))
        sd = sd[[not (dd < 1.0 and (st5.station.iloc[ii], day) in key5) for dd, ii, day in zip(d5, i5, sd.day)]]
    ai = a.drop_duplicates("station")
    d, _ = cKDTree(km(ai.lat, ai.lon)).query(km(s5.lat, s5.lon), k=1)
    s5 = s5[d > 2.0]                                                                 # as v1: AEMET wins within 2 km
    base = pd.concat([a, sd, s5], ignore_index=True)
    # partner stations: 29 Oct 06 UTC - 30 Oct 06 UTC = (accum at 23 UTC - accum at 06 UTC) / share of the
    # nearest complete SAIH5 station's day falling in 06-23 UTC (the file ends at 23 UTC)
    p = pd.read_csv(os.path.join(OBS, "partner_hourly.csv"), parse_dates=["time_utc"])
    d29 = pd.Timestamp("2024-10-29")
    w5 = h5[(h5.index > d29 + pd.Timedelta("6h")) & (h5.index <= d29 + pd.Timedelta("30h"))]
    share = (w5[w5.index <= d29 + pd.Timedelta("23h")].sum(min_count=17) / w5.sum(min_count=24))
    ok5 = st5.set_index("station").loc[share.dropna().index]
    tree5 = cKDTree(km(ok5.lat, ok5.lon))
    prow = []
    for sid, g in p.groupby("station"):
        g = g.set_index("time_utc").sort_index()
        a06, a23 = g.accum_mm.get(d29 + pd.Timedelta("6h"), np.nan), g.accum_mm.get(d29 + pd.Timedelta("23h"), np.nan)
        if g.source.iloc[0] == "AEMET10":
            continue                                                                 # Turís is in the AEMET daily table
        if not (np.isfinite(a06) and np.isfinite(a23)) or (g.precip_mm.dropna() < -0.5).any():
            continue
        dd, ii = tree5.query(km([g.lat.iloc[0]], [g.lon.iloc[0]])[0])
        sh = share.loc[ok5.index[ii]] if dd < 15 and w5[ok5.index[ii]].sum() > 5 else 1.0
        prow.append(("PARTNER_" + g.source.iloc[0], sid, g.name.iloc[0], g.lat.iloc[0], g.lon.iloc[0], d29,
                     (a23 - a06) / max(sh, 0.5)))
    pt = pd.DataFrame(prow, columns=base.columns)
    b29 = base[base.day == d29]
    dd, _ = cKDTree(km(b29.lat, b29.lon)).query(km(pt.lat, pt.lon), k=1)
    pt = pt[dd > 1.0]
    # neighbour check (the partner networks include amateur stations): drop where the station and the
    # IDW of the official stations around it differ by more than x3 on a wet day
    nb = idw(km(b29.lat, b29.lon), b29.precip_mm.values, km(pt.lat, pt.lon))
    keep = ~((np.maximum(pt.precip_mm, nb) > 20) & ((pt.precip_mm > 3 * nb) | (pt.precip_mm < nb / 3)))
    print(f"partner stations for 29 Oct: {len(pt)} new (> 1 km), {int((~keep).sum())} dropped by the neighbour check")
    g = pd.concat([base, pt[keep.values]], ignore_index=True)
    g = g[g.day.isin(DAYS)].dropna(subset=["precip_mm"])
    g.to_csv(os.path.join(OBS, "dana_gauges_daily_v2.csv"), index=False)
    return g


def timing_stations():
    """Hourly fractions (06-06 UTC days) at the timing stations: complete SAIH5 days; on 29 Oct also the
    complete partner series and Turís 10-min (06-23 UTC, the files end there; rain after 23 UTC is small)."""
    h5, st5 = _saih5_hourly()
    out = {}
    for d in DAYS:
        hrs = pd.date_range(d + pd.Timedelta("7h"), d + pd.Timedelta("30h"), freq="h")
        w = h5.reindex(hrs)
        good = w.columns[w.notna().all() & (w.sum() >= 1.0)]
        fr = w[good] / w[good].sum()
        meta = st5.set_index("station").loc[good, ["lat", "lon"]]
        out[d] = (fr, meta, w[good].sum())
    d29 = pd.Timestamp("2024-10-29")
    p = pd.read_csv(os.path.join(OBS, "partner_hourly.csv"), parse_dates=["time_utc"])
    hrs = pd.date_range(d29 + pd.Timedelta("7h"), d29 + pd.Timedelta("30h"), freq="h")
    extra, meta, tot = {}, [], {}
    for sid, g in p.groupby("station"):
        s = g.set_index("time_utc").precip_mm.reindex(hrs)
        part = s[s.index <= d29 + pd.Timedelta("23h")]
        if part.isna().any() or part.sum() < 1.0 or (part < -0.5).any():
            continue
        s = s.fillna(0.0).clip(lower=0)
        extra[sid] = s / s.sum(); tot[sid] = s.sum()
        meta.append((sid, g.lat.iloc[0], g.lon.iloc[0]))
    fr, m, t = out[d29]
    m2 = pd.DataFrame(meta, columns=["station", "lat", "lon"]).set_index("station")
    d, _ = cKDTree(km(m.lat, m.lon)).query(km(m2.lat, m2.lon), k=1)             # SAIH5 wins within 1 km
    m2 = m2[d > 1.0]
    out[d29] = (pd.concat([fr, pd.DataFrame(extra)[m2.index]], axis=1), pd.concat([m, m2]), pd.concat([t, pd.Series(tot)[m2.index]]))
    return out


def timing_weights(meta, dst, method):
    """(cells, k) station indices and weights for the hourly fractions."""
    tree = cKDTree(km(meta.lat, meta.lon))
    if method == "nearest":
        d, i = tree.query(dst, k=1)
        return d[:, None], i[:, None], np.ones((len(dst), 1))
    d, i = tree.query(dst, k=min(4, len(meta)))
    w = 1.0 / np.maximum(d, 0.05) ** 2
    return d, i, w / w.sum(1, keepdims=True)


def build2():
    """v2 daily grid (with partner stations), timing check (leave-one-out), and the hourly reference."""
    os.makedirs(FIG, exist_ok=True)
    g = gauges_v2()
    g1 = gauges()
    d29 = pd.Timestamp("2024-10-29")
    # 1) totals: leave-one-out on 29 Oct at every station, v1 network vs v2 network
    x = g[g.day == d29].reset_index(drop=True)
    xy = km(x.lat, x.lon)
    v1_net = g1[g1.day == d29]
    rows = []
    for j in range(len(x)):
        m = np.ones(len(x), bool); m[j] = False
        far1 = np.hypot(*(km(v1_net.lat, v1_net.lon) - xy[j]).T) > 0.5           # the station itself out of v1 too
        rows.append({"station": x.station[j], "source": x.source[j], "name": x["name"][j], "lat": x.lat[j], "lon": x.lon[j],
                     "gauge": x.precip_mm[j],
                     "v1 network": idw(km(v1_net.lat[far1], v1_net.lon[far1]), v1_net.precip_mm[far1].values, xy[j:j + 1])[0],
                     "v2 network": idw(xy[m], x.precip_mm[m].values, xy[j:j + 1])[0]})
    loo = pd.DataFrame(rows)
    loo.to_csv(os.path.join(FIG, "loo_29oct_v2.csv"), index=False, float_format="%.2f")
    res = []
    for sub, msk in (("all", loo.gauge >= 0), ("partner stations", loo.source.str.startswith("PARTNER")),
                     ("official (AEMET, SAIH)", ~loo.source.str.startswith("PARTNER")), ("core (>= 300 mm)", loo.gauge >= 300)):
        for c in ("v1 network", "v2 network"):
            o, e = loo.gauge[msk].values, loo[c][msk].values
            ok = np.isfinite(o) & np.isfinite(e)
            res.append({"subset": sub, "estimate": c, "n": int(ok.sum()), "ratio_of_sums": e[ok].sum() / o[ok].sum(),
                        "mae_mm": np.mean(np.abs(e[ok] - o[ok])), "r": np.corrcoef(o[ok], e[ok])[0, 1]})
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(FIG, "loo_29oct_v2_summary.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 200)
    print("leave-one-out, 29 Oct (06-06 UTC):\n" + res.round(2).to_string(index=False))

    # 2) timing: leave-one-out at the timing stations, wet days (>= 20 mm), hourly fractions x the station's own total
    ts = timing_stations()
    im = hourly("IMERG Final")
    trows = []
    for d, (fr, meta, tot) in ts.items():
        if len(meta) < 5:                                                            # no 5-min coverage (27 Oct)
            continue
        hrs = fr.index
        imd = im.sel(time=hrs).interp(latitude=xr.DataArray(meta.lat.values, dims="pt"),
                                      longitude=xr.DataArray(meta.lon.values, dims="pt"), method="linear").values
        imf = imd / np.where(imd.sum(0) > 0, imd.sum(0), 1)
        for j, sid in enumerate(meta.index):
            if tot[sid] < 20:
                continue
            o = fr[sid].values * tot[sid]
            others = meta.drop(index=sid)
            est = {"IMERG Final timing": imf[:, j]}
            for meth in ("nearest", "idw4"):
                dd, ii, ww = timing_weights(others, km([meta.lat.iloc[j]], [meta.lon.iloc[j]]), meth)
                est[meth + " gauge timing"] = (fr[others.index].values[:, ii[0]] * ww[0]).sum(1)
            for k, f_ in est.items():
                e = f_ * tot[sid]
                trows.append({"day": d, "station": sid, "method": k, "dist_km": float(dd[0, 0]) if k != "IMERG Final timing" else np.nan,
                              "obs_max1h": o.max(), "est_max1h": e.max(),
                              "obs_max3h": pd.Series(o).rolling(3).sum().max(), "est_max3h": pd.Series(e).rolling(3).sum().max(),
                              "r": np.corrcoef(o, e)[0, 1] if e.std() > 0 else np.nan})
    tl = pd.DataFrame(trows)
    tl.to_csv(os.path.join(FIG, "timing_loo.csv"), index=False, float_format="%.3f")
    tsum = tl.groupby("method").agg(n=("r", "size"), r_median=("r", "median"),
                                    max1h_ratio=("est_max1h", "sum"), max3h_ratio=("est_max3h", "sum"))
    tsum["max1h_ratio"] /= tl.groupby("method").obs_max1h.sum(); tsum["max3h_ratio"] /= tl.groupby("method").obs_max3h.sum()
    core = tl[tl.obs_max1h >= 30].groupby("method").agg(n=("r", "size"), r_median=("r", "median"), est=("est_max1h", "sum"), obs=("obs_max1h", "sum"))
    tsum["core_n"], tsum["core_r_median"], tsum["core_max1h_ratio"] = core.n, core.r_median, core.est / core.obs
    tsum.to_csv(os.path.join(FIG, "timing_loo_summary.csv"), float_format="%.3f")
    print("\ntiming leave-one-out (wet station-days; core = observed max 1 h >= 30 mm):\n" + tsum.round(2).to_string())
    gm = tsum.loc[[m for m in tsum.index if "gauge" in m]]
    score = gm.core_r_median - (gm.core_max1h_ratio - 1).abs()                      # timing right and intensity kept
    best = score.idxmax().split()[0]
    print(f"gauge timing method used: {best}")

    # 3) daily grid v2 and the hourly reference
    sm = xr.open_dataset(os.path.join(REPO, "data", "wflow", "dana_2024", "model", "staticmaps.nc"))
    lat, lon = sm.latitude.values, sm.longitude.values
    LON, LAT = np.meshgrid(lon, lat)
    dst = km(LAT.ravel(), LON.ravel())
    grid = np.full((len(DAYS),) + LAT.shape, np.nan, "float32")
    for i, d in enumerate(DAYS):
        xd = g[g.day == d]
        grid[i] = idw(km(xd.lat, xd.lon), xd.precip_mm.values, dst).reshape(LAT.shape)
    gd = xr.DataArray(grid, dims=("day", "latitude", "longitude"), coords={"day": DAYS, "latitude": lat, "longitude": lon},
                      name="precip", attrs={"units": "mm per day", "note": "day = 06-06 UTC; IDW of AEMET + SAIH (5-min where complete) + partner (AVAMET, SISRITEL, Torrent; 29 Oct) daily totals; scripts/dana_reference_rain.py build2"})
    gd.to_netcdf(os.path.join(OBS, "gauge_grid_daily_v2.nc"), encoding={"precip": {"zlib": True, "complevel": 4}})
    imc = im.interp(latitude=lat, longitude=lon, method="linear").fillna(0.0).clip(min=0)
    parts, timing_src = [], []
    for i, d in enumerate(DAYS):
        hrs = pd.date_range(d + pd.Timedelta("7h"), d + pd.Timedelta("30h"), freq="h")
        h = imc.sel(time=hrs).values
        itot = h.sum(0)
        dom = h.mean((1, 2)); dom = dom / dom.sum() if dom.sum() > 0 else np.full(24, 1 / 24)
        frac = np.where(itot[None] >= 0.2, h / np.where(itot > 0, itot, 1)[None], dom[:, None, None])   # IMERG fallback
        if d in ts and len(ts[d][1]) >= 5:
            fr, meta, _ = ts[d]
            dd, ii, ww = timing_weights(meta, dst, best)
            gf = np.einsum("tck,ck->tc", fr.values[:, ii], ww).reshape((24,) + LAT.shape)
            near = (dd[:, 0] <= 25.0).reshape(LAT.shape)
            frac = np.where(near[None], gf, frac)
            timing_src.append((d, float(near.mean())))
        g_ = np.nan_to_num(grid[i])
        parts.append(xr.DataArray((g_[None] * frac).astype("float32"), dims=("time", "latitude", "longitude"),
                                  coords={"time": hrs, "latitude": lat, "longitude": lon}))
    ref = xr.concat(parts, dim="time")
    end_grid = pd.Timestamp(ref.time.values[-1])
    ms = hourly("MSWEP").interp(latitude=lat, longitude=lon, method="linear").fillna(0.0)
    ms = ms.sel(time=ms.time > np.datetime64(end_grid)).astype("float32")
    out = xr.concat([ref, ms.drop_vars([c for c in ms.coords if c not in ("time", "latitude", "longitude")])], dim="time")
    out.name = "precip"
    out.attrs = {"units": "mm per hour", "time_convention": "end of hour",
                 "note": f"v2: AEMET + SAIH + partner daily gauge grid with gauge timing ({best}; SAIH 5-min, partner hourly, "
                         f"Turís 10-min; IMERG Final beyond 25 km of a timing gauge) to {end_grid}, then MSWEP Past; "
                         f"scripts/dana_reference_rain.py build2"}
    path = os.path.join(OBS, "gauge_grid_hourly_gauges.nc")
    out.to_dataset().to_netcdf(path, encoding={"precip": {"zlib": True, "complevel": 4}})
    print("share of grid cells with gauge timing per day:", [(f"{d:%d %b}", round(s, 2)) for d, s in timing_src])
    print(f"hourly reference v2 {pd.Timestamp(out.time.values[0])} -> {pd.Timestamp(out.time.values[-1])}: {path}")

    # 4) checks against the partner/CHJ numbers and the v1 reference
    v1 = xr.open_dataset(os.path.join(OBS, "gauge_grid_hourly_imergfinal.nc"))["precip"]
    m = xr.open_dataset(os.path.join(REPO, "data", "wflow", "dana_2024", "gauge_upstream.nc"))["mask"]
    names = pd.read_csv(os.path.join(REPO, "data", "wflow", "dana_2024", "gauges.csv")).set_index("id")["name"]
    chk = []
    for lab, ds in (("v1 (IMERG timing)", v1), ("v2 (gauge timing)", out)):
        tot = ds.sel(time=slice("2024-10-28T01", "2024-11-05T00")).sum("time").interp(latitude=m.latitude, longitude=m.longitude, method="nearest")
        for gid in m.gauge.values:
            w = m.sel(gauge=gid).astype(bool) * np.cos(np.deg2rad(m.latitude))
            chk.append({"reference": lab, "gauge": names.get(gid, gid), "episode_mm": float((tot * w).sum() / w.sum())})
        for nm, la, lo in (("TURIS (AEMET)", 39.39222, -0.62472), ("CHIVA (SAIH)", 39.4575, -0.7355)):
            s = ds.sel(latitude=la, longitude=lo, method="nearest").sel(time=slice("2024-10-29T07", "2024-10-30T06")).to_series()
            chk.append({"reference": lab, "gauge": nm, "day29_mm": s.sum(), "max1h_mm": s.max(), "max3h_mm": s.rolling(3).sum().max()})
    chk = pd.DataFrame(chk)
    chk.to_csv(os.path.join(FIG, "reference_v1_v2_checks.csv"), index=False, float_format="%.1f")
    print("\n" + chk.round(0).to_string(index=False))
    print("observed: Turís 710.8 mm (29 Oct 06 UTC - 30 Oct 06 UTC), max 1 h 179 mm; CHJ areal: Poyo 392 mm, Magro 306 mm (28 Oct - 4 Nov)")

    # figure: v1 vs v2 event totals and the Turís / Poyo hourly series
    fig, axs = plt.subplots(1, 3, figsize=(18 / 2.54, 6 / 2.54), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, (lab, ds) in zip(axs[:2], (("a  v1: gauges × IMERG timing", v1), ("b  v2: gauges (+partner) × gauge timing", out))):
        ev = ds.sel(time=slice("2024-10-29T07", "2024-10-30T06"))
        pc = ax.pcolormesh(lon, lat, ev.max("time"), cmap="Blues", vmin=0, vmax=120, shading="nearest")
        ax.set_xlim(-1.3, -0.2); ax.set_ylim(39.0, 39.8)
        ax.set_title(lab + "\nmax 1 h, 29 Oct (mm)", fontsize=6.5, loc="left")
        ax.tick_params(labelsize=5.5)
    fig.colorbar(pc, ax=axs[:2], shrink=0.8).ax.tick_params(labelsize=6)
    obs = pd.read_csv(os.path.join(OBS, "partner_hourly.csv"), parse_dates=["time_utc"])
    tu = obs[obs.station == "AEMET10_8337X"].set_index("time_utc").precip_mm
    axs[2].bar(tu.index, tu.values, width=1 / 24, color="#bbbbbb", label="Turís gauge (AEMET 10-min)")
    for lab, ds, col in (("v1", v1, "#1f77b4"), ("v2", out, "k")):
        s = ds.sel(latitude=39.39222, longitude=-0.62472, method="nearest").sel(time=slice("2024-10-29T00", "2024-10-30T00")).to_series()
        axs[2].plot(s.index, s.values, color=col, lw=1, label=f"{lab} at Turís")
    axs[2].set_title("c  Turís, 29 Oct (mm/h, UTC)", fontsize=6.5, loc="left")
    axs[2].legend(fontsize=5, frameon=False); axs[2].tick_params(labelsize=5.5)
    axs[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H"))
    fig.savefig(os.path.join(FIG, "reference_v1_v2.png"), dpi=200)
    plt.close(fig)
    print(FIG)


if __name__ == "__main__":
    {"download": download, "build": build, "hourly": hourly_reference, "partner": partner, "build2": build2}[sys.argv[1]]()
