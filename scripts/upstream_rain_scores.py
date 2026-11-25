#!/usr/bin/env python3
"""
Forecast rainfall over the upstream area of each selected gauge, scored against
the validated reference (step 4 of the extreme-gauge framework).

Every product is put on wflow's 3 h end-stamped slots and regridded bilinearly
to the model grid exactly as the wflow forcing is (wflow_prep_forcing), then
averaged over each gauge's full upstream area (gauge_upstream.py). So the rain
scored here is the rain the discharge runs of step 5 receive.

  truth      : the event's validated reference (Catalonia: XEMA gauge grid)
  also shown : ERA5 (deterministic; its CRPS is the absolute error)
  forecasts  : IFS, AIFS, WN2 members, every init at or before event_start
  window     : event_start -> event_end (3 h slots ending inside it), or with
               --band core the event core: the shortest stretch of that window
               holding >= 80 % of the reference's gauge-mean areal rain (at least
               24 h). Either way the band is the same for every start, so each
               issue time is scored on exactly the same valid times; the core band
               stops low-intensity rain from diluting the score of one start but
               not another (user, 16 Oct 2026).

Metrics per (gauge, product, init): fair CRPS of the window total (mm),
ensemble-mean and -median totals, ensemble-mean bias (%) and absolute error,
the truth's rank in the ensemble (fraction of members below it), the share of
members that reach half of the truth, and on the 3 h series in the window:
RMSE and correlation of the ensemble mean, slot-averaged fair CRPS and the
peak ratio. Head-to-head: the best product per (gauge, init, metric), tie when
the runner-up is within 5 % (0.02 for correlation).

Outputs in figures/<slug>/upstream_rain/:
    upstream_rain_scores.csv
    upstream_areas_reference_total.png   reference total with the gauge areas
    upstream_rain_vs_lead.png            member spread vs init time, per gauge
    upstream_rain_crps_vs_lead.png       CRPS / truth vs lead, per product
    upstream_rain_best.csv, upstream_rain_wins.csv, upstream_rain_wins.png

    python scripts/upstream_rain_scores.py catalonia_2026
    python scripts/upstream_rain_scores.py catalonia_2026 --band core   # -> upstream_rain_core/
    python scripts/upstream_rain_scores.py catalonia_2026 --ifs025      # IFS averaged to 0.25 deg -> upstream_rain_ifs025/

Every run also writes upstream_rain_members.csv.gz: the window total of every member per
(gauge, product, init), for the threshold (Brier) scores of synthesis_stats.py.
"""
import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wflow_prep_forcing as wpf                                   # noqa: E402
from event_config import EVENTS, EXCLUDED_PRODUCTS                 # noqa: E402
from plot_style import IFS_COLOR, AIFS_COLOR, WN2_COLOR, ERA5_COLOR  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Validated reference per event (hourly, end-stamped, mm per hour)
REFERENCE = {
    "hans_2023": {"label": "MSWEP (reference run forcing)", "kind": "wflow_base"},
    # Kristin (user, 20 Oct): ERA5 is the reference (closest to the 11 IPMA gauges, and its
    # reference runs come closest to the reported Coimbra peak); AIFS and WN2 are trained on
    # ERA5, so MSWEP and ERA5 scaled to the gauges are kept as sensitivities (ALT_REFERENCE).
    # "run": the wflow case forced by it (MSWEP to 26 Jan 00Z, then ERA5), scored by gauge_discharge_scores.
    **{s: {"label": "ERA5 (AIFS and WN2 are trained on ERA5)", "path": "data/processed/kristin_2026/era5_wflow.nc",
           "var": "tp", "run": "era5ref_jan26"} for s in ("kristin_mondego", "kristin_lis")},
    "boris_sudetes": {"label": "CHMI + IMGW gauge grid",
                      "path": "data/raw/boris_sudetes/obs/gauge_grid_hourly.nc", "var": "precip"},
    "texas_2025": {"label": "MRMS scaled to GHCN-D gauges", "path": "data/raw/texas_2025/obs/mrms_qpe_1h_gaugescaled.nc", "var": "precip"},
    "california_2023": {"label": "MRMS scaled to GHCN-D gauges", "path": "data/raw/california_2023/obs/mrms_qpe_1h_gaugescaled.nc", "var": "precip"},
    "zermatt_2024": {"label": "station grid scaled to RhiresD",
                     "path": "data/raw/zermatt_2024/obs/gauge_grid_hourly_rhiresd.nc", "var": "precip"},
    "texas_2026": {"label": "MRMS scaled to GHCN-D gauges", "path": "data/raw/texas_2026/obs/mrms_qpe_1h_gaugescaled.nc", "var": "precip"},
    "mississippi_2026": {"label": "MRMS scaled to GHCN-D gauges", "path": "data/raw/mississippi_2026/obs/mrms_qpe_1h_gaugescaled.nc", "var": "precip"},
    "helene_2024": {"label": "MRMS scaled to GHCN-D gauges", "path": "data/raw/helene_2024/obs/mrms_qpe_1h_gaugescaled.nc", "var": "precip"},
    "dana_2024": {"label": "AEMET + SAIH + AVAMET/SISRITEL gauge grid, SAIH 5-min gauge timing",
                  "path": "data/raw/dana_2024/obs/gauge_grid_hourly_gauges.nc", "var": "precip"},   # v2, 23 Oct
    "catalonia_2026": {"label": "XEMA gauge grid",
                       "path": "data/raw/catalonia_2026/obs/xema_grid_hourly.nc", "var": "precip"},
}
# Alternative references for a sensitivity run (--ref NAME): rain file and the wflow case
# forced by it (gauge_discharge_scores.py --ref NAME scores against that case's run).
ALT_REFERENCE = {
    slug: {"mswep": {"label": "MSWEP NRT (sensitivity; 18 % low at Coimbra, 41 % at Monte Real)",
                     "kind": "wflow_base", "run": "hist_3h"},
           "era5_scaled": {"label": "ERA5 scaled to 11 IPMA gauges (sensitivity)",
                           "path": "data/processed/kristin_2026/era5_wflow_stationscaled.nc", "var": "tp",
                           "run": "era5scaled_jan26"}}
    for slug in ("kristin_mondego", "kristin_lis")}
ALT_REFERENCE["dana_2024"] = {          # v1 (20 Oct): daily AEMET + SAIH grid with IMERG Final timing
    "imerg_timing": {"label": "AEMET + SAIH daily gauge grid, IMERG Final timing (v1, sensitivity)",
                     "path": "data/raw/dana_2024/obs/gauge_grid_hourly_imergfinal.nc", "var": "precip"}}
# US events: main reference is MRMS scaled to GHCN-D gauges since 21 Oct; unscaled MRMS as sensitivity
for _s in ("california_2023", "texas_2025", "texas_2026", "mississippi_2026", "helene_2024"):
    ALT_REFERENCE[_s] = {"mrms_raw": {"label": "MRMS (unscaled, sensitivity)",
                                     "path": f"data/raw/{_s}/obs/mrms_qpe_1h.nc", "var": "precip"}}
# Events whose gauges flooded at different times: per gauge group, the rain and
# discharge windows and the wflow starts scored there (gauge_discharge_scores).
# Everything else uses event_start -> event_end and every start.
# A window opens at its latest start (the -1 d one, 16 Oct), so every start is scored
# on the same valid times (see open action 9 in docs/event_impact_catchments.md).
WINDOWS = {
    # Two episodes eight days apart (16 Oct): the Matter arm floods on 21 Jun, the Saas
    # arm and the outlet on 29-30 Jun (the runs end 3 Jul since 17 Oct); a single
    # band cannot hold both.
    "zermatt_2024": [
        {"gauges": [2, 4], "label": "Matter Vispa (21 Jun storm)",
         "rain": ("2024-06-21 00:00", "2024-06-23 00:00"), "flow": ("2024-06-21 00:00", "2024-06-25 00:00"),
         "cases": ["jun15", "jun19", "jun20"]},
        {"gauges": [1, 3], "label": "Saaser Vispa and Visp outlet (29-30 Jun)",
         "rain": ("2024-06-29 00:00", "2024-06-30 00:00"), "flow": ("2024-06-29 00:00", "2024-07-03 00:00"),
         "cases": ["jun22", "jun27", "jun28"]},   # -7/-2/-1 d since 19 Nov (was jun19/jun20, 9-10 d)
    ],
    # Kristin (20 Oct): the reference runs rise again from 2 Feb with the next storm, so
    # the flow window ends on 2 Feb 00Z (Coimbra peaks 30 Jan 00Z); the rain window is
    # the event window.
    "kristin_mondego": [
        {"gauges": [6114300, 9112013, 9112007, 6114500, 9112008], "label": "Mondego (Kristin, 27 Jan - 1 Feb)",
         "rain": ("2026-01-27 00:00", "2026-02-01 00:00"), "flow": ("2026-01-27 00:00", "2026-02-02 00:00"),
         "cases": ["jan20", "jan25", "jan26"]},   # -7 d since 3 Nov (was jan22, -5 d)
    ],
    "kristin_lis": [
        {"gauges": [9112019], "label": "Lis (Kristin, 27 Jan - 1 Feb)",
         "rain": ("2026-01-27 00:00", "2026-02-01 00:00"), "flow": ("2026-01-27 00:00", "2026-02-02 00:00"),
         "cases": ["jan20", "jan25", "jan26"]},   # -7 d since 3 Nov (was jan22, -5 d)
    ],
    "california_2023": [
        {"gauges": [2, 4], "label": "Cosumnes / Soquel (peaks 31 Dec - 1 Jan)",
         "rain": ("2022-12-30 00:00", "2023-01-02 00:00"), "flow": ("2022-12-30 00:00", "2023-01-04 00:00"),
         "cases": ["dec24", "dec29", "dec30"]},
        {"gauges": [1, 3], "label": "Arroyo Seco / San Lorenzo (peaks 9 Jan)",
         "rain": ("2023-01-08 00:00", "2023-01-11 00:00"), "flow": ("2023-01-08 00:00", "2023-01-14 00:00"),
         "cases": ["jan02", "jan07", "jan08"]},
    ],
}


def gauge_windows(slug, ids, kind="rain"):
    """[(t0, t1, [gauge ids], case suffixes or None)] for an event."""
    if slug not in WINDOWS:
        t0, t1 = window(slug)
        return [(t0, t1, list(ids), None)]
    return [(pd.Timestamp(w[kind][0]), pd.Timestamp(w[kind][1]), [g for g in ids if g in w["gauges"]], w["cases"])
            for w in WINDOWS[slug]]


PRODUCTS = {"ifs": ("ifs_processed.nc", "IFS", IFS_COLOR),
            "aifs": ("aifs_processed.nc", "AIFS", AIFS_COLOR),
            "wn2": ("weathernext_processed.nc", "WN2", WN2_COLOR)}


def gauge_label(row):
    """Short gauge label: the code after '|' in `station` (Catalonia 'CALC001219|BE-20'
    -> 'BE-20'), otherwise a shortened name (Sudetes, Texas)."""
    st = str(row.get("station", ""))
    if "|" in st:
        return st.split("|")[-1]
    n = str(row.get("name", st))
    for a_, b_ in ((" at ", " "), (" nr ", " "), ("Guadalupe R.", "Guadalupe"), (" (outlet)", ""), (" – ", " ")):
        n = n.replace(a_, b_)
    return n[:22]


def gauge_area(row):
    """Official upstream area if given, else the model's."""
    for k in ("official_km2", "uparea_km2", "model_km2"):
        v = row.get(k)
        if v is not None and np.isfinite(pd.to_numeric(v, errors="coerce")):
            return float(v)
    return np.nan


def fair_crps(ens, obs):
    m = len(ens)
    return np.abs(ens - obs).mean() - np.abs(ens[:, None] - ens[None]).sum() / (2 * m * (m - 1))


class Areas:
    """Gauge upstream masks on the model grid, and area means of any field.

    Lat/lon models: products are regridded bilinearly as in wflow_prep_forcing.
    Projected models (the Drammen UTM grid): sampled bilinearly at the cell
    centres as in wflow_drammen_forcing. Fields already on the model grid are
    averaged directly.
    """

    def __init__(self, slug):
        m = xr.open_dataset(os.path.join(REPO, "data", "wflow", slug, "gauge_upstream.nc"))["mask"].load()
        self.projected = "x" in m.dims
        if self.projected:
            import wflow_drammen_forcing as wdf
            self._wdf = wdf
            x, y, self.lat2d, self.lon2d, _ = wdf.load_model_grid()
            self.x, self.y = m["x"].values, m["y"].values
            assert np.allclose(x, self.x) and np.allclose(y, self.y), "gauge masks are not on the model grid"
            w = m.astype(float)                                                  # equal-area cells
            self.grid_dims = ("y", "x")
        else:
            self.lat, self.lon = m["latitude"].values, m["longitude"].values
            w = m.astype(float) * np.cos(np.deg2rad(m["latitude"]))
            self.grid_dims = ("latitude", "longitude")
        self.shape = (m.sizes[self.grid_dims[0]], m.sizes[self.grid_dims[1]])
        self.w = (w / w.sum(list(self.grid_dims))).values.reshape(m.sizes["gauge"], -1)   # (g, cells)
        self.ids = m["gauge"].values
        self._ops = {}

    def _to_model(self, flat):
        """flat (time, lat, lon) on a lat/lon grid -> (time, model cells)."""
        if self.projected:
            r = self._wdf.sample_to_model(flat.rename({"lat": "latitude", "lon": "longitude"}),
                                          self.y, self.x, self.lat2d, self.lon2d).fillna(0.0)
            return r.transpose("time", "y", "x").values.reshape(flat.sizes["time"], -1)
        r = wpf.regrid_to_wflow(flat, self.lat, self.lon)
        return r.transpose("time", "latitude", "longitude").values.reshape(flat.sizes["time"], -1)

    def operator(self, lat, lon):
        """(gauge, source cell) weights: regrid to the model grid, then area mean."""
        key = (lat.tobytes(), lon.tobytes())
        if key not in self._ops:
            n = len(lat) * len(lon)
            eye = xr.DataArray(np.eye(n).reshape(n, len(lat), len(lon)), dims=("time", "lat", "lon"),
                               coords={"time": np.arange(n), "lat": lat, "lon": lon})
            self._ops[key] = self.w @ self._to_model(eye).T                       # (g, n)
        return self._ops[key]

    def mean(self, da):
        """Area means (gauge, ...) of da(..., lat, lon), or of a field on the model grid."""
        if all(d in da.dims for d in self.grid_dims) and tuple(da.sizes[d] for d in self.grid_dims) == self.shape:
            da = da.transpose(..., *self.grid_dims)
            v = da.values.reshape(*da.shape[:-2], -1)
            return np.tensordot(self.w, v, axes=(1, v.ndim - 1))
        lat = next(d for d in da.dims if d.lower() in ("lat", "latitude"))
        lon = next(d for d in da.dims if d.lower() in ("lon", "longitude"))
        da = da.transpose(..., lat, lon)
        if da.sizes[lat] * da.sizes[lon] > 4000:
            # fine source grid (radar, gauge grid): the operator would be n x n,
            # so regrid the field itself (the same step, done once)
            lead_shape = da.shape[:-2]
            k = int(np.prod(lead_shape)) if lead_shape else 1
            flat = xr.DataArray(da.values.reshape(k, da.sizes[lat], da.sizes[lon]), dims=("time", "lat", "lon"),
                                coords={"time": np.arange(k), "lat": da[lat].values, "lon": da[lon].values})
            out = self.w @ self._to_model(flat).T                                # (g, k)
            return out.reshape(len(self.ids), *lead_shape) if lead_shape else out[:, 0]
        op = self.operator(da[lat].values, da[lon].values)
        v = da.values.reshape(*da.shape[:-2], -1)
        return np.tensordot(op, v, axes=(1, v.ndim - 1))


def window(slug):
    c = EVENTS[slug]
    return pd.Timestamp(c["event_start"]), pd.Timestamp(c["event_end"]).ceil("3h")


def slots_in(t0, t1):
    return pd.date_range(t0 + pd.Timedelta("3h"), t1, freq="3h")


CORE_MASS, CORE_MIN_SLOTS = 0.8, 8      # share of the reference rain; minimum 24 h


def core_band(slug, areas, w0, w1, gids, mass=CORE_MASS):
    """Shortest stretch of (w0, w1] holding `mass` of the gauge-mean areal reference rain.

    Returns (start, end) so that slots_in(start, end) are the band's 3 h slots."""
    tw = np.asarray(reference_totals(slug, areas, w0, w1)[0])           # (gauge, slot)
    pos = {g: i for i, g in enumerate(areas.ids)}
    s = np.nan_to_num(tw[[pos[g] for g in gids]].mean(axis=0))
    slots = slots_in(w0, w1)
    cs = np.r_[0.0, np.cumsum(s)]
    best = None
    for i in range(len(s)):
        j = int(np.searchsorted(cs, cs[i] + mass * cs[-1], side="left"))
        if j <= len(s) and (best is None or j - i < best[0]):
            best = (j - i, i, j)
    n, i, j = best
    if n < CORE_MIN_SLOTS:                                                # widen to 24 h, centred
        i = max(0, min(i - (CORE_MIN_SLOTS - n) // 2, len(s) - CORE_MIN_SLOTS))
        j = i + CORE_MIN_SLOTS
    share = float(s[i:j].sum() / s.sum()) if s.sum() > 0 else np.nan
    print(f"  core band {slots[i] - pd.Timedelta('3h')} -> {slots[j - 1]} ({(j - i) * 3} h of {len(s) * 3} h, {100 * share:.0f} % of the rain)")
    return slots[i] - pd.Timedelta("3h"), slots[j - 1]


def reference_totals(slug, areas, t0, t1):
    r = REFERENCE[slug]
    if r.get("kind") == "wflow_base":
        # the reference run's own precipitation: 3 h, end-stamped, on the model grid
        import glob
        files = sorted(glob.glob(os.path.join(REPO, "data", "wflow", slug, "model", "inmaps_base", "base_*.nc")))
        parts = []
        for f in files:
            with xr.open_dataset(f) as d:
                sel = d["precip"].sel(time=d["time"].isin(slots_in(t0, t1).values))
                if sel.sizes["time"]:
                    parts.append(sel.load())
        h3 = xr.concat(parts, dim="time").sel(time=slots_in(t0, t1))
        return areas.mean(h3.fillna(0.0)), h3.sum("time")
    with xr.open_dataset(os.path.join(REPO, r["path"])) as d:
        h = d[r["var"]].load()
    h3 = h.resample(time="3h", closed="right", label="right").sum(min_count=3).sel(time=slots_in(t0, t1))
    if bool(h3.isnull().all(dim=[x for x in h3.dims if x != "time"]).any()):
        raise ValueError("reference has empty slots in the window")
    series = areas.mean(h3.fillna(0.0))                                       # (gauge, slot)
    return series, h3.sum("time")


def era5_totals(slug, areas, t0, t1):
    f = os.path.join(REPO, "data", "processed", EVENTS[slug].get("proc_slug", slug), "era5_processed.nc")
    if not os.path.exists(f):
        return None
    tp = xr.open_dataset(f)["tp"]
    tp3 = tp.resample(time="3h", closed="right", label="right").sum(min_count=3)
    s = slots_in(t0, t1)
    if not set(s).issubset(set(pd.DatetimeIndex(tp3["time"].values))):
        return None
    return areas.mean(tp3.sel(time=s))                                         # (gauge, slot)


def forecast_series(ds, init, areas, t0, t1):
    """(gauge, member, slot) 3 h area means in the window for one init, or None if it does not cover it."""
    tp = ds["tp"].sel(init_time=init).transpose("step", "number", ...).load()
    valid = np.isfinite(tp.values).reshape(tp.sizes["step"], -1).all(axis=1)
    n_ok = int(np.argmin(valid)) if not valid.all() else len(valid)
    tp = tp.isel(step=slice(0, n_ok))
    step_h = pd.to_timedelta(tp["step"].values) / pd.Timedelta("1h")
    init = pd.Timestamp(init)
    s = slots_in(t0, t1)
    slot_h = ((s - init) / pd.Timedelta("1h")).values
    if n_ok == 0 or slot_h.min() - 3 < 0 or slot_h.max() > step_h.max():
        return None
    v = wpf.accum_to_slots(tp.values, step_h.values, slot_h).clip(min=0.0)    # (slot, member, lat, lon)
    lat = next(d for d in tp.dims if d.lower() in ("lat", "latitude"))
    lon = next(d for d in tp.dims if d.lower() in ("lon", "longitude"))
    da = xr.DataArray(v, dims=("slot", "number", lat, lon), coords={lat: tp[lat].values, lon: tp[lon].values})
    return areas.mean(da).transpose(0, 2, 1)                                 # (gauge, member, slot)


def coarsen025(ds):
    """IFS on a 0.25 deg lattice: mean of the native cells inside each 0.25 deg cell (scale test)."""
    lat = next(d for d in ds["tp"].dims if d.lower() in ("lat", "latitude"))
    lon = next(d for d in ds["tp"].dims if d.lower() in ("lon", "longitude"))
    out = ds["tp"]
    for dim in (lat, lon):
        v = out[dim].values
        edges = np.arange(np.floor(v.min() / 0.25) * 0.25, v.max() + 0.25 + 1e-6, 0.25)
        cen = edges[:-1] + 0.125
        g = out.groupby_bins(dim, edges, labels=cen).mean()
        out = g.rename({f"{dim}_bins": dim})
        out = out.assign_coords({dim: np.asarray(cen, dtype=float)})
        out = out.dropna(dim, how="all")
    return out.to_dataset(name="tp")


def series_scores(xs, ys):
    """Scores of the 3 h series in the window. xs (member, slot), ys (slot).

    rmse/corr use the ensemble mean (for ERA5 the single series); crps_3h is the
    fair CRPS per slot averaged over the window (timing-aware, probabilistic);
    peak_ratio = wettest 3 h slot of the ensemble mean / of the truth.
    """
    m = xs.mean(axis=0)
    out = {"rmse_3h_mm": float(np.sqrt(np.mean((m - ys) ** 2))),
           "corr_3h": float(np.corrcoef(m, ys)[0, 1]) if m.std() > 0 and ys.std() > 0 else np.nan,
           "peak_ratio": float(m.max() / ys.max()) if ys.max() > 0 else np.nan}
    out["crps_3h_mm"] = (float(np.mean([fair_crps(xs[:, k], ys[k]) for k in range(len(ys))]))
                         if len(xs) > 1 else float(np.mean(np.abs(xs[0] - ys))))
    return out


# metric: (column, better, tie rule) -- relative 5 % as in obs_precip_skill, 0.02 for correlations
METRICS = {"CRPS (total)": ("crps_mm", "low"), "CRPS (3 h series)": ("crps_3h_mm", "low"),
           "RMSE (3 h series)": ("rmse_3h_mm", "low"), "|bias| (total)": ("abs_err_mean_mm", "low"),
           "correlation (3 h)": ("corr_3h", "high")}
TIE_REL, TIE_CORR = 0.05, 0.02


def head_to_head(f, out):
    """Best product per (gauge, init, metric) among inits where every product is present."""
    prods = sorted(f["product"].unique())
    rows = []
    for (gid, init), g in f.groupby(["gauge", "init"]):
        if set(g["product"]) != set(prods):
            continue
        g = g.set_index("product")
        for mname, (col, better) in METRICS.items():
            v = g[col].dropna()
            if len(v) < len(prods):
                continue
            order = v.sort_values(ascending=(better == "low"))
            a, b = order.iloc[0], order.iloc[1]
            tie = (abs(a - b) <= TIE_CORR) if better == "high" else (abs(b - a) <= TIE_REL * max(abs(a), 1e-9))
            rows.append({"gauge": gid, "code": g["code"].iloc[0], "init": init, "lead_d": g["lead_d"].iloc[0],
                         "metric": mname, "best": "tie" if tie else order.index[0],
                         **{f"{p}": v[p] for p in prods}})
    w = pd.DataFrame(rows)
    w.to_csv(os.path.join(out, "upstream_rain_best.csv"), index=False, float_format="%.3f")
    lead_sets = {"all inits": w, "7-day lead (wflow)": w[np.isclose(w.lead_d, 7)],
                 "2-day lead (wflow)": w[np.isclose(w.lead_d, 2)]}
    tab = []
    for lname, ww in lead_sets.items():
        c = ww.groupby(["metric", "best"]).size().unstack(fill_value=0)
        c.insert(0, "leads", lname)
        tab.append(c)
    tab = pd.concat(tab).reset_index()
    cols = ["leads", "metric"] + [p for p in prods + ["tie"] if p in tab]
    tab = tab[cols].fillna(0)
    tab.to_csv(os.path.join(out, "upstream_rain_wins.csv"), index=False)
    print("\nWins per metric (catchment x init cases; tie = within 5 % / 0.02 r):")
    print(tab.to_string(index=False))
    return w


def plot_wins(f, w, code, ids, ref_label, out):
    """(a) CRPS winner per catchment and init, (b) wins per metric over all inits."""
    colors = {"IFS": IFS_COLOR, "AIFS": AIFS_COLOR, "WN2": WN2_COLOR, "tie": "#c8c8c8"}
    order = ["IFS", "AIFS", "WN2", "tie"]
    fig, ax = plt.subplots(1, 2, figsize=(18 / 2.54, 7.5 / 2.54), gridspec_kw={"width_ratios": [1.6, 1]})
    c = w[w.metric == "CRPS (total)"]
    inits = sorted(c["init"].unique())
    for yi, gid in enumerate(ids):
        for xi, it in enumerate(inits):
            r = c[(c.gauge == gid) & (c.init == it)]
            if len(r):
                ax[0].add_patch(plt.Rectangle((xi, yi), 0.92, 0.92, color=colors[r.best.iloc[0]], lw=0))
    ax[0].set_xlim(0, len(inits)); ax[0].set_ylim(len(ids), 0)
    ax[0].set_xticks(np.arange(len(inits)) + 0.46, [pd.Timestamp(i).strftime("%d %Hz") for i in inits],
                     rotation=90, fontsize=5.5)
    ax[0].set_yticks(np.arange(len(ids)) + 0.46, [code[i] for i in ids], fontsize=6)
    ax[0].set_xlabel("forecast start", fontsize=6.5)
    ax[0].set_title(f"(a) lowest CRPS of the event total vs {ref_label}", fontsize=7, loc="left")
    for s_ in ax[0].spines.values():
        s_.set_visible(False)
    cnt = w.groupby(["metric", "best"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    cnt = cnt.reindex([m for m in METRICS if m in cnt.index])
    left = np.zeros(len(cnt))
    for p in order:
        ax[1].barh(np.arange(len(cnt)), cnt[p], left=left, color=colors[p], label=p, height=0.6)
        left += cnt[p].values
    ax[1].set_yticks(np.arange(len(cnt)), cnt.index, fontsize=6); ax[1].invert_yaxis()
    ax[1].tick_params(axis="x", labelsize=6); ax[1].set_xlabel("catchment × start cases won", fontsize=6.5)
    ax[1].set_title("(b) wins per metric, all starts", fontsize=7, loc="left")
    ax[1].legend(fontsize=6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "upstream_rain_wins.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slug")
    ap.add_argument("--band", choices=["event", "core"], default="event")
    ap.add_argument("--ifs025", action="store_true", help="score IFS averaged to 0.25 deg (scale test) and stop")
    ap.add_argument("--ref", default=None, help="alternative reference from ALT_REFERENCE (sensitivity run)")
    a = ap.parse_args()
    slug = a.slug
    out = os.path.join(REPO, "figures", slug, "upstream_rain" + ("_ifs025" if a.ifs025 else "_core" if a.band == "core" else ""))
    if a.ref:
        REFERENCE[slug] = ALT_REFERENCE[slug][a.ref]
        out += f"_ref-{a.ref}"
    os.makedirs(out, exist_ok=True)
    areas = Areas(slug)
    gdf = gpd.read_file(os.path.join(REPO, "data", "wflow", slug, "gauge_upstream.geojson")).set_index("id")
    code = {i: gauge_label(r) for i, r in gdf.iterrows()}
    wins = gauge_windows(slug, areas.ids, "rain")
    if a.band == "core":
        wins = [(*core_band(slug, areas, w0, w1, gids), gids, suf) for w0, w1, gids, suf in wins]
    t0, t1 = (wins[0][0], wins[0][1]) if len(wins) == 1 else window(slug)
    truth_s, ref_map = reference_totals(slug, areas, t0, t1)
    truth = truth_s.sum(axis=1)
    era5_s = era5_totals(slug, areas, t0, t1)
    era5 = era5_s.sum(axis=1) if era5_s is not None else None
    ref_label = REFERENCE[slug]["label"]
    print(f"{slug}: window {t0} -> {t1}; {ref_label} totals:",
          {code[i]: round(float(v)) for i, v in zip(areas.ids, truth)})

    rows, mrows = [], []
    pos = {g: i for i, g in enumerate(areas.ids)}
    prods = {"ifs025": ("ifs_processed.nc", "IFS-0.25", IFS_COLOR)} if a.ifs025 else PRODUCTS
    for w0, w1, gids, _ in wins:
        tw_s = reference_totals(slug, areas, w0, w1)[0] if (w0, w1) != (t0, t1) else truth_s
        e5w = era5_totals(slug, areas, w0, w1) if (w0, w1) != (t0, t1) else era5_s
        for key, (fn, label, _) in prods.items():
            f = os.path.join(REPO, "data", "processed", EVENTS[slug].get("proc_slug", slug), fn)
            if key in EXCLUDED_PRODUCTS or not os.path.exists(f):
                continue
            ds = xr.open_dataset(f)
            if a.ifs025:
                ds = coarsen025(ds.load() if ds["tp"].nbytes < 6e9 else ds)
            for init in ds["init_time"].values:
                if pd.Timestamp(init) > w0:
                    continue
                XS = forecast_series(ds, init, areas, w0, w1)
                if XS is None:
                    print(f"  {label} {pd.Timestamp(init)}: does not cover {w0} -> {w1}, skipped")
                    continue
                lead = (w0 - pd.Timestamp(init)) / pd.Timedelta("1D")
                X = XS.sum(axis=2)
                for gid in gids:
                    gi = pos[gid]
                    x, y = X[gi], float(tw_s[gi].sum())
                    mrows.append({"gauge": int(gid), "product": label, "init": pd.Timestamp(init), "window_start": w0,
                                  "truth_mm": y, **{f"m{k}": float(v) for k, v in enumerate(x)}})
                    ser = series_scores(XS[gi], tw_s[gi])
                    rows.append({"gauge": int(gid), "code": code[gid], "area_km2": gdf.loc[gid, "model_km2"],
                                 "product": label, "init": pd.Timestamp(init), "lead_d": round(lead, 2),
                                 "window_start": w0, "window_end": w1,
                                 "n_members": len(x), "truth_mm": y,
                                 "era5_mm": float(e5w[gi].sum()) if e5w is not None else np.nan,
                                 "ens_mean_mm": x.mean(), "ens_median_mm": np.median(x),
                                 "p10_mm": np.percentile(x, 10), "p90_mm": np.percentile(x, 90), "max_mm": x.max(),
                                 "crps_mm": fair_crps(x, y), "bias_pct": 100 * (x.mean() - y) / y,
                                 "abs_err_mean_mm": abs(x.mean() - y),
                                 "truth_rank": float((x < y).mean()), "frac_ge_half_truth": float((x >= 0.5 * y).mean()),
                                 **ser})
    df = pd.DataFrame(rows)
    pd.DataFrame(mrows).to_csv(os.path.join(out, "upstream_rain_members.csv.gz"), index=False, float_format="%.3f")
    if a.ifs025:
        df.to_csv(os.path.join(out, "upstream_rain_scores.csv"), index=False, float_format="%.3f")
        print(out)
        return
    e5rows = []
    for w0, w1, gids, _ in wins:
        tw_s = reference_totals(slug, areas, w0, w1)[0]
        e5w = era5_totals(slug, areas, w0, w1)
        if e5w is None:
            continue
        for gid in gids:
            gi = pos[gid]
            y, e = float(tw_s[gi].sum()), float(e5w[gi].sum())
            e5rows.append({"gauge": int(gid), "code": code[gid], "area_km2": gdf.loc[gid, "model_km2"],
                           "product": "ERA5", "window_start": w0, "window_end": w1, "truth_mm": y, "era5_mm": e,
                           "ens_mean_mm": e, "crps_mm": abs(e - y), "bias_pct": 100 * (e - y) / y,
                           "abs_err_mean_mm": abs(e - y), **series_scores(e5w[gi][None], tw_s[gi])})
    if e5rows:
        df = pd.concat([df, pd.DataFrame(e5rows)], ignore_index=True)
    df.to_csv(os.path.join(out, "upstream_rain_scores.csv"), index=False, float_format="%.3f")

    pd.set_option("display.width", 220)
    f = df[df["product"] != "ERA5"]
    print("\nEnsemble-mean total (mm) by init, per gauge, vs", ref_label)
    piv = f.pivot_table(index=["code", "truth_mm"], columns=["product", "init"], values="ens_mean_mm")
    piv.columns = [f"{p} {pd.Timestamp(i):%d%H}" for p, i in piv.columns]
    print(piv.round(0).to_string())
    print("\nMean over gauges: CRPS / truth, per product and lead")
    print(f.assign(rel=f.crps_mm / f.truth_mm).pivot_table(index="lead_d", columns="product", values="rel").round(2).to_string())

    wins = head_to_head(f, out)
    plot_wins(f, wins, code, areas.ids, ref_label, out)
    plot_map(slug, gdf, code, ref_map, truth, era5, ref_label, t0, t1, out, areas)
    plot_vs_lead(df, code, areas.ids, ref_label, t0, t1, out)
    plot_crps(f, ref_label, out)
    print(out)


def plot_map(slug, gdf, code, ref_map, truth, era5, ref_label, t0, t1, out, areas=None):
    x0, y0, x1, y1 = gdf.total_bounds
    pad = 0.08
    fig, ax = plt.subplots(1, 2, figsize=(18 / 2.54, 8.5 / 2.54), gridspec_kw={"width_ratios": [1.35, 1]})
    if "x" in ref_map.dims and areas is not None and areas.projected:      # field on a projected model grid
        im = ax[0].pcolormesh(areas.lon2d, areas.lat2d, ref_map.transpose("y", "x").values,
                              cmap="YlGnBu", vmin=0, shading="auto")
    else:
        lat = "lat" if "lat" in ref_map.dims else "latitude"
        lon = "lon" if "lon" in ref_map.dims else "longitude"
        sub = ref_map.sortby(lat).sortby(lon).sel({lon: slice(x0 - pad, x1 + pad), lat: slice(y0 - pad, y1 + pad)})
        im = ax[0].pcolormesh(sub[lon], sub[lat], sub, cmap="YlGnBu", vmin=0, shading="auto")
    order = gdf.sort_values("model_km2", ascending=False)
    for gid, g in order.iterrows():
        gpd.GeoSeries([g.geometry]).boundary.plot(ax=ax[0], color="k", lw=0.9 if g.model_km2 > 500 else 0.5)
    pts = gpd.GeoSeries(gpd.points_from_xy(gdf["x"], gdf["y"]), index=gdf.index)
    ax[0].scatter(pts.x, pts.y, marker="^", s=18, c="#d1495b", edgecolor="k", lw=0.4, zorder=4)
    placed = []
    for gid, p in pts.items():
        near = any(abs(p.x - q.x) < 0.06 and abs(p.y - q.y) < 0.03 for q in placed)
        ax[0].annotate(code[gid], (p.x, p.y), xytext=(3, -8) if near else (3, 2),
                       textcoords="offset points", fontsize=5.5, zorder=5)
        placed.append(p)
    ax[0].set_xlim(x0 - pad, x1 + pad); ax[0].set_ylim(y0 - pad, y1 + pad)
    ax[0].set_aspect(1 / np.cos(np.deg2rad((y0 + y1) / 2)))
    ax[0].tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=ax[0], shrink=0.8, pad=0.02)
    cb.set_label(f"{ref_label}, {t0:%d %b %H}–{t1:%d %b %H} UTC (mm)", fontsize=6); cb.ax.tick_params(labelsize=6)
    ax[0].set_title("(a) reference total and gauge upstream areas", fontsize=7, loc="left")

    ids = list(order.index)
    yy = np.arange(len(ids))
    tr = dict(zip(gdf.index, truth))
    ax[1].barh(yy, [tr[i] for i in ids], height=0.38, color="#1f6f8b", label=ref_label)
    if era5 is not None:
        e5 = dict(zip(gdf.index, era5))
        ax[1].barh(yy + 0.4, [e5[i] for i in ids], height=0.38, color=ERA5_COLOR, label="ERA5")
    ax[1].set_yticks(yy + 0.2, [f"{code[i]} ({gdf.loc[i, 'model_km2']:.0f} km²)" for i in ids], fontsize=6)
    ax[1].invert_yaxis(); ax[1].tick_params(axis="x", labelsize=6)
    ax[1].set_xlabel("upstream-area mean total (mm)", fontsize=6.5)
    ax[1].legend(fontsize=6, frameon=False, loc="lower right")
    ax[1].set_title("(b) rain over each gauge's upstream area", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "upstream_areas_reference_total.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_vs_lead(df, code, ids, ref_label, t0, t1, out):
    n = len(ids)
    nc = 5 if n > 6 else n
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(18 / 2.54, 4.2 * nr / 2.54 + 1), sharex=True, squeeze=False)
    for ax, gid in zip(axs.ravel(), ids):
        g = df[df.gauge == gid]
        y = g["truth_mm"].iloc[0]
        for k, (_, label, color) in PRODUCTS.items():
            p = g[g["product"] == label].sort_values("lead_d")
            if p.empty:
                continue
            ax.fill_between(p.lead_d, p.p10_mm, p.p90_mm, color=color, alpha=0.18, lw=0)
            ax.plot(p.lead_d, p.ens_median_mm, color=color, lw=1.2, label=f"{label} median, 10–90 %")
        ax.axhline(y, color="k", lw=1.4, label=ref_label)
        e = g["era5_mm"].dropna()
        if len(e):
            ax.axhline(e.iloc[0], color=ERA5_COLOR, lw=1.0, ls="--", label="ERA5")
        ax.set_title(f"{code[gid]} ({g['area_km2'].iloc[0]:.0f} km²)", fontsize=6.5, loc="left")
        ax.set_ylim(0, max(y, g["p90_mm"].max()) * 1.08)
        ax.tick_params(labelsize=5.5)
    axs[0, 0].invert_xaxis()            # shared x: inverting each panel would flip it back and forth
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("lead before event start (days)", fontsize=6)
    for ax in axs[:, 0]:
        ax.set_ylabel("total (mm)", fontsize=6)
    h, l = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Rain over each gauge's upstream area, {t0:%d %b %H}–{t1:%d %b %H} UTC", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "upstream_rain_vs_lead.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_crps(f, ref_label, out):
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    rel = f.assign(rel=f.crps_mm / f.truth_mm)
    for k, (_, label, color) in PRODUCTS.items():
        p = rel[rel["product"] == label].groupby("lead_d")["rel"].mean()
        if len(p):
            ax.plot(p.index, p.values, color=color, lw=1.4, marker="o", ms=2.5, label=label)
    ax.invert_xaxis(); ax.set_ylim(0, None)
    ax.set_xlabel("lead before event start (days)", fontsize=6.5)
    ax.set_ylabel(f"fair CRPS / {ref_label} total\n(mean over gauges)", fontsize=6.5)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "upstream_rain_crps_vs_lead.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
