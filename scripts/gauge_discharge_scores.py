#!/usr/bin/env python3
"""
Forecast discharge at the selected gauges, scored against the reference run
(step 5 of the extreme-gauge framework).

  truth      : the wflow run forced by the validated reference rain (hist_3h;
               Catalonia: XEMA gauge grid). wflow is a black box here, so the
               forecasts are judged against what the model makes of the truth.
  shown      : observed discharge (context only, not a score), ERA5-forced run
  forecasts  : every member run of every forecast case (wflow_ensemble.py,
               members_output.nc), warm-started from the reference spin-up
  window     : event_start -> sim_end

Metrics per (gauge, product, case):
  peak       fair CRPS of the member peaks (m3/s and / reference peak), bias of
             the mean member peak (%), the reference peak's rank among the
             member peaks, share of members reaching half of it, and the
             timing error of the median member peak time (h; only members
             peaking at >= 10 % of the reference, else NaN)
  hydrograph RMSE of the member-mean hydrograph (/ reference peak), NSE of the
             member mean, and fair CRPS per step averaged over the window
  volume     flood volume in mm over the upstream area: discharge above the
             reference run's pre-event flow (vol) and above half the reference
             peak (volhalf, the crest), each with fair CRPS and bias of the mean
Head-to-head: best product per (gauge, case, metric); tie when the runner-up
is within 5 % (0.02 for NSE), as for the rain.

Outputs in figures/<slug>/gauge_discharge/:
    discharge_scores.csv, discharge_best.csv, discharge_wins.csv
    hydrographs_<case lead>.png     one panel per gauge, per forecast start
    discharge_wins.png

    python scripts/gauge_discharge_scores.py catalonia_2026
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wflow_ensemble as we                                        # noqa: E402
from event_config import EVENTS                                    # noqa: E402
from plot_style import IFS_COLOR, AIFS_COLOR, WN2_COLOR, ERA5_COLOR  # noqa: E402
from upstream_rain_scores import fair_crps, REFERENCE, ALT_REFERENCE, gauge_label, gauge_area, gauge_windows, WINDOWS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCOL = {"hans_2023": "Q_", "kristin_mondego": "river_q_grdc_", "kristin_lis": "river_q_grdc_", "dana_2024": "river_q_grdc_"}   # output.csv column prefix per model (default river_q_)
PRODUCT = {"ifs": ("IFS", IFS_COLOR), "aifs": ("AIFS", AIFS_COLOR), "wn2": ("WN2", WN2_COLOR)}
METRICS = {"CRPS (peak)": ("crps_peak", "low"), "|bias| (peak)": ("abs_bias_peak_pct", "low"),
           "peak timing": ("abs_timing_h", "low"), "RMSE (hydrograph)": ("rmse_rel", "low"),
           "CRPS (hydrograph)": ("crps_series", "low"), "NSE (member mean)": ("nse_mean", "high")}
TIE_REL, TIE_ABS = 0.05, 0.02


def run_csv(slug, case):
    f = os.path.join(REPO, "data", "wflow", slug, "model", f"run_{slug}_{case}", "output.csv")
    return pd.read_csv(f, index_col=0, parse_dates=True) if os.path.exists(f) else None


def load_obs(slug, g):
    """Observed discharge for a gauges.csv row: discharge_<station code>.csv or discharge_<id>.csv."""
    obs_dir = os.path.join(REPO, "data", "raw", slug, "obs")
    names = [f"discharge_{str(g.get('station', '')).split('|')[0]}.csv", f"discharge_{g['id']}.csv"]
    for n in names:
        f = os.path.join(obs_dir, n)
        if os.path.exists(f):
            d = pd.read_csv(f)
            tcol = next(c for c in d.columns if "time" in c.lower())
            qcol = next(c for c in d.columns if c.lower().startswith("q") or "discharge" in c.lower())
            t = pd.to_datetime(d[tcol], utc=True).dt.tz_localize(None)       # all model times are naive UTC
            return pd.Series(d[qcol].values, index=t, name="obs").sort_index()
    return None


def volume_scores(X, y, times, area_km2, q0):
    """Flood volumes in mm over the upstream area, members X (member, time) against the reference y.

    vol     : discharge above the pre-event flow q0 (reference run at the window start), whole window
    volhalf : discharge above half the reference peak (the flood crest), whole window
    Same thresholds for members and reference, so a member is judged on the water it adds."""
    dt = np.diff(times.values).astype("timedelta64[s]").astype(float)
    dt = np.r_[dt[0], dt]                                   # end-stamped steps: each value covers the step before it
    mm = 1000.0 / (area_km2 * 1e6)
    out = {}
    for name, thr in (("vol", q0), ("volhalf", 0.5 * float(y.max()))):
        v = (np.clip(X - thr, 0, None) * dt).sum(axis=1) * mm
        yv = float((np.clip(y - thr, 0, None) * dt).sum() * mm)
        c = fair_crps(v, yv) if len(v) > 1 else abs(v[0] - yv)
        out.update({f"ref_{name}_mm": yv, f"mean_member_{name}_mm": v.mean(), f"crps_{name}_mm": c,
                    f"crps_{name}_rel": c / yv if yv > 0 else np.nan,
                    f"bias_{name}_pct": 100 * (v.mean() - yv) / yv if yv > 0 else np.nan})
    return out


def peak_scores(X, y, times):
    """X (member, time), y (time) on the same window."""
    pk, ypk = X.max(axis=1), float(y.max())
    tpk = times[X.argmax(axis=1)]
    t_ref = times[int(y.argmax())]
    # timing only from members with a real flood peak (>= 10 % of the reference):
    # a flat near-zero hydrograph has no meaningful peak time
    ok = pk >= 0.1 * ypk
    dt = (pd.Series(tpk[ok]).median() - t_ref) / pd.Timedelta("1h") if ok.any() else np.nan
    m = X.mean(axis=0)
    crps_t = np.mean([fair_crps(X[:, k], y[k]) for k in range(len(y))]) if len(X) > 1 else np.mean(np.abs(X[0] - y))
    return {"ref_peak": ypk, "ref_peak_time": t_ref, "mean_member_peak": pk.mean(),
            "p10_peak": np.percentile(pk, 10), "p90_peak": np.percentile(pk, 90), "max_member_peak": pk.max(),
            "crps_peak": fair_crps(pk, ypk) if len(pk) > 1 else abs(pk[0] - ypk),
            "crps_peak_rel": (fair_crps(pk, ypk) if len(pk) > 1 else abs(pk[0] - ypk)) / ypk,
            "bias_peak_pct": 100 * (pk.mean() - ypk) / ypk, "abs_bias_peak_pct": abs(100 * (pk.mean() - ypk) / ypk),
            "ref_rank": float((pk < ypk).mean()), "frac_ge_half_peak": float((pk >= 0.5 * ypk).mean()),
            "timing_h": dt, "abs_timing_h": abs(dt),
            "rmse_rel": float(np.sqrt(np.mean((m - y) ** 2)) / ypk),
            "nse_mean": float(1 - np.sum((m - y) ** 2) / np.sum((y - y.mean()) ** 2)),
            "crps_series": float(crps_t) / ypk}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slug")
    ap.add_argument("--ref", default=None, help="score against the run forced by ALT_REFERENCE[slug][REF] (sensitivity)")
    a = ap.parse_args()
    slug = a.slug
    out = os.path.join(REPO, "figures", slug, "gauge_discharge" + (f"_ref-{a.ref}" if a.ref else ""))
    os.makedirs(out, exist_ok=True)
    c = EVENTS[slug]
    sim_start, sim_end, cases = we.event_cfg(slug)
    t0 = pd.Timestamp(c["event_start"])
    gauges = pd.read_csv(os.path.join(REPO, "data", "wflow", slug, "gauges.csv"))
    code = {r["id"]: gauge_label(r) for _, r in gauges.iterrows()}
    gauges["area_km2"] = [gauge_area(r) for _, r in gauges.iterrows()]
    ref = run_csv(slug, (ALT_REFERENCE[slug][a.ref] if a.ref else REFERENCE.get(slug, {})).get("run", "hist_3h"))
    era5_case = next((k for k in cases if k.startswith("era5")), None)
    era5 = run_csv(slug, era5_case) if era5_case else None
    win = ref.loc[(ref.index > t0) & (ref.index <= sim_end)].index
    ref_label = (ALT_REFERENCE[slug][a.ref] if a.ref else REFERENCE.get(slug, {})).get("label", "reference") + " run"

    fc = {}
    for case, cc in we.forecast_cases(slug).items():
        if case.split("_")[0] not in PRODUCT:      # excluded products (WN3 runs of Kristin, DANA, ...)
            continue
        f = os.path.join(we.ens_dir(slug), case, "members_output.nc")
        if not os.path.exists(f):
            print(f"  {case}: no members_output.nc, skipped")
            continue
        fstart = we.fstart_of(cc)
        fc[case] = (xr.open_dataset(f).load(), PRODUCT[case.split("_")[0]][0],
                    round((t0 - fstart) / pd.Timedelta("1D"), 2), fstart)

    # per-gauge window and starts (WINDOWS: events whose gauges peaked days apart);
    # there the lead is counted to the gauge's reference peak, elsewhere to event_start
    gw = {}
    if slug in WINDOWS:          # otherwise every gauge: event_start -> sim_end (keeps the recession)
        for w0, w1, gids, suffixes in gauge_windows(slug, gauges["id"].tolist(), "flow"):
            for gid in gids:
                gw[gid] = (w0, min(w1, sim_end), suffixes)
    rows = []
    for g in gauges.itertuples():
        col = f"{QCOL.get(slug, 'river_q_')}{g.id}"
        g0, g1, suffixes = gw.get(g.id, (t0, sim_end, None))
        gwin = ref.loc[(ref.index > g0) & (ref.index <= g1)].index
        y = ref.loc[gwin, col].values
        q0 = float(ref.loc[ref.index <= g0, col].iloc[-1])           # pre-event flow of the reference run
        t_pk = gwin[int(np.argmax(y))]
        base = {"gauge": g.id, "code": code[g.id], "area_km2": g.area_km2, "window_start": g0, "window_end": g1}
        obs = load_obs(slug, g._asdict())
        if obs is not None:
            o = obs[(obs.index > g0) & (obs.index <= g1)]
            if len(o):
                base.update(obs_peak=o.max(), obs_peak_time=o.idxmax())
        for case, (ds, label, lead, fstart) in fc.items():
            if suffixes is not None and case.split("_", 1)[1] not in suffixes:
                continue
            if slug in WINDOWS:
                lead = round((t_pk - fstart) / pd.Timedelta("1D"), 1)
            X = ds[col].sel(time=gwin).transpose("member", "time").values
            rows.append({**base, "product": label, "case": case, "lead_d": lead, "n_members": X.shape[0],
                         **peak_scores(X, y, gwin), **volume_scores(X, y, gwin, g.area_km2, q0)})
        if era5 is not None:
            Xe = era5.loc[gwin, col].values[None]
            rows.append({**base, "product": "ERA5", "case": era5_case, "lead_d": np.nan, "n_members": 1,
                         **peak_scores(Xe, y, gwin), **volume_scores(Xe, y, gwin, g.area_km2, q0)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "discharge_scores.csv"), index=False, float_format="%.3f")

    pd.set_option("display.width", 220)
    print(f"{slug}: window {t0} -> {sim_end}" + (" (per-gauge windows, see WINDOWS)" if slug in WINDOWS else ""))
    print("\nReference-run peak vs observed peak (m3/s; context only):")
    print(df.drop_duplicates("gauge")[[c_ for c_ in ("code", "area_km2", "ref_peak", "obs_peak", "ref_peak_time", "obs_peak_time") if c_ in df]]
          .round(1).to_string(index=False))
    f = df[df["product"] != "ERA5"]
    print("\nMean over gauges, per product and lead:")
    print(f.groupby(["lead_d", "product"])[["crps_peak_rel", "bias_peak_pct", "ref_rank", "frac_ge_half_peak",
                                            "timing_h", "rmse_rel", "nse_mean", "crps_series"]].mean().round(2).to_string())
    print("\nERA5-forced run, mean over gauges:")
    print(df[df["product"] == "ERA5"][["crps_peak_rel", "bias_peak_pct", "timing_h", "rmse_rel", "nse_mean"]]
          .mean().round(2).to_string())

    w = head_to_head(f, out)
    for case in sorted({(v[2], v[3]) for v in fc.values()}, reverse=True):
        plot_hydrographs(slug, gauges, code, ref, era5, fc, case, win, t0, sim_end, ref_label, out)
    plot_wins(w, out)
    print(out)


def head_to_head(f, out):
    prods = sorted(f["product"].unique())
    rows = []
    for (gid, lead), g in f.groupby(["gauge", "lead_d"]):
        if set(g["product"]) != set(prods):
            continue
        g = g.set_index("product")
        for mname, (colm, better) in METRICS.items():
            v = g[colm].dropna()
            if len(v) < len(prods):
                continue
            order = v.sort_values(ascending=(better == "low"))
            a, b = order.iloc[0], order.iloc[1]
            tie = abs(a - b) <= TIE_ABS if better == "high" else abs(b - a) <= max(TIE_REL * abs(a), 1e-9)
            if mname == "peak timing":
                tie = abs(b - a) <= 3.0          # within one model step
            rows.append({"gauge": gid, "code": g["code"].iloc[0], "lead_d": lead, "metric": mname,
                         "best": "tie" if tie else order.index[0], **{p: v[p] for p in prods}})
    w = pd.DataFrame(rows)
    w.to_csv(os.path.join(out, "discharge_best.csv"), index=False, float_format="%.3f")
    tab = w.groupby(["lead_d", "metric", "best"]).size().unstack(fill_value=0).reset_index()
    tab.to_csv(os.path.join(out, "discharge_wins.csv"), index=False)
    print("\nWins per metric (gauges; tie = within 5 %, 0.02 NSE, 3 h timing):")
    print(tab.to_string(index=False))
    return w


def plot_hydrographs(slug, gauges, code, ref, era5, fc, case_key, win, t0, sim_end, ref_label, out):
    lead, fstart = case_key
    sel = {k: v for k, v in fc.items() if (v[2], v[3]) == case_key}
    n = len(gauges)
    nc = 5 if n > 6 else n
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(18 / 2.54, 4.4 * nr / 2.54 + 1.2), squeeze=False)
    t_lo = fstart if fstart > t0 - pd.Timedelta("2D") else t0 - pd.Timedelta("1D")
    for ax, g in zip(axs.ravel(), gauges.itertuples()):
        col = f"{QCOL.get(slug, 'river_q_')}{g.id}"
        obs = load_obs(slug, g._asdict())
        if obs is not None:
            o = obs[(obs.index >= t_lo) & (obs.index <= sim_end)]
            ax.plot(o.index, o.values, color="#9a9a9a", lw=0.8, label="observed (ACA)" if slug == "catalonia_2026" else "observed")
        for case, (ds, label, _, _) in sel.items():
            color = PRODUCT[case.split("_")[0]][1]
            q = ds[col].sel(time=slice(t_lo, sim_end))
            ax.fill_between(q.time.values, q.quantile(0.1, "member"), q.quantile(0.9, "member"), color=color, alpha=0.2, lw=0)
            ax.plot(q.time.values, q.median("member"), color=color, lw=1.1, label=f"{label} median, 10–90 %")
        r = ref.loc[(ref.index >= t_lo) & (ref.index <= sim_end), col]
        ax.plot(r.index, r.values, color="k", lw=1.4, label=ref_label)
        if era5 is not None:
            e = era5.loc[(era5.index >= t_lo) & (era5.index <= sim_end), col]
            ax.plot(e.index, e.values, color=ERA5_COLOR, lw=0.9, ls="--", label="ERA5 run")
        ax.set_title(f"{code[g.id]} ({g.area_km2:.0f} km²)", fontsize=6.5, loc="left")
        ax.tick_params(labelsize=5.5)
        ax.set_xlim(t_lo, sim_end)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d"))
        ax.set_ylim(0, None)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("Q (m³/s)", fontsize=6)
    h, l = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Discharge, forecasts from {fstart:%d %b %H}Z ({lead:.0f}-day lead), day of month on the x-axis",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, f"hydrographs_{fstart:%Y%m%d%H}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_wins(w, out):
    colors = {"IFS": IFS_COLOR, "AIFS": AIFS_COLOR, "WN2": WN2_COLOR, "tie": "#c8c8c8"}
    leads = sorted(w["lead_d"].unique(), reverse=True)
    fig, axs = plt.subplots(1, len(leads), figsize=(18 / 2.54, 6 / 2.54), sharey=True, squeeze=False)
    for ax, lead in zip(axs[0], leads):
        cnt = w[w.lead_d == lead].groupby(["metric", "best"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(columns=list(colors), fill_value=0).reindex([m for m in METRICS if m in cnt.index])
        left = np.zeros(len(cnt))
        for p, colr in colors.items():
            ax.barh(np.arange(len(cnt)), cnt[p], left=left, color=colr, label=p, height=0.6)
            left += cnt[p].values
        ax.set_yticks(np.arange(len(cnt)), cnt.index, fontsize=6)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel("gauges won", fontsize=6.5)
        ax.set_title(f"{lead:.0f}-day lead", fontsize=7, loc="left")
    axs[0, 0].legend(fontsize=6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(1.1, -0.22))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "discharge_wins.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
