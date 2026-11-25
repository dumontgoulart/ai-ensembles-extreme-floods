#!/usr/bin/env python3
"""
Numbers and figures for docs/research_backbone.md (co-author backbone, 20 Oct 2026).

Scoring points are INDEPENDENT subcatchments: within each event, gauges whose upstream
areas overlap are merged, keeping the most extreme one (largest reference-run peak per
km2). Every number below is over these points only, from the existing score files:
    figures/<slug>/upstream_rain/upstream_rain_scores.csv (+ members)   rain, event band
    figures/<slug>/gauge_discharge/discharge_scores.csv                 discharge vs the reference run

Starts: within each gauge window the IFS starts are ranked, latest = short (-1 d),
then medium (-2 d), then long (-7 d; Zermatt 21 Jun -6 d); starts more than 5 d before the rain window
count as long. Starts without an IFS run (Texas AIFS 1 Jul) are left out.
Best product at a point: lowest mean fair CRPS; a tie when the runner-up is within 5 %.

    python scripts/backbone_analysis.py

Outputs: figures/backbone/ (points.csv, meteo_*.csv, discharge_*.csv, trends.csv, link.csv,
fig1..fig4 png, event_<key>.png)
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wflow_ensemble as we                                                   # noqa: E402
from plot_style import IFS_COLOR, AIFS_COLOR, WN2_COLOR                        # noqa: E402
from gauge_discharge_scores import run_csv, QCOL, load_obs                             # noqa: E402
from upstream_rain_scores import REFERENCE                                     # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figures", "backbone")
AI_BOTH = "AIFS & WN2"   # three-way: the two AI ensembles tie with each other and both beat IFS ENS
COL = {"IFS": IFS_COLOR, "AIFS": AIFS_COLOR, "WN2": WN2_COLOR, AI_BOTH: "#c4566f", "tie": "#bbbbbb"}
TIE = 0.05
# key, title, models
EVENTS = [("catalonia", "Catalonia, Oct 2026", ["catalonia_2026"]),
          ("jesenik", "Storm Boris, Jeseníky/Kłodzko, Sep 2024", ["boris_sudetes"]),
          ("texas25", "Central Texas, Jul 2025", ["texas_2025"]),
          ("texas26", "Texas Hill Country, Jul 2026", ["texas_2026"]),
          ("mississippi", "Mississippi Gulf Coast, Jun 2026", ["mississippi_2026"]),
          ("helene", "Hurricane Helene, Sep 2024", ["helene_2024"]),
          ("hans", "Storm Hans, Aug 2023", ["hans_2023"]),
          ("zermatt", "Valais / Vispa, Jun 2024", ["zermatt_2024"]),
          ("california", "California ARs, Dec 2022 - Jan 2023", ["california_2023"]),
          ("kristin", "Storm Kristin, Jan 2026", ["kristin_mondego", "kristin_lis"]),
          ("dana", "DANA, Valencia, Oct 2024", ["dana_2024"])]
RAIN_BINS = [(0, 1.5, "<1.5 d"), (1.5, 3.5, "1.5-3.5 d"), (3.5, 5.5, "3.5-5.5 d"), (5.5, 99, ">=5.5 d")]
STARTS = ["short (-1 d)", "medium (-2 d)", "long (-7 d)"]


def lead_bin(x):
    for lo, hi, lab in RAIN_BINS:
        if lo <= x < hi:
            return lab
    return None


# ── points ──────────────────────────────────────────────────────────────────
def points():
    rows = []
    for key, title, slugs in EVENTS:
        for slug in slugs:
            g = pd.read_csv(os.path.join(REPO, "data", "wflow", slug, "gauges.csv"))
            q = pd.read_csv(os.path.join(REPO, "figures", slug, "gauge_discharge", "discharge_scores.csv"))
            r = pd.read_csv(os.path.join(REPO, "figures", slug, "upstream_rain", "upstream_rain_scores.csv"))
            m = xr.open_dataset(os.path.join(REPO, "data", "wflow", slug, "gauge_upstream.nc"))["mask"].load()
            for _, row in g.iterrows():
                qq, rr = q[q.gauge == row.id], r[r.gauge == row.id]
                if qq.empty or rr.empty:
                    continue
                area = float(qq.area_km2.iloc[0])
                rows.append({"event": key, "title": title, "slug": slug, "gauge": row.id,
                             "name": qq.code.iloc[0], "area_km2": area,
                             "ref_rain_mm": float(rr.truth_mm.iloc[0]),
                             "ref_peak": float(qq.ref_peak.iloc[0]),
                             "ref_spec_peak": float(qq.ref_peak.iloc[0]) / area,
                             "obs_peak": float(qq.obs_peak.dropna().iloc[0]) if "obs_peak" in qq and qq.obs_peak.notna().any() else np.nan,
                             "mask": m.sel(gauge=row.id).values.astype(bool)})
    return select(pd.DataFrame(rows), "ref_spec_peak")


def select(p, key):
    """Greedy: highest `key` first, drop anything overlapping an already kept area."""
    p = p.copy()
    p["independent"], p["merged_into"] = False, ""
    for slug, grp in p.groupby("slug"):
        kept = []
        for i in grp.sort_values(key, ascending=False).index:
            hit = [k for k in kept if (p.at[i, "mask"] & p.at[k, "mask"]).any()]
            if hit:
                p.at[i, "merged_into"] = str(p.at[hit[0], "name"])
            else:
                kept.append(i)
                p.at[i, "independent"] = True
    return p


def start_classes(slug, q):
    """case tag -> start class, per gauge window (rank of the IFS starts)."""
    cfg = we.forecast_cases(slug)
    init = {c.split("_", 1)[1]: we.fstart_of(cc) for c, cc in cfg.items()}
    out = {}
    for ws, grp in q[q["product"] == "IFS"].groupby("window_start"):
        tags = sorted(grp.case.str.split("_", n=1).str[1].unique(), key=lambda t: init[t], reverse=True)
        for i, t in enumerate(tags):
            cls = STARTS[min(i, 2)]
            if (pd.Timestamp(ws) - init[t]) / pd.Timedelta("1D") > 5:
                cls = STARTS[2]
            out[(ws, t)] = cls
    return out, init


def load_scores(p):
    ind = p[p.independent]
    R, Q, M = [], [], []
    for slug in ind.slug.unique():
        gids = set(ind[ind.slug == slug].gauge)
        ev = ind[ind.slug == slug].event.iloc[0]
        r = pd.read_csv(os.path.join(REPO, "figures", slug, "upstream_rain", "upstream_rain_scores.csv"))
        r = r[r.gauge.isin(gids) & (r["product"] != "ERA5")].assign(event=ev, slug=slug)
        R.append(r)
        mm = pd.read_csv(os.path.join(REPO, "figures", slug, "upstream_rain", "upstream_rain_members.csv.gz"))
        M.append(mm[mm.gauge.isin(gids)].assign(event=ev, slug=slug))
        q = pd.read_csv(os.path.join(REPO, "figures", slug, "gauge_discharge", "discharge_scores.csv"))
        q = q[q.gauge.isin(gids) & q["product"].isin(["IFS", "AIFS", "WN2"])].copy()
        cls, init = start_classes(slug, q)
        q["tag"] = q.case.str.split("_", n=1).str[1]
        q["start"] = [cls.get((w, t)) for w, t in zip(q.window_start, q.tag)]
        q["init"] = q.tag.map(init)
        Q.append(q.dropna(subset=["start"]).assign(event=ev, slug=slug))
    R = pd.concat(R, ignore_index=True)
    R["bin"] = R.lead_d.map(lead_bin)
    return R, pd.concat(Q, ignore_index=True), pd.concat(M, ignore_index=True)


def best(vals):
    """vals {product: score}, lower is better -> winner, AI_BOTH or 'tie'.
    Products within TIE of the lowest score share the lead. A tie always involves IFS: when the shared lead is
    AIFS and WN2 only (both more than TIE better than IFS), the result is AI_BOTH."""
    v = {k: x for k, x in vals.items() if np.isfinite(x)}
    if len(v) < 2:
        return None
    lo = min(v.values())
    lead = {k for k, x in v.items() if x <= lo * (1 + TIE) or x - lo < 1e-9}
    if len(lead) == 1:
        return lead.pop()
    return "tie" if "IFS" in lead else AI_BOTH


def tally(df, by, col, products):
    rows = []
    for k, g in df.groupby(["event", "slug", "gauge", by]):
        mean = g.groupby("product")[col].mean()
        if not all(pp in mean for pp in products):
            continue
        rows.append({"event": k[0], "slug": k[1], "gauge": k[2], by: k[3],
                     "winner": best({pp: mean[pp] for pp in products}),
                     **{pp: mean[pp] for pp in products}})
    return pd.DataFrame(rows)


def counts(t, by, order):
    c = t.groupby([by, "winner"]).size().unstack(fill_value=0)
    return c.reindex([o for o in order if o in c.index])


# ── meteo metrics ───────────────────────────────────────────────────────────
def meteo_metrics(R, M):
    mc = [c for c in M.columns if c.startswith("m") and c[1:].isdigit()]
    M = M.merge(R[["slug", "gauge", "product", "init", "bin"]].drop_duplicates(), on=["slug", "gauge", "product", "init"])
    X = M[mc].values.astype(float)
    M["ens_mean"] = np.nanmean(X, axis=1)
    M["ens_var"] = np.nanvar(X, axis=1, ddof=1)
    rows = []
    for (pr, b), g in M.groupby(["product", "bin"]):
        rr = R[(R["product"] == pr) & (R.bin == b)]
        rows.append({"product": pr, "bin": b, "cases": len(g),
                     "crps_mm": rr.crps_mm.mean(), "crps_3h_mm": rr.crps_3h_mm.mean(),
                     "bias_pct": 100 * (g.ens_mean.sum() - g.truth_mm.sum()) / g.truth_mm.sum(),
                     "spread_skill": np.sqrt(g.ens_var.mean()) / np.sqrt(((g.ens_mean - g.truth_mm) ** 2).mean()),
                     "truth_above_all_pct": 100 * (rr.truth_rank >= 1).mean(),
                     "truth_below_all_pct": 100 * (rr.truth_rank <= 0).mean(),
                     "corr_3h": rr.corr_3h.mean()})
    return pd.DataFrame(rows)


TAIL_T = [100, 200, 300]


def tail_metrics(R, M, aifs_ev):
    """Extreme tail: among the cases (gauge x initialisation) whose reference total reaches T mm, the share in
    which at least 10 % of the members reach T as well. IFS ENS and WN2 on all events, all three on the AIFS events."""
    mc = [c for c in M.columns if c.startswith("m") and c[1:].isdigit()]
    M = M.merge(R[["slug", "gauge", "product", "init", "bin"]].drop_duplicates(), on=["slug", "gauge", "product", "init"])
    X = M[mc].values.astype(float)
    rows, detail = [], []
    for T in TAIL_T:
        M["p"] = (X >= T).sum(axis=1) / np.maximum(np.isfinite(X).sum(axis=1), 1)
        hit = M[M.truth_mm >= T]
        for evset, h in (("all", hit), ("AIFS events", hit[hit.event.isin(aifs_ev)])):
            for (pr, b), g in h.groupby(["product", "bin"]):
                if evset == "all" and pr == "AIFS":
                    continue
                rows.append({"events": evset, "product": pr, "bin": b, "T_mm": T, "cases": len(g),
                             "n_events": g.event.nunique(), "hit_pct": 100 * (g.p >= 0.1).mean(),
                             "mean_member_share_pct": 100 * g.p.mean()})
        if T == TAIL_T[-1]:
            detail = hit.assign(hit=hit.p >= 0.1).groupby(["event", "product", "bin"]).agg(
                cases=("hit", "size"), hits=("hit", "sum")).reset_index()
    return pd.DataFrame(rows), detail


def tail_panel(ax, tl, T, dotted=True):
    """Share of the cases reaching T mm in which >= 10 % of the members reach T, by lead."""
    rb = [x[2] for x in RAIN_BINS]
    x = np.arange(len(rb))
    lines = (("all", "IFS", "-"), ("all", "WN2", "-"), ("AIFS events", "AIFS", "-"))
    if dotted:
        lines += (("AIFS events", "IFS", ":"), ("AIFS events", "WN2", ":"))
    for evset, pr, ls in lines:
        s = tl[(tl.T_mm == T) & (tl.events == evset) & (tl["product"] == pr)].set_index("bin").reindex(rb)
        lab = pr if evset == "all" or pr == "AIFS" else f"{pr} (AIFS events)"
        ax.plot(x, s.hit_pct, color=COL[pr], ls=ls, marker="o" if ls == "-" else None, ms=3, lw=1.2 if ls == "-" else 0.9,
                label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace(" d", "") for b in rb], fontsize=6)
    ax.set_xlabel("lead to the start of the event window (d)", fontsize=6)
    ax.set_ylim(-3, 100)


def fig_tail(tl):
    """Fig. S13 (file fig6_tail_hits.png): share of the cases reaching T mm in which >= 10 % of the members reach T."""
    fig, axs = plt.subplots(1, len(TAIL_T), figsize=(18 / 2.54, 5.4 / 2.54), constrained_layout=True, sharey=True)
    for ax, T in zip(axs, TAIL_T):
        tail_panel(ax, tl, T)
        ax.set_title(f"({'abc'[TAIL_T.index(T)]}) reference ≥ {T} mm", fontsize=7, loc="left")
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("cases with ≥ 10 % of members\nalso reaching the threshold (%)", fontsize=6)
    axs[-1].legend(fontsize=5.5, frameon=False, loc="upper right")
    fig.savefig(os.path.join(OUT, "fig6_tail_hits.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def crpss_ci(R, ai, n=2000, seed=1, col="crps_mm", by="bin", order=None, key=("event", "gauge", "init")):
    """CRPSS of ai against IFS per lead bin (or discharge start) on paired cases.

    CI by a two-stage bootstrap: events resampled with replacement, then the points inside each
    picked event (points of one event share the storm, so they are not independent)."""
    rng = np.random.default_rng(seed)
    out = []
    for b in (order or [x[2] for x in RAIN_BINS]):
        s = R[R[by] == b]
        a = s[s["product"] == ai].set_index(list(key))[col]
        i = s[s["product"] == "IFS"].set_index(list(key))[col]
        k = a.index.intersection(i.index)
        if len(k) == 0:
            continue
        d = pd.DataFrame({"a": a.loc[k], "i": i.loc[k]}).reset_index()
        ev = [g[["a", "i"]].values for _, g in d.groupby(["event", "gauge"])[["a", "i"]].sum().groupby("event")]
        bs = []
        for _ in range(n):
            tot = np.zeros(2)
            for e in rng.integers(0, len(ev), len(ev)):
                tot += ev[e][rng.integers(0, len(ev[e]), len(ev[e]))].sum(0)
            bs.append(1 - tot[0] / tot[1])
        # per-event CRPSS: how many events the AI product is better in, and a two-sided sign test over events
        ev_ss = np.array([1 - x[:, 0].sum() / x[:, 1].sum() for x in ev])
        n_better = int((ev_ss > 0).sum())
        out.append({"ai": ai, "bin": b, "events": len(ev), "points": int(sum(len(x) for x in ev)),
                    "crpss": 1 - d.a.sum() / d.i.sum(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                    "p_better": float(np.mean(np.array(bs) > 0)), "events_better": n_better,
                    "sign_p": stats.binomtest(n_better, len(ev), 0.5).pvalue})
    return pd.DataFrame(out)


# ── figures ─────────────────────────────────────────────────────────────────
def bars(ax, c, title, products):
    order = [x for x in products + [AI_BOTH, "tie"] if x in c.columns]
    left = np.zeros(len(c))
    for pr in order:
        v = c[pr].values if pr in c else np.zeros(len(c))
        ax.barh(range(len(c)), v, left=left, color=COL[pr], label=pr, height=0.6, edgecolor="white", linewidth=1)
        for j, (l_, x) in enumerate(zip(left, v)):
            if x:
                ax.text(l_ + x / 2, j, str(int(x)), ha="center", va="center", fontsize=7 if x > 1 else 5,
                        color="white" if pr != "tie" else "k")
        left += v
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("independent gauges", fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    p = points()
    p.drop(columns="mask").to_csv(os.path.join(OUT, "points.csv"), index=False, float_format="%.3f")
    ind = p[p.independent]
    print(f"{len(p)} scored gauges -> {len(ind)} independent points")
    print(ind.groupby("event").size().to_string())
    R, Q, M = load_scores(p)
    aifs_ev = sorted(R[R["product"] == "AIFS"].event.unique())

    # rain: best per point and lead bin
    rb = [x[2] for x in RAIN_BINS]
    t2r = tally(R, "bin", "crps_mm", ["IFS", "WN2"])
    t3r = tally(R[R.event.isin(aifs_ev)], "bin", "crps_mm", ["IFS", "AIFS", "WN2"])
    t2q = tally(Q, "start", "crps_peak", ["IFS", "WN2"])
    t3q = tally(Q[Q.event.isin(aifs_ev)], "start", "crps_peak", ["IFS", "AIFS", "WN2"])
    t2h = tally(Q, "start", "crps_series", ["IFS", "WN2"])
    t3h = tally(Q[Q.event.isin(aifs_ev)], "start", "crps_series", ["IFS", "AIFS", "WN2"])
    # flood volume above the pre-event flow (vol) and above half the reference peak (volhalf, the crest)
    t2v = tally(Q, "start", "crps_vol_mm", ["IFS", "WN2"])
    t3v = tally(Q[Q.event.isin(aifs_ev)], "start", "crps_vol_mm", ["IFS", "AIFS", "WN2"])
    t2c = tally(Q, "start", "crps_volhalf_mm", ["IFS", "WN2"])
    t3c = tally(Q[Q.event.isin(aifs_ev)], "start", "crps_volhalf_mm", ["IFS", "AIFS", "WN2"])
    for name, t in (("meteo_best_ifs_wn2", t2r), ("meteo_best_3way", t3r), ("discharge_best_ifs_wn2", t2q),
                    ("discharge_best_3way", t3q), ("hydrograph_best_ifs_wn2", t2h), ("hydrograph_best_3way", t3h),
                    ("volume_best_ifs_wn2", t2v), ("volume_best_3way", t3v),
                    ("crestvolume_best_ifs_wn2", t2c), ("crestvolume_best_3way", t3c)):
        t.to_csv(os.path.join(OUT, f"{name}.csv"), index=False, float_format="%.3f")
    for name, t, by, order in (("rain IFS/WN2", t2r, "bin", rb), ("rain 3-way", t3r, "bin", rb),
                               ("peak IFS/WN2", t2q, "start", STARTS), ("peak 3-way", t3q, "start", STARTS),
                               ("hydrograph IFS/WN2", t2h, "start", STARTS), ("hydrograph 3-way", t3h, "start", STARTS),
                               ("flood volume IFS/WN2", t2v, "start", STARTS), ("flood volume 3-way", t3v, "start", STARTS),
                               ("crest volume IFS/WN2", t2c, "start", STARTS), ("crest volume 3-way", t3c, "start", STARTS)):
        print(f"\n{name}:\n{counts(t, by, order).to_string()}")

    mm = meteo_metrics(R, M)
    mm.to_csv(os.path.join(OUT, "meteo_metrics.csv"), index=False, float_format="%.3f")
    mm3 = meteo_metrics(R[R.event.isin(aifs_ev)], M[M.event.isin(aifs_ev)])
    mm3.to_csv(os.path.join(OUT, "meteo_metrics_aifs_events.csv"), index=False, float_format="%.3f")
    tl, tld = tail_metrics(R, M, aifs_ev)
    tl.to_csv(os.path.join(OUT, "tail_hits.csv"), index=False, float_format="%.2f")
    tld.to_csv(os.path.join(OUT, "tail_hits_300mm_by_event.csv"), index=False)
    print("\ntail (share of cases reaching T with >= 10 % of members reaching T):\n" + tl.round(1).to_string(index=False))
    print(tld.to_string(index=False))
    cs = pd.concat([crpss_ci(R, "WN2"), crpss_ci(R[R.event.isin(aifs_ev)], "AIFS"),
                    crpss_ci(R[R.event.isin(aifs_ev)], "WN2").assign(ai="WN2 (AIFS events)")])
    cs.to_csv(os.path.join(OUT, "meteo_crpss.csv"), index=False, float_format="%.3f")
    # discharge: CRPSS by start for the peak (CRPS relative to the reference peak, so every gauge weighs
    # the same) and the event volume (mm over the catchment, like the rain total)
    qk = dict(by="start", order=STARTS, key=("event", "slug", "gauge", "window_start", "tag"))
    qa = Q[Q.event.isin(aifs_ev)]
    qcs = pd.concat([crpss_ci(q, ai, col=col, **qk).assign(var=var, ai=lab)
                     for col, var in (("crps_peak_rel", "peak"), ("crps_vol_mm", "volume"))
                     for q, ai, lab in ((Q, "WN2", "WN2"), (qa, "AIFS", "AIFS"), (qa, "WN2", "WN2 (AIFS events)"))])
    qcs.to_csv(os.path.join(OUT, "discharge_crpss.csv"), index=False, float_format="%.3f")
    print("\ndischarge CRPSS by start:\n" + qcs.round(3).to_string(index=False))
    print("\n", mm.round(2).to_string(index=False)); print(mm3.round(2).to_string(index=False)); print(cs.round(3).to_string(index=False))

    # trends: WN2 vs IFS relative difference per point
    def rel(t, a="WN2"):
        return 100 * (t[a] - t["IFS"]) / t["IFS"]
    tr = []
    for t, by, kind in ((t2r, "bin", "rain"), (t2q, "start", "peak")):
        x = t.assign(rel=rel(t), kind=kind).rename(columns={by: "lead"})
        tr.append(x[["event", "slug", "gauge", "lead", "kind", "rel"]])
    tr = pd.concat(tr).merge(ind[["slug", "gauge", "area_km2", "ref_rain_mm", "ref_spec_peak"]], on=["slug", "gauge"])
    tr.to_csv(os.path.join(OUT, "trends.csv"), index=False, float_format="%.3f")
    print("\nTrends (Spearman rho of the WN2-IFS difference with):")
    for kind, g in tr.groupby("kind"):
        for v in ("area_km2", "ref_rain_mm", "ref_spec_peak"):
            print(f"  {kind:5s} {v:14s} rho = {g[v].rank().corr(g.rel.rank()):+.2f} (n = {len(g)})")
        big = g.area_km2 >= 300
        print(f"  {kind:5s} median diff: >=300 km2 {g[big].rel.median():+.1f} % (n {big.sum()}), <300 km2 {g[~big].rel.median():+.1f} % (n {(~big).sum()})")
        print("  by lead:", g.groupby("lead").rel.median().round(1).to_dict())

    # meteo -> discharge link: rain at the start's init vs the discharge peak, per point x start x product
    rr = R.assign(init=pd.to_datetime(R.init)).set_index(["slug", "gauge", "product", "init"]).crps_mm
    lk = []
    for k, g in Q.groupby(["slug", "gauge", "start"]):
        for ai in ("WN2", "AIFS"):
            a, i = g[g["product"] == ai], g[g["product"] == "IFS"]
            if a.empty or i.empty:
                continue
            ini_a, ini_i = a.init.iloc[0], i.init.iloc[0]
            try:
                ra, ri = rr.loc[(k[0], k[1], ai, ini_a)], rr.loc[(k[0], k[1], "IFS", ini_i)]
            except KeyError:
                continue
            lk.append({"slug": k[0], "gauge": k[1], "start": k[2], "ai": ai, "event": g.event.iloc[0],
                       "rain_rel": 100 * (float(np.mean(ra)) - float(np.mean(ri))) / float(np.mean(ri)),
                       "peak_rel": 100 * (a.crps_peak.mean() - i.crps_peak.mean()) / i.crps_peak.mean(),
                       "hydro_rel": 100 * (a.crps_series.mean() - i.crps_series.mean()) / i.crps_series.mean(),
                       "vol_rel": 100 * (a.crps_vol_mm.mean() - i.crps_vol_mm.mean()) / i.crps_vol_mm.mean(),
                       "volhalf_rel": 100 * (a.crps_volhalf_mm.mean() - i.crps_volhalf_mm.mean()) / i.crps_volhalf_mm.mean(),
                       "ifs_peak_bias": i.bias_peak_pct.mean(), "ai_peak_bias": a.bias_peak_pct.mean(),
                       "ifs_vol_bias": i.bias_vol_pct.mean(), "ai_vol_bias": a.bias_vol_pct.mean()})
    lk = pd.DataFrame(lk)
    lk.to_csv(os.path.join(OUT, "link.csv"), index=False, float_format="%.3f")
    fig_link(lk)
    for ai, g in lk.groupby("ai"):
        same = (np.sign(g.rain_rel) == np.sign(g.peak_rel)).mean()
        rc = g.rain_rel.clip(-100, 300)
        qc = g.peak_rel.clip(-100, 300)
        slope = np.polyfit(rc, qc, 1)[0]
        amp = (g.peak_rel.abs() / g.rain_rel.abs().replace(0, np.nan)).median()
        print(f"\nlink {ai}: n {len(g)}, same sign {100 * same:.0f} %, rho {g.rain_rel.rank().corr(g.peak_rel.rank()):+.2f}, "
              f"OLS slope (clipped) {slope:.2f}, median |dQ|/|dR| {amp:.2f}")
        for st, h in g.groupby("start"):
            print(f"   {st}: same sign {100 * (np.sign(h.rain_rel) == np.sign(h.peak_rel)).mean():.0f} %, "
                  f"median |dQ|/|dR| {(h.peak_rel.abs() / h.rain_rel.abs().replace(0, np.nan)).median():.2f}, n {len(h)}")
        # the same link for the flood volumes, and the asymmetry (winner of rain -> winner of peak / volume)
        win = lambda x: np.where(x > 100 * TIE, "IFS", np.where(x < -100 * (1 - 1 / (1 + TIE)), ai, "tie"))
        for col, lab in (("vol_rel", "flood volume"), ("volhalf_rel", "crest volume")):
            h = g.dropna(subset=[col])
            print(f"   link {ai} -> {lab}: same sign {100 * (np.sign(h.rain_rel) == np.sign(h[col])).mean():.0f} %, "
                  f"rho {h.rain_rel.rank().corr(h[col].rank()):+.2f}, median |dV|/|dR| "
                  f"{(h[col].abs() / h.rain_rel.abs().replace(0, np.nan)).median():.2f}, n {len(h)}")
        for col, lab in (("peak_rel", "peak"), ("vol_rel", "flood volume"), ("volhalf_rel", "crest volume")):
            print(f"   rain winner -> {lab} winner ({ai}):\n" +
                  pd.crosstab(pd.Series(win(g.rain_rel), name="rain"), pd.Series(win(g[col]), name=lab)).to_string())

    # sensitivity: keep the largest gauge of each nested group instead of the most extreme
    pa = select(p, "area_km2")
    Ra, Qa, _ = load_scores(pa)
    sens = []
    for name, t, by, order in (("rain IFS/WN2", tally(Ra, "bin", "crps_mm", ["IFS", "WN2"]), "bin", rb),
                               ("peak IFS/WN2", tally(Qa, "start", "crps_peak", ["IFS", "WN2"]), "start", STARTS),
                               ("peak 3-way", tally(Qa[Qa.event.isin(aifs_ev)], "start", "crps_peak", ["IFS", "AIFS", "WN2"]), "start", STARTS)):
        c = counts(t, by, order)
        print(f"\nsensitivity (largest kept, {int(pa.independent.sum())} points) {name}:\n{c.to_string()}")
        sens.append(c.assign(table=name).reset_index())
    pd.concat(sens).to_csv(os.path.join(OUT, "sensitivity_keep_largest.csv"), index=False)
    pa[pa.independent][["event", "gauge", "name", "area_km2"]].to_csv(os.path.join(OUT, "points_keep_largest.csv"), index=False)

    figs(p, R, Q, t2r, t3r, t2q, t3q, t2h, t3h, mm, cs, tr, lk, aifs_ev, t2v, t3v)
    for key, title, slugs in EVENTS:
        # Figs. S1-S11 also need the simulated hydrographs (data/wflow/<slug>/model, ensemble/), which
        # are not part of this repository; without them the event figures are skipped.
        if not all(os.path.isdir(os.path.join(REPO, "data", "wflow", s, "ensemble")) for s in slugs):
            print(f"event figure {key}: model outputs not found, skipped")
            continue
        event_fig(key, title, slugs, p, R, Q)
    print(OUT)


def figs(p, R, Q, t2r, t3r, t2q, t3q, t2h, t3h, mm, cs, tr, lk, aifs_ev, t2v, t3v):
    rb = [x[2] for x in RAIN_BINS]
    # Fig 1 (rain) and Fig 3 (discharge): who is best at how many independent points
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("IFS", "AIFS", "WN2", AI_BOTH, "tie")]
    labels = ["IFS", "AIFS", "WN2", "AIFS & WN2", "tie with IFS (within 5 %)"]
    fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 4.6 / 2.54), constrained_layout=True)
    bars(axs[0], counts(t2r, "bin", rb), "a  Rain total CRPS, IFS vs WN2 (all events)", ["IFS", "WN2"])
    bars(axs[1], counts(t3r, "bin", rb), "b  Rain total CRPS, 3-way (AIFS events)", ["IFS", "AIFS", "WN2"])
    fig.legend(h, labels, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(os.path.join(OUT, "fig1_best_rain.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(18 / 2.54, 8 / 2.54), constrained_layout=True)
    bars(axs[0, 0], counts(t2q, "start", STARTS), "a  Discharge peak CRPS, IFS vs WN2 (all events)", ["IFS", "WN2"])
    bars(axs[0, 1], counts(t2h, "start", STARTS), "b  Hydrograph CRPS, IFS vs WN2 (all events)", ["IFS", "WN2"])
    bars(axs[1, 0], counts(t3q, "start", STARTS), "c  Discharge peak CRPS, 3-way (AIFS events)", ["IFS", "AIFS", "WN2"])
    bars(axs[1, 1], counts(t3h, "start", STARTS), "d  Hydrograph CRPS, 3-way (AIFS events)", ["IFS", "AIFS", "WN2"])
    fig.legend(h, labels, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.savefig(os.path.join(OUT, "fig3_best_discharge.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    fig_peak_volume(t2q, t2v, t3q, t3v)
    fig_discharge_skill(pd.read_csv(os.path.join(OUT, "discharge_crpss.csv")), t2q, t2v)

    tl = pd.read_csv(os.path.join(OUT, "tail_hits.csv"))
    fig_rain_lead(cs, mm, tr, tl)
    fig_trends(tr)
    fig_tail(tl)

    # Fig 5: rain difference -> discharge difference
    fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 6.6 / 2.54), constrained_layout=True)
    for ax, ai in zip(axs, ("WN2", "AIFS")):
        g = lk[lk.ai == ai]
        for st, mk in zip(STARTS, ("o", "s", "^")):
            h = g[g.start == st]
            ax.scatter(h.rain_rel.clip(-100, 200), h.peak_rel.clip(-100, 200), s=10, marker=mk,
                       color=COL[ai], alpha=0.7, label=st, edgecolor="none")
        lim = [-100, 200]
        ax.plot(lim, lim, color="k", lw=0.6, ls="--")
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        same = (np.sign(g.rain_rel) == np.sign(g.peak_rel)).mean()
        ax.set_title(f"{ai} vs IFS ({len(g)} point × start; same sign {100 * same:.0f} %)", fontsize=7, loc="left")
        ax.set_xlabel("rain total CRPS difference (%)", fontsize=6)
        ax.set_ylabel("discharge peak CRPS difference (%)", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, frameon=False, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(OUT, "fig5_rain_to_discharge.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


_rvo = pd.read_csv(os.path.join(REPO, "figures", "aggregate_2", "synthesis", "reference_run_vs_observed.csv"))
OBS_FLAGS = {(e, int(g)): str(f) for e, g, f in zip(_rvo.event, _rvo.gauge, _rvo.flag.fillna(""))}


def fig_peak_volume(t2q, t2v, t3q, t3v):
    """Fig. 2 of the paper (file fig3b_peak_volume.png): best ensemble per gauge for the peak and the event
    volume, two-way (all events) and three-way (AIFS events). All panels share one x scale, so the bar
    lengths show the sample sizes (46-47 gauges against 19-21)."""
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("IFS", "AIFS", "WN2", AI_BOTH, "tie")]
    labels = ["IFS", "AIFS", "WN2", "AIFS & WN2", "tie with IFS (within 5 %)"]
    fig, axs = plt.subplots(2, 2, figsize=(18 / 2.54, 8 / 2.54), constrained_layout=True, sharex=True)
    bars(axs[0, 0], counts(t2q, "start", STARTS), "(a) Peak discharge, IFS vs WN2 (all events)", ["IFS", "WN2"])
    bars(axs[0, 1], counts(t2v, "start", STARTS), "(b) Event volume, IFS vs WN2 (all events)", ["IFS", "WN2"])
    bars(axs[1, 0], counts(t3q, "start", STARTS), "(c) Peak discharge, 3-way (AIFS events)", ["IFS", "AIFS", "WN2"])
    bars(axs[1, 1], counts(t3v, "start", STARTS), "(d) Event volume, 3-way (AIFS events)", ["IFS", "AIFS", "WN2"])
    n = max(len(t.drop_duplicates(["slug", "gauge"])) for t in (t2q, t2v))
    for ax in axs.ravel():
        ax.set_xlim(0, n + 1)
        ax.tick_params(axis="x", labelbottom=True)
    fig.legend(h, labels, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.savefig(os.path.join(OUT, "fig3b_peak_volume.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def fig_discharge_skill(qcs, t2q, t2v):
    """Draft alternative to Fig. 2 (file fig2alt_discharge_crpss.png): CRPSS against IFS ENS by start for the
    peak and the event volume, as Fig. 1a for rainfall, with the per-gauge skill of WN2 as faint dots."""
    x = np.arange(len(STARTS))
    YL = 0.5
    fig, axs = plt.subplots(1, 2, figsize=(16 / 2.54, 5.6 / 2.54), constrained_layout=True, sharey=True)
    for ax, var, t, col, ttl in ((axs[0], "peak", t2q, None, "(a) peak discharge"),
                                 (axs[1], "volume", t2v, None, "(b) event volume")):
        # per gauge and start: skill of WN2 from the gauge's mean CRPS (the values behind Fig. 2a-b)
        for j, st in enumerate(STARTS):
            g = t[t.start == st]
            v = (1 - g.WN2 / g.IFS).values
            xj = np.full(len(v), j - 0.1) + np.random.default_rng(j).uniform(-0.12, 0.12, len(v))
            inr = np.abs(v) <= YL
            ax.scatter(xj[inr], v[inr], s=5, color=WN2_COLOR, alpha=0.3, edgecolor="none", zorder=1)
            # beyond the axis: drawn at the edge as small triangles
            ax.scatter(xj[v > YL], np.full((v > YL).sum(), YL), s=7, marker="^", color=WN2_COLOR, alpha=0.5,
                       edgecolor="none", zorder=1, clip_on=False)
            ax.scatter(xj[v < -YL], np.full((v < -YL).sum(), -YL), s=7, marker="v", color=WN2_COLOR, alpha=0.5,
                       edgecolor="none", zorder=1, clip_on=False)
            ax.text(j - 0.1, -YL * 0.93, f"n = {len(v)}", ha="center", fontsize=5, color="#666666")
        for ai, c, mk in (("WN2", WN2_COLOR, "o"), ("AIFS", AIFS_COLOR, "s"), ("WN2 (AIFS events)", WN2_COLOR, "^")):
            s = qcs[(qcs["var"] == var) & (qcs.ai == ai)].set_index("bin").reindex(STARTS)
            dx = x + {"WN2": -0.1, "AIFS": 0.1, "WN2 (AIFS events)": 0.2}[ai]
            ax.errorbar(dx, s.crpss, yerr=[s.crpss - s.lo, s.hi - s.crpss], color=c, lw=1, capsize=2,
                        ls="-" if ai == "WN2" else ":", zorder=2)
            # one marker style, as Fig. 1a (user, 25 Nov); the bars show the interval
            ax.scatter(dx, s.crpss, color=c, marker=mk, s=16, zorder=3, label=ai)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([s_.replace(" (", "\n(") for s_ in STARTS], fontsize=6)
        ax.set_xlabel("forecast start", fontsize=6)
        ax.set_title(ttl, fontsize=7, loc="left")
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylim(-YL, YL)
    axs[0].set_ylabel("CRPSS vs IFS (–)\n> 0: better than IFS", fontsize=6)
    hd, lb = axs[0].get_legend_handles_labels()
    fig.legend(hd, [l_ + (" (5 events)" if l_ == "AIFS" else " (11 events)" if l_ == "WN2" else "") for l_ in lb],
               loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.17),
               title="dots: WN2 at each gauge", title_fontsize=5.5)
    fig.savefig(os.path.join(OUT, "fig2alt_discharge_crpss.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def case_skill(rel):
    """Per-case skill score of WN2 against IFS ENS, 1 - CRPS_WN2 / CRPS_IFS, from the relative
    CRPS difference in %. > 0: WN2 better, the same sign as the CRPSS in Fig. 1a."""
    return -rel / 100


def fig_rain_lead(cs, mm, tr, tl):
    """Fig. 1 of the paper (file fig2_rain_metrics_vs_lead.png): (a) CRPSS by lead, (b) bias,
    (c) per-case WN2 skill against the reference rainfall total, (d) cases reaching 300 mm with >= 10 % of the
    members there too.
    Lead = initialisation to the start of the event window."""
    rb = [x[2] for x in RAIN_BINS]
    fig, axs = plt.subplots(2, 2, figsize=(16 / 2.54, 10 / 2.54), constrained_layout=True)
    axs = axs.ravel()
    x = np.arange(len(rb))
    for ai, c, mk in (("WN2", WN2_COLOR, "o"), ("AIFS", AIFS_COLOR, "s"), ("WN2 (AIFS events)", WN2_COLOR, "^")):
        s = cs[cs.ai == ai].set_index("bin").reindex(rb)
        dx = x + {"WN2": -0.1, "AIFS": 0.1, "WN2 (AIFS events)": 0.2}[ai]
        axs[0].errorbar(dx, s.crpss, yerr=[s.crpss - s.lo, s.hi - s.crpss], color=c, marker=None, lw=1, capsize=2,
                        ls="-" if ai == "WN2" else ":")
        # one marker style (user, 24 Nov: filled vs open symbols confused readers; the bars show the interval)
        axs[0].scatter(dx, s.crpss, color=c, marker=mk, s=16, zorder=3, label=ai)
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_ylabel("CRPSS vs IFS (–)\n> 0: better than IFS", fontsize=6)
    axs[0].legend(fontsize=5.5, frameon=False, loc="lower right")
    for pr in ("IFS", "AIFS", "WN2"):
        s = mm[mm["product"] == pr].set_index("bin").reindex(rb)
        axs[1].plot(x, s["bias_pct"], color=COL[pr], marker="o", ms=3, lw=1.2, label=pr)
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_ylabel("ensemble-mean bias (%)", fontsize=6)
    axs[1].legend(fontsize=6, frameon=False, loc="upper right")
    for ax in axs[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace(" d", "") for b in rb], fontsize=6)
        ax.set_xlabel("lead to the start of the event window (d)", fontsize=6)
    # (d) the most extreme totals
    tail_panel(axs[3], tl, TAIL_T[-1], dotted=False)
    axs[3].set_ylabel(f"cases ≥ {TAIL_T[-1]} mm with ≥ 10 % of\nmembers also ≥ {TAIL_T[-1]} mm (%)", fontsize=6)
    axs[3].set_ylim(-2, 60)
    # (c) one dot per gauge and lead bin, shaded light (short lead) to dark (long lead); values below -1 drawn at -1
    ax = axs[2]
    g = tr[tr.kind == "rain"].assign(skill=lambda d: case_skill(d.rel))
    shade = dict(zip(rb, matplotlib.colormaps["Purples"](np.linspace(0.35, 0.95, len(rb)))))
    for o in rb[::-1]:
        h = g[g.lead == o]
        ax.scatter(h.ref_rain_mm, h.skill.clip(lower=-1), s=7, color=shade[o], edgecolor="white", linewidth=0.3,
                   label=o.replace(" d", ""))
    q = pd.qcut(g.ref_rain_mm, 6)
    med = g.groupby(q, observed=True).agg(xm=("ref_rain_mm", "median"), ym=("skill", "median"))
    ax.plot(med.xm, med.ym, color="k", lw=1, marker="o", ms=2.5, label="median")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-1.05, 1.0)
    rho = g.ref_rain_mm.rank().corr(g.skill.rank())
    print(f"  fig1c: rho {rho:+.2f} (n {len(g)}), below -1: {(g.skill < -1).sum()}; medians by sextile:\n"
          f"{med.round(2).to_string()}")
    ax.text(0.97, 0.95, f"ρ = {rho:+.2f}", transform=ax.transAxes, fontsize=6, va="top", ha="right")
    hd, lb = ax.get_legend_handles_labels()
    ax.legend(hd[-2::-1] + hd[-1:], lb[-2::-1] + lb[-1:], title="lead (d)", fontsize=5, title_fontsize=5,
              frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.2, ncol=2, columnspacing=0.6)
    ax.set_ylabel("CRPSS of WN2 vs IFS per case (–)\n> 0: better than IFS", fontsize=6)
    ax.set_xlabel("reference rainfall total over the catchment (mm)", fontsize=6)
    for ax, ttl in zip(axs, ("(a) skill against IFS, rain total", "(b) bias", "(c) WN2 skill against storm rainfall",
                             f"(d) reaching the most extreme totals (≥ {TAIL_T[-1]} mm)")):
        ax.set_title(ttl, fontsize=7, loc="left")
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(OUT, "fig2_rain_metrics_vs_lead.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def fig_trends(tr):
    """Fig. S12 (file fig4_trends.png): per-case WN2 skill against IFS ENS (same sign as the CRPSS) against
    catchment area, event size (reference rainfall total for rain, reference peak per unit area for the peak) and
    lead or start. Values below -1 are drawn at -1."""
    rb = [x[2] for x in RAIN_BINS]
    fig, axs = plt.subplots(2, 3, figsize=(18 / 2.54, 8.8 / 2.54), constrained_layout=True)
    for r_, kind, lab in ((0, "rain", "rain total"), (1, "peak", "discharge peak")):
        g = tr[tr.kind == kind].assign(skill=lambda d: case_skill(d.rel))
        order = rb if kind == "rain" else STARTS
        shade = dict(zip(order, matplotlib.colormaps["Greens"](np.linspace(0.4, 0.95, len(order)))))
        short = lambda o: o.split(" ")[0] if kind == "peak" else o.replace(" d", "")
        size = (("ref_rain_mm", "reference rainfall total (mm)", False) if kind == "rain" else
                ("ref_spec_peak", "reference peak per unit area (m³ s⁻¹ km⁻²)", True))
        for c_, v, xl, logx in ((0, "area_km2", "catchment area (km²)", True), (1,) + size,
                                (2, "lead", "lead (d)" if kind == "rain" else "start", False)):
            ax = axs[r_, c_]
            if v == "lead":
                data = [g[g.lead == o].skill.clip(lower=-1).values for o in order]
                ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": "k"})
                for j, (o, d) in enumerate(zip(order, data)):
                    ax.scatter(np.full(len(d), j + 1) + np.random.default_rng(j).uniform(-0.15, 0.15, len(d)), d, s=6,
                               color=shade[o], edgecolor="white", linewidth=0.3)
                ax.set_xticks(range(1, len(order) + 1))
                ax.set_xticklabels([short(o) for o in order], fontsize=6)
            else:
                # longest lead drawn first, so the short-lead dots sit on top
                for o in order[::-1]:
                    h = g[g.lead == o]
                    ax.scatter(h[v], h.skill.clip(lower=-1), s=7, color=shade[o], edgecolor="white", linewidth=0.3,
                               label=short(o) + (" d" if kind == "rain" else ""))
                if logx:
                    ax.set_xscale("log")
                rho = g[v].rank().corr(g.skill.rank())
                ax.text(0.03, 0.05, f"ρ = {rho:+.2f}", transform=ax.transAxes, fontsize=6, va="bottom")
                print(f"  fig4 {kind} {v}: rho {rho:+.2f}; by {'lead' if kind == 'rain' else 'start'} " + ", ".join(
                    f"{o} {g[g.lead == o][v].rank().corr(g[g.lead == o].skill.rank()):+.2f}" for o in order))
                if c_ == 1:
                    hd, lb = ax.get_legend_handles_labels()
                    ax.legend(hd[::-1], lb[::-1], title="lead" if kind == "rain" else "start", fontsize=5, title_fontsize=5,
                              frameon=False, loc="upper right", handletextpad=0.2, borderaxespad=0.2)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylim(-1.05, 1.0)
            ax.set_xlabel(xl, fontsize=6)
            ax.tick_params(labelsize=6)
            if c_ == 0:
                ax.set_ylabel(f"CRPSS of WN2 vs IFS, {lab} (–)\n> 0: WN2 better", fontsize=6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(OUT, "fig4_trends.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def fig_link(lk):
    """Fig. S15: WN2-IFS ENS CRPS difference of the discharge against that of the rainfall total, per
    independent gauge and start, for (a) the event volume and (b) the peak. In (b), open markers are the
    cases where both ensembles miss more than 80 % of the reference peak."""
    g = lk[lk.ai == "WN2"].copy()
    mk = {"short (-1 d)": "o", "medium (-2 d)": "s", "long (-7 d)": "^"}
    fig, axs = plt.subplots(1, 2, figsize=(16 / 2.54, 7 / 2.54), constrained_layout=True, sharex=True, sharey=True)
    for ax, col, lab in ((axs[0], "vol_rel", "(a) event volume"), (axs[1], "peak_rel", "(b) peak discharge")):
        ax.axhspan(-100 * (1 - 1 / (1 + TIE)), 100 * TIE, color="#eeeeee", zorder=0)
        ax.axvspan(-100 * (1 - 1 / (1 + TIE)), 100 * TIE, color="#eeeeee", zorder=0)
        ax.axhline(0, color="#999999", lw=0.6); ax.axvline(0, color="#999999", lw=0.6)
        ax.plot([-100, 200], [-100, 200], ls="--", color="#666666", lw=0.8)
        miss = (g.ifs_peak_bias < -80) & (g.ai_peak_bias < -80)
        for st, m in mk.items():
            h = g[g.start == st]
            x, y = h.rain_rel.clip(-100, 200), h[col].clip(-100, 200)
            if col == "peak_rel":
                mm = miss[h.index]
                ax.scatter(x[~mm], y[~mm], marker=m, s=14, color=WN2_COLOR, edgecolor="white", lw=0.4, label=st)
                ax.scatter(x[mm], y[mm], marker=m, s=14, facecolor="none", edgecolor=WN2_COLOR, lw=0.7)
            else:
                ax.scatter(x, y, marker=m, s=14, color=WN2_COLOR, edgecolor="white", lw=0.4, label=st)
        same = 100 * (np.sign(g.rain_rel) == np.sign(g[col])).mean()
        amp = (g[col].abs() / g.rain_rel.abs().replace(0, np.nan)).median()
        ax.set_title(f"{lab}\nsame sign {same:.0f} %, median size ratio {amp:.2f}", fontsize=8, loc="left")
        ax.set_xlim(-100, 200); ax.set_ylim(-100, 200)
        ax.set_xlabel("rainfall total: CRPS difference (%)", fontsize=7)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("discharge: CRPS difference (%)", fontsize=7)
    h1, l1 = axs[0].get_legend_handles_labels()
    h1.append(matplotlib.lines.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=WN2_COLOR))
    l1.append("(b) both miss > 80 % of the peak")
    fig.legend(h1, l1, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=6.5, frameon=False)
    fig.savefig(os.path.join(OUT, "fig7_rain_to_discharge_wn2.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def figs_only():
    """Redraw Fig. 1, Fig. 2, Fig. S12 and Fig. S13 from the saved tables (seconds; no score loading)."""
    rd = lambda f: pd.read_csv(os.path.join(OUT, f))
    fig_rain_lead(rd("meteo_crpss.csv"), rd("meteo_metrics.csv"), rd("trends.csv"), rd("tail_hits.csv"))
    fig_trends(rd("trends.csv"))
    fig_tail(rd("tail_hits.csv"))
    fig_peak_volume(rd("discharge_best_ifs_wn2.csv"), rd("volume_best_ifs_wn2.csv"),
                    rd("discharge_best_3way.csv"), rd("volume_best_3way.csv"))
    fig_discharge_skill(rd("discharge_crpss.csv"), rd("discharge_best_ifs_wn2.csv"), rd("volume_best_ifs_wn2.csv"))
    fig_link(rd("link.csv"))


def event_fig(key, title, slugs, p, R, Q):
    """One figure per event for the supplement: (a) upstream areas, (b) rain CRPS and (c) rain
    bias against lead, (d) peak and (e) event-volume CRPS per start, (f) hydrographs one day
    ahead at the most extreme independent point. Independent points only, as in the paper."""
    pe, re_, qe = p[p.event == key], R[R.event == key], Q[Q.event == key]
    prods = [x for x in ("IFS", "AIFS", "WN2") if x in set(re_["product"])]
    fig, axs = plt.subplots(2, 3, figsize=(17 / 2.54, 10 / 2.54), constrained_layout=True)
    axs = axs.ravel()
    # a: upstream areas, colour = reference rain, independent points outlined
    ax = axs[0]
    gdfs = []
    for slug in slugs:
        g = gpd.read_file(os.path.join(REPO, "data", "wflow", slug, "gauge_upstream.geojson"))
        g = g.merge(pe[pe.slug == slug][["gauge", "ref_rain_mm", "independent"]], left_on="id", right_on="gauge")
        gdfs.append(g)
    g = pd.concat(gdfs).sort_values("model_km2" if "model_km2" in gdfs[0] else "ref_rain_mm", ascending=False)
    norm = matplotlib.colors.Normalize(g.ref_rain_mm.min(), g.ref_rain_mm.max())
    g.plot(ax=ax, column="ref_rain_mm", cmap="Blues", norm=norm, edgecolor="#666666", linewidth=0.3)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="Blues"), ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("reference rainfall (mm)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    g[g.independent].boundary.plot(ax=ax, color="#d62728", linewidth=0.9)
    ax.set_axis_off()
    c = g[g.independent].to_crs(4326).geometry.union_all().centroid
    ax.text(0.0, 0.0, f"{abs(c.y):.1f}°{'N' if c.y >= 0 else 'S'}, {abs(c.x):.1f}°{'E' if c.x >= 0 else 'W'}",
            transform=ax.transAxes, fontsize=5.5, color="#444444")
    ax.set_title(f"(a) {len(g)} gauges, {int(pe.independent.sum())} independent (red)", fontsize=6.5, loc="left")
    # b, c: rain CRPS and ensemble-mean bias against lead, over the independent points
    for ax, lab in ((axs[1], "(b) Rainfall CRPS"), (axs[2], "(c) Rainfall bias of the ensemble mean")):
        wins = sorted(re_.window_start.unique())
        for pr in prods:
            for k, ws in enumerate(wins):
                r = re_[(re_["product"] == pr) & (re_.window_start == ws)].groupby("lead_d")
                y = r.crps_mm.mean() if ax is axs[1] else 100 * (r.ens_mean_mm.sum() / r.truth_mm.sum() - 1)
                lbl = pr if len(wins) == 1 else f"{pr}, {pd.Timestamp(ws):%-d %b} window"
                ax.plot(y.index, y.values, color=COL[pr], marker="o" if k == 0 else "s", ms=2.2, lw=1,
                        ls="-" if k == 0 else "--", label=lbl)
        ax.invert_xaxis()
        ax.set_xlabel("lead to the start of the event window (d)", fontsize=6)
        ax.set_title(lab, fontsize=6.5, loc="left")
    axs[1].set_ylabel("CRPS (mm)", fontsize=6)
    axs[1].legend(fontsize=5.5, frameon=False)
    axs[2].set_ylabel("bias (%)", fontsize=6)
    axs[2].axhline(0, color="#999999", lw=0.6)
    # d, e: peak and event-volume CRPS relative to the reference, per start
    st = [s_ for s_ in STARTS if s_ in set(qe.start)]
    w = 0.8 / max(len(prods), 1)
    for ax, col, lab in ((axs[3], "crps_peak_rel", "(d) Peak discharge"), (axs[4], "crps_vol_rel", "(e) Event volume")):
        for j, pr in enumerate(prods):
            v = [qe[(qe["product"] == pr) & (qe.start == s_)][col].mean() for s_ in st]
            ax.bar(np.arange(len(st)) + (j - (len(prods) - 1) / 2) * w, v, w, color=COL[pr], label=pr)
        ax.set_xticks(range(len(st)))
        ax.set_xticklabels([s_.split(" ", 1)[1].strip("()") for s_ in st], fontsize=6)
        ax.set_xlabel("forecast start", fontsize=6)
        ax.set_ylabel("CRPS / reference value", fontsize=6)
        ax.set_title(lab, fontsize=6.5, loc="left")
    # f: hydrographs at the most extreme independent point, -1 d start (else the shortest available)
    ax = axs[5]
    cand = pe[pe.independent].sort_values("ref_spec_peak", ascending=False)
    has1 = [((qe.slug == c.slug) & (qe.gauge == c.gauge) & (qe.start == STARTS[0])).any() for c in cand.itertuples()]
    top = cand[has1].iloc[0] if any(has1) else cand.iloc[0]          # most extreme point with a -1 d start
    q1 = qe[(qe.slug == top.slug) & (qe.gauge == top.gauge)]
    sname = STARTS[0] if STARTS[0] in set(q1.start) else next(s_ for s_ in STARTS if s_ in set(q1.start))
    col = f"{QCOL.get(top.slug, 'river_q_')}{top.gauge}"
    ref = run_csv(top.slug, REFERENCE.get(top.slug, {}).get("run", "hist_3h"))
    w0, w1 = pd.Timestamp(q1.window_start.iloc[0]), pd.Timestamp(q1.window_end.iloc[0])
    t0 = w0 - pd.Timedelta("1D")
    for pr in prods:
        c = q1[(q1["product"] == pr) & (q1.start == sname)]
        if c.empty:
            continue
        f = os.path.join(we.ens_dir(top.slug), c.case.iloc[0], "members_output.nc")
        if not os.path.exists(f):
            continue
        with xr.open_dataset(f) as ds:
            x = ds[col].sel(time=slice(t0, w1)).transpose("member", "time")
            t = pd.DatetimeIndex(x.time.values)
            v = x.values
        ax.fill_between(t, np.nanpercentile(v, 10, axis=0), np.nanpercentile(v, 90, axis=0), color=COL[pr], alpha=0.2, lw=0)
        ax.plot(t, np.nanmedian(v, axis=0), color=COL[pr], lw=1, label=f"{pr} median, 10–90 %")
    r = ref.loc[(ref.index >= t0) & (ref.index <= w1), col]
    ax.plot(r.index, r.values, color="k", lw=1.2, label="reference run")
    gr = pd.read_csv(os.path.join(REPO, "data", "wflow", top.slug, "gauges.csv"))
    o = load_obs(top.slug, gr[gr["id"] == top.gauge].iloc[0].to_dict())
    fl = OBS_FLAGS.get((top.slug, int(top.gauge)), "")              # reference_run_vs_observed.csv
    if fl.startswith("exclude"):
        o = None
    if o is not None and len(o[(o.index >= t0) & (o.index <= w1)]):
        o = o[(o.index >= t0) & (o.index <= w1)]
        ax.plot(o.index, o.values, color="#777777", lw=0.8,
                label="observed (lower bound)" if fl.startswith("lower bound") else "observed")
    elif np.isfinite(top.obs_peak) and not fl.startswith("exclude"):
        ax.axhline(top.obs_peak, color="k", ls=":", lw=0.8, label="observed peak")
    ax.set_title(f"(f) {top['name'].split(' (')[0]} ({top.area_km2:.0f} km²), {sname.split(' ', 1)[1].strip('()')} start",
                 fontsize=6.5, loc="left")
    ax.set_ylabel("discharge (m³ s⁻¹)", fontsize=6)
    ax.legend(fontsize=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(np.ceil((w1 - t0).days / 3)))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    for a in axs[1:]:
        a.tick_params(labelsize=5.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(title, fontsize=8, x=0.01, ha="left")
    fig.savefig(os.path.join(OUT, f"event_{key}.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figs_only() if "--figs-only" in sys.argv else main()
