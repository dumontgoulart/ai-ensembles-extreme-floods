#!/usr/bin/env python3
"""
Robustness numbers of Sect. S3 of the HESS Brief Communication supplement, from the same score
files and point selection as backbone_analysis.py (so they match Figs. 1, 2 and S14):
  - leave one event out: WN2 rainfall CRPSS per lead bin without each event in turn (range)
  - tie threshold (Table S3): IFS ENS / WN2 / tie summed over lead bins (rainfall) and starts
    (peak, event volume) for 2, 5 and 10 %
  - event-volume CRPSS of WN2 by start without the events where IFS ENS wins by large margins
    (Helene, Valencia), and the count behind the three-way "AIFS & WN2" class
  - Table S4 (rainfall bias and reference above every member, per lead bin) and Table S5
    (median peak and event-volume bias from -1 d per event)
    python scripts/si_robustness.py
Outputs: figures/backbone/si_robustness.txt (and stdout).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backbone_analysis as b   # noqa: E402

OUT = os.path.join(b.OUT, "si_robustness.txt")


def crpss(df, ai, col, by, key):
    """{bin/start: CRPSS of ai against IFS on paired cases} (no bootstrap)."""
    out = {}
    for k, s in df.groupby(by):
        a = s[s["product"] == ai].set_index(list(key))[col]
        i = s[s["product"] == "IFS"].set_index(list(key))[col]
        j = a.index.intersection(i.index)
        if len(j):
            out[k] = 1 - a.loc[j].sum() / i.loc[j].sum()
    return out


def main():
    p = b.points()
    R, Q, _ = b.load_scores(p)
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    rkey, qkey = ("event", "gauge", "init"), ("event", "slug", "gauge", "window_start", "tag")
    bins = [x[2] for x in b.RAIN_BINS]

    say("Leave one event out, WN2 rainfall CRPSS (min to max over the events left out):")
    loo = {bn: [] for bn in bins}
    for ev in sorted(R.event.unique()):
        for bn, v in crpss(R[R.event != ev], "WN2", "crps_mm", "bin", rkey).items():
            loo[bn].append((v, ev))
    for bn in bins:
        lo, hi = min(loo[bn]), max(loo[bn])
        say(f"  {bn:10s} {lo[0]:+.3f} (without {lo[1]}) to {hi[0]:+.3f} (without {hi[1]})")

    say("\nTable S3, IFS / WN2 / tie summed over lead bins or starts:")
    tie0 = b.TIE
    for t in (0.02, 0.05, 0.10):
        b.TIE = t
        cells = []
        for df, by, col in ((R, "bin", "crps_mm"), (Q, "start", "crps_peak"), (Q, "start", "crps_vol_mm")):
            w = b.tally(df, by, col, ["IFS", "WN2"]).winner.value_counts()
            cells.append(f"{w.get('IFS', 0)} / {w.get('WN2', 0)} / {w.get('tie', 0)}")
        say(f"  {100 * t:.0f} %: rainfall {cells[0]} | peak {cells[1]} | volume {cells[2]}")
    b.TIE = tie0

    say("\nThree-way 'AIFS & WN2' cases (5 % rule):")
    qa = Q[Q.event.isin(Q[Q["product"] == "AIFS"].event.unique())]
    for col, lab in (("crps_peak", "peak"), ("crps_vol_mm", "volume")):
        w = b.tally(qa, "start", col, ["IFS", "AIFS", "WN2"]).winner
        say(f"  {lab}: {int((w == b.AI_BOTH).sum())} of {len(w)} gauge-start cases")

    say("\nEvent-volume CRPSS of WN2 by start, all events / without Helene and Valencia:")
    a = crpss(Q, "WN2", "crps_vol_mm", "start", qkey)
    c = crpss(Q[~Q.event.isin(["helene", "dana"])], "WN2", "crps_vol_mm", "start", qkey)
    for st in b.STARTS:
        say(f"  {st:14s} {a[st]:+.3f} / {c[st]:+.3f}")
    say("\nTable S4, ensemble-mean rainfall bias (%) / reference above every member (%), per lead bin:")
    for f, lab, prods in (("meteo_metrics.csv", "all events", ["IFS", "WN2"]),
                          ("meteo_metrics_aifs_events.csv", "AIFS ENS events", ["IFS", "AIFS", "WN2"])):
        mt = pd.read_csv(os.path.join(b.OUT, f))
        for pr in prods:
            r = mt[mt["product"] == pr].set_index("bin").reindex(bins)
            say(f"  {pr:5s} {lab:16s} " + " | ".join(f"{x:+.0f} / {y:.0f}" for x, y in zip(r.bias_pct, r.truth_above_all_pct)))

    say("\nTable S5, median bias (%) of the ensemble-mean peak / event volume from -1 d, per event:")
    s1 = Q[Q.start == b.STARTS[0]]
    t = s1.groupby(["event", "product"])[["bias_peak_pct", "bias_vol_pct"]].median().unstack()
    for ev, r in t.iterrows():
        f_ = lambda col: " / ".join("–" if pd.isna(r[(col, pr)]) else f"{r[(col, pr)]:+.0f}" for pr in ("IFS", "AIFS", "WN2"))
        say(f"  {ev:12s} peak {f_('bias_peak_pct'):18s} volume {f_('bias_vol_pct')}")
    lk = pd.read_csv(os.path.join(b.OUT, "link.csv"))
    lk = lk[(lk.ai == "WN2") & (lk.start == b.STARTS[0])]
    say(f"  all gauges   peak {lk.ifs_peak_bias.median():+.0f} / – / {lk.ai_peak_bias.median():+.0f}   "
        f"volume {lk.ifs_vol_bias.median():+.0f} / – / {lk.ai_vol_bias.median():+.0f}")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
