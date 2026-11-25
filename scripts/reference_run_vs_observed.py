#!/usr/bin/env python3
"""
Reference-run peak vs observed peak at every scored gauge with an observation (caveat table).

Reads the obs_peak and ref_peak columns that gauge_discharge_scores.py writes per event, adds the
observed peaks that only exist as published or digitised values (MANUAL), and flags the gauges
whose observed peak is not a usable value (FLAGS). Ratios are reference-run peak / observed peak;
"lower bound" rows (sensor lost at the peak) are counted separately.

    python scripts/reference_run_vs_observed.py

Output: figures/aggregate_2/synthesis/reference_run_vs_observed.csv (one row per gauge and flow
window) and a per-event summary on stdout.
"""
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figures", "aggregate_2", "synthesis", "reference_run_vs_observed.csv")
SLUGS = ["catalonia_2026", "boris_sudetes", "texas_2025", "texas_2026", "mississippi_2026", "helene_2024",
         "hans_2023", "zermatt_2024", "california_2023", "kristin_mondego", "dana_2024"]
# observed peaks that are not in an observation file (m3/s, source)
MANUAL = {("kristin_mondego", 6114300): (1750.0, "OE/APA report, digitised (±20 m3/s)"),
          ("dana_2024", 9000404): (4198.0, "published (Muñoz et al. 2025)"),
          ("dana_2024", 9000402): (5700.0, "reconstructed 5,700-7,700 (lower end)")}
# observed value at the peak not usable as a peak
FLAGS = {("boris_sudetes", 8): "exclude: record lost at the peak",
         ("dana_2024", 9000401): "lower bound: sensor lost",
         ("dana_2024", 6227130): "lower bound: sensor lost",
         ("dana_2024", 9000402): "lower bound: reconstruction range"}


def main():
    rows = []
    for slug in SLUGS:
        f = os.path.join(REPO, "figures", slug, "gauge_discharge", "discharge_scores.csv")
        if not os.path.exists(f):
            print(f"missing (skipped): {f}")
            continue
        d = pd.read_csv(f).drop_duplicates(["gauge", "window_start"])
        if "obs_peak" not in d:
            d["obs_peak"] = np.nan
        d["source"] = np.where(d.obs_peak.notna(), "observation file", "")
        for (s, g), (v, src) in MANUAL.items():
            if s == slug:
                d.loc[d.gauge == g, ["obs_peak", "source"]] = [v, src]
        d["flag"] = [FLAGS.get((slug, g), "") for g in d.gauge]
        rows.append(d.assign(event=slug)[["event", "gauge", "code", "area_km2", "window_start", "ref_peak",
                                          "obs_peak", "source", "flag"]])
    t = pd.concat(rows, ignore_index=True).dropna(subset=["obs_peak"])
    t["ref_over_obs"] = t.ref_peak / t.obs_peak
    t.to_csv(OUT, index=False, float_format="%.2f")
    use = t[~t.flag.str.startswith("exclude") & ~t.flag.str.startswith("lower")]
    ok = use.ref_over_obs.between(0.5, 2)
    print(f"{len(use)} gauges with a usable observed peak; {ok.sum()} within a factor of 2; "
          f"{(t.flag.str.startswith('lower')).sum()} lower bounds, {(t.flag.str.startswith('exclude')).sum()} excluded")
    s = use.groupby("event").ref_over_obs.agg(["count", "median", "min", "max"])
    s["within_x2"] = use.assign(ok=ok).groupby("event").ok.sum()
    print(s.round(2).to_string())
    print(OUT)


if __name__ == "__main__":
    main()
