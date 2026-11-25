#!/usr/bin/env python3
"""
Table S2 of the HESS Brief Communication supplement: per event, the rainfall and discharge
windows scored at the independent gauges, the initialisations behind each lead bin and the
three discharge starts. Read from the same score files as backbone_analysis.py, so the counts
are the ones behind Figs. 1 and 2.

    python scripts/si_windows_table.py

Outputs: figures/backbone/windows_counts.csv and windows_counts.md (Markdown table for the SI).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import backbone_analysis as b                    # noqa: E402

ORDER = ["catalonia", "jesenik", "texas25", "texas26", "mississippi", "helene", "hans", "zermatt",
         "california", "kristin", "dana"]
NAMES = {"catalonia": "Catalonia", "jesenik": "Storm Boris", "texas25": "Central Texas",
         "texas26": "Texas Hill Country", "mississippi": "Mississippi Gulf Coast", "helene": "Hurricane Helene",
         "hans": "Storm Hans", "zermatt": "Valais/Vispa", "california": "California ARs",
         "kristin": "Storm Kristin", "dana": "DANA Valencia"}
BINS = [x[2] for x in b.RAIN_BINS]


def span(a, z):
    a, z = pd.Timestamp(a), pd.Timestamp(z)
    return f"{a:%d %b %H}Z – {z:%d %b %H}Z"


def main():
    p = b.points()
    R, Q, _ = b.load_scores(p)
    rows = []
    for ev in ORDER:
        r, q = R[R.event == ev], Q[Q.event == ev]
        rw = "; ".join(sorted({span(a, z) for a, z in zip(r.window_start, r.window_end)}))
        qw = "; ".join(sorted({span(a, z) for a, z in zip(q.window_start, q.window_end)}))
        inits = r[r["product"] == "IFS"].init
        cases = r[r["product"] == "IFS"].groupby("bin").size().reindex(BINS).fillna(0).astype(int)
        starts = q[q["product"] == "IFS"].groupby("start").init.agg(lambda s: ", ".join(sorted({f"{pd.Timestamp(x):%d %b}" for x in s})))
        rows.append({"event": NAMES[ev], "rain_window": rw, "discharge_window": qw,
                     "inits": f"{pd.Timestamp(inits.min()):%d %b} – {pd.Timestamp(inits.max()):%d %b} ({inits.nunique()})",
                     **{f"cases {k}": int(cases[k]) for k in BINS},
                     **{s: starts.get(s, "–") for s in b.STARTS}})
    t = pd.DataFrame(rows)
    out = os.path.join(b.OUT, "windows_counts")
    t.to_csv(out + ".csv", index=False)
    tot = {f"cases {k}": int(t[f"cases {k}"].sum()) for k in BINS}
    head = ["Event", "Rainfall window", "Discharge window", "IFS ENS initialisations (n)",
            "Cases per lead bin (<1.5 / 1.5–3.5 / 3.5–5.5 / ≥5.5 d)", "Discharge starts (−1 / −2 / −7 d)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for x in t.itertuples(index=False):
        c = " / ".join(str(v) for k, v in zip(t.columns, x) if k.startswith("cases "))
        s = " / ".join(str(v) for k, v in zip(t.columns, x) if k in b.STARTS)
        lines.append(f"| {x.event} | {x.rain_window} | {x.discharge_window} | {x.inits} | {c} | {s} |")
    lines.append(f"| Total | | | | {' / '.join(str(v) for v in tot.values())} | |")
    open(out + ".md", "w", encoding="utf8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
