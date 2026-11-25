#!/usr/bin/env python3
"""
Run the hydrological part and the scoring for one or more events, from the processed forecasts
and the reference rainfall to the score files that the analysis reads.

Per event (the same steps, in the same order, as in the paper):
  1. base forcing and reference run forcing (hist_3h)
  2. one spin-up per forecast start, cold start -> initialisation, and a check that a warm restart
     reproduces an uninterrupted run
  3. every ensemble member of every forecast case (plus the ensemble-mean run), warm-started from
     the spin-up of its start, then collected into <case>/members_output.nc
  4. the continuous cases (reference run, ERA5 and ensemble-mean forcing)
  5. scores: catchment-average rainfall (event window, core band, IFS at 0.25 deg) and discharge

    python scripts/run_event.py texas_2026 --workers 4
    python scripts/run_event.py texas_2026 --only score          # rescore only
    python scripts/run_event.py --list                            # the paper events

Steps 1-4 need the processed forecasts (scripts/preprocess.py), the reference rainfall and the
wflow_sbm models (see README); step 5 needs their outputs. Member runs are independent, so
--workers runs that many side by side (one Julia thread each). Finished members are skipped,
so an interrupted run can simply be started again.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PAPER_EVENTS = ["catalonia_2026", "boris_sudetes", "texas_2025", "texas_2026", "mississippi_2026",
                "helene_2024", "hans_2023", "zermatt_2024", "california_2023", "kristin_mondego",
                "kristin_lis", "dana_2024"]
HANS = "hans_2023"     # calibrated Drammen model with its own forcing script


def py(*args, check=True, quiet=False):
    cmd = [sys.executable, os.path.join(HERE, args[0]), *map(str, args[1:])]
    print("$", " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=REPO, text=True, capture_output=quiet)
    if check and r.returncode:
        sys.exit(f"failed: {' '.join(cmd[1:])}\n{r.stderr if quiet else ''}")
    return r


def lines(*args):
    r = py(*args, quiet=True)
    return [ln.split("\t") for ln in r.stdout.splitlines() if ln.strip()]


def base(slug):
    if slug == HANS:
        py("wflow_drammen_forcing.py", "hist_3h")
    else:
        py("wflow_setup.py", slug, "hist_3h")
        py("wflow_prep_forcing.py", slug, "hist_3h")


def spinups(slug):
    for _, init in lines("wflow_ensemble.py", "inits", slug):
        py("wflow_ensemble.py", "spinup", slug, init)
        py("wflow_ensemble.py", "validate-run", slug, init)


def members(slug, workers):
    os.environ.setdefault("WFLOW_THREADS", "1")
    for _, case, n, _ in lines("wflow_ensemble.py", "plan", slug):
        tasks = [str(i) for i in range(int(n))] + (["mean"] if int(n) > 1 else [])
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(lambda t: py("wflow_ensemble.py", "member", slug, case, t, check=False, quiet=True), tasks))
        bad = [t for t, r in zip(tasks, res) if r.returncode]
        if bad:
            print(f"  {case}: {len(bad)} member runs failed ({', '.join(bad[:5])}...); rerun to retry")
        py("wflow_ensemble.py", "collect", slug, case)


def cases(slug):
    mdir = os.path.join(REPO, "data", "wflow", slug, "model")
    tags = sorted({d[len(f"run_{slug}_"):] for d in os.listdir(mdir) if d.startswith(f"run_{slug}_")} - {"hist"})
    for c in tags:
        if slug == HANS:
            py("wflow_drammen_forcing.py", c)
        else:
            py("wflow_setup.py", slug, c)
            py("wflow_prep_forcing.py", slug, c)
            py("wflow_setup.py", slug, c)
        py("wflow_run.py", slug, c)


def score(slug):
    py("upstream_rain_scores.py", slug)
    py("upstream_rain_scores.py", slug, "--band", "core")
    py("upstream_rain_scores.py", slug, "--ifs025")
    py("gauge_discharge_scores.py", slug)


STEPS = {"base": base, "spinup": spinups, "members": members, "cases": cases, "score": score}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="*", help="event slugs (default: none; see --list)")
    ap.add_argument("--only", nargs="+", choices=list(STEPS), help="run only these steps")
    ap.add_argument("--workers", type=int, default=1, help="member runs side by side (default 1)")
    ap.add_argument("--list", action="store_true", help="print the paper events and exit")
    a = ap.parse_args()
    if a.list or not a.events:
        print("\n".join(PAPER_EVENTS))
        return
    for slug in a.events:
        print(f"=== {slug}", flush=True)
        for name in (a.only or list(STEPS)):
            STEPS[name](slug, a.workers) if name == "members" else STEPS[name](slug)


if __name__ == "__main__":
    main()
