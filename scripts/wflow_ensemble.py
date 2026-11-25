#!/usr/bin/env python3
"""
Per-member wflow forecast runs, warm-started from a shared spin-up state.

The mean-forced cases (wflow_setup.py + wflow_prep_forcing.py) run the whole
spin-up (months to two years) for every case. Per member that would be
thousands of identical spin-ups, so here the spin-up runs ONCE per model and
forecast start, and every member restarts from its saved state and simulates
only the forecast window.

Exactness. Wflow.jl (1.0.x) advances the clock and THEN reads forcing, so the
forcing stamped T drives the step (T-dt, T], and the state file is written at
`endtime`. Every precipitation source is put on that end-stamped convention
(wflow_prep_forcing.accum_to_slots); the forecast owns the slots after its
actual init time (fstart) and MSWEP everything up to and including fstart. So:
    spin-up : cold start at sim_start  -> endtime = fstart   (MSWEP base)
    member  : warm start from that state, fstart -> sim_end   (member precip)
reproduces an uninterrupted run step for step (`validate-run`).

Layout, under data/wflow/{slug}/ensemble/:
    states/{YYYYMMDDHH}/   spin-up to fstart: outstates.nc, output.csv, toml, COMPLETE
    {case}/m{NN}/          one member, NN = the product's ensemble number
                           (IFS/AIFS 0 = control, 1-50 perturbed; WN2 0-63)
    {case}/mean/           forced with the ensemble mean of all members
    {case}/members_output.nc   every member's output.csv, collected
Member forcing is built per job on node-local scratch and deleted afterwards;
only the TOML, log and output.csv are kept.

Usage:
    python scripts/wflow_ensemble.py plan [slug ...]
    python scripts/wflow_ensemble.py inits [slug ...]
    python scripts/wflow_ensemble.py missing [slug ...]
    python scripts/wflow_ensemble.py spinup   <slug> <init_date>
    python scripts/wflow_ensemble.py spinup-status <slug> <init_date>
    python scripts/wflow_ensemble.py member   <slug> <case> <index|mean>
    python scripts/wflow_ensemble.py collect  <slug> <case>
    python scripts/wflow_ensemble.py compare-forcing <slug> <case>
    python scripts/wflow_ensemble.py validate-run    <slug> <init YYYY-MM-DDTHH>
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wflow_prep_forcing as wpf                          # noqa: E402
from wflow_run import detect_julia, strip_conda_ld_path  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANS = "hans_2023"
ENC = {"dtype": "float32", "zlib": True, "complevel": 1}

# Catchments whose members must run the full cold-start period instead of a
# warm restart. Empty since the Hans restart mismatch was traced to
# fews_run__flag = true (Wflow then starts one step after `starttime`, so the
# restart skipped a step); with the flag forced off every model restarts exactly.
COLD_START = set()

# Events with per-member runs. kristin_2026 is the full 1.77 M-cell basin; the
# clipped sub-basins share its forcing config (see wflow_prep_forcing.py).
EVENTS = ["texas_2025", "gezani_2026", "california_ar_2023", "bc_ar_2024",
          "zermatt_2024", "kristin_lis", "kristin_mondego", "dana_2024",
          "boris_2024", "boris_sudetes", HANS, "kristin_2026", "catalonia_2026",
          "california_2023", "helene_2024", "texas_2026", "mississippi_2026"]


# ── configuration lookup (generic lat/lon pipeline or the Drammen one) ───────
def event_cfg(slug):
    """(sim_start, sim_end, {case: case_cfg}) for an event."""
    if slug == HANS:
        import wflow_drammen_forcing as wdf
        return pd.Timestamp(wdf.SIM_START), pd.Timestamp(wdf.SIM_END), wdf.CASES
    e = wpf.EVENT_CONFIG[slug]
    return pd.Timestamp(e["sim_start"]), pd.Timestamp(e["sim_end"]), e["cases"]


def forecast_cases(slug):
    """{case: case_cfg} for the forecast (blend) cases of an event.

    Cases whose init is not (yet) in the processed file are left out with a
    warning, so the other starts can run (e.g. a start still downloading).
    """
    out = {}
    for c, cc in event_cfg(slug)[2].items():
        if cc["precip_mode"] != "blend":
            continue
        if slug != HANS:
            try:
                fstart_of(cc)
            except (ValueError, FileNotFoundError, OSError) as e:
                print(f"[WARN] {slug}/{c} left out: {e}", file=sys.stderr)
                continue
        out[c] = cc
    return out


def model_dir(slug):
    return os.path.join(REPO_ROOT, "data", "wflow", slug, "model")


def ens_dir(slug):
    return os.path.join(REPO_ROOT, "data", "wflow", slug, "ensemble")


def case_toml(slug, case):
    return os.path.join(model_dir(slug), f"wflow_sbm_{slug}_{case}.toml")


def timestep(slug, case):
    m = re.search(r"^timestepsecs\s*=\s*(\d+)", open(case_toml(slug, case)).read(), re.M)
    return pd.Timedelta(seconds=int(m.group(1)))


def _starts_table():
    """data/forecast_starts.csv: the actual init time of every paper case, read once from the
    processed forecast files of the full project. Used when those files are not present, so the
    analysis can run from the score files alone (the processed forecasts are not redistributed)."""
    f = os.path.join(REPO_ROOT, "data", "forecast_starts.csv")
    if not os.path.exists(f):
        return {}
    t = pd.read_csv(f, dtype=str)
    return {(r.forecast_src, r.init_date): pd.Timestamp(r.fstart) for r in t.itertuples()}


_STARTS = None


def fstart_of(case_cfg):
    """Actual forecast init time (e.g. 12Z for texas aifs_jul01)."""
    global _STARTS
    try:
        return wpf.forecast_init(case_cfg)
    except (FileNotFoundError, OSError):
        if _STARTS is None:
            _STARTS = _starts_table()
        src = str(case_cfg.get("forecast_src", ""))
        key = (os.path.relpath(src, REPO_ROOT) if os.path.isabs(src) else src, str(case_cfg.get("init_date")))
        if key in _STARTS:
            return _STARTS[key]
        raise


def state_dir(slug, fstart):
    return os.path.join(ens_dir(slug), "states", fstart.strftime("%Y%m%d%H"))


def spinup_done(slug, fstart, dt):
    """True once the spin-up for fstart has FINISHED.

    Wflow creates outstates.nc when the run starts and fills it at the end, so
    the file existing means nothing. New spin-ups leave a COMPLETE marker; for
    older ones, the state must be written after the output reached its end.
    """
    d = state_dir(slug, fstart)
    state, csv = os.path.join(d, "outstates.nc"), os.path.join(d, "output.csv")
    if os.path.exists(os.path.join(d, "COMPLETE")):
        return True
    return (os.path.exists(state) and output_complete(csv, fstart)
            and os.path.getmtime(state) >= os.path.getmtime(csv))


def members_of(case_cfg):
    """Ensemble numbers in the processed forecast file, in file order."""
    with xr.open_dataset(os.path.join(REPO_ROOT, case_cfg["forecast_src"])) as ds:
        return [int(n) for n in ds["number"].values]


# "subset" = the members used before the full-ensemble download: IFS/AIFS
# control + perturbed 1-10, WN2 samples 0-9. Its mean, with today's forcing
# rules, separates the effect of ensemble size from that of the timing fixes.
SUBSET_SIZE = {"ifs": 11, "aifs": 11, "wn2": 10}


def member_label(case_cfg, index):
    if index == "mean":
        return "mean"
    if index == "subset":
        return f"mean_first{_subset_slice(case_cfg).stop}"
    return f"m{members_of(case_cfg)[int(index)]:02d}"


def _subset_slice(case_cfg):
    product = os.path.basename(case_cfg["forecast_src"]).split("_")[0]
    product = {"weathernext": "wn2"}.get(product, product)
    if product not in SUBSET_SIZE:
        raise ValueError(f"no pre-download subset defined for {case_cfg['forecast_src']}")
    nums = members_of(case_cfg)
    k = SUBSET_SIZE[product]
    if nums[:k] != list(range(k)):
        raise ValueError(f"first {k} members are {nums[:k]}, expected 0..{k - 1}")
    return slice(0, k)


# ── TOML patching ─────────────────────────────────────────────────────────────
def _strip_sections(content, prefix):
    out, skip = [], False
    for line in content.splitlines(keepends=True):
        s = line.strip()
        if s.startswith("["):
            skip = s.strip("[]").startswith(prefix)
        if not skip:
            out.append(line)
    return "".join(out)


def write_toml(slug, template_case, dst, *, start, end, forcing_glob,
               cold, state_in=None, state_out, out_dir):
    """Patch the case TOML into a self-contained one with absolute paths."""
    c = open(case_toml(slug, template_case)).read()
    static = re.search(r'^path_static\s*=\s*"([^"]+)"', c, re.M).group(1)
    fmt = "%Y-%m-%dT%H:%M:%S"
    patches = {
        r"^(dir_output\s*=\s*).*$":        f'\\g<1>"{out_dir}"',
        r"^(starttime\s*=\s*).*$":         f"\\g<1>{start.strftime(fmt)}",
        r"^(endtime\s*=\s*).*$":           f"\\g<1>{end.strftime(fmt)}",
        r"^(cold_start__flag\s*=\s*).*$":  f"\\g<1>{'true' if cold else 'false'}",
        r"^(path_forcing\s*=\s*).*$":      f'\\g<1>"{forcing_glob}"',
        r"^(path_static\s*=\s*).*$":       f'\\g<1>"{os.path.join(model_dir(slug), static)}"',
        r"^(path_output\s*=\s*).*$":       f'\\g<1>"{state_out}"',
    }
    if state_in:
        patches[r"^(path_input\s*=\s*).*$"] = f'\\g<1>"{state_in}"'
    for pat, rep in patches.items():
        c, n = re.subn(pat, rep, c, flags=re.M)
        if n != 1:
            raise RuntimeError(f"TOML patch {pat!r} matched {n} lines in {template_case}")
    c = re.sub(r"^(fews_run__flag\s*=\s*).*$", r"\g<1>false", c, flags=re.M)   # see wflow_setup.patch_toml
    c = _strip_sections(c, "output.netcdf_grid")
    with open(dst, "w") as fh:
        fh.write(c)


def run_wflow(toml):
    julia = detect_julia()
    if julia is None:
        sys.exit("Julia not found (~/.juliaup/bin/julia or PATH).")
    # WFLOW_THREADS: one thread per run when a bundle runs several members side by side
    threads = os.environ.get("WFLOW_THREADS") or os.environ.get("SLURM_CPUS_PER_TASK", "auto")
    cmd = [str(julia), "--threads", threads, "--project=@wflow1",
           "-e", "using Wflow; Wflow.run()", toml]
    env = strip_conda_ld_path(os.environ.copy())
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    print("  $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=os.path.dirname(toml), env=env)
    if r.returncode:
        raise RuntimeError(f"wflow exited with {r.returncode}")


def output_complete(csv, end):
    if not os.path.exists(csv):
        return False
    try:
        last = pd.read_csv(csv, usecols=[0]).iloc[-1, 0]
    except Exception:
        return False
    return pd.Timestamp(last) >= end


# ── forcing for one member over the forecast window ──────────────────────────
def _base_files(slug):
    files = sorted(glob.glob(os.path.join(model_dir(slug), "inmaps_base", "base_*.nc")))
    if not files:
        raise FileNotFoundError(f"No inmaps_base for {slug}; build the mean-forced cases first.")
    return files


def _read_base(slug, times):
    """Base months covering `times`, read eagerly one file at a time.

    No open_mfdataset/dask here: with the 500 MB full-Kristin base months, the
    dask-threaded HDF5 reads deadlocked in ~1 in 5 member jobs (each hung at
    this step until the time limit). Only the 1-2 months of the window are read.
    """
    months = {f"{t:%Y%m}" for t in times}
    parts, sref = [], None
    for f in _base_files(slug):
        if os.path.basename(f)[5:11] not in months:
            continue
        with xr.open_dataset(f) as d:
            if sref is None and "spatial_ref" in d:
                sref = d["spatial_ref"].load()
            d = d.drop_vars("spatial_ref", errors="ignore")
            parts.append(d.sel(time=d["time"].isin(times.values)).load())
    base = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    missing = times.difference(pd.DatetimeIndex(base["time"].values))
    if len(missing):
        raise ValueError(f"base forcing lacks {len(missing)} slots, first {missing[0]}")
    base.attrs["_spatial_ref"] = sref
    return base


def _blend_window(fcast, times):
    """Forecast precip on `times` (all after the init); zero past the horizon.

    Same rule as the event-window overlay in wflow_prep_forcing.build_forcing:
    the exact 3 h forecast slot up to the horizon, zero after it.
    """
    horizon = pd.DatetimeIndex(fcast["time"].values).max()
    ev_in, ev_out = times[times <= horizon], times[times > horizon]
    parts = []
    if len(ev_in):
        parts.append(fcast.sel(time=ev_in))
    if len(ev_out):
        parts.append(xr.zeros_like(fcast.isel(time=0, drop=True)).expand_dims(time=ev_out))
    return (xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]).astype("float32")


def build_member_forcing(slug, case, index, out_path, start=None):
    """Write precip (member) + temp/pet (base) for the slots start..sim_end.

    `start` defaults to the first slot after the forecast start (warm start);
    an earlier start (cold-start catchments) takes base MSWEP precip for the
    slots up to and including fstart, as the blended cases do.
    """
    _, sim_end, cases = event_cfg(slug)
    cc = cases[case]
    fstart = fstart_of(cc)
    dt = timestep(slug, case)
    times = pd.date_range(start or fstart + dt, sim_end, freq=dt)
    member = (None if index == "mean" else
              _subset_slice(cc) if index == "subset" else int(index))

    base = _read_base(slug, times)
    spatial_ref = base.attrs.pop("_spatial_ref", None)
    print(f"  base temp/pet read ({base.sizes['time']} slots)", flush=True)

    if slug == HANS:
        import wflow_drammen_forcing as wdf
        x, y, lat2d, lon2d, _ = wdf.load_model_grid()
        fc = wdf.sample_to_model(wdf._resample_forecast_3h_latlon(cc, member),
                                 y, x, lat2d, lon2d).clip(min=0.0)
    else:
        fc = wpf.extract_forecast_3h(cc, base["latitude"].values,
                                     base["longitude"].values, member=member)
    print("  forecast regridded", flush=True)
    horizon = pd.DatetimeIndex(fc["time"].values).max()
    if horizon < sim_end and not (wpf.ALLOW_TRUNCATED or wpf.EVENT_CONFIG.get(slug, {}).get("allow_truncated")):
        raise RuntimeError(f"sim_end {sim_end} is beyond the forecast horizon {horizon}")

    precip = _blend_window(fc, times[times > fstart]).transpose(*base["precip"].dims)
    if (times <= fstart).any():
        precip = xr.concat([base["precip"].sel(time=times[times <= fstart]).astype("float32"),
                            precip], dim="time")
    out = xr.Dataset({
        "precip": precip,
        "temp":   base["temp"].sel(time=times).astype("float32"),
        "pet":    base["pet"].sel(time=times).astype("float32"),
    })
    if spatial_ref is not None:
        out["spatial_ref"] = spatial_ref
    out.to_netcdf(out_path, encoding={v: ENC for v in ("precip", "temp", "pet")})
    print("  forcing written", flush=True)
    return times


# ── deterministic forecast = mean of the member hydrographs ─────────────────
def member_mean_output(slug, case):
    """A forecast case's deterministic discharge: the mean over its member
    hydrographs (members_output.nc), as a DataFrame shaped like output.csv.

    The analyses used to score the continuous run forced with the ensemble-mean
    rain; averaging rain first smooths each member's intense bursts away, and
    runoff responds nonlinearly, so that run is not the ensemble's mean
    discharge (user decision 2026-10-09: score the member mean). Rows up to and
    including the forecast start come from that continuous run, which is the
    MSWEP spin-up there and identical to every member's warm-start state (see
    validate-run); rows after it are the member mean. None if not collected.
    Before the forecast start every run equals the MSWEP reference (hist_3h),
    so those rows are read from it: it carries any output gauges added later.
    """
    cc = forecast_cases(slug).get(case)
    nc = os.path.join(ens_dir(slug), case, "members_output.nc")
    csv = os.path.join(model_dir(slug), f"run_{slug}_hist_3h", "output.csv")
    if not os.path.exists(csv):
        csv = os.path.join(model_dir(slug), f"run_{slug}_{case}", "output.csv")
    if cc is None or not os.path.exists(nc) or not os.path.exists(csv):
        return None
    fstart = fstart_of(cc)
    pre = pd.read_csv(csv, parse_dates=["time"], index_col="time")
    with xr.open_dataset(nc) as ds:
        mean = ds.mean("member").to_dataframe()
    mean.index.name = "time"
    cols = [c for c in pre.columns if c in mean.columns]
    out = pd.concat([pre.loc[pre.index <= fstart, cols], mean.loc[mean.index > fstart, cols]])
    return out


# ── commands ──────────────────────────────────────────────────────────────────
def cmd_plan(slugs):
    """One line per case: slug case n_members fstart(YYYYMMDDHH) — for SLURM."""
    for slug in slugs or EVENTS:
        for case, cc in forecast_cases(slug).items():
            n = len(members_of(cc))
            print(f"{slug}\t{case}\t{n}\t{fstart_of(cc).strftime('%Y%m%d%H')}")


def cmd_missing(slugs):
    """One line per case with unfinished work: slug case n_members array-indices
    (member positions, plus n for the ensemble-mean run) -- for resubmission."""
    for slug in slugs or EVENTS:
        _, sim_end, cases = event_cfg(slug)
        for case, cc in forecast_cases(slug).items():
            nums = members_of(cc)
            d = os.path.join(ens_dir(slug), case)
            idx = [i for i, n in enumerate(nums)
                   if not output_complete(os.path.join(d, f"m{n:02d}", "output.csv"), sim_end)]
            if len(nums) > 1 and not output_complete(os.path.join(d, "mean", "output.csv"), sim_end):
                idx.append(len(nums))
            if idx:
                print(f"{slug}\t{case}\t{len(nums)}\t{','.join(map(str, idx))}")


def cmd_inits(slugs):
    """One line per spin-up: slug and the actual forecast init (YYYY-MM-DDTHH)."""
    for slug in slugs or EVENTS:
        for t in sorted({fstart_of(cc) for cc in forecast_cases(slug).values()}):
            print(f"{slug}\t{t.strftime('%Y-%m-%dT%H')}")


def cmd_spinup_status(slug, init_date):
    """Exit 0 if the spin-up for init_date has finished (used by the submit script)."""
    fstart = pd.Timestamp(init_date)
    case = next(c for c, cc in forecast_cases(slug).items() if fstart_of(cc) == fstart)
    done = spinup_done(slug, fstart, timestep(slug, case))
    print("done" if done else "not-done")
    sys.exit(0 if done else 1)


def cmd_spinup(slug, init_date):
    sim_start, _, _ = event_cfg(slug)
    fstart = pd.Timestamp(init_date)
    cases = [c for c, cc in forecast_cases(slug).items() if fstart_of(cc) == fstart]
    if not cases:
        sys.exit(f"No forecast case of {slug} starts at {fstart}")
    dt = timestep(slug, cases[0])
    end = fstart
    d = state_dir(slug, fstart)
    os.makedirs(d, exist_ok=True)
    state = os.path.join(d, "outstates.nc")
    if spinup_done(slug, fstart, dt):
        print(f"[SKIP] spin-up already complete: {state}")
        return
    toml = os.path.join(d, "wflow_spinup.toml")
    write_toml(slug, cases[0], toml, start=sim_start, end=end,
               forcing_glob=os.path.join(model_dir(slug), "inmaps_base", "base_*.nc"),
               cold=True, state_out=state, out_dir=d)
    print(f"Spin-up {slug}: {sim_start} -> {end} (cases {cases})")
    run_wflow(toml)
    open(os.path.join(d, "COMPLETE"), "w").close()
    print(f"[DONE] {state}")


def cmd_member(slug, case, index):
    sim_start, sim_end, cases = event_cfg(slug)
    cc = cases[case]
    fstart = fstart_of(cc)
    dt = timestep(slug, case)
    cold = slug in COLD_START
    state = os.path.join(state_dir(slug, fstart), "outstates.nc")
    if not cold and not spinup_done(slug, fstart, dt):
        sys.exit(f"Spin-up not finished: {state}")
    label = member_label(cc, index)
    d = os.path.join(ens_dir(slug), case, label)
    # WFLOW_FORCE=1 reruns finished members (e.g. after adding output gauges;
    # the spin-up states do not depend on outputs and are reused).
    if output_complete(os.path.join(d, "output.csv"), sim_end) and os.environ.get("WFLOW_FORCE") != "1":
        print(f"[SKIP] complete: {d}/output.csv")
        return
    os.makedirs(d, exist_ok=True)

    scratch = tempfile.mkdtemp(prefix=f"wfe_{slug}_{case}_{label}_",
                               dir=os.environ.get("WFLOW_ENS_SCRATCH") or os.environ.get("TMPDIR") or "/tmp")
    try:
        toml = os.path.join(d, "wflow.toml")
        if cold:
            # Full cold-start run: whole base months before the forecast month are
            # linked unchanged; the forecast month onwards is written with the
            # member overlaid from fstart (MSWEP before it, as in the blend cases).
            mstart = max(fstart.to_period("M").start_time, sim_start)
            for f in _base_files(slug):
                if os.path.basename(f)[5:11] < fstart.strftime("%Y%m"):
                    os.symlink(f, os.path.join(scratch, os.path.basename(f)))
            forcing = os.path.join(scratch, "forecast.nc")
            print(f"{slug} / {case} / {label}: cold start {sim_start}, member from {fstart}", flush=True)
            build_member_forcing(slug, case, index, forcing, start=mstart)
            write_toml(slug, case, toml, start=sim_start, end=sim_end,
                       forcing_glob=os.path.join(scratch, "*.nc"), cold=True,
                       state_out=os.path.join(scratch, "outstates.nc"), out_dir=d)
        else:
            forcing = os.path.join(scratch, "forcing.nc")
            print(f"{slug} / {case} / {label}: forcing {fstart} -> {sim_end} into {scratch}", flush=True)
            build_member_forcing(slug, case, index, forcing)
            write_toml(slug, case, toml, start=fstart, end=sim_end, forcing_glob=forcing,
                       cold=False, state_in=state, state_out=os.path.join(scratch, "outstates.nc"),
                       out_dir=d)
        print(f"  forcing {os.path.getsize(forcing) / 1e6:.1f} MB", flush=True)
        run_wflow(toml)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if not output_complete(os.path.join(d, "output.csv"), sim_end):
        sys.exit(f"output.csv incomplete: {d}")
    print(f"[DONE] {d}/output.csv")


def cmd_collect(slug, case):
    _, sim_end, cases = event_cfg(slug)
    cc = cases[case]
    numbers = members_of(cc)
    frames, missing = {}, []
    for n in numbers:
        csv = os.path.join(ens_dir(slug), case, f"m{n:02d}", "output.csv")
        if output_complete(csv, sim_end):
            frames[n] = pd.read_csv(csv, index_col=0, parse_dates=True)
        else:
            missing.append(n)
    if not frames:
        sys.exit(f"No complete members for {slug}/{case}")
    first = next(iter(frames.values()))
    ds = xr.Dataset(
        {col: (("member", "time"), np.stack([frames[n][col].values for n in frames]))
         for col in first.columns},
        coords={"member": list(frames), "time": first.index.values})
    ds.attrs.update(event=slug, case=case, forecast_start=str(fstart_of(cc)),
                    n_members=len(frames), missing_members=str(missing),
                    source="wflow_ensemble.py: warm start from ensemble/states/")
    out = os.path.join(ens_dir(slug), case, "members_output.nc")
    ds.to_netcdf(out)
    print(f"{slug}/{case}: {len(frames)}/{len(numbers)} members -> {out}"
          + (f"  MISSING {missing}" if missing else ""))
    if missing:
        sys.exit(1)


def cmd_compare_forcing(slug, case):
    """Old inmaps_{case} precip vs the corrected (end-stamped) ensemble mean.

    Reports basin-grid totals over the forecast slots and the lag (h) at which
    the old domain-mean series best matches the new one (negative = old early).
    """
    _, sim_end, cases = event_cfg(slug)
    fstart, dt = fstart_of(cases[case]), timestep(slug, case)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or "/tmp") as tmp:
        p = os.path.join(tmp, "f.nc")
        times = build_member_forcing(slug, case, "mean", p)
        new = xr.open_dataset(p)["precip"].load()
    files = sorted(glob.glob(os.path.join(model_dir(slug), f"inmaps_{case}", "*.nc")))
    old = xr.open_mfdataset(files, combine="by_coords",
                            preprocess=lambda d: d.drop_vars("spatial_ref", errors="ignore"))["precip"]
    sp = [d for d in new.dims if d != "time"]
    n = new.mean(sp).to_series()
    o = old.mean(sp).sel(time=slice(fstart - dt, sim_end)).to_series()
    lags = {}
    for L in range(-6, 7, 3):
        oo = o.copy(); oo.index = oo.index + pd.Timedelta(hours=L)
        j = pd.concat([oo, n], axis=1, join="inner").dropna()
        lags[L] = j.iloc[:, 0].corr(j.iloc[:, 1])
    best = max(lags, key=lags.get)
    tot_o = float(o.loc[o.index > fstart - dt].sum()); tot_n = float(n.sum())
    print(f"COMPARE_FORCING {slug} {case}: total old {tot_o:.2f} new {tot_n:.2f} mm "
          f"({(tot_o - tot_n) / max(tot_n, 1e-9) * 100:+.1f}% old vs new); "
          f"best alignment when old is shifted {best:+d} h (r={lags[best]:.3f}, r at 0 h={lags[0]:.3f})")


def cmd_validate_run(slug, init):
    """Warm restart from the spin-up state vs an uninterrupted run, same forcing.

    Both use the MSWEP base forcing through sim_end, so the only difference is
    the restart at fstart. Self-contained: needs no earlier outputs.
    """
    sim_start, sim_end, _ = event_cfg(slug)
    fstart = pd.Timestamp(init)
    case = next(c for c, cc in forecast_cases(slug).items() if fstart_of(cc) == fstart)
    if not spinup_done(slug, fstart, timestep(slug, case)):
        sys.exit(f"spin-up for {fstart} not finished")
    base = os.path.join(model_dir(slug), "inmaps_base", "base_*.nc")
    root = os.path.join(ens_dir(slug), "_validate", fstart.strftime("%Y%m%d%H"))
    runs = {}
    for name, start, cold, state_in in (
            ("continuous", sim_start, True, None),
            ("restart", fstart, False, os.path.join(state_dir(slug, fstart), "outstates.nc"))):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        toml = os.path.join(d, "wflow.toml")
        write_toml(slug, case, toml, start=start, end=sim_end, forcing_glob=base, cold=cold,
                   state_in=state_in, state_out=os.path.join(d, "outstates.nc"), out_dir=d)
        run_wflow(toml)
        runs[name] = pd.read_csv(os.path.join(d, "output.csv"), index_col=0, parse_dates=True)
    new, ref = runs["restart"], runs["continuous"].reindex(runs["restart"].index)
    rel = ((new - ref).abs().max() / ref.abs().max().replace(0, np.nan)).max()
    print(f"  steps compared: {len(new)}; max relative difference over columns: {rel:.3e}")
    ok = rel < 1e-6
    print("VALIDATE_RUN", "OK" if ok else "MISMATCH", slug, fstart)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").add_argument("slugs", nargs="*")
    sub.add_parser("inits").add_argument("slugs", nargs="*")
    sub.add_parser("missing").add_argument("slugs", nargs="*")
    p = sub.add_parser("spinup"); p.add_argument("slug"); p.add_argument("init_date")
    p = sub.add_parser("spinup-status"); p.add_argument("slug"); p.add_argument("init_date")
    p = sub.add_parser("member"); p.add_argument("slug"); p.add_argument("case"); p.add_argument("index")
    for name in ("collect", "compare-forcing"):
        p = sub.add_parser(name); p.add_argument("slug"); p.add_argument("case")
    p = sub.add_parser("validate-run"); p.add_argument("slug"); p.add_argument("init")
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a.slugs)
    elif a.cmd == "inits":
        cmd_inits(a.slugs)
    elif a.cmd == "missing":
        cmd_missing(a.slugs)
    elif a.cmd == "spinup":
        cmd_spinup(a.slug, a.init_date)
    elif a.cmd == "spinup-status":
        cmd_spinup_status(a.slug, a.init_date)
    elif a.cmd == "member":
        cmd_member(a.slug, a.case, a.index)
    elif a.cmd == "collect":
        cmd_collect(a.slug, a.case)
    elif a.cmd == "compare-forcing":
        cmd_compare_forcing(a.slug, a.case)
    else:
        cmd_validate_run(a.slug, a.init)


if __name__ == "__main__":
    main()
