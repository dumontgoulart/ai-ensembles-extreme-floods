#!/usr/bin/env python3
"""
Apply a nested gauge ladder to an EXISTING wflow model, in place.

Why this script exists
----------------------
The from-scratch models (Russian River, Ivondro) were built with a single gauge
at the basin outlet, so every forecast-verification metric we have is scored at
one large, slow-integrating catchment. To test whether IFS's higher resolution
matters more at smaller, faster-responding scales we need several nested gauges
per basin — but re-running wflow_build_*.py is not an option: it opens the model
with mode="w" and would rebuild everything from MERIT/SoilGrids and overwrite
inmaps.nc. This script instead reopens the model in "r+" and only touches the
gauge map, the derived subcatchments and the output section of the TOML.

What it changes
---------------
  * staticmaps: gauges_<basename> (N cells) + subcatchment_<basename>
  * staticgeoms: gauges_<basename>.geojson + subcatchment_<basename>.geojson
  * wflow_sbm.toml: one [output.csv] column per gauge, plus [output.netcdf_grid]
    (the from-scratch build recipe never wrote that section, so the spatial
    peak-discharge map could only be produced by hand-patching the TOML)

What it deliberately does NOT change: forcing, physics, or gauge id 1. Gauge 1
is asserted to stay on the existing outlet cell so `river_q_1` — and therefore
every previously published metric — is reproduced bit-for-bit.

Usage:
    python scripts/wflow_add_gauges.py <event_slug> [--dry-run] [--no-backup]
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydromt_wflow import WflowSbmModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# basename MUST match the map already in staticmaps, otherwise setup_gauges
# writes a second gauge map and the old one keeps emitting river_q columns.
GAUGE_CONFIG = {
    "california_ar_2023": {"basename": "california"},
    "gezani_2026":        {"basename": "toamasina"},
    "bc_ar_2024":         {"basename": "chilliwack"},
    "zermatt_2024":       {"basename": "vispa"},
    "texas_2025":         {"basename": "texas"},
    "boris_sudetes":      {"basename": "sudetes"},
}

UPAREA_TOL_PCT = 2.0   # CSV uparea_km2 must match the grid within this


def load_gauge_csv(path):
    df = pd.read_csv(path)
    missing = {"id", "x", "y", "name"} - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing required column(s) {sorted(missing)}")
    if df["id"].duplicated().any():
        sys.exit(f"{path}: duplicate gauge ids")
    if 1 not in set(df["id"]):
        sys.exit(f"{path}: id 1 must exist and be the basin outlet")
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:4326")


def preflight(model_dir, gdf, basename):
    """Fail loudly before touching anything if the points are not where we think."""
    sm = xr.open_dataset(os.path.join(model_dir, "staticmaps.nc"))
    try:
        ua, rm = sm["meta_upstream_area"], sm["river_mask"]
        existing = sm.get(f"gauges_{basename}")
        problems = []

        for _, r in gdf.iterrows():
            sel = dict(longitude=r["x"], latitude=r["y"])
            on_river = float(rm.sel(**sel, method="nearest"))
            area = float(ua.sel(**sel, method="nearest"))
            if on_river != 1:
                problems.append(f"  id {r['id']} ({r['name']}): not on a river cell")
            if "uparea_km2" in gdf.columns and np.isfinite(r["uparea_km2"]):
                err = 100 * abs(area - r["uparea_km2"]) / r["uparea_km2"]
                if err > UPAREA_TOL_PCT:
                    problems.append(
                        f"  id {r['id']} ({r['name']}): uparea {area:.1f} km2 differs "
                        f"from CSV {r['uparea_km2']:.1f} km2 by {err:.1f}%")
            print(f"  id {int(r['id'])}: {r['name']:<32} uparea {area:8.1f} km²  river={on_river:.0f}")

        # The whole backward-compatibility guarantee rests on gauge 1 not moving.
        if existing is not None:
            old = existing.where(existing.notnull() & (existing > 0), drop=True)
            if old.sizes.get("latitude") and old.sizes.get("longitude"):
                oy = float(old["latitude"].values[0])
                ox = float(old["longitude"].values[0])
                g1 = gdf[gdf["id"] == 1].iloc[0]
                ny = float(ua.sel(latitude=g1["y"], method="nearest").latitude)
                nx = float(ua.sel(longitude=g1["x"], method="nearest").longitude)
                if not (np.isclose(ox, nx) and np.isclose(oy, ny)):
                    problems.append(
                        f"  id 1 lands on ({nx:.6f},{ny:.6f}) but the existing gauge "
                        f"is at ({ox:.6f},{oy:.6f}) — river_q_1 would change meaning")
                else:
                    print(f"  id 1 matches the existing gauge cell "
                          f"({ox:.6f},{oy:.6f}) — river_q_1 preserved")
        return problems
    finally:
        sm.close()


def backup(model_dir):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = os.path.join(os.path.dirname(model_dir), f"model_backup_{stamp}")
    os.makedirs(dst, exist_ok=True)
    for name in ("staticmaps.nc", "wflow_sbm.toml"):
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, name))
    sg = os.path.join(model_dir, "staticgeoms")
    if os.path.isdir(sg):
        shutil.copytree(sg, os.path.join(dst, "staticgeoms"), dirs_exist_ok=True)
    print(f"Backup written to {dst}")
    return dst


def drop_outlet_csv_row(model):
    """Remove the duplicate `river_q` column emitted by the `outlets` map.

    Gauge id 1 and the single `outlets` cell are the same cell, so both rows
    write a column literally named `river_q_1`. pandas silently mangles the
    second to `river_q_1.1`. Dropping the outlets row leaves exactly one column
    per gauge; the `outlets` map itself stays in staticmaps and [input].
    """
    cols = model.config.get_value("output.csv.column") or []
    keep = [c for c in cols if c.get("map") != "outlets"]
    if len(keep) != len(cols):
        model.config.set("output.csv.column", keep)
        print(f"  dropped {len(cols) - len(keep)} duplicate 'outlets' output.csv row(s)")


def ensure_gridded_output(model):
    """The from-scratch build recipe never wrote [output.netcdf_grid]."""
    if model.config.get_value("output.netcdf_grid.path") is None:
        model.config.set("output.netcdf_grid.path", "output.nc")
        model.config.set("output.netcdf_grid.compressionlevel", 1)
        model.config.set(
            "output.netcdf_grid.variables.river_water__volume_flow_rate", "river_q")
        print("  added [output.netcdf_grid] (river_q) to the base TOML")


def postflight(model_dir, basename, n_expected, backup_dir):
    sm = xr.open_dataset(os.path.join(model_dir, "staticmaps.nc"))
    try:
        gname, sname = f"gauges_{basename}", f"subcatchment_{basename}"
        assert gname in sm, f"{gname} missing from staticmaps"
        g = sm[gname]
        ids = np.unique(g.values[np.isfinite(g.values) & (g.values > 0)]).astype(int)
        print(f"  {gname}: {len(ids)} cells, ids {sorted(ids)}")
        assert len(ids) == n_expected, f"expected {n_expected} gauges, found {len(ids)}"
        assert sname in sm, f"{sname} missing (derive_subcatch failed)"
        nsub = len(np.unique(sm[sname].values[np.isfinite(sm[sname].values)]))
        print(f"  {sname}: {nsub} distinct subcatchment ids")

        # Everything except the two new maps must be untouched.
        old = xr.open_dataset(os.path.join(backup_dir, "staticmaps.nc"))
        try:
            changed = [v for v in old.data_vars
                       if v in sm and not old[v].equals(sm[v])]
            unchanged = [v for v in old.data_vars if v not in changed]
            print(f"  {len(unchanged)}/{len(old.data_vars)} pre-existing variables "
                  f"bit-identical to the backup")
            unexpected = [v for v in changed if v not in (gname, sname)]
            assert not unexpected, f"unexpectedly modified: {unexpected}"
        finally:
            old.close()
    finally:
        sm.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event_slug", help=f"one of {list(GAUGE_CONFIG)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the pre-flight checks and stop")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--allow-move-gauge1", action="store_true",
                    help="new model without results yet: gauge 1 may move")
    args = ap.parse_args()

    if args.event_slug not in GAUGE_CONFIG:
        sys.exit(f"Unknown event '{args.event_slug}'. Options: {list(GAUGE_CONFIG)}")
    basename = GAUGE_CONFIG[args.event_slug]["basename"]

    model_dir = os.path.join(REPO, "data", "wflow", args.event_slug, "model")
    csv_path = os.path.join(REPO, "data", "wflow", args.event_slug, "gauges.csv")
    for p in (model_dir, csv_path):
        if not os.path.exists(p):
            sys.exit(f"Not found: {p}")

    gdf = load_gauge_csv(csv_path)
    print(f"=== {args.event_slug}: {len(gdf)} gauges from {os.path.basename(csv_path)} ===")
    print(f"Model:    {model_dir}")
    print(f"Basename: gauges_{basename}\n")

    print("Pre-flight checks:")
    problems = preflight(model_dir, gdf, basename)
    if args.allow_move_gauge1:
        problems = [p for p in problems if "river_q_1 would change meaning" not in p]
    if problems:
        sys.exit("\nPre-flight FAILED:\n" + "\n".join(problems))
    print("Pre-flight OK.\n")

    if args.dry_run:
        print("--dry-run: stopping before any modification.")
        return

    backup_dir = None if args.no_backup else backup(model_dir)

    print("\nOpening model in r+ …")
    m = WflowSbmModel(root=model_dir, mode="r+", data_libs=["deltares_data"])
    m.setup_gauges(
        gauges_fn=gdf, index_col="id", snap_to_river=True, derive_subcatch=True,
        basename=basename, gauge_toml_header=["river_q"],
        gauge_toml_param=["river_water__volume_flow_rate"],
    )
    drop_outlet_csv_row(m)
    ensure_gridded_output(m)

    print("\nWriting …")
    m.staticmaps.write()
    m.geoms.write(folder="staticgeoms")
    m.config.write()
    # close() is REQUIRED, not cosmetic: staticmaps.nc is open lazily for reading,
    # so hydromt cannot overwrite it in place. It writes a hashed temp file and
    # defers the rename to finish_write(), which only runs via close(). Skip this
    # and you get an orphan staticmaps_<hash>.nc and an unchanged staticmaps.nc.
    m.close()

    if backup_dir:
        print("\nPost-flight checks:")
        postflight(model_dir, basename, len(gdf), backup_dir)

    print(f"\nDone. Next: regenerate the per-case TOMLs and re-run wflow —\n"
          f"  for c in <cases>; do python scripts/wflow_setup.py {args.event_slug} $c; done")


if __name__ == "__main__":
    main()
