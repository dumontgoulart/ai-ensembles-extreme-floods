#!/usr/bin/env python3
"""Clip an existing wflow_sbm model to a sub-basin upstream of an outlet point.

Uses HydroMT-Wflow's network-aware clip (delineates the catchment upstream of
the outlet from the model's own flow-direction grid, then clips all static maps
and geoms to it — routing stays consistent).

Forcing/states are NOT clipped here: the AIFS forcing pipeline rebuilds forcing
from the clipped model grid automatically (wflow_prep_forcing.py).

Usage:
    python scripts/wflow_clip.py \
        --src  data/wflow/kristin_2026/model \
        --dst  data/wflow/kristin_mondego/model \
        --lon -8.430833 --lat 40.204167 --uparea 4929 \
        --config wflow_sbm_run.toml
"""

import argparse
import os
import xarray as xr
from hydromt_wflow import WflowSbmModel

# HydroMT-Wflow's clip drops the full reservoir-map set when the clipped region
# has no reservoir, but errors if some of those names aren't present in the
# source staticmaps (this model carries only a subset). Make drop_vars tolerant
# of missing names — benign, and lets reservoir-free sub-basins clip cleanly.
_orig_drop_vars = xr.Dataset.drop_vars


def _drop_vars_ignore(self, names, *a, **k):
    k.setdefault("errors", "ignore")
    return _orig_drop_vars(self, names, *a, **k)


xr.Dataset.drop_vars = _drop_vars_ignore


def main():
    ap = argparse.ArgumentParser(description="Clip a wflow model to a sub-basin")
    ap.add_argument("--src", required=True, help="source model dir")
    ap.add_argument("--dst", required=True, help="destination (clipped) model dir")
    ap.add_argument("--lon", type=float, required=True, help="outlet longitude")
    ap.add_argument("--lat", type=float, required=True, help="outlet latitude")
    ap.add_argument("--uparea", type=float, default=None,
                    help="snap outlet to cell with ~this upstream area (km2)")
    ap.add_argument("--config", default="wflow_sbm_run.toml",
                    help="config filename within src to read")
    args = ap.parse_args()

    region = {"subbasin": [args.lon, args.lat]}
    if args.uparea is not None:
        region["uparea"] = args.uparea

    print(f"Source : {args.src}")
    print(f"Dest   : {args.dst}")
    print(f"Region : {region}")

    m = WflowSbmModel(root=args.src, config_filename=args.config, mode="r")
    # Read only what clipping needs — avoid loading the large forcing component.
    m.config.read()
    m.staticmaps.read()
    m.geoms.read()
    m.tables.read()

    # Basin delineation (get_basin_geometry) expects an "uparea" variable for
    # outlet snapping; this model stores upstream area as "meta_upstream_area".
    sm = m.staticmaps.data
    if "uparea" not in sm and "meta_upstream_area" in sm:
        m.staticmaps.set(sm["meta_upstream_area"].rename("uparea"), name="uparea")

    m.clip(region=region, clip_forcing=False, clip_states=False)

    # Point at the new root and write the clipped model.
    os.makedirs(args.dst, exist_ok=True)
    m.root.set(args.dst, mode="w")

    m.write(config_filename=args.config)

    # Report size of clipped grid
    import numpy as np
    ds = xr.open_dataset(os.path.join(args.dst, "staticmaps.nc"))
    yd = [d for d in ds.dims if d.lower() in ("y", "lat", "latitude")][0]
    xd = [d for d in ds.dims if d.lower() in ("x", "lon", "longitude")][0]
    sub = [v for v in ds.data_vars if "subcatch" in v.lower()]
    nact = int(np.isfinite(ds[sub[0]]).sum()) if sub else -1
    print(f"\nClipped grid: {ds.sizes[yd]} x {ds.sizes[xd]} = "
          f"{ds.sizes[yd]*ds.sizes[xd]:,} cells | active={nact:,}")

    # Make the clipped model self-contained: subset the parent's daily inmaps
    # (temp/pet/precip source) onto the clipped grid so the forcing pipeline
    # (wflow_prep_forcing.py) can derive the grid + build forcing unchanged.
    import glob
    clip_lat = ds["latitude"]
    clip_lon = ds["longitude"]
    src_inmaps = sorted(glob.glob(os.path.join(args.src, "inmaps",
                                               "mswep_precip_era5_temp_pet_daily_*.nc")))
    dst_inmaps_dir = os.path.join(args.dst, "inmaps")
    os.makedirs(dst_inmaps_dir, exist_ok=True)
    print(f"Subsetting {len(src_inmaps)} daily inmaps onto clipped grid…")
    for f in src_inmaps:
        di = xr.open_dataset(f)
        sub_di = di.reindex(latitude=clip_lat, longitude=clip_lon, method="nearest")
        out = os.path.join(dst_inmaps_dir, os.path.basename(f))
        enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
               for v in sub_di.data_vars if sub_di[v].ndim >= 2}
        sub_di.to_netcdf(out, encoding=enc)
        di.close()
    ds.close()
    print(f"Daily inmaps: {len(src_inmaps)} files → {dst_inmaps_dir}")
    print(f"Done: {args.dst}")


if __name__ == "__main__":
    main()
