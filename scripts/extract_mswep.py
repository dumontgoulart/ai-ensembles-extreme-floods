#!/usr/bin/env python3
"""
Extract MSWEP V3.16 NRT/Past 3-hourly precipitation for the active event window.

Event and spatial parameters are read from event_config.py.
Change ACTIVE_EVENT there to switch between configured events.

Reads from the shared MSWEP archive, subsets spatially and temporally,
and saves a single NetCDF to data/raw/{slug}/mswep/.
"""

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from event_config import cfg

import xarray as xr

# =============================================================================
# Configuration (derived from event_config)
# =============================================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "data", "raw", cfg["slug"], "mswep")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"mswep_tp_{cfg['slug']}.nc")

MSWEP_BASE = os.environ.get("MSWEP_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "MSWEP_V316"))
# The 3hourly product (mm/3h, stamped at the START of each 3-h window). Reading
# the Hourly folder at 00/03/../21 instead keeps one hour in three: that is what
# this script did from 2026-04-07 to 2026-10-09 (see mswep_fix_report.py).
MSWEP_DIR  = os.path.join(MSWEP_BASE, cfg["mswep_subdir"], "3hourly")

# Parse ref window dates from config strings (YYYY-MM-DD)
_ref_start = cfg["ref_start"]
_ref_end   = cfg["ref_end"]
EVENT_START = date(int(_ref_start[:4]), int(_ref_start[5:7]), int(_ref_start[8:10]))
EVENT_END   = date(int(_ref_end[:4]),   int(_ref_end[5:7]),   int(_ref_end[8:10]))

LAT_MIN, LAT_MAX = cfg["lat_min"], cfg["lat_max"]
LON_MIN, LON_MAX = cfg["lon_min"], cfg["lon_max"]


def main():
    print("=" * 60)
    print(f"MSWEP V3.16 — Extract {cfg['name']}")
    print(f"  Slug:    {cfg['slug']}")
    print(f"  Source:  {MSWEP_DIR}")
    print(f"  Period:  {EVENT_START} → {EVENT_END}")
    print(f"  Bbox:    lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
    print(f"  Output:  {OUTPUT_FILE}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    want_end = f"{EVENT_END:%Y-%m-%d}T21"
    if os.path.exists(OUTPUT_FILE):
        with xr.open_dataset(OUTPUT_FILE) as old:
            last = str(old.time.values[-1])[:13]
        if last >= want_end:
            print(f"[SKIP] Already exists and covers {EVENT_END}: {OUTPUT_FILE}")
            return
        print(f"[REFETCH] Existing file stops at {last}; re-extracting")

    files = []
    d = EVENT_START
    while d <= EVENT_END:
        doy = d.timetuple().tm_yday
        for hh in range(0, 24, 3):
            fname = os.path.join(MSWEP_DIR, f"{d.year}{doy:03d}.{hh:02d}.nc")
            if os.path.exists(fname):
                files.append(fname)
        d += timedelta(days=1)

    if not files:
        print(f"[ERROR] No MSWEP files found in {MSWEP_DIR}")
        return

    print(f"  Found {len(files)} 3-hourly files")
    print(f"  Subsetting to lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]...")

    ds = xr.open_mfdataset(files, combine="by_coords")

    # MSWEP lat is descending
    ds = ds.sel(
        lat=slice(LAT_MAX, LAT_MIN),
        lon=slice(LON_MIN, LON_MAX),
    )

    print(f"  Saving to {OUTPUT_FILE}...")
    ds = ds.load()
    ds.to_netcdf(OUTPUT_FILE)
    if str(ds.time.values[-1])[:13] < want_end:
        print(f"[WARN] MSWEP stops at {str(ds.time.values[-1])[:13]}, short of {want_end}; re-run later")
    print(f"[DONE] Saved: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
