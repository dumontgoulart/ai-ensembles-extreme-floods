#!/usr/bin/env python3
"""
Build 3-hourly wflow forcing files for an AIFS project event.

Output: monthly NetCDF files in data/wflow/{slug}/model/inmaps_{case}/
        matching the wflow model grid (coords from existing daily inmaps).

Forcing strategy per case
--------------------------
hist_3h        MSWEP 3-hourly precip (full spinup + event); events with a
               "gauge_grid" switch to that gauge-based grid after its "from"
               time (Catalonia: XEMA grid from Sep 15, MSWEP NRT ends Sep 30)
               ERA5 daily temp/PET disaggregated to 3h

ifs_{mmdd}     MSWEP 3-hourly precip for spinup (Apr 1 – init date)
               IFS hourly/3h/6h precip aggregated to 3h for event window
               ERA5 daily temp/PET disaggregated to 3h (full period)

wn2_{mmdd}     MSWEP 3-hourly precip for spinup
               WeatherNext 6-hourly precip resampled to 3h for event window
               ERA5 daily temp/PET disaggregated to 3h (full period)

aifs_{mmdd}    Same as wn2_{mmdd} but using AIFS ENS (6h native steps)

era5_{mmdd}    MSWEP 3-hourly precip for spinup
               ERA5 hourly precip aggregated to 3h from the given start date
               ERA5 daily temp/PET disaggregated to 3h (full period)

Usage:
    python scripts/wflow_prep_forcing.py dana_2024 hist_3h
    python scripts/wflow_prep_forcing.py dana_2024 ifs_oct25
    python scripts/wflow_prep_forcing.py kristin_2026 ifs_jan22
"""

import sys
import os
import glob
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSWEP_PAST = os.path.join(os.environ.get("MSWEP_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "MSWEP_V316")), "Past", "3hourly")
MSWEP_NRT  = os.path.join(os.environ.get("MSWEP_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "MSWEP_V316")), "NRT", "3hourly")

# Set by --allow-truncated-forecast. When False (default) a sim_end that runs
# past the forecast horizon is a hard error instead of silently zero-filling.
ALLOW_TRUNCATED = False

# Buffer (degrees) added around the wflow model grid when subsetting MSWEP,
# so bilinear regridding always has source points surrounding every target cell.
BBOX_BUFFER_DEG = 1.0


# ── per-event configuration ────────────────────────────────────────────────────
# Each event entry has:
#   sim_start          : simulation start (spinup)
#   sim_end            : simulation end
#   mswep_root         : path to MSWEP 3h directory (Past or NRT)
#   daily_inmaps_glob  : glob relative to model_dir for daily temp/PET files
#   cases              : dict of case configs
#
# NOTE: the MSWEP read bbox is derived dynamically from the actual wflow model
# grid (± BBOX_BUFFER_DEG) at build time — see build_forcing(). This guarantees
# the forcing subset always fully covers the model grid, even for large
# pfafstetter tiles whose corners reach the open Atlantic.

EVENT_CONFIG = {

    "dana_2024": {
        "sim_start":         "2024-04-01 00:00",
        # Trimmed to the shortest horizon across ALL cases. ifs_oct25 reaches
        # 2024-11-04 00Z but the ERA5 file behind era5_oct28 stops one 3 h slot
        # earlier, so 11-03 21Z is the last time every case has real data.
        # Peak is 2024-10-31, well inside.
        "sim_end":           "2024-11-09 00:00",
        "mswep_root":        MSWEP_PAST,
        "daily_inmaps_glob": "inmaps/mswep_precip_era5_temp_pet_daily_2024*.nc",
        "cases": {
            "hist_3h": {
                "precip_mode": "mswep",
                "forcing_dir": "inmaps_hist_3h",
                "basename":    "mswep_era5_3h",
                "label":       "MSWEP 3h + ERA5 (disaggregated)",
            },
            "era5_oct28": {
                "precip_mode": "era5_blend",
                "era5_src":    "data/processed/dana_2024/era5_processed.nc",
                "era5_start":  "2024-10-28",
                "forcing_dir": "inmaps_era5_oct28",
                "basename":    "era5_oct28_mswep_spinup_3h",
                "label":       "ERA5 reanalysis (event) + MSWEP spinup",
            },
            # IFS — long-lead (init Oct 25, ~5 days before event)
            "ifs_oct25": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/dana_2024/ifs_processed.nc",
                "init_date":    "2024-10-25",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_ifs_oct25",
                "basename":     "ifs_oct25_era5_3h",
                "label":        "IFS ENS (init 2024-10-25 00Z) + MSWEP spinup",
            },
            # IFS — near-lead (init Oct 27, ~2 days before event)
            "ifs_oct27": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/dana_2024/ifs_processed.nc",
                "init_date":    "2024-10-27",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_ifs_oct27",
                "basename":     "ifs_oct27_era5_3h",
                "label":        "IFS ENS (init 2024-10-27 00Z) + MSWEP spinup",
            },
            # WeatherNext 2 — long-lead (init Oct 25)
            "wn2_oct25": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/dana_2024/weathernext_processed.nc",
                "init_date":    "2024-10-25",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_wn2_oct25",
                "basename":     "wn2_oct25_era5_3h",
                "label":        "WeatherNext 2 (init 2024-10-25 00Z) + MSWEP spinup",
            },
            # WeatherNext 2 — near-lead (init Oct 27)
            "wn2_oct27": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/dana_2024/weathernext_processed.nc",
                "init_date":    "2024-10-27",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_wn2_oct27",
                "basename":     "wn2_oct27_era5_3h",
                "label":        "WeatherNext 2 (init 2024-10-27 00Z) + MSWEP spinup",
            },
        },
    },

    "kristin_2026": {
        "sim_start":         "2025-07-01 00:00",
        "sim_end":           "2026-02-05 21:00",
        "mswep_root":        MSWEP_NRT,
        # 2025 files only (Jan-Dec 2025 available); 2026 months fall back to
        # nearest Dec 2025 for temp/PET — acceptable for flood-dominated runs.
        "daily_inmaps_glob": "inmaps/mswep_precip_era5_temp_pet_daily_2025*.nc",
        "cases": {
            "hist_3h": {
                "precip_mode": "mswep",
                "forcing_dir": "inmaps_hist_3h",
                "basename":    "mswep_era5_3h",
                "label":       "MSWEP NRT 3h + ERA5 (disaggregated)",
            },
            # ERA5 reanalysis — event from Jan 22 (aligns with long-lead init)
            "era5_jan22": {
                "precip_mode": "era5_blend",
                "era5_src":    "data/processed/kristin_2026/era5_wflow.nc",   # extended back to era5_start (extend_era5_for_wflow.py)
                "era5_start":  "2026-01-22",
                "forcing_dir": "inmaps_era5_jan22",
                "basename":    "era5_jan22_mswep_spinup_3h",
                "label":       "ERA5 reanalysis (event Jan 22+) + MSWEP spinup",
            },
            # IFS ENS — long-lead (init Jan 22, ~5 days before Jan 27 event)
            "ifs_jan22": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/kristin_2026/ifs_processed.nc",
                "init_date":    "2026-01-22",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_ifs_jan22",
                "basename":     "ifs_jan22_era5_3h",
                "label":        "IFS ENS (init 2026-01-22 00Z) + MSWEP spinup",
            },
            # IFS ENS — near-lead (init Jan 25, ~2 days before Jan 27 event)
            "ifs_jan25": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/kristin_2026/ifs_processed.nc",
                "init_date":    "2026-01-25",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_ifs_jan25",
                "basename":     "ifs_jan25_era5_3h",
                "label":        "IFS ENS (init 2026-01-25 00Z) + MSWEP spinup",
            },
            # AIFS ENS — long-lead (init Jan 22); uniform 6h steps
            "aifs_jan22": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/kristin_2026/aifs_processed.nc",
                "init_date":    "2026-01-22",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_aifs_jan22",
                "basename":     "aifs_jan22_era5_3h",
                "label":        "AIFS ENS (init 2026-01-22 00Z) + MSWEP spinup",
            },
            # AIFS ENS — near-lead (init Jan 25)
            "aifs_jan25": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/kristin_2026/aifs_processed.nc",
                "init_date":    "2026-01-25",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_aifs_jan25",
                "basename":     "aifs_jan25_era5_3h",
                "label":        "AIFS ENS (init 2026-01-25 00Z) + MSWEP spinup",
            },
            # WeatherNext 2 — long-lead (init Jan 22)
            "wn2_jan22": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/kristin_2026/weathernext_processed.nc",
                "init_date":    "2026-01-22",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_wn2_jan22",
                "basename":     "wn2_jan22_era5_3h",
                "label":        "WeatherNext 2 (init 2026-01-22 00Z) + MSWEP spinup",
            },
            # WeatherNext 3 — long-lead (init Jan 22). Statistics product: the
            # single "number" entry is the 64-member ensemble mean, so the usual
            # mean over member_dim is a no-op returning the right field.
            "wn3_jan22": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/kristin_2026/weathernext3_processed.nc",
                "init_date":    "2026-01-22",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_wn3_jan22",
                "basename":     "wn3_jan22_era5_3h",
                "label":        "WeatherNext 3 (init 2026-01-22 00Z) + MSWEP spinup",
            },
            # WeatherNext 3 — near-lead (init Jan 25)
            "wn3_jan25": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/kristin_2026/weathernext3_processed.nc",
                "init_date":    "2026-01-25",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_wn3_jan25",
                "basename":     "wn3_jan25_era5_3h",
                "label":        "WeatherNext 3 (init 2026-01-25 00Z) + MSWEP spinup",
            },
            # WeatherNext 2 — near-lead (init Jan 25)
            "wn2_jan25": {
                "precip_mode":  "blend",
                "forecast_src": "data/processed/kristin_2026/weathernext_processed.nc",
                "init_date":    "2026-01-25",
                "member_dim":   "number",
                "forecast_var": "tp",
                "forcing_dir":  "inmaps_wn2_jan25",
                "basename":     "wn2_jan25_era5_3h",
                "label":        "WeatherNext 2 (init 2026-01-25 00Z) + MSWEP spinup",
            },
        },
    },

    "texas_2025": {
        "sim_start":         "2025-01-01 00:00",
        # Shortest horizon: ifs_jun27 ends 2025-07-07 00Z. Peak is 2025-07-04.
        "sim_end":           "2025-07-12 00:00",
        "mswep_root":        MSWEP_NRT,
        # Reference rain (extreme-gauge framework, 2026-10-14): MSWEP spin-up, then
        # MRMS MultiSensor QPE Pass2 (hourly, end-stamped) from Jun 25, before the
        # first forecast start, so spin-up states and members use it too. Since 21 Oct
        # scaled to GHCN-D daily gauges (mrms_gaugescaled.py; MRMS ~10 % low); unscaled
        # results in model/_archive_mrms_raw/.
        "gauge_grid":        {"path": "data/raw/texas_2025/obs/mrms_qpe_1h_gaugescaled.nc",
                              "from": "2025-06-25 00:00"},
        # ERA5-Land daily temp/PET inmaps built by wflow_build_meteo.py
        "daily_inmaps_glob": "inmaps/era5land_temp_pet_daily_2025*.nc",
        "cases": {
            "hist_3h": {
                "precip_mode": "mswep",
                "forcing_dir": "inmaps_hist_3h",
                "basename":    "mswep_era5land_3h",
                "label":       "MSWEP NRT 3h + ERA5-Land (disaggregated)",
            },
            "era5_jul03": {
                "precip_mode": "era5_blend",
                "era5_src":    "data/processed/texas_2025/era5_processed.nc",
                "era5_start":  "2025-07-03",
                "forcing_dir": "inmaps_era5_jul03",
                "basename":    "era5_jul03_mswep_spinup_3h",
                "label":       "ERA5 reanalysis (event Jul 3+) + MSWEP spinup",
            },
            # IFS ENS — long lead (init Jun 27, ~7 days) and near lead (Jul 2, ~2 days)
            "ifs_jun27": {
                "precip_mode": "blend", "forecast_src": "data/processed/texas_2025/ifs_processed.nc",
                "init_date": "2025-06-27", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_jun27", "basename": "ifs_jun27_era5land_3h",
                "label": "IFS ENS (init 2025-06-27 00Z, 7-day lead) + MSWEP spinup",
            },
            "ifs_jul02": {
                "precip_mode": "blend", "forecast_src": "data/processed/texas_2025/ifs_processed.nc",
                "init_date": "2025-07-02", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_jul02", "basename": "ifs_jul02_era5land_3h",
                "label": "IFS ENS (init 2025-07-02 00Z, 2-day lead) + MSWEP spinup",
            },
            # AIFS ENS — first archived init is Jul 1 12Z, so leads are ~3-day / ~2-day
            "aifs_jul01": {
                "precip_mode": "blend", "forecast_src": "data/processed/texas_2025/aifs_processed.nc",
                "init_date": "2025-07-01", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_aifs_jul01", "basename": "aifs_jul01_era5land_3h",
                "label": "AIFS ENS (init 2025-07-01 12Z, ~3-day lead) + MSWEP spinup",
            },
            "aifs_jul02": {
                "precip_mode": "blend", "forecast_src": "data/processed/texas_2025/aifs_processed.nc",
                "init_date": "2025-07-02", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_aifs_jul02", "basename": "aifs_jul02_era5land_3h",
                "label": "AIFS ENS (init 2025-07-02 00Z, 2-day lead) + MSWEP spinup",
            },
            # WeatherNext 2 — long lead (Jun 27) and near lead (Jul 2)
            "wn2_jun27": {
                "precip_mode": "blend", "forecast_src": "data/processed/texas_2025/weathernext_processed.nc",
                "init_date": "2025-06-27", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_jun27", "basename": "wn2_jun27_era5land_3h",
                "label": "WeatherNext 2 (init 2025-06-27 00Z, 7-day lead) + MSWEP spinup",
            },
            "wn2_jul02": {
                "precip_mode": "blend", "forecast_src": "data/processed/texas_2025/weathernext_processed.nc",
                "init_date": "2025-07-02", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_jul02", "basename": "wn2_jul02_era5land_3h",
                "label": "WeatherNext 2 (init 2025-07-02 00Z, 2-day lead) + MSWEP spinup",
            },
        },
    },

    "boris_2024": {
        "sim_start":         "2024-04-01 00:00",
        # Shortest horizon after the 360 h IFS re-pull: ifs_sep05 / wn2_sep05 both
        # end 2024-09-20 00Z. At the old 240 h this was 09-15, i.e. BEFORE the
        # Elbe peak (Děčín crests 09-17 18Z) — which is why boris was re-pulled
        # rather than trimmed.
        "sim_end":           "2024-09-20 00:00",
        "mswep_root":        MSWEP_PAST,   # Sep 2024 is in the historical archive
        # ERA5-Land daily temp/PET inmaps built by wflow_build_meteo.py
        "daily_inmaps_glob": "inmaps/era5land_temp_pet_daily_2024*.nc",
        "cases": {
            "hist_3h": {
                "precip_mode": "mswep",
                "forcing_dir": "inmaps_hist_3h",
                "basename":    "mswep_era5land_3h",
                "label":       "MSWEP 3h + ERA5-Land (disaggregated)",
            },
            "era5_sep11": {
                "precip_mode": "era5_blend",
                "era5_src":    "data/processed/boris_2024/era5_processed.nc",
                "era5_start":  "2024-09-11",
                "forcing_dir": "inmaps_era5_sep11",
                "basename":    "era5_sep11_mswep_spinup_3h",
                "label":       "ERA5 reanalysis (event Sep 11+) + MSWEP spinup",
            },
            # IFS ENS — long lead (init Sep 5, ~7 days) and near lead (Sep 10, ~2 days)
            "ifs_sep05": {
                "precip_mode": "blend", "forecast_src": "data/processed/boris_2024/ifs_processed.nc",
                "init_date": "2024-09-05", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_sep05", "basename": "ifs_sep05_era5land_3h",
                "label": "IFS ENS (init 2024-09-05 00Z, 7-day lead) + MSWEP spinup",
            },
            "ifs_sep10": {
                "precip_mode": "blend", "forecast_src": "data/processed/boris_2024/ifs_processed.nc",
                "init_date": "2024-09-10", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_sep10", "basename": "ifs_sep10_era5land_3h",
                "label": "IFS ENS (init 2024-09-10 00Z, 2-day lead) + MSWEP spinup",
            },
            # WeatherNext 2 — long lead (Sep 5) and near lead (Sep 10)
            "wn2_sep05": {
                "precip_mode": "blend", "forecast_src": "data/processed/boris_2024/weathernext_processed.nc",
                "init_date": "2024-09-05", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_sep05", "basename": "wn2_sep05_era5land_3h",
                "label": "WeatherNext 2 (init 2024-09-05 00Z, 7-day lead) + MSWEP spinup",
            },
            "wn2_sep10": {
                "precip_mode": "blend", "forecast_src": "data/processed/boris_2024/weathernext_processed.nc",
                "init_date": "2024-09-10", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_sep10", "basename": "wn2_sep10_era5land_3h",
                "label": "WeatherNext 2 (init 2024-09-10 00Z, 2-day lead) + MSWEP spinup",
            },
        },
    },

    "gezani_2026": {
        "sim_start":         "2025-10-01 00:00",
        # Shortest horizon: ifs_feb03 / aifs_feb03 end 2026-02-13 00Z. Peak is
        # 2026-02-11 06Z. (wn2_feb03 reaches 02-18; trimming to the common horizon
        # removes WN2's horizon advantage from the head-to-head.)
        "sim_end":           "2026-02-18 00:00",
        "mswep_root":        MSWEP_NRT,
        "daily_inmaps_glob": "inmaps/era5land_temp_pet_daily_202*.nc",
        "cases": {
            "hist_3h": {
                "precip_mode": "mswep",
                "forcing_dir": "inmaps_hist_3h",
                "basename":    "mswep_era5land_3h",
                "label":       "MSWEP NRT 3h + ERA5-Land (disaggregated)",
            },
            "era5_feb09": {
                "precip_mode": "era5_blend",
                "era5_src":    "data/processed/gezani_2026/era5_processed.nc",
                "era5_start":  "2026-02-09",
                "forcing_dir": "inmaps_era5_feb09",
                "basename":    "era5_feb09_mswep_spinup_3h",
                "label":       "ERA5 reanalysis (event Feb 9+) + MSWEP spinup",
            },
            "ifs_feb03": {
                "precip_mode": "blend", "forecast_src": "data/processed/gezani_2026/ifs_processed.nc",
                "init_date": "2026-02-03", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_feb03", "basename": "ifs_feb03_era5land_3h",
                "label": "IFS ENS (init 2026-02-03 00Z, 7-day lead) + MSWEP spinup",
            },
            "ifs_feb08": {
                "precip_mode": "blend", "forecast_src": "data/processed/gezani_2026/ifs_processed.nc",
                "init_date": "2026-02-08", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_feb08", "basename": "ifs_feb08_era5land_3h",
                "label": "IFS ENS (init 2026-02-08 00Z, 2-day lead) + MSWEP spinup",
            },
            "aifs_feb03": {
                "precip_mode": "blend", "forecast_src": "data/processed/gezani_2026/aifs_processed.nc",
                "init_date": "2026-02-03", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_aifs_feb03", "basename": "aifs_feb03_era5land_3h",
                "label": "AIFS ENS (init 2026-02-03 00Z, 7-day lead) + MSWEP spinup",
            },
            "aifs_feb08": {
                "precip_mode": "blend", "forecast_src": "data/processed/gezani_2026/aifs_processed.nc",
                "init_date": "2026-02-08", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_aifs_feb08", "basename": "aifs_feb08_era5land_3h",
                "label": "AIFS ENS (init 2026-02-08 00Z, 2-day lead) + MSWEP spinup",
            },
            "wn2_feb03": {
                "precip_mode": "blend", "forecast_src": "data/processed/gezani_2026/weathernext_processed.nc",
                "init_date": "2026-02-03", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_feb03", "basename": "wn2_feb03_era5land_3h",
                "label": "WeatherNext 2 (init 2026-02-03 00Z, 7-day lead) + MSWEP spinup",
            },
            "wn2_feb08": {
                "precip_mode": "blend", "forecast_src": "data/processed/gezani_2026/weathernext_processed.nc",
                "init_date": "2026-02-08", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_feb08", "basename": "wn2_feb08_era5land_3h",
                "label": "WeatherNext 2 (init 2026-02-08 00Z, 2-day lead) + MSWEP spinup",
            },
            # WeatherNext 3 (statistics product: the single "number" entry is the
            # 64-member ensemble mean, so averaging over member_dim is a no-op).
            "wn3_feb03": {
                "precip_mode": "blend", "forecast_src": "data/processed/gezani_2026/weathernext3_processed.nc",
                "init_date": "2026-02-03", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn3_feb03", "basename": "wn3_feb03_era5land_3h",
                "label": "WeatherNext 3 (init 2026-02-03 00Z, 7-day lead) + MSWEP spinup",
            },
            "wn3_feb08": {
                "precip_mode": "blend", "forecast_src": "data/processed/gezani_2026/weathernext3_processed.nc",
                "init_date": "2026-02-08", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn3_feb08", "basename": "wn3_feb08_era5land_3h",
                "label": "WeatherNext 3 (init 2026-02-08 00Z, 2-day lead) + MSWEP spinup",
            },
        },
    },

    "california_ar_2023": {
        "sim_start":         "2022-10-01 00:00",
        # Shortest horizon: ifs_dec28 / wn2_dec28 both end 2023-01-12 00Z (15-day
        # limit from the Dec 28 init). Peak is 2023-01-08 21Z.
        "sim_end":           "2023-01-12 00:00",
        "mswep_root":        MSWEP_PAST,
        "daily_inmaps_glob": "inmaps/era5land_temp_pet_daily_202*.nc",
        "cases": {
            "hist_3h": {
                "precip_mode": "mswep",
                "forcing_dir": "inmaps_hist_3h",
                "basename":    "mswep_era5land_3h",
                "label":       "MSWEP 3h + ERA5 (disaggregated)",
            },
            "era5_jan03": {
                "precip_mode": "era5_blend",
                "era5_src":    "data/processed/california_ar_2023/era5_processed.nc",
                "era5_start":  "2023-01-03",
                "forcing_dir": "inmaps_era5_jan03",
                "basename":    "era5_jan03_mswep_spinup_3h",
                "label":       "ERA5 reanalysis (event Jan 3+) + MSWEP spinup",
            },
            "ifs_dec28": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_ar_2023/ifs_processed.nc",
                "init_date": "2022-12-28", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_dec28", "basename": "ifs_dec28_era5land_3h",
                "label": "IFS ENS (init 2022-12-28 00Z, 7-day lead) + MSWEP spinup",
            },
            "ifs_jan02": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_ar_2023/ifs_processed.nc",
                "init_date": "2023-01-02", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_jan02", "basename": "ifs_jan02_era5land_3h",
                "label": "IFS ENS (init 2023-01-02 00Z, 2-day lead) + MSWEP spinup",
            },
            "wn2_dec28": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_ar_2023/weathernext_processed.nc",
                "init_date": "2022-12-28", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_dec28", "basename": "wn2_dec28_era5land_3h",
                "label": "WeatherNext 2 (init 2022-12-28 00Z, 7-day lead) + MSWEP spinup",
            },
            "wn2_jan02": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_ar_2023/weathernext_processed.nc",
                "init_date": "2023-01-02", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_jan02", "basename": "wn2_jan02_era5land_3h",
                "label": "WeatherNext 2 (init 2023-01-02 00Z, 2-day lead) + MSWEP spinup",
            },
        },
    },

    "bc_ar_2024": {
        "sim_start":         "2023-10-01 00:00",
        "sim_end":           "2024-10-26 00:00",   # ifs_oct11 + 360 h
        "mswep_root":        MSWEP_PAST,
        "daily_inmaps_glob": "inmaps/era5land_temp_pet_daily_202*.nc",
        "cases": {
            "hist_3h": {
                "precip_mode": "mswep",
                "forcing_dir": "inmaps_hist_3h",
                "basename":    "mswep_era5_3h",
                "label":       "MSWEP 3h + ERA5 (disaggregated)",
            },
            "era5_oct16": {
                "precip_mode": "era5_blend",
                "era5_src":    "data/processed/bc_ar_2024/era5_wflow.nc",   # extended back to era5_start (extend_era5_for_wflow.py)
                "era5_start":  "2024-10-16",
                "forcing_dir": "inmaps_era5_oct16",
                "basename":    "era5_oct16_mswep_spinup_3h",
                "label":       "ERA5 reanalysis (event Oct 16+) + MSWEP spinup",
            },
            "ifs_oct11": {
                "precip_mode": "blend", "forecast_src": "data/processed/bc_ar_2024/ifs_processed.nc",
                "init_date": "2024-10-11", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_oct11", "basename": "ifs_oct11_era5_3h",
                "label": "IFS ENS (init 2024-10-11 00Z, 7-day lead) + MSWEP spinup",
            },
            "ifs_oct16": {
                "precip_mode": "blend", "forecast_src": "data/processed/bc_ar_2024/ifs_processed.nc",
                "init_date": "2024-10-16", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_oct16", "basename": "ifs_oct16_era5_3h",
                "label": "IFS ENS (init 2024-10-16 00Z, 2-day lead) + MSWEP spinup",
            },
            "wn2_oct11": {
                "precip_mode": "blend", "forecast_src": "data/processed/bc_ar_2024/weathernext_processed.nc",
                "init_date": "2024-10-11", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_oct11", "basename": "wn2_oct11_era5_3h",
                "label": "WeatherNext 2 (init 2024-10-11 00Z, 7-day lead) + MSWEP spinup",
            },
            "wn2_oct16": {
                "precip_mode": "blend", "forecast_src": "data/processed/bc_ar_2024/weathernext_processed.nc",
                "init_date": "2024-10-16", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_oct16", "basename": "wn2_oct16_era5_3h",
                "label": "WeatherNext 2 (init 2024-10-16 00Z, 2-day lead) + MSWEP spinup",
            },
        },
    },

    "zermatt_2024": {
        "sim_start":         "2022-10-01 00:00",
        "sim_end":           "2024-07-03 00:00",   # 3 d after the 30 Jun record peak at Visp; ifs_jun15 (ends 30 Jun) is zero-filled after
        "allow_truncated":   True,
        "mswep_root":        MSWEP_PAST,
        # Reference rain (user, 2026-10-16): the MeteoSwiss station grid rescaled to
        # RhiresD daily totals (scripts/rhiresd_rescale.py; the station grid alone
        # is 40 % too dry, MSWEP 79 %) from Jun 10, MSWEP before.
        "gauge_grid":        {"path": "data/raw/zermatt_2024/obs/gauge_grid_hourly_rhiresd.nc",
                              "from": "2024-06-10 06:00"},
        "daily_inmaps_glob": "inmaps/era5land_temp_pet_daily_202*.nc",
        "cases": {
            "hist_3h": {
                "precip_mode": "mswep",
                "forcing_dir": "inmaps_hist_3h",
                "basename":    "mswep_era5_3h",
                "label":       "RhiresD-scaled station grid (MSWEP spinup) + ERA5 (disaggregated)",
            },
            "era5_jun19": {
                "precip_mode": "era5_blend",
                "era5_src":    "data/processed/zermatt_2024/era5_processed.nc",
                "era5_start":  "2024-06-19",
                "forcing_dir": "inmaps_era5_jun19",
                "basename":    "era5_jun19_mswep_spinup_3h",
                "label":       "ERA5 reanalysis (event Jun 19+) + MSWEP spinup",
            },
            "ifs_jun15": {
                "precip_mode": "blend", "forecast_src": "data/processed/zermatt_2024/ifs_processed.nc",
                "init_date": "2024-06-15", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_jun15", "basename": "ifs_jun15_era5_3h",
                "label": "IFS ENS (init 2024-06-15 00Z, 15-day lead) + MSWEP spinup",
            },
            "ifs_jun19": {
                "precip_mode": "blend", "forecast_src": "data/processed/zermatt_2024/ifs_processed.nc",
                "init_date": "2024-06-19", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_jun19", "basename": "ifs_jun19_era5_3h",
                "label": "IFS ENS (init 2024-06-19 00Z, 2-day lead) + MSWEP spinup",
            },
            "wn2_jun15": {
                "precip_mode": "blend", "forecast_src": "data/processed/zermatt_2024/weathernext_processed.nc",
                "init_date": "2024-06-15", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_jun15", "basename": "wn2_jun15_era5_3h",
                "label": "WeatherNext 2 (init 2024-06-15 00Z, 15-day lead) + MSWEP spinup",
            },
            "wn2_jun19": {
                "precip_mode": "blend", "forecast_src": "data/processed/zermatt_2024/weathernext_processed.nc",
                "init_date": "2024-06-19", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_jun19", "basename": "wn2_jun19_era5_3h",
                "label": "WeatherNext 2 (init 2024-06-19 00Z, 2-day lead) + MSWEP spinup",
            },
        },
    },

}

EVENT_CONFIG["catalonia_2026"] = {
    "sim_start":         "2026-04-01 00:00",
    # Rain 3 Oct ~10 UTC -> 4 Oct evening; small basins recede within days.
    # Set by the reference: XEMA has no data network-wide on Oct 8 00:00-01:30
    # and 05:30-10:00 UTC (gap at the source), when a second, minor rain began.
    # 3.5 days of recession after the Oct 4 peak; ERA5 reaches Oct 8 23 UTC and
    # every forecast at least Oct 12.
    "sim_end":           "2026-10-07 21:00",
    "mswep_root":        MSWEP_NRT,
    # Reference rain: MSWEP NRT (ends Sep 30 03 UTC) up to Sep 15, then the
    # XEMA gauge grid (scripts/xema_grid.py; leave-one-out bias -1 %, ERA5 -60 %).
    "gauge_grid":        {"path": "data/raw/catalonia_2026/obs/xema_grid_hourly.nc",
                          "from": "2026-09-15 00:00"},
    # ERA5 daily temp/PET (EDH to Sep 30, CDS ERA5T after; wflow_build_meteo.py)
    "daily_inmaps_glob": "inmaps/era5_temp_pet_daily_2026*.nc",
    "cases": {
        "hist_3h": {
            "precip_mode": "mswep",
            "forcing_dir": "inmaps_hist_3h",
            "basename":    "xema_era5_3h",
            "label":       "XEMA gauge grid (MSWEP NRT spinup) + ERA5 temp/PET",
        },
        "era5_oct01": {
            "precip_mode": "era5_blend",
            "era5_src":    "data/processed/catalonia_2026/era5_processed.nc",
            "era5_start":  "2026-10-01",
            "forcing_dir": "inmaps_era5_oct01",
            "basename":    "era5_oct01_spinup_3h",
            "label":       "ERA5 reanalysis (Oct 1+) + MSWEP/XEMA spinup",
        },
        # long lead (Sep 26, 7 days before event_start) and near lead (Oct 1, 2 days)
            "ifs_sep26": {
                "precip_mode": "blend", "forecast_src": "data/processed/catalonia_2026/ifs_processed.nc",
                "init_date": "2026-09-26", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_sep26", "basename": "ifs_sep26_era5_3h",
                "label": "IFS ENS (init 2026-09-26 00Z, 7-day lead) + MSWEP/XEMA spinup",
            },
            "ifs_oct01": {
                "precip_mode": "blend", "forecast_src": "data/processed/catalonia_2026/ifs_processed.nc",
                "init_date": "2026-10-01", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_oct01", "basename": "ifs_oct01_era5_3h",
                "label": "IFS ENS (init 2026-10-01 00Z, 2-day lead) + MSWEP/XEMA spinup",
            },
            "aifs_sep26": {
                "precip_mode": "blend", "forecast_src": "data/processed/catalonia_2026/aifs_processed.nc",
                "init_date": "2026-09-26", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_aifs_sep26", "basename": "aifs_sep26_era5_3h",
                "label": "AIFS ENS (init 2026-09-26 00Z, 7-day lead) + MSWEP/XEMA spinup",
            },
            "aifs_oct01": {
                "precip_mode": "blend", "forecast_src": "data/processed/catalonia_2026/aifs_processed.nc",
                "init_date": "2026-10-01", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_aifs_oct01", "basename": "aifs_oct01_era5_3h",
                "label": "AIFS ENS (init 2026-10-01 00Z, 2-day lead) + MSWEP/XEMA spinup",
            },
            "wn2_sep26": {
                "precip_mode": "blend", "forecast_src": "data/processed/catalonia_2026/weathernext_processed.nc",
                "init_date": "2026-09-26", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_sep26", "basename": "wn2_sep26_era5_3h",
                "label": "WeatherNext 2 (init 2026-09-26 00Z, 7-day lead) + MSWEP/XEMA spinup",
            },
            "wn2_oct01": {
                "precip_mode": "blend", "forecast_src": "data/processed/catalonia_2026/weathernext_processed.nc",
                "init_date": "2026-10-01", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_oct01", "basename": "wn2_oct01_era5_3h",
                "label": "WeatherNext 2 (init 2026-10-01 00Z, 2-day lead) + MSWEP/XEMA spinup",
            },
    },
}

EVENT_CONFIG["california_2023"] = {
    # Arroyo Seco (record, peak 2023-01-09 18:30 UTC) + Cosumnes (3rd of 119,
    # 2022-12-31 17:45) + San Lorenzo + Soquel. Spin-up from 1 Jul 2022 since 18 Nov 2026
    # (was the water year, 1 Oct: 2.8 months; the spin-up test, scripts/spinup_test.py,
    # gave 25-40 % more Nov-Dec baseflow and 2-7 % higher flood peaks from 1 Oct).
    "sim_start":         "2022-07-01 00:00",
    "sim_end":           "2023-01-14 00:00",   # Arroyo Seco recession
    "mswep_root":        MSWEP_PAST,
    # Reference rain: MRMS MultiSensor QPE Pass2 (hourly, end-stamped) from Dec 20,
    # before the first forecast start, scaled to independent GHCN-D daily gauges per
    # flood period (21 Oct 2026; california_mrms_gaugescaled.py: MRMS was 17-30 % low
    # in the coastal ranges). The unscaled-MRMS results are in model/_archive_mrms_raw/.
    "gauge_grid":        {"path": "data/raw/california_2023/obs/mrms_qpe_1h_gaugescaled.nc",
                          "from": "2022-12-20 00:00"},
    # Two floods 9 days apart, one shared sim_end: the Dec 24 / Dec 29 forecasts
    # (for Cosumnes) end before Jan 14 and are zero-filled after their horizon;
    # Cosumnes is scored over its own window, which ends well before.
    "allow_truncated":   True,
    "daily_inmaps_glob": "inmaps/era5_temp_pet_daily_202*.nc",
    "cases": {
        "hist_3h": {
            "precip_mode": "mswep",
            "forcing_dir": "inmaps_hist_3h",
            "basename":    "mrms_era5_3h",
            "label":       "MRMS (MSWEP spinup) + ERA5 temp/PET",
        },
        "era5_dec24": {
            "precip_mode": "era5_blend",
            "era5_src":    "data/processed/california_2023/era5_processed.nc",
            "era5_start":  "2022-12-24",
            "forcing_dir": "inmaps_era5_dec24",
            "basename":    "era5_dec24_spinup_3h",
            "label":       "ERA5 reanalysis (Dec 24+) + MSWEP/MRMS spinup",
        },
        # ~7 and ~2 days before each flood peak (user decision 2026-10-15); no AIFS for 2022-23
            "ifs_dec24": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_2023/ifs_processed.nc",
                "init_date": "2022-12-24", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_dec24", "basename": "ifs_dec24_era5_3h",
                "label": "IFS ENS (init 2022-12-24 00Z, Cosumnes -7 d) + MSWEP/MRMS spinup",
            },
            "ifs_dec29": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_2023/ifs_processed.nc",
                "init_date": "2022-12-29", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_dec29", "basename": "ifs_dec29_era5_3h",
                "label": "IFS ENS (init 2022-12-29 00Z, Cosumnes -2 d) + MSWEP/MRMS spinup",
            },
            "ifs_jan02": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_2023/ifs_processed.nc",
                "init_date": "2023-01-02", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_jan02", "basename": "ifs_jan02_era5_3h",
                "label": "IFS ENS (init 2023-01-02 00Z, Arroyo Seco -7 d) + MSWEP/MRMS spinup",
            },
            "ifs_jan07": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_2023/ifs_processed.nc",
                "init_date": "2023-01-07", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_jan07", "basename": "ifs_jan07_era5_3h",
                "label": "IFS ENS (init 2023-01-07 00Z, Arroyo Seco -2 d) + MSWEP/MRMS spinup",
            },
            "wn2_dec24": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_2023/weathernext_processed.nc",
                "init_date": "2022-12-24", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_dec24", "basename": "wn2_dec24_era5_3h",
                "label": "WeatherNext 2 (init 2022-12-24 00Z, Cosumnes -7 d) + MSWEP/MRMS spinup",
            },
            "wn2_dec29": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_2023/weathernext_processed.nc",
                "init_date": "2022-12-29", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_dec29", "basename": "wn2_dec29_era5_3h",
                "label": "WeatherNext 2 (init 2022-12-29 00Z, Cosumnes -2 d) + MSWEP/MRMS spinup",
            },
            "wn2_jan02": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_2023/weathernext_processed.nc",
                "init_date": "2023-01-02", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_jan02", "basename": "wn2_jan02_era5_3h",
                "label": "WeatherNext 2 (init 2023-01-02 00Z, Arroyo Seco -7 d) + MSWEP/MRMS spinup",
            },
            "wn2_jan07": {
                "precip_mode": "blend", "forecast_src": "data/processed/california_2023/weathernext_processed.nc",
                "init_date": "2023-01-07", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_jan07", "basename": "wn2_jan07_era5_3h",
                "label": "WeatherNext 2 (init 2023-01-07 00Z, Arroyo Seco -2 d) + MSWEP/MRMS spinup",
            },
    },
}

EVENT_CONFIG["helene_2024"] = {
    # French Broad at Asheville (record of 130 yr, peak 2024-09-27 21:38 UTC; Mills
    # River and Swannanoa nested), South Toe, Cataloochee, Catawba (all records).
    # Six-month spin-up from April.
    "sim_start":         "2024-04-01 00:00",
    # Sep 18 forecasts end Oct 3 00Z (15 days), so no member is truncated; five
    # days of recession after the Asheville peak.
    "sim_end":           "2024-10-03 00:00",
    "mswep_root":        MSWEP_PAST,
    # Reference rain: MRMS MultiSensor QPE Pass2 (hourly, end-stamped) from Sep 20
    # (bias about -5 % overall, 10-25 % undercatch in the headwaters). Since 21 Oct scaled
    # to GHCN-D daily gauges (mrms_gaugescaled.py); unscaled results in model/_archive_mrms_raw/.
    "gauge_grid":        {"path": "data/raw/helene_2024/obs/mrms_qpe_1h_gaugescaled.nc",
                          "from": "2024-09-20 00:00"},
    "daily_inmaps_glob": "inmaps/era5_temp_pet_daily_2024*.nc",
    "cases": {
        "hist_3h": {
            "precip_mode": "mswep",
            "forcing_dir": "inmaps_hist_3h",
            "basename":    "mrms_era5_3h",
            "label":       "MRMS (MSWEP spinup) + ERA5 temp/PET",
        },
        "era5_sep18": {
            "precip_mode": "era5_blend",
            "era5_src":    "data/processed/helene_2024/era5_processed.nc",
            "era5_start":  "2024-09-20",
            "forcing_dir": "inmaps_era5_sep18",
            "basename":    "era5_sep18_spinup_3h",
            "label":       "ERA5 reanalysis (Sep 20+) + MSWEP/MRMS spinup",
        },
        # ~7 and ~2 days before event_start (Sep 25); no AIFS for 2024
            "ifs_sep18": {
                "precip_mode": "blend", "forecast_src": "data/processed/helene_2024/ifs_processed.nc",
                "init_date": "2024-09-18", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_sep18", "basename": "ifs_sep18_era5_3h",
                "label": "IFS ENS (init 2024-09-18 00Z, 7-day lead) + MSWEP/MRMS spinup",
            },
            "ifs_sep23": {
                "precip_mode": "blend", "forecast_src": "data/processed/helene_2024/ifs_processed.nc",
                "init_date": "2024-09-23", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_ifs_sep23", "basename": "ifs_sep23_era5_3h",
                "label": "IFS ENS (init 2024-09-23 00Z, 2-day lead) + MSWEP/MRMS spinup",
            },
            "wn2_sep18": {
                "precip_mode": "blend", "forecast_src": "data/processed/helene_2024/weathernext_processed.nc",
                "init_date": "2024-09-18", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_sep18", "basename": "wn2_sep18_era5_3h",
                "label": "WeatherNext 2 (init 2024-09-18 00Z, 7-day lead) + MSWEP/MRMS spinup",
            },
            "wn2_sep23": {
                "precip_mode": "blend", "forecast_src": "data/processed/helene_2024/weathernext_processed.nc",
                "init_date": "2024-09-23", "member_dim": "number", "forecast_var": "tp",
                "forcing_dir": "inmaps_wn2_sep23", "basename": "wn2_sep23_era5_3h",
                "label": "WeatherNext 2 (init 2024-09-23 00Z, 2-day lead) + MSWEP/MRMS spinup",
            },
    },
}

def _aifs_era_event(slug, sim_end, grid_from, grid_path, tags, era5_tag, label_ref):
    """Forcing config of an AIFS-era event (texas_2026, mississippi_2026): MSWEP NRT spin-up,
    MRMS reference from `grid_from`, IFS / AIFS / WN2 forecast starts `tags` (init date
    by tag), ERA5 case at `era5_tag`. Tags are (name, init date, lead label)."""
    cases = {
        "hist_3h": {"precip_mode": "mswep", "forcing_dir": "inmaps_hist_3h", "basename": "mrms_era5_3h",
                    "label": label_ref},
        f"era5_{era5_tag[0]}": {"precip_mode": "era5_blend",
                                "era5_src": f"data/processed/{slug}/era5_processed.nc",
                                "era5_start": era5_tag[1], "forcing_dir": f"inmaps_era5_{era5_tag[0]}",
                                "basename": f"era5_{era5_tag[0]}_spinup_3h",
                                "label": f"ERA5 reanalysis ({era5_tag[1]}+) + MSWEP/MRMS spinup"},
    }
    for m, fn, nm in (("ifs", "ifs_processed.nc", "IFS ENS"), ("aifs", "aifs_processed.nc", "AIFS ENS"),
                      ("wn2", "weathernext_processed.nc", "WeatherNext 2")):
        for tag, init, lead in tags:
            cases[f"{m}_{tag}"] = {"precip_mode": "blend", "forecast_src": f"data/processed/{slug}/{fn}",
                                   "init_date": init, "member_dim": "number", "forecast_var": "tp",
                                   "forcing_dir": f"inmaps_{m}_{tag}", "basename": f"{m}_{tag}_era5_3h",
                                   "label": f"{nm} (init {init} 00Z, {lead}) + MSWEP/MRMS spinup"}
    # MSWEP NRT spin-up from 1 Oct 2025 since 18 Nov 2026 (was 1 Apr 2026, inherited from Catalonia:
    # 2.5-3.4 months before the floods); the spin-up test (scripts/spinup_test.py) changed peaks and
    # volumes by less than 2 %, but every event now spins up for at least six months.
    return {"sim_start": "2025-10-01 00:00", "sim_end": sim_end, "mswep_root": MSWEP_NRT,
            "gauge_grid": {"path": grid_path, "from": grid_from},
            "daily_inmaps_glob": "inmaps/era5_temp_pet_daily_20*.nc", "cases": cases}


# AIFS-era events (chosen 18 Oct 2026): reference MRMS (scaled to GHCN-D daily gauges since
# 21 Oct, mrms_gaugescaled.py), starts -7 / -2 / -1 d before event_start
EVENT_CONFIG["texas_2026"] = _aifs_era_event(
    "texas_2026", "2026-07-19 00:00", "2026-07-06 00:00", "data/raw/texas_2026/obs/mrms_qpe_1h_gaugescaled.nc",
    [("jul07", "2026-07-07", "7-day lead"), ("jul12", "2026-07-12", "2-day lead"), ("jul13", "2026-07-13", "1-day lead")],
    ("jul12", "2026-07-12"), "MRMS (MSWEP spinup) + ERA5 temp/PET")
EVENT_CONFIG["mississippi_2026"] = _aifs_era_event(
    "mississippi_2026", "2026-06-22 00:00", "2026-06-09 00:00", "data/raw/mississippi_2026/obs/mrms_qpe_1h_gaugescaled.nc",
    [("jun10", "2026-06-10", "7-day lead"), ("jun15", "2026-06-15", "2-day lead"), ("jun16", "2026-06-16", "1-day lead")],
    ("jun15", "2026-06-15"), "MRMS (MSWEP spinup) + ERA5 temp/PET")

# -1 d start (user decision 2026-10-16, after the Jeseníky and Hans tests): one more
# forecast start per event, the 00Z init one day before the peak / event start.
# Each is a copy of the event's existing case of the same model, with the init
# date, directories and label changed. (event, tag, init date, template tag)
MINUS_ONE_DAY = [
    ("catalonia_2026", "oct02", "2026-10-02", "oct01"),
    ("texas_2025", "jul03", "2025-07-03", "jul02"),
    ("california_2023", "dec30", "2022-12-30", "dec29"),
    ("california_2023", "jan08", "2023-01-08", "jan07"),
    ("helene_2024", "sep24", "2024-09-24", "sep23"),
    ("zermatt_2024", "jun20", "2024-06-20", "jun19"),
    ("kristin_2026", "jan26", "2026-01-26", "jan25"),     # also the kristin_mondego / kristin_lis cases (same dict)
]
for _ev, _tag, _init, _tpl in MINUS_ONE_DAY:
    _cases = EVENT_CONFIG[_ev]["cases"]
    for _m in ("ifs", "aifs", "wn2"):
        if f"{_m}_{_tpl}" not in _cases:
            continue
        _c = dict(_cases[f"{_m}_{_tpl}"])
        _c["init_date"] = _init
        _c["forcing_dir"] = f"inmaps_{_m}_{_tag}"
        _c["basename"] = _c["basename"].replace(_tpl, _tag)
        _c["label"] = f"{_m.upper()} (init {_init} 00Z, 1-day lead) + spinup"
        _cases[f"{_m}_{_tag}"] = _c

# Zermatt second flood (user, 18 Nov 2026): the 29-30 Jun Saaser Vispa flood gets its own
# -7/-2/-1 d starts (22, 27, 28 Jun 00Z; inits downloaded 18-19 Nov). Copies of the jun19
# cases; the 14-20 Jun starts stay for the 21 Jun Matter Vispa flood.
for _tag, _init, _lead in (("jun22", "2024-06-22", "7-day lead"), ("jun27", "2024-06-27", "2-day lead"),
                           ("jun28", "2024-06-28", "1-day lead")):
    _cases = EVENT_CONFIG["zermatt_2024"]["cases"]
    for _m in ("ifs", "wn2"):
        _c = dict(_cases[f"{_m}_jun19"])
        _c.update(init_date=_init, forcing_dir=f"inmaps_{_m}_{_tag}", basename=_c["basename"].replace("jun19", _tag),
                  label=_c["label"].replace("2024-06-19", _init).replace("2-day lead", _lead + ", Saaser flood"))
        _cases[f"{_m}_{_tag}"] = _c

# DANA (20 Oct 2026): reference = AEMET + CHJ-SAIH daily gauge grid with IMERG Final hourly
# timing (dana_reference_rain.py; leave-one-out bias -4 %, r 0.80; MSWEP Past -77 % in the
# storm core) from 27 Oct 06Z. Starts -7/-2/-1 d (22, 27, 28 Oct) replace the 25 Oct start;
# sim_end is the horizon of the 22 Oct forecasts (360 h).
# 23 Oct: v2 reference from the partner data (dana_reference_rain.py build2): + AVAMET/SISRITEL
# stations on 29 Oct (core leave-one-out 0.68 -> 0.99) and nearest-gauge timing from the SAIH
# 5-min gauges (wettest hour 0.73 -> 1.03 of observed). v1 results in model/_archive_imerg_timing/.
_d = EVENT_CONFIG["dana_2024"]
_d["gauge_grid"] = {"path": "data/raw/dana_2024/obs/gauge_grid_hourly_gauges.nc", "from": "2024-10-27 06:00"}
_d["sim_end"] = "2024-11-06 00:00"
for _m in ("ifs", "wn2"):
    for _tag, _init in (("oct22", "2024-10-22"), ("oct28", "2024-10-28")):
        _c = dict(_d["cases"][f"{_m}_oct27"])
        _c.update(init_date=_init, forcing_dir=f"inmaps_{_m}_{_tag}", basename=_c["basename"].replace("oct27", _tag),
                  label=_c["label"].replace("2024-10-27", _init))
        _d["cases"][f"{_m}_{_tag}"] = _c
    _d["cases"].pop(f"{_m}_oct25")

# Kristin reference-rain sensitivity (20 Oct): reference runs with MSWEP up to the latest
# forecast start (26 Jan 00Z), so every start keeps its warm state, then ERA5 raw or ERA5
# scaled to the 11 IPMA gauges (kristin_era5_reference.py). Scored with --ref era5 / era5_scaled.
for _tag, _src, _lab in (("era5ref", "era5_wflow.nc", "ERA5"),
                         ("era5scaled", "era5_wflow_stationscaled.nc", "ERA5 scaled to 11 IPMA gauges")):
    EVENT_CONFIG["kristin_2026"]["cases"][f"{_tag}_jan26"] = {
        "precip_mode": "era5_blend",
        "era5_src":    f"data/processed/kristin_2026/{_src}",
        "era5_start":  "2026-01-26",
        "forcing_dir": f"inmaps_{_tag}_jan26",
        "basename":    f"{_tag}_jan26_mswep_spinup_3h",
        "label":       f"{_lab} from 2026-01-26 + MSWEP spinup (reference sensitivity)",
    }

# Kristin -7 d start (3 Nov 2026): the pilot download began on 22 Jan (-5 d); 20 Jan
# inits were added so Kristin has a -7 d start like the other events. Copies of the
# jan22 cases. The 20 Jan forecasts end on 4 Feb 00Z, before sim_end (5 Feb 21Z);
# the scored flow window ends on 2 Feb, so the members may run past the horizon
# with zero rain (allow_truncated).
_k = EVENT_CONFIG["kristin_2026"]
for _m in ("ifs", "aifs", "wn2"):
    _c = dict(_k["cases"][f"{_m}_jan22"])
    _c.update(init_date="2026-01-20", forcing_dir=f"inmaps_{_m}_jan20", basename=_c["basename"].replace("jan22", "jan20"),
              label=_c["label"].replace("2026-01-22", "2026-01-20"))
    _k["cases"][f"{_m}_jan20"] = _c
_k["allow_truncated"] = True

# Spin-up length test (18 Nov 2026): reference run only, with an earlier start, in separate
# model dirs data/wflow/<slug>_spin (static files only, so no forcing cache is reused).
# The AIFS-era events had inherited Catalonia's 1 Apr 2026 start (2.5-3.4 months before
# the floods); California starts on 1 Oct 2022 (2.8 months). Compare with spinup_test.py.
SPIN_TEST = {"mississippi_2026": "2025-10-01", "texas_2026": "2025-10-01", "california_2023": "2022-07-01"}
for _ev, _start in SPIN_TEST.items():
    _c = dict(EVENT_CONFIG[_ev])
    _c.update(sim_start=f"{_start} 00:00", daily_inmaps_glob="inmaps/era5_temp_pet_daily_20*.nc",
              cases={"hist_3h": EVENT_CONFIG[_ev]["cases"]["hist_3h"]})
    EVENT_CONFIG[f"{_ev}_spin"] = _c

# Clipped sub-basins of kristin_2026 (Mondego @ Coimbra, Lis @ Leiria) share all
# forcing settings — same NRT MSWEP, sim period, daily inmaps glob, and cases.
# Only the model dir (slug) differs; build_forcing derives the grid + bbox from
# each clipped model's own daily inmaps, so the entries are identical aliases.
for _sub in ("kristin_mondego", "kristin_lis"):
    EVENT_CONFIG[_sub] = EVENT_CONFIG["kristin_2026"]

# Jeseníky / Kłodzko model: same storm, sim period, cases and processed forecasts
# as boris_2024; the grid comes from its own daily ERA5-Land inmaps
# (wflow_build_meteo.py --basename era5land_temp_pet_daily).
# Its own copy (not an alias): the reference rain differs from the Elbe model's.
# Reference (extreme-gauge framework, 2026-10-14): MSWEP spin-up, then the
# CHMI + IMGW hourly gauge grid (gauge_grid.py; leave-one-out bias +2 %, r 0.90;
# ERA5 -28 %, MSWEP -21 %) from Sep 8, the start of the station download.
EVENT_CONFIG["boris_sudetes"] = {
    **EVENT_CONFIG["boris_2024"],
    "gauge_grid": {"path": "data/raw/boris_sudetes/obs/gauge_grid_hourly.nc", "from": "2024-09-08 00:00"},
    # -1 d start added 2026-10-15 (lead-selection check: IFS beats WN2 at 0.5-1.5 d
    # here, which the -7/-2 d starts miss); tested on Jeseníky and Hans first.
    "cases": {
        **EVENT_CONFIG["boris_2024"]["cases"],
        "ifs_sep11": {
            "precip_mode": "blend", "forecast_src": "data/processed/boris_2024/ifs_processed.nc",
            "init_date": "2024-09-11", "member_dim": "number", "forecast_var": "tp",
            "forcing_dir": "inmaps_ifs_sep11", "basename": "ifs_sep11_era5land_3h",
            "label": "IFS ENS (init 2024-09-11 00Z, 1-day lead) + MSWEP/gauge-grid spinup",
        },
        "wn2_sep11": {
            "precip_mode": "blend", "forecast_src": "data/processed/boris_2024/weathernext_processed.nc",
            "init_date": "2024-09-11", "member_dim": "number", "forecast_var": "tp",
            "forcing_dir": "inmaps_wn2_sep11", "basename": "wn2_sep11_era5land_3h",
            "label": "WeatherNext 2 (init 2024-09-11 00Z, 1-day lead) + MSWEP/gauge-grid spinup",
        },
    },
}


# =============================================================================
# MSWEP helpers
# =============================================================================

def mswep_filename(dt: pd.Timestamp) -> str:
    """YYYYDDD.HH.nc for a given 3-hourly timestamp."""
    doy = dt.day_of_year
    return f"{dt.year}{doy:03d}.{dt.hour:02d}.nc"


def read_mswep_period(start: pd.Timestamp, end: pd.Timestamp,
                      bbox: dict, mswep_root: str) -> xr.DataArray:
    """Read all MSWEP 3h files in [start, end], subset to bbox.

    Returns DataArray (time, lat, lon) in mm/3h.
    """
    times = pd.date_range(start, end, freq="3h")
    chunks = []
    missing = []
    for t in times:
        fpath = os.path.join(mswep_root, mswep_filename(t))
        if not os.path.exists(fpath):
            missing.append(t)
            continue
        ds = xr.open_dataset(fpath)
        sub = (ds["precipitation"]
               .sel(lat=slice(bbox["lat_max"], bbox["lat_min"]),
                    lon=slice(bbox["lon_min"], bbox["lon_max"]))
               .load()
               .squeeze("time")
               .expand_dims({"time": [t]}))
        chunks.append(sub)
        ds.close()

    if missing:
        print(f"  [WARN] {len(missing)} MSWEP files not found (first: {missing[0]})")
    if not chunks:
        raise FileNotFoundError(f"No MSWEP files found for {start} – {end}")

    da = xr.concat(chunks, dim="time")
    da.attrs["units"] = "mm/3h"
    return da


# =============================================================================
# Regrid coarse → wflow model grid (bilinear)
# =============================================================================

def regrid_to_wflow(da: xr.DataArray, wflow_lat, wflow_lon) -> xr.DataArray:
    """Bilinear interpolation to the wflow model grid."""
    lat_dim = next(d for d in da.dims if d.lower() in ("lat", "latitude"))
    lon_dim = next(d for d in da.dims if d.lower() in ("lon", "longitude"))
    da = da.rename({lat_dim: "lat", lon_dim: "lon"})

    lats = da["lat"].values
    flip = lats[0] > lats[-1]
    if flip:
        da = da.isel(lat=slice(None, None, -1))
        target_lat = wflow_lat[::-1]
    else:
        target_lat = wflow_lat

    # fill_value=0.0 (not "extrapolate"): any target cell outside the source
    # coverage gets 0 rather than a wild linear extrapolation. Critical when the
    # wflow model grid extends beyond the forcing subset (e.g. Atlantic corners
    # of a large pfafstetter tile) — extrapolation there produced absurd precip
    # (>10,000 mm/3h) that crashed wflow. The MSWEP read bbox is also derived
    # from the model grid (build_forcing), so genuine MSWEP cases never hit this.
    da_interp = da.interp(
        lat=target_lat,
        lon=wflow_lon,
        method="linear",
        kwargs={"fill_value": 0.0, "bounds_error": False},
    )
    if flip:
        da_interp = da_interp.isel(lat=slice(None, None, -1))

    return da_interp.rename({"lat": "latitude", "lon": "longitude"}).clip(min=0.0)


# =============================================================================
# Disaggregate daily ERA5 temp / PET → 3h
# =============================================================================

def disaggregate_daily_to_3h(daily_ds: xr.Dataset,
                              times_3h: pd.DatetimeIndex) -> xr.Dataset:
    """Convert daily temp/PET to 3-hourly.

    temp: linear interpolation between daily means.
    pet:  uniform distribution (daily / 8).
    """
    t0 = pd.Timestamp("2000-01-01")   # arbitrary epoch for interpolation arithmetic

    days_needed = pd.date_range(times_3h[0].normalize(),
                                times_3h[-1].normalize(), freq="D")
    daily_sub = daily_ds.sel(time=days_needed, method="nearest")

    daily_sec  = np.array([(d - t0).total_seconds() for d in days_needed])
    threeh_sec = np.array([(t - t0).total_seconds() for t in times_3h])

    temp_daily = daily_sub["temp"].values        # (days, lat, lon)
    n_3h  = len(times_3h)
    nlat  = temp_daily.shape[1]
    nlon  = temp_daily.shape[2]

    temp_3h = np.empty((n_3h, nlat, nlon), dtype="float32")
    for j in range(nlat):
        for k in range(nlon):
            temp_3h[:, j, k] = np.interp(threeh_sec, daily_sec, temp_daily[:, j, k])

    pet_daily = daily_sub["pet"].values
    pet_3h = np.empty((n_3h, nlat, nlon), dtype="float32")
    for i, t in enumerate(times_3h):
        day_idx = int(np.argmin(np.abs(days_needed - t.normalize())))
        pet_3h[i] = np.maximum(pet_daily[day_idx] / 8.0, 0.0)

    coords = {
        "time":      times_3h,
        "latitude":  daily_sub["latitude"].values,
        "longitude": daily_sub["longitude"].values,
    }
    return xr.Dataset({
        "temp": xr.DataArray(temp_3h, dims=["time", "latitude", "longitude"],
                             coords=coords, attrs={"units": "°C"}),
        "pet":  xr.DataArray(pet_3h,  dims=["time", "latitude", "longitude"],
                             coords=coords, attrs={"units": "mm/3h"}),
    })


# =============================================================================
# Extract 3-hourly forecast precip (IFS, WN2, AIFS) for a specific init date
# =============================================================================

def accum_to_slots(values, step_hours, slot_hours, width=3.0):
    """Re-bin accumulations onto wflow's 3-hourly slots, conserving mass.

    values[i] is the precipitation accumulated over (s[i-1], s[i]] hours after
    init (s[-1] = 0), which is how every processed product stores it (IFS/WN3
    hourly then IFS 3- and 6-hourly, AIFS/WN2 6-hourly). Wflow applies the
    forcing stamped T to the step (T-3h, T], so the value for slot T must be the
    accumulation over exactly that interval. With the rate held constant within
    each native step, the cumulative curve is piecewise linear and
        P(T) = A(T) - A(T - 3h).
    Exact for 1 h and 3 h input, an even split for 6 h input, and nothing is
    lost or moved at the 3h->6h change of the IFS step (the old nearest/resample
    route shifted hourly input 2 h early, dropped half of the 144-150 h IFS
    total and put forecast rain in the slot ending AT the init).
    `width` is the slot length in hours (3 for the standard runs, 1 for the
    hourly-IFS experiment, wflow_hourly_ifs.py): exact for input at that
    resolution or finer, an even split of coarser steps.
    """
    s = np.concatenate([[0.0], np.asarray(step_hours, dtype=float)])
    cum = np.concatenate([np.zeros_like(values[:1]), np.cumsum(values, axis=0)], axis=0)

    def weights(x):
        w = np.zeros((len(x), len(s)))
        for k, xv in enumerate(x):
            if xv < 0 or xv > s[-1] + 1e-9:
                raise ValueError(f"slot edge {xv} h outside forecast range 0-{s[-1]} h")
            i = min(int(np.searchsorted(s, xv, side="right")) - 1, len(s) - 2)
            f = (xv - s[i]) / (s[i + 1] - s[i])
            w[k, i], w[k, i + 1] = 1.0 - f, f
        return w

    slot_hours = np.asarray(slot_hours, dtype=float)
    return np.tensordot(weights(slot_hours) - weights(slot_hours - width), cum, axes=(1, 0))


def forecast_init(case_cfg: dict) -> pd.Timestamp:
    """Actual init time of a case: the first init on its init_date (e.g. the
    AIFS archive starts at 2025-07-01 12Z, so texas aifs_jul01 is a 12Z run)."""
    with xr.open_dataset(os.path.join(REPO_ROOT, case_cfg["forecast_src"])) as ds:
        inits = pd.DatetimeIndex(ds["init_time"].values)
    hit = inits.normalize() == pd.Timestamp(case_cfg["init_date"]).normalize()
    if not hit.any():   # argmax of all-False is 0: would silently use the first init
        raise ValueError(f"no init on {case_cfg['init_date']} in {case_cfg['forecast_src']}")
    return inits[int(hit.argmax())]


def forecast_slots_latlon(case_cfg: dict, member=None, width=3) -> xr.DataArray:
    """3-hourly forecast precip (mm/3h) on the product's own lat/lon grid
    (`width`=1: hourly slots in mm/h, for the hourly-IFS experiment).

    Slots are stamped at the END of their 3 h interval, from init+3h to the
    last forecast step. `member` (position along the member dim) selects one
    member; None gives the ensemble mean.
    """
    src = os.path.join(REPO_ROOT, case_cfg["forecast_src"])
    init = forecast_init(case_cfg)
    with xr.open_dataset(src) as ds:
        tp = ds[case_cfg.get("forecast_var", "tp")].sel(init_time=init)
        mdim = case_cfg.get("member_dim", "number")
        if member is None:
            tp = tp.transpose("step", ...).load()
        elif isinstance(member, slice):      # mean over a subset of members
            tp = tp.isel({mdim: member}).transpose("step", ...).load()
        else:
            tp = tp.isel({mdim: member}, drop=True).transpose("step", ...).load()
    # A forecast downloaded only to the released lead (download_ifs.py
    # --available-lead) is NaN after its last step. accum_to_slots weights every
    # step, so one NaN step would turn every slot NaN (0 * NaN): cut at the last
    # step where all cells (and, for a mean, all members) are valid. NaN inside
    # that range is a real data gap and stays an error.
    valid = np.isfinite(tp.values).reshape(tp.sizes["step"], -1).all(axis=1)
    n_ok = int(np.argmin(valid)) if not valid.all() else len(valid)
    if n_ok == 0 or valid[n_ok:].any():
        raise ValueError(f"forecast {init} has NaN steps before its last valid step")
    tp = tp.isel(step=slice(0, n_ok))
    if mdim in tp.dims:
        tp = tp.mean(dim=mdim)
    step_h = pd.to_timedelta(tp["step"].values) / pd.Timedelta("1h")
    slot_h = np.arange(float(width), float(step_h.max()) + 1e-9, float(width))
    vals = accum_to_slots(tp.values, step_h.values, slot_h, width=float(width)).clip(min=0.0)
    lat = next(d for d in tp.dims if d.lower() in ("lat", "latitude"))
    lon = next(d for d in tp.dims if d.lower() in ("lon", "longitude"))
    out = xr.DataArray(vals.astype("float32"), dims=("time", lat, lon),
                       coords={"time": init + pd.to_timedelta(slot_h, "h"),
                               lat: tp[lat].values, lon: tp[lon].values})
    out.attrs.update(units=f"mm/{width}h", init_time=str(init))
    return out


def extract_forecast_3h(case_cfg: dict, wflow_lat, wflow_lon, member=None) -> xr.DataArray:
    """3-hourly forecast precip on the wflow grid (mm/3h), slots stamped at the
    end of their interval (see accum_to_slots). Ensemble mean by default;
    `member` selects one member (used by wflow_ensemble.py).
    """
    tp_3h = forecast_slots_latlon(case_cfg, member)
    print(f"  Forecast init: {tp_3h.attrs['init_time']}  ({tp_3h.sizes['time']} slots)")
    lat = next(d for d in tp_3h.dims if d.lower() in ("lat", "latitude"))
    lon = next(d for d in tp_3h.dims if d.lower() in ("lon", "longitude"))
    tp_regrid = regrid_to_wflow(tp_3h.rename({lat: "lat", lon: "lon"}), wflow_lat, wflow_lon)
    tp_regrid.attrs["units"] = "mm/3h"
    return tp_regrid


# =============================================================================
# Extract ERA5 reanalysis 3-hourly precip (hourly → 3h sum)
# =============================================================================

def extract_era5_3h(case_cfg: dict, wflow_lat, wflow_lon) -> xr.DataArray:
    """Return 3-hourly ERA5 precip on the wflow grid (mm/3h).

    ERA5 hourly tp stamped t is the accumulation over (t-1h, t]; three hours are
    summed into (T-3h, T] and stamped T (end of interval), matching wflow and
    the forecast slots. A left-labelled resample would shift ERA5 2 h early.
    """
    src = os.path.join(REPO_ROOT, case_cfg["era5_src"])
    ds  = xr.open_dataset(src)
    tp  = ds["tp"]

    tp_3h = tp.resample(time="3h", closed="right", label="right").sum(skipna=False).clip(min=0.0)
    # keep only complete (3-hour) bins: a partial bin at either end of the ERA5
    # record would under-count that slot
    n_h = pd.Series(1, index=pd.DatetimeIndex(tp["time"].values)).resample(
        "3h", closed="right", label="right").sum()
    tp_3h = tp_3h.sel(time=n_h.index[n_h.values == 3])

    lat_dim = next(d for d in tp_3h.dims if d.lower() in ("lat", "latitude"))
    lon_dim = next(d for d in tp_3h.dims if d.lower() in ("lon", "longitude"))
    tp_3h   = tp_3h.rename({lat_dim: "lat", lon_dim: "lon"})

    tp_regrid = regrid_to_wflow(tp_3h, wflow_lat, wflow_lon)
    tp_regrid.attrs["units"] = "mm/3h"
    ds.close()
    t = pd.DatetimeIndex(tp_regrid["time"].values)
    print(f"  ERA5 3h available: {t[0]} → {t[-1]}  ({len(t)} steps)")
    return tp_regrid


# =============================================================================
# Shared per-event "base" forcing cache (MSWEP precip + temp + pet)
# =============================================================================
# temp/pet are identical across every case of an event, and the MSWEP spinup
# precip is identical across every forecast case (only the post-init event
# precip differs). We therefore build the full MSWEP+temp+pet forcing ONCE per
# event into inmaps_base/, then:
#   - spinup months (and all-MSWEP cases) hardlink the base file (zero extra
#     disk, no recomputation)
#   - event-window months reuse the cached temp/pet and only overlay the
#     case-specific forecast precip after the init/event-start time.
# This removes the dominant redundant cost: the cell-by-cell temp/pet
# disaggregation and the repeated MSWEP read/regrid of the long spinup.

_ENC = {"dtype": "float32", "zlib": True, "complevel": 4}


def _write_nc_atomic(ds: xr.Dataset, path: str):
    """Write NetCDF via a unique temp file + atomic rename.

    Safe under parallel jobs sharing inmaps_base/: a reader never sees a
    partially written file, and concurrent identical writes are harmless.
    """
    enc = {v: _ENC for v in ("precip", "temp", "pet") if v in ds}
    tmp = f"{path}.tmp.{os.getpid()}"
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)


def _link_or_copy(src: str, dst: str):
    """Hardlink src→dst (no extra disk); fall back to copy across devices."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def base_is_current(base_path: str) -> bool:
    """A base month exists AND its precip is on the end-of-interval stamps."""
    if not os.path.exists(base_path):
        return False
    with xr.open_dataset(base_path) as d:
        return d["precip"].attrs.get("time_convention") == "end"


def mswep_slots(times_3h, bbox, mswep_root, wflow_lat, wflow_lon) -> xr.DataArray:
    """MSWEP precip on wflow's 3 h slots (mm/3h).

    MSWEP file YYYYDDD.HH holds the accumulation over (HH, HH+3h] -- its stamp
    is the START of the interval (confirmed by lagged correlation with ERA5,
    scripts/check_forcing_timing.py: best match at +3 h for every event, and
    preprocess.process_mswep applies the same shift). Wflow slot T is
    (T-3h, T], so slot T reads the file stamped T-3h.
    """
    src = read_mswep_period(times_3h[0] - pd.Timedelta("3h"),
                            times_3h[-1] - pd.Timedelta("3h"), bbox, mswep_root)
    if src.sizes["time"] != len(times_3h):
        raise FileNotFoundError(f"MSWEP incomplete for {times_3h[0]} - {times_3h[-1]}")
    precip = regrid_to_wflow(src, wflow_lat, wflow_lon).assign_coords(time=times_3h).astype("float32")
    precip.attrs.update(units="mm/3h", time_convention="end",
                        note="slot T = MSWEP file stamped T-3h (MSWEP stamps interval start)")
    return precip


def gauge_grid_slots(times_3h, gauge_grid, wflow_lat, wflow_lon) -> xr.DataArray:
    """Gauge-based precip on wflow's 3 h slots (mm/3h).

    gauge_grid["path"] is an hourly grid stamped at the END of each hour (e.g.
    xema_grid.py), so slot T sums the hours stamped T-2h, T-1h and T.
    """
    with xr.open_dataset(os.path.join(REPO_ROOT, gauge_grid["path"])) as d:
        h = d[gauge_grid.get("var", "precip")].load()
    h3 = h.resample(time="3h", closed="right", label="right").sum(min_count=3)
    missing = pd.DatetimeIndex(times_3h).difference(pd.DatetimeIndex(h3["time"].values))
    if len(missing):
        raise FileNotFoundError(f"gauge grid lacks {len(missing)} slots, first {missing[0]}")
    h3 = h3.sel(time=times_3h)
    if bool(h3.isnull().all(dim=[x for x in h3.dims if x != "time"]).any()):
        raise ValueError("gauge grid has empty 3 h slots in the requested period")
    precip = regrid_to_wflow(h3.fillna(0.0), wflow_lat, wflow_lon).astype("float32")
    precip.attrs.update(units="mm/3h", time_convention="end", note=f"gauge grid {gauge_grid['path']}")
    return precip


def base_precip_slots(times_3h, bbox, mswep_root, wflow_lat, wflow_lon, gauge_grid=None):
    """Base precip: MSWEP, or MSWEP up to gauge_grid["from"] and the gauge grid after it."""
    times_3h = pd.DatetimeIndex(times_3h)
    if not gauge_grid:
        return mswep_slots(times_3h, bbox, mswep_root, wflow_lat, wflow_lon)
    switch = pd.Timestamp(gauge_grid["from"])
    early, late = times_3h[times_3h <= switch], times_3h[times_3h > switch]
    parts = []
    if len(early):
        parts.append(mswep_slots(early, bbox, mswep_root, wflow_lat, wflow_lon))
    if len(late):
        parts.append(gauge_grid_slots(late, gauge_grid, wflow_lat, wflow_lon))
    precip = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    precip.attrs.update(units="mm/3h", time_convention="end",
                        note=f"MSWEP to {switch}, then gauge grid {gauge_grid['path']}")
    return precip


def build_base_month(base_path: str, mon_start, mon_end, times_3h,
                     bbox: dict, mswep_root: str, wflow_lat, wflow_lon,
                     daily_ds: xr.Dataset, spatial_ref, gauge_grid=None):
    """Build (once) the all-MSWEP base forcing for one month: precip+temp+pet.

    A base month written before the MSWEP time fix gets its precip replaced;
    temp/pet (the slow part) are kept as they are.
    """
    if base_is_current(base_path):
        return
    if os.path.exists(base_path):
        with xr.open_dataset(base_path) as old:
            out_ds = old.load()
        own = pd.DatetimeIndex(out_ds["time"].values)   # may exceed times_3h
        out_ds["precip"] = base_precip_slots(own, bbox, mswep_root, wflow_lat, wflow_lon,
                                             gauge_grid).rename("precip")
        _write_nc_atomic(out_ds, base_path)
        return
    precip = base_precip_slots(times_3h, bbox, mswep_root, wflow_lat, wflow_lon, gauge_grid)
    tp = disaggregate_daily_to_3h(daily_ds, times_3h)
    out_ds = xr.Dataset({
        "precip": precip.rename("precip"),
        "temp":   tp["temp"].astype("float32"),
        "pet":    tp["pet"].astype("float32"),
    })
    if spatial_ref is not None:
        out_ds["spatial_ref"] = spatial_ref
    _write_nc_atomic(out_ds, base_path)


# =============================================================================
# Main build function
# =============================================================================

def build_forcing(slug: str, case: str):
    evt_cfg  = EVENT_CONFIG[slug]
    case_cfg = evt_cfg["cases"][case]

    SIM_START  = pd.Timestamp(evt_cfg["sim_start"])
    SIM_END    = pd.Timestamp(evt_cfg["sim_end"])
    mswep_root = evt_cfg["mswep_root"]

    model_dir   = os.path.join(REPO_ROOT, "data", "wflow", slug, "model")
    daily_glob  = os.path.join(model_dir, evt_cfg["daily_inmaps_glob"])
    daily_files = sorted(glob.glob(daily_glob))

    if not daily_files:
        sys.exit(
            f"No daily inmaps files found matching:\n  {daily_glob}\n"
            "Run wflow_setup.py first to copy daily inmaps."
        )

    out_dir = os.path.join(model_dir, case_cfg["forcing_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # Shared per-event base cache (MSWEP precip + temp + pet), built once.
    base_dir = os.path.join(model_dir, "inmaps_base")
    os.makedirs(base_dir, exist_ok=True)

    # Load daily ERA5 temp/PET reference (lazy, multi-file)
    print(f"Loading daily ERA5 reference forcing ({len(daily_files)} files)…")
    # `spatial_ref` is CRS metadata that lands as either a coordinate or a data
    # variable depending on how the model's staticmaps stores it. Rebuilding only
    # some months (e.g. after a staticmaps resync) leaves a mixed set, and the
    # merge then dies with "unable to determine if these variables should be
    # coordinates or not" — which is a packaging detail, not a data conflict. Drop
    # it on open and take the CRS from the first file instead.
    daily_ds  = xr.open_mfdataset(
        daily_files, combine="by_coords", chunks={"time": 30},
        preprocess=lambda d: d.drop_vars("spatial_ref", errors="ignore"))
    wflow_lat = daily_ds["latitude"].values
    wflow_lon = daily_ds["longitude"].values

    # Derive the MSWEP read bbox from the actual model grid (± buffer) so the
    # forcing subset always fully covers the grid — no bilinear extrapolation.
    bbox = {
        "lat_min": float(np.min(wflow_lat)) - BBOX_BUFFER_DEG,
        "lat_max": float(np.max(wflow_lat)) + BBOX_BUFFER_DEG,
        "lon_min": float(np.min(wflow_lon)) - BBOX_BUFFER_DEG,
        "lon_max": float(np.max(wflow_lon)) + BBOX_BUFFER_DEG,
    }
    print(f"  Model grid: lat {wflow_lat.min():.2f}–{wflow_lat.max():.2f}, "
          f"lon {wflow_lon.min():.2f}–{wflow_lon.max():.2f}")
    print(f"  MSWEP read bbox (+{BBOX_BUFFER_DEG}° buffer): "
          f"lat {bbox['lat_min']:.2f}–{bbox['lat_max']:.2f}, "
          f"lon {bbox['lon_min']:.2f}–{bbox['lon_max']:.2f}")

    # Preload forecast/ERA5 event precip (blend cases only)
    fcast_da    = None
    fcast_start = None

    if case_cfg["precip_mode"] == "blend":
        print("Extracting forecast precip…")
        fcast_da    = extract_forecast_3h(case_cfg, wflow_lat, wflow_lon)
        fcast_start = forecast_init(case_cfg)
        fcast_times = pd.DatetimeIndex(fcast_da["time"].values)
        print(f"  Forecast 3h available: {fcast_times[0]} → {fcast_times[-1]}")

    elif case_cfg["precip_mode"] == "era5_blend":
        print("Extracting ERA5 reanalysis precip…")
        fcast_da    = extract_era5_3h(case_cfg, wflow_lat, wflow_lon)
        fcast_start = pd.Timestamp(case_cfg["era5_start"])

    # Last time step the forecast (or ERA5 file) actually provides. Everything
    # after this is zero-filled rather than persisted — see _blend_event_precip.
    fcast_horizon = None
    if fcast_da is not None:
        fcast_horizon = pd.DatetimeIndex(fcast_da["time"].values).max()
        n_beyond = int((pd.date_range(SIM_START, SIM_END, freq="3h") > fcast_horizon).sum())
        if n_beyond:
            msg = (f"  sim_end {SIM_END} is {n_beyond} slots ({n_beyond * 3 / 24:.1f} d) "
                   f"beyond the forecast horizon {fcast_horizon}")
            if not (ALLOW_TRUNCATED or evt_cfg.get("allow_truncated", False)):
                sys.exit(
                    f"[ERROR]{msg}\n"
                    "  Refusing to fabricate precipitation past the horizon. Either shorten\n"
                    "  sim_end / pick an earlier-reaching init, or pass\n"
                    "  --allow-truncated-forecast to zero-fill the remainder."
                )
            print(f"  [WARN]{msg} → zero-filled")

    # dropped from daily_ds above, so read the CRS straight from the first file
    with xr.open_dataset(daily_files[0]) as _d0:
        spatial_ref = _d0["spatial_ref"].load() if "spatial_ref" in _d0 else None

    # Process month by month
    months = pd.period_range(SIM_START, SIM_END, freq="M")
    for period in months:
        mon_start = max(pd.Timestamp(period.start_time).floor("3h"), SIM_START)
        mon_end   = min(pd.Timestamp(period.end_time).floor("3h"),   SIM_END)
        times_3h  = pd.date_range(mon_start, mon_end, freq="3h")

        ym        = f"{period.year}{period.month:02d}"
        base_path = os.path.join(base_dir, f"base_{ym}.nc")
        out_fname = f"{case_cfg['basename']}_{ym}.nc"
        out_path  = os.path.join(out_dir, out_fname)

        # 1. Ensure the shared base month (MSWEP precip + temp + pet) exists.
        if not base_is_current(base_path):
            print(f"\nMonth {period}: building base (MSWEP + temp/pet)…")
            build_base_month(base_path, mon_start, mon_end, times_3h,
                             bbox, mswep_root, wflow_lat, wflow_lon,
                             daily_ds, spatial_ref, evt_cfg.get("gauge_grid"))
        else:
            print(f"\nMonth {period}: base cached.")

        if os.path.exists(out_path):
            print(f"  [SKIP] {out_fname} already exists.")
            continue

        # 2. Derive the case month from the base.
        is_event = fcast_start is not None and mon_end > fcast_start
        if not is_event:
            # Pure spinup / all-MSWEP month → identical to base. Hardlink it.
            print(f"  Reusing base (spinup) → {out_fname}")
            _link_or_copy(base_path, out_path)
            continue

        # Event-window month: keep cached temp/pet, overlay forecast precip on
        # the slots AFTER fcast_start. Slots are stamped at the end of their 3 h
        # interval, so the slot stamped fcast_start (the 3 h before the init)
        # keeps the base MSWEP precip.
        base_ds  = xr.open_dataset(base_path)
        sp_times = times_3h[times_3h <= fcast_start]
        ev_times = times_3h[times_3h >  fcast_start]

        parts = []
        if len(sp_times):
            parts.append(base_ds["precip"].sel(time=sp_times))
        if len(ev_times):
            # Split at the forecast horizon. `sel(method="nearest")` has no upper
            # bound, so without this split every slot past the last forecast step
            # would silently repeat that step forever — fabricating rain that then
            # lands inside the scored window. Zero-fill is the honest extension:
            # MSWEP would leak truth into a forecast case, persistence invents
            # rain, zero lets the hydrograph recess, which is what "no forecast
            # information" actually means.
            ev_in  = ev_times[ev_times <= fcast_horizon]
            ev_out = ev_times[ev_times >  fcast_horizon]
            if len(ev_in):
                parts.append(fcast_da.sel(time=ev_in))   # exact 3 h slots
            if len(ev_out):
                zero = xr.zeros_like(fcast_da.isel(time=0, drop=True))
                parts.append(zero.expand_dims(time=ev_out))
        precip = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        precip = precip.reindex(time=times_3h).astype("float32")

        # Subset temp/pet to times_3h too. The base cache is shared and may have
        # been built under a longer sim_end; without this, xarray aligns the
        # Dataset on the UNION of the two time axes and silently pads precip with
        # NaN out to the cached month's end — which wflow reads as missing forcing.
        out_ds = xr.Dataset({
            "precip": precip.rename("precip"),
            "temp":   base_ds["temp"].sel(time=times_3h),
            "pet":    base_ds["pet"].sel(time=times_3h),
        })
        if "spatial_ref" in base_ds:
            out_ds["spatial_ref"] = base_ds["spatial_ref"]

        print(f"  Overlaying forecast precip ({len(ev_times)} event slots) → {out_fname}")
        _write_nc_atomic(out_ds, out_path)
        base_ds.close()
        print(f"  → {os.path.getsize(out_path) / 1e6:.0f} MB")

    daily_ds.close()
    print(f"\nDone. Forcing written to: {out_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build 3-hourly wflow forcing for an AIFS event")
    parser.add_argument("event_slug", help="e.g. dana_2024 | kristin_2026")
    parser.add_argument("case",       help="e.g. hist_3h | ifs_oct25 | ifs_jan22")
    parser.add_argument(
        "--allow-truncated-forecast", action="store_true",
        help="Zero-fill precip past the forecast horizon instead of aborting. "
             "Only use when the scored window is trimmed to the horizon anyway.")
    args = parser.parse_args()

    global ALLOW_TRUNCATED
    ALLOW_TRUNCATED = args.allow_truncated_forecast

    slug = args.event_slug
    case = args.case

    if slug not in EVENT_CONFIG:
        sys.exit(f"Unknown event '{slug}'. Options: {list(EVENT_CONFIG)}")
    if case not in EVENT_CONFIG[slug]["cases"]:
        sys.exit(f"Unknown case '{case}' for '{slug}'. "
                 f"Options: {list(EVENT_CONFIG[slug]['cases'])}")

    print(f"=== wflow forcing prep: {slug} / {case} ===")
    print(f"Label: {EVENT_CONFIG[slug]['cases'][case]['label']}\n")
    build_forcing(slug, case)


if __name__ == "__main__":
    main()
