#!/usr/bin/env python3
"""Set up a wflow model directory for an AIFS project event + case.

Copies model static files from the source archive, selects only the forcing
(inmaps) files needed for the simulation period, and writes a patched TOML
with event-specific time bounds and output directory.

Usage:
    python scripts/wflow_setup.py dana_2024 hist_3h
    python scripts/wflow_setup.py kristin_2026 ifs_jan22
"""

import sys
import re
import shutil
import argparse
from pathlib import Path

import os

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── helpers ───────────────────────────────────────────────────────────────────

def _case(forcing_dir: str, timestepsecs: int = 10800) -> dict:
    """Return a case config dict from a forcing directory name."""
    slug_placeholder = "{slug}"
    return {
        "run_dir":      f"run_{slug_placeholder}_{forcing_dir.lstrip('inmaps_') or 'hist'}",
        "toml_out":     "wflow_sbm_{slug}_" + forcing_dir.lstrip("inmaps_").lstrip("_") + ".toml",
        "forcing_dir":  forcing_dir,
        "timestepsecs": timestepsecs,
    }


# ── per-event config ──────────────────────────────────────────────────────────
# Each event defines:
#   source_model   : path to the wflow_sbm dir in the archive
#   spinup_start   : ISO datetime for simulation start (includes spinup)
#   sim_end        : ISO datetime for simulation end
#   forcing_years  : list of int years whose inmaps files to copy
#   cases          : dict of case_name → case config (run_dir, toml_out, forcing_dir, timestepsecs)

WFLOW_CONFIG = {

    "dana_2024": {
        "source_model": Path(
            os.environ.get("WFLOW_EUROPE_MODELS", os.path.join(REPO_ROOT, "data", "external", "wflow_europe"))
            +             "/models/pfafstetter_Europe_12000uparea_291741/wflow_sbm"
        ),
        "spinup_start":  "2024-04-01T00:00:00",
        "sim_end":       "2024-11-09T00:00:00",   # shortest forecast horizon (360 h)
        "forcing_years": [2024],
        "cases": {
            "hist": {
                "run_dir":      "run_{slug}_hist",
                "toml_out":     "wflow_sbm_{slug}_hist.toml",
                "forcing_dir":  "inmaps",
                "timestepsecs": 86400,
            },
            "hist_3h": {
                "run_dir":         "run_{slug}_hist_3h",
                "toml_out":        "wflow_sbm_{slug}_hist_3h.toml",
                "forcing_dir":     "inmaps_hist_3h",
                "timestepsecs":    10800,
                "gridded_output":  True,   # reference case → keep output.nc for spatial peak map
            },
            "era5_oct28": {
                "run_dir":      "run_{slug}_era5_oct28",
                "toml_out":     "wflow_sbm_{slug}_era5_oct28.toml",
                "forcing_dir":  "inmaps_era5_oct28",
                "timestepsecs": 10800,
            },
            # IFS ENS — long-lead (init Oct 25, ~5 days before event)
            "ifs_oct25": {
                "run_dir":      "run_{slug}_ifs_oct25",
                "toml_out":     "wflow_sbm_{slug}_ifs_oct25.toml",
                "forcing_dir":  "inmaps_ifs_oct25",
                "timestepsecs": 10800,
            },
            # IFS ENS — near-lead (init Oct 27, ~2 days before event)
            "ifs_oct27": {
                "run_dir":      "run_{slug}_ifs_oct27",
                "toml_out":     "wflow_sbm_{slug}_ifs_oct27.toml",
                "forcing_dir":  "inmaps_ifs_oct27",
                "timestepsecs": 10800,
            },
            # WeatherNext 2 — long-lead (init Oct 25)
            "wn2_oct25": {
                "run_dir":      "run_{slug}_wn2_oct25",
                "toml_out":     "wflow_sbm_{slug}_wn2_oct25.toml",
                "forcing_dir":  "inmaps_wn2_oct25",
                "timestepsecs": 10800,
            },
            # WeatherNext 2 — near-lead (init Oct 27)
            "wn2_oct27": {
                "run_dir":      "run_{slug}_wn2_oct27",
                "toml_out":     "wflow_sbm_{slug}_wn2_oct27.toml",
                "forcing_dir":  "inmaps_wn2_oct27",
                "timestepsecs": 10800,
            },
        },
    },

    "kristin_2026": {
        "source_model": Path(
            os.environ.get("WFLOW_EUROPE_MODELS", os.path.join(REPO_ROOT, "data", "external", "wflow_europe"))
            +             "/models/pfafstetter_Europe_12000uparea_295311/wflow_sbm"
        ),
        "spinup_start":  "2025-07-01T00:00:00",
        "sim_end":       "2026-02-05T21:00:00",
        "forcing_years": [2025, 2026],
        "cases": {
            "hist_3h": {
                "run_dir":         "run_{slug}_hist_3h",
                "toml_out":        "wflow_sbm_{slug}_hist_3h.toml",
                "forcing_dir":     "inmaps_hist_3h",
                "timestepsecs":    10800,
                "gridded_output":  True,   # reference case → keep output.nc for spatial peak map
            },
            # ERA5 reanalysis — event from Jan 22 (matches long-lead init)
            "era5_jan22": {
                "run_dir":      "run_{slug}_era5_jan22",
                "toml_out":     "wflow_sbm_{slug}_era5_jan22.toml",
                "forcing_dir":  "inmaps_era5_jan22",
                "timestepsecs": 10800,
            },
            # IFS ENS — long-lead (init Jan 22, ~5 days before Jan 27 event)
            "ifs_jan22": {
                "run_dir":      "run_{slug}_ifs_jan22",
                "toml_out":     "wflow_sbm_{slug}_ifs_jan22.toml",
                "forcing_dir":  "inmaps_ifs_jan22",
                "timestepsecs": 10800,
            },
            # IFS ENS — near-lead (init Jan 25, ~2 days before Jan 27 event)
            "ifs_jan25": {
                "run_dir":      "run_{slug}_ifs_jan25",
                "toml_out":     "wflow_sbm_{slug}_ifs_jan25.toml",
                "forcing_dir":  "inmaps_ifs_jan25",
                "timestepsecs": 10800,
            },
            # AIFS ENS — long-lead (init Jan 22)
            "aifs_jan22": {
                "run_dir":      "run_{slug}_aifs_jan22",
                "toml_out":     "wflow_sbm_{slug}_aifs_jan22.toml",
                "forcing_dir":  "inmaps_aifs_jan22",
                "timestepsecs": 10800,
            },
            # AIFS ENS — near-lead (init Jan 25)
            "aifs_jan25": {
                "run_dir":      "run_{slug}_aifs_jan25",
                "toml_out":     "wflow_sbm_{slug}_aifs_jan25.toml",
                "forcing_dir":  "inmaps_aifs_jan25",
                "timestepsecs": 10800,
            },
            # WeatherNext 2 — long-lead (init Jan 22)
            "wn2_jan22": {
                "run_dir":      "run_{slug}_wn2_jan22",
                "toml_out":     "wflow_sbm_{slug}_wn2_jan22.toml",
                "forcing_dir":  "inmaps_wn2_jan22",
                "timestepsecs": 10800,
            },
            # WeatherNext 2 — near-lead (init Jan 25)
            "wn2_jan25": {
                "run_dir":      "run_{slug}_wn2_jan25",
                "toml_out":     "wflow_sbm_{slug}_wn2_jan25.toml",
                "forcing_dir":  "inmaps_wn2_jan25",
                "timestepsecs": 10800,
            },
            # WeatherNext 3 — long lead (Jan 22) and near lead (Jan 25)
            "wn3_jan22": {
                "run_dir":      "run_{slug}_wn3_jan22",
                "toml_out":     "wflow_sbm_{slug}_wn3_jan22.toml",
                "forcing_dir":  "inmaps_wn3_jan22",
                "timestepsecs": 10800,
            },
            "wn3_jan25": {
                "run_dir":      "run_{slug}_wn3_jan25",
                "toml_out":     "wflow_sbm_{slug}_wn3_jan25.toml",
                "forcing_dir":  "inmaps_wn3_jan25",
                "timestepsecs": 10800,
            },
        },
    },

    "texas_2025": {
        # Guadalupe headwaters / Kerr County (July 4 2025 Hill Country flood).
        # Model copied to data/basins/; uses wflow_sbm.toml (not _run) and
        # ERA5-Land daily temp/PET inmaps built by wflow_build_meteo.py.
        "source_model": Path(REPO_ROOT) / "data" / "basins" / "Guadelupe_river",
        # Template with the 5 gauge outputs: the source TOML writes the outlet only,
        # and regenerating the case TOMLs from it dropped gauges 2-5 (Oct 2026).
        # An absolute path: Path / absolute -> the absolute path.
        "source_toml":  Path(REPO_ROOT) / "data" / "wflow" / "texas_2025" / "model" / "wflow_sbm_gauges.toml",
        "spinup_start": "2025-01-01T00:00:00",
        "sim_end":      "2025-07-12T00:00:00",   # shortest forecast horizon (360 h)
        "forcing_years": [2025],
        "cases": {
            "hist_3h": {
                "run_dir":        "run_{slug}_hist_3h",
                "toml_out":       "wflow_sbm_{slug}_hist_3h.toml",
                "forcing_dir":    "inmaps_hist_3h",
                "timestepsecs":   10800,
                "gridded_output": True,   # reference → keep output.nc for peak map
            },
            "era5_jul03": {
                "run_dir":     "run_{slug}_era5_jul03",
                "toml_out":    "wflow_sbm_{slug}_era5_jul03.toml",
                "forcing_dir": "inmaps_era5_jul03",
                "timestepsecs": 10800,
            },
            "ifs_jun27":  {"run_dir": "run_{slug}_ifs_jun27",  "toml_out": "wflow_sbm_{slug}_ifs_jun27.toml",  "forcing_dir": "inmaps_ifs_jun27",  "timestepsecs": 10800},
            "ifs_jul02":  {"run_dir": "run_{slug}_ifs_jul02",  "toml_out": "wflow_sbm_{slug}_ifs_jul02.toml",  "forcing_dir": "inmaps_ifs_jul02",  "timestepsecs": 10800},
            "aifs_jul01": {"run_dir": "run_{slug}_aifs_jul01", "toml_out": "wflow_sbm_{slug}_aifs_jul01.toml", "forcing_dir": "inmaps_aifs_jul01", "timestepsecs": 10800},
            "aifs_jul02": {"run_dir": "run_{slug}_aifs_jul02", "toml_out": "wflow_sbm_{slug}_aifs_jul02.toml", "forcing_dir": "inmaps_aifs_jul02", "timestepsecs": 10800},
            "wn2_jun27":  {"run_dir": "run_{slug}_wn2_jun27",  "toml_out": "wflow_sbm_{slug}_wn2_jun27.toml",  "forcing_dir": "inmaps_wn2_jun27",  "timestepsecs": 10800},
            "wn2_jul02":  {"run_dir": "run_{slug}_wn2_jul02",  "toml_out": "wflow_sbm_{slug}_wn2_jul02.toml",  "forcing_dir": "inmaps_wn2_jul02",  "timestepsecs": 10800},
        },
    },

    "boris_2024": {
        # Upper Elbe (CzechRep9) model. Uses wflow_sbm.toml (Wflow.jl v1, cold start)
        # and ERA5-Land daily temp/PET inmaps built by wflow_build_meteo.py.
        "source_model": Path(REPO_ROOT) / "data" / "basins" / "CzechRep9",
        "source_toml":  "wflow_sbm.toml",
        "spinup_start": "2024-04-01T00:00:00",
        "sim_end":      "2024-09-20T00:00:00",   # shortest horizon (360 h IFS)
        "forcing_years": [2024],
        "cases": {
            "hist_3h": {
                "run_dir":        "run_{slug}_hist_3h",
                "toml_out":       "wflow_sbm_{slug}_hist_3h.toml",
                "forcing_dir":    "inmaps_hist_3h",
                "timestepsecs":   10800,
                "gridded_output": True,   # reference → keep output.nc for peak map
            },
            "era5_sep11": {
                "run_dir":     "run_{slug}_era5_sep11",
                "toml_out":    "wflow_sbm_{slug}_era5_sep11.toml",
                "forcing_dir": "inmaps_era5_sep11",
                "timestepsecs": 10800,
            },
            # IFS ENS — long lead (init Sep 5, ~7 days) and near lead (Sep 10, ~2 days)
            "ifs_sep05": {"run_dir": "run_{slug}_ifs_sep05", "toml_out": "wflow_sbm_{slug}_ifs_sep05.toml", "forcing_dir": "inmaps_ifs_sep05", "timestepsecs": 10800},
            "ifs_sep10": {"run_dir": "run_{slug}_ifs_sep10", "toml_out": "wflow_sbm_{slug}_ifs_sep10.toml", "forcing_dir": "inmaps_ifs_sep10", "timestepsecs": 10800},
            # WeatherNext 2 — long lead (Sep 5) and near lead (Sep 10)
            "wn2_sep05": {"run_dir": "run_{slug}_wn2_sep05", "toml_out": "wflow_sbm_{slug}_wn2_sep05.toml", "forcing_dir": "inmaps_wn2_sep05", "timestepsecs": 10800},
            "wn2_sep10": {"run_dir": "run_{slug}_wn2_sep10", "toml_out": "wflow_sbm_{slug}_wn2_sep10.toml", "forcing_dir": "inmaps_wn2_sep10", "timestepsecs": 10800},
        },
    },

    "gezani_2026": {
        # Uncalibrated Ivondro @ Toamasina model built from scratch by
        # scripts/wflow_build_toamasina.py (lat/lon, Wflow.jl v1, cold start).
        # ERA5-Land daily temp/PET inmaps built by wflow_build_meteo.py.
        "source_model": Path(REPO_ROOT) / "data" / "wflow" / "gezani_2026" / "model",
        "source_toml":  "wflow_sbm.toml",
        "spinup_start": "2025-10-01T00:00:00",
        "sim_end":      "2026-02-18T00:00:00",   # shortest forecast horizon (360 h)
        "forcing_years": [2025, 2026],
        "cases": {
            "hist_3h": {
                "run_dir":        "run_{slug}_hist_3h",
                "toml_out":       "wflow_sbm_{slug}_hist_3h.toml",
                "forcing_dir":    "inmaps_hist_3h",
                "timestepsecs":   10800,
                "gridded_output": True,
            },
            "era5_feb09": {
                "run_dir":     "run_{slug}_era5_feb09",
                "toml_out":    "wflow_sbm_{slug}_era5_feb09.toml",
                "forcing_dir": "inmaps_era5_feb09",
                "timestepsecs": 10800,
            },
            # IFS ENS — long lead (init Feb 3, ~7 days) and near lead (Feb 8, ~2 days)
            "ifs_feb03":  {"run_dir": "run_{slug}_ifs_feb03",  "toml_out": "wflow_sbm_{slug}_ifs_feb03.toml",  "forcing_dir": "inmaps_ifs_feb03",  "timestepsecs": 10800},
            "ifs_feb08":  {"run_dir": "run_{slug}_ifs_feb08",  "toml_out": "wflow_sbm_{slug}_ifs_feb08.toml",  "forcing_dir": "inmaps_ifs_feb08",  "timestepsecs": 10800},
            # AIFS ENS — available for 2026 (long lead Feb 3, near lead Feb 8)
            "aifs_feb03": {"run_dir": "run_{slug}_aifs_feb03", "toml_out": "wflow_sbm_{slug}_aifs_feb03.toml", "forcing_dir": "inmaps_aifs_feb03", "timestepsecs": 10800},
            "aifs_feb08": {"run_dir": "run_{slug}_aifs_feb08", "toml_out": "wflow_sbm_{slug}_aifs_feb08.toml", "forcing_dir": "inmaps_aifs_feb08", "timestepsecs": 10800},
            # WeatherNext 2 — long lead (Feb 3) and near lead (Feb 8)
            "wn2_feb03":  {"run_dir": "run_{slug}_wn2_feb03",  "toml_out": "wflow_sbm_{slug}_wn2_feb03.toml",  "forcing_dir": "inmaps_wn2_feb03",  "timestepsecs": 10800},
            "wn2_feb08":  {"run_dir": "run_{slug}_wn2_feb08",  "toml_out": "wflow_sbm_{slug}_wn2_feb08.toml",  "forcing_dir": "inmaps_wn2_feb08",  "timestepsecs": 10800},
            # WeatherNext 3 — long lead (Feb 3) and near lead (Feb 8)
            "wn3_feb03":  {"run_dir": "run_{slug}_wn3_feb03",  "toml_out": "wflow_sbm_{slug}_wn3_feb03.toml",  "forcing_dir": "inmaps_wn3_feb03",  "timestepsecs": 10800},
            "wn3_feb08":  {"run_dir": "run_{slug}_wn3_feb08",  "toml_out": "wflow_sbm_{slug}_wn3_feb08.toml",  "forcing_dir": "inmaps_wn3_feb08",  "timestepsecs": 10800},
        },
    },

    "california_ar_2023": {
        # Uncalibrated Russian River @ Guerneville model built from scratch by
        # scripts/wflow_build_california.py (lat/lon, Wflow.jl v1, cold start).
        "source_model": Path(REPO_ROOT) / "data" / "wflow" / "california_ar_2023" / "model",
        "source_toml":  "wflow_sbm.toml",
        "spinup_start": "2022-10-01T00:00:00",
        "sim_end":      "2023-01-12T00:00:00",   # shortest forecast horizon
        "forcing_years": [2022, 2023],
        "cases": {
            "hist_3h": {
                "run_dir":        "run_{slug}_hist_3h",
                "toml_out":       "wflow_sbm_{slug}_hist_3h.toml",
                "forcing_dir":    "inmaps_hist_3h",
                "timestepsecs":   10800,
                "gridded_output": True,
            },
            "era5_jan03": {
                "run_dir":     "run_{slug}_era5_jan03",
                "toml_out":    "wflow_sbm_{slug}_era5_jan03.toml",
                "forcing_dir": "inmaps_era5_jan03",
                "timestepsecs": 10800,
            },
            # IFS ENS — long lead (init Dec 28, ~7 days) and near lead (Jan 2, ~2 days)
            "ifs_dec28": {"run_dir": "run_{slug}_ifs_dec28", "toml_out": "wflow_sbm_{slug}_ifs_dec28.toml", "forcing_dir": "inmaps_ifs_dec28", "timestepsecs": 10800},
            "ifs_jan02": {"run_dir": "run_{slug}_ifs_jan02", "toml_out": "wflow_sbm_{slug}_ifs_jan02.toml", "forcing_dir": "inmaps_ifs_jan02", "timestepsecs": 10800},
            # WeatherNext 2 — long lead (Dec 28) and near lead (Jan 2)  [no AIFS for 2023]
            "wn2_dec28": {"run_dir": "run_{slug}_wn2_dec28", "toml_out": "wflow_sbm_{slug}_wn2_dec28.toml", "forcing_dir": "inmaps_wn2_dec28", "timestepsecs": 10800},
            "wn2_jan02": {"run_dir": "run_{slug}_wn2_jan02", "toml_out": "wflow_sbm_{slug}_wn2_jan02.toml", "forcing_dir": "inmaps_wn2_jan02", "timestepsecs": 10800},
        },
    },

    "bc_ar_2024": {
        # Uncalibrated Chilliwack River @ Vedder Crossing model built from scratch
        # by scripts/wflow_build_chilliwack.py (lat/lon, Wflow.jl v1, cold start).
        "source_model": Path(REPO_ROOT) / "data" / "wflow" / "bc_ar_2024" / "model",
        "source_toml":  "wflow_sbm.toml",
        # Full water year: the event is 18 Oct and the BC water year starts 1 Oct,
        # so an April cold start would give bone-dry post-summer soils and miss the
        # early-October re-wetting that primes the AR response.
        "spinup_start": "2023-10-01T00:00:00",
        "sim_end":      "2024-10-26T00:00:00",   # ifs_oct11 + 360 h
        "forcing_years": [2023, 2024],
        "cases": {
            "hist_3h": {
                "run_dir":        "run_{slug}_hist_3h",
                "toml_out":       "wflow_sbm_{slug}_hist_3h.toml",
                "forcing_dir":    "inmaps_hist_3h",
                "timestepsecs":   10800,
                "gridded_output": True,
            },
            "era5_oct16": {
                "run_dir":     "run_{slug}_era5_oct16",
                "toml_out":    "wflow_sbm_{slug}_era5_oct16.toml",
                "forcing_dir": "inmaps_era5_oct16",
                "timestepsecs": 10800,
            },
            # IFS ENS — long lead (init Oct 11, ~7 days) and near lead (Oct 16, ~2 days)
            "ifs_oct11": {"run_dir": "run_{slug}_ifs_oct11", "toml_out": "wflow_sbm_{slug}_ifs_oct11.toml", "forcing_dir": "inmaps_ifs_oct11", "timestepsecs": 10800},
            "ifs_oct16": {"run_dir": "run_{slug}_ifs_oct16", "toml_out": "wflow_sbm_{slug}_ifs_oct16.toml", "forcing_dir": "inmaps_ifs_oct16", "timestepsecs": 10800},
            # WeatherNext 2 — long lead (Oct 11) and near lead (Oct 16)  [no AIFS for 2024]
            "wn2_oct11": {"run_dir": "run_{slug}_wn2_oct11", "toml_out": "wflow_sbm_{slug}_wn2_oct11.toml", "forcing_dir": "inmaps_wn2_oct11", "timestepsecs": 10800},
            "wn2_oct16": {"run_dir": "run_{slug}_wn2_oct16", "toml_out": "wflow_sbm_{slug}_wn2_oct16.toml", "forcing_dir": "inmaps_wn2_oct16", "timestepsecs": 10800},
        },
    },

    "zermatt_2024": {
        # Uncalibrated Vispa @ Visp model built from scratch by
        # scripts/wflow_build_vispa.py (lat/lon, Wflow.jl v1, glaciers ON).
        "source_model": Path(REPO_ROOT) / "data" / "wflow" / "zermatt_2024" / "model",
        "source_toml":  "wflow_sbm.toml",
        # Two accumulation seasons: an April 2024 cold start would carry NO winter
        # snowpack, removing the June melt contribution entirely. Glacier ice is
        # initialised separately (glacier_initial_leq_depth); it is the seasonal
        # pack that needs the runway.
        "spinup_start": "2022-10-01T00:00:00",
        "sim_end":      "2024-07-03T00:00:00",   # 3 d after the 30 Jun record peak; ifs_jun15 ends 30 Jun
        "forcing_years": [2022, 2023, 2024],
        "cases": {
            "hist_3h": {
                "run_dir":        "run_{slug}_hist_3h",
                "toml_out":       "wflow_sbm_{slug}_hist_3h.toml",
                "forcing_dir":    "inmaps_hist_3h",
                "timestepsecs":   10800,
                "gridded_output": True,
            },
            "era5_jun19": {
                "run_dir":     "run_{slug}_era5_jun19",
                "toml_out":    "wflow_sbm_{slug}_era5_jun19.toml",
                "forcing_dir": "inmaps_era5_jun19",
                "timestepsecs": 10800,
            },
            # IFS ENS — long lead (init Jun 15, 15-day) and near lead (Jun 19)
            "ifs_jun15": {"run_dir": "run_{slug}_ifs_jun15", "toml_out": "wflow_sbm_{slug}_ifs_jun15.toml", "forcing_dir": "inmaps_ifs_jun15", "timestepsecs": 10800},
            "ifs_jun19": {"run_dir": "run_{slug}_ifs_jun19", "toml_out": "wflow_sbm_{slug}_ifs_jun19.toml", "forcing_dir": "inmaps_ifs_jun19", "timestepsecs": 10800},
            # WeatherNext 2 — long lead (Jun 15) and near lead (Jun 19)  [no AIFS for 2024]
            "wn2_jun15": {"run_dir": "run_{slug}_wn2_jun15", "toml_out": "wflow_sbm_{slug}_wn2_jun15.toml", "forcing_dir": "inmaps_wn2_jun15", "timestepsecs": 10800},
            "wn2_jun19": {"run_dir": "run_{slug}_wn2_jun19", "toml_out": "wflow_sbm_{slug}_wn2_jun19.toml", "forcing_dir": "inmaps_wn2_jun19", "timestepsecs": 10800},
        },
    },

}

# DANA starts -7/-2/-1 d (20 Oct 2026, see wflow_prep_forcing.py); 25 Oct dropped
WFLOW_CONFIG["dana_2024"]["sim_end"] = "2024-11-06T00:00:00"
for _m in ("ifs", "wn2"):
    for _tag in ("oct22", "oct28"):
        WFLOW_CONFIG["dana_2024"]["cases"][f"{_m}_{_tag}"] = {
            "run_dir": f"run_{{slug}}_{_m}_{_tag}", "toml_out": f"wflow_sbm_{{slug}}_{_m}_{_tag}.toml",
            "forcing_dir": f"inmaps_{_m}_{_tag}", "timestepsecs": 10800}
    WFLOW_CONFIG["dana_2024"]["cases"].pop(f"{_m}_oct25")

# Kristin reference-rain sensitivity runs (see wflow_prep_forcing.py)
for _tag in ("era5ref_jan26", "era5scaled_jan26"):
    WFLOW_CONFIG["kristin_2026"]["cases"][_tag] = {
        "run_dir": f"run_{{slug}}_{_tag}", "toml_out": f"wflow_sbm_{{slug}}_{_tag}.toml",
        "forcing_dir": f"inmaps_{_tag}", "timestepsecs": 10800}

# Kristin -7 d start (3 Nov 2026; see wflow_prep_forcing.py): copies of the jan22 cases
for _m in ("ifs", "aifs", "wn2"):
    WFLOW_CONFIG["kristin_2026"]["cases"][f"{_m}_jan20"] = {
        k: (v.replace("jan22", "jan20") if isinstance(v, str) else v)
        for k, v in WFLOW_CONFIG["kristin_2026"]["cases"][f"{_m}_jan22"].items()}

# Clipped sub-basins of kristin_2026: the model already exists (created by
# wflow_clip.py), so source_model points at the clipped dir itself — the static
# copy is skipped (dst already populated) and the clipped wflow_sbm_run.toml is
# used as the patch template. Spinup/sim/cases are inherited from kristin_2026.
for _sub in ("kristin_mondego", "kristin_lis"):
    WFLOW_CONFIG[_sub] = {
        **WFLOW_CONFIG["kristin_2026"],
        "source_model": REPO_ROOT / "data" / "wflow" / _sub / "model",
    }

# Jeseníky / Kłodzko model (wflow_build_sudetes.py): built in place, so
# source_model is its own dir; cases and periods as boris_2024.
WFLOW_CONFIG["boris_sudetes"] = {
    **WFLOW_CONFIG["boris_2024"],
    "source_model": REPO_ROOT / "data" / "wflow" / "boris_sudetes" / "model",
    "source_toml":  "wflow_sbm.toml",
    # own case list: the -1 d start (2026-10-15) is not run for the Elbe model
    "cases": {
        **WFLOW_CONFIG["boris_2024"]["cases"],
        "ifs_sep11": {"run_dir": "run_{slug}_ifs_sep11", "toml_out": "wflow_sbm_{slug}_ifs_sep11.toml", "forcing_dir": "inmaps_ifs_sep11", "timestepsecs": 10800},
        "wn2_sep11": {"run_dir": "run_{slug}_wn2_sep11", "toml_out": "wflow_sbm_{slug}_wn2_sep11.toml", "forcing_dir": "inmaps_wn2_sep11", "timestepsecs": 10800},
    },
}

# Catalonia (wflow_build_catalonia.py): Besos + Tordera + Maresme rieres, built in
# place. Reference rain is the XEMA gauge grid (wflow_prep_forcing.py).
WFLOW_CONFIG["catalonia_2026"] = {
    "source_model": REPO_ROOT / "data" / "wflow" / "catalonia_2026" / "model",
    "source_toml":  "wflow_sbm.toml",
    "spinup_start": "2026-04-01T00:00:00",
    "sim_end":      "2026-10-07T21:00:00",   # XEMA gap on Oct 8 (see wflow_prep_forcing.py)
    "forcing_years": [2026],
    "cases": {
        "hist_3h": {
            "run_dir":        "run_{slug}_hist_3h",
            "toml_out":       "wflow_sbm_{slug}_hist_3h.toml",
            "forcing_dir":    "inmaps_hist_3h",
            "timestepsecs":   10800,
            "gridded_output": True,
        },
        "era5_oct01": {"run_dir": "run_{slug}_era5_oct01", "toml_out": "wflow_sbm_{slug}_era5_oct01.toml", "forcing_dir": "inmaps_era5_oct01", "timestepsecs": 10800},
        # long lead (Sep 26, ~7 days) and near lead (Oct 1, ~2 days)
        "ifs_sep26": {"run_dir": "run_{slug}_ifs_sep26", "toml_out": "wflow_sbm_{slug}_ifs_sep26.toml", "forcing_dir": "inmaps_ifs_sep26", "timestepsecs": 10800},
        "ifs_oct01": {"run_dir": "run_{slug}_ifs_oct01", "toml_out": "wflow_sbm_{slug}_ifs_oct01.toml", "forcing_dir": "inmaps_ifs_oct01", "timestepsecs": 10800},
        "aifs_sep26": {"run_dir": "run_{slug}_aifs_sep26", "toml_out": "wflow_sbm_{slug}_aifs_sep26.toml", "forcing_dir": "inmaps_aifs_sep26", "timestepsecs": 10800},
        "aifs_oct01": {"run_dir": "run_{slug}_aifs_oct01", "toml_out": "wflow_sbm_{slug}_aifs_oct01.toml", "forcing_dir": "inmaps_aifs_oct01", "timestepsecs": 10800},
        "wn2_sep26": {"run_dir": "run_{slug}_wn2_sep26", "toml_out": "wflow_sbm_{slug}_wn2_sep26.toml", "forcing_dir": "inmaps_wn2_sep26", "timestepsecs": 10800},
        "wn2_oct01": {"run_dir": "run_{slug}_wn2_oct01", "toml_out": "wflow_sbm_{slug}_wn2_oct01.toml", "forcing_dir": "inmaps_wn2_oct01", "timestepsecs": 10800},
    },
}

# California 2023 impact catchments (wflow_build_california_2023.py), built in place.
WFLOW_CONFIG["california_2023"] = {
    "source_model": REPO_ROOT / "data" / "wflow" / "california_2023" / "model",
    "source_toml":  "wflow_sbm.toml",
    "spinup_start": "2022-07-01T00:00:00",   # 1 Oct before 18 Nov 2026 (spin-up test)
    "sim_end":      "2023-01-14T00:00:00",
    "forcing_years": [2022, 2023],
    "cases": {
        "hist_3h": {"run_dir": "run_{slug}_hist_3h", "toml_out": "wflow_sbm_{slug}_hist_3h.toml", "forcing_dir": "inmaps_hist_3h", "timestepsecs": 10800, "gridded_output": True},
        "era5_dec24": {"run_dir": "run_{slug}_era5_dec24", "toml_out": "wflow_sbm_{slug}_era5_dec24.toml", "forcing_dir": "inmaps_era5_dec24", "timestepsecs": 10800},
        "ifs_dec24": {"run_dir": "run_{slug}_ifs_dec24", "toml_out": "wflow_sbm_{slug}_ifs_dec24.toml", "forcing_dir": "inmaps_ifs_dec24", "timestepsecs": 10800},
        "ifs_dec29": {"run_dir": "run_{slug}_ifs_dec29", "toml_out": "wflow_sbm_{slug}_ifs_dec29.toml", "forcing_dir": "inmaps_ifs_dec29", "timestepsecs": 10800},
        "ifs_jan02": {"run_dir": "run_{slug}_ifs_jan02", "toml_out": "wflow_sbm_{slug}_ifs_jan02.toml", "forcing_dir": "inmaps_ifs_jan02", "timestepsecs": 10800},
        "ifs_jan07": {"run_dir": "run_{slug}_ifs_jan07", "toml_out": "wflow_sbm_{slug}_ifs_jan07.toml", "forcing_dir": "inmaps_ifs_jan07", "timestepsecs": 10800},
        "wn2_dec24": {"run_dir": "run_{slug}_wn2_dec24", "toml_out": "wflow_sbm_{slug}_wn2_dec24.toml", "forcing_dir": "inmaps_wn2_dec24", "timestepsecs": 10800},
        "wn2_dec29": {"run_dir": "run_{slug}_wn2_dec29", "toml_out": "wflow_sbm_{slug}_wn2_dec29.toml", "forcing_dir": "inmaps_wn2_dec29", "timestepsecs": 10800},
        "wn2_jan02": {"run_dir": "run_{slug}_wn2_jan02", "toml_out": "wflow_sbm_{slug}_wn2_jan02.toml", "forcing_dir": "inmaps_wn2_jan02", "timestepsecs": 10800},
        "wn2_jan07": {"run_dir": "run_{slug}_wn2_jan07", "toml_out": "wflow_sbm_{slug}_wn2_jan07.toml", "forcing_dir": "inmaps_wn2_jan07", "timestepsecs": 10800},
    },
}

# Helene 2024 (wflow_build_helene.py): French Broad, South Toe, Cataloochee,
# Catawba headwaters, built in place. Reference rain is MRMS (wflow_prep_forcing.py).
WFLOW_CONFIG["helene_2024"] = {
    "source_model": REPO_ROOT / "data" / "wflow" / "helene_2024" / "model",
    "source_toml":  "wflow_sbm.toml",
    "spinup_start": "2024-04-01T00:00:00",
    "sim_end":      "2024-10-03T00:00:00",
    "forcing_years": [2024],
    "cases": {
        "hist_3h": {"run_dir": "run_{slug}_hist_3h", "toml_out": "wflow_sbm_{slug}_hist_3h.toml", "forcing_dir": "inmaps_hist_3h", "timestepsecs": 10800, "gridded_output": True},
        "era5_sep18": {"run_dir": "run_{slug}_era5_sep18", "toml_out": "wflow_sbm_{slug}_era5_sep18.toml", "forcing_dir": "inmaps_era5_sep18", "timestepsecs": 10800},
        "ifs_sep18": {"run_dir": "run_{slug}_ifs_sep18", "toml_out": "wflow_sbm_{slug}_ifs_sep18.toml", "forcing_dir": "inmaps_ifs_sep18", "timestepsecs": 10800},
        "ifs_sep23": {"run_dir": "run_{slug}_ifs_sep23", "toml_out": "wflow_sbm_{slug}_ifs_sep23.toml", "forcing_dir": "inmaps_ifs_sep23", "timestepsecs": 10800},
        "wn2_sep18": {"run_dir": "run_{slug}_wn2_sep18", "toml_out": "wflow_sbm_{slug}_wn2_sep18.toml", "forcing_dir": "inmaps_wn2_sep18", "timestepsecs": 10800},
        "wn2_sep23": {"run_dir": "run_{slug}_wn2_sep23", "toml_out": "wflow_sbm_{slug}_wn2_sep23.toml", "forcing_dir": "inmaps_wn2_sep23", "timestepsecs": 10800},
    },
}

# -1 d start (see MINUS_ONE_DAY in wflow_prep_forcing.py): copies of the
# existing cases of each model for the new tag.
for _ev, _tag, _tpl in (("catalonia_2026", "oct02", "oct01"), ("texas_2025", "jul03", "jul02"),
                        ("california_2023", "dec30", "dec29"), ("california_2023", "jan08", "jan07"),
                        ("helene_2024", "sep24", "sep23"), ("zermatt_2024", "jun20", "jun19"),
                        ("kristin_2026", "jan26", "jan25")):
    _cases = WFLOW_CONFIG[_ev]["cases"]
    for _m in ("ifs", "aifs", "wn2"):
        if f"{_m}_{_tpl}" in _cases:
            _cases[f"{_m}_{_tag}"] = {k: (v.replace(_tpl, _tag) if isinstance(v, str) else v)
                                      for k, v in _cases[f"{_m}_{_tpl}"].items()}

# Zermatt second flood (18 Nov 2026; see wflow_prep_forcing.py): 22/27/28 Jun starts, copies of jun19
for _tag in ("jun22", "jun27", "jun28"):
    for _m in ("ifs", "wn2"):
        WFLOW_CONFIG["zermatt_2024"]["cases"][f"{_m}_{_tag}"] = {
            k: (v.replace("jun19", _tag) if isinstance(v, str) else v)
            for k, v in WFLOW_CONFIG["zermatt_2024"]["cases"][f"{_m}_jun19"].items()}

# AIFS-era events (18 Oct 2026): built in place; cases as wflow_prep_forcing._aifs_era_event
def _aifs_era_setup(slug, sim_end, tags, era5_tag):
    cases = {"hist_3h": {"run_dir": "run_{slug}_hist_3h", "toml_out": "wflow_sbm_{slug}_hist_3h.toml",
                         "forcing_dir": "inmaps_hist_3h", "timestepsecs": 10800, "gridded_output": True}}
    for c in [f"era5_{era5_tag}"] + [f"{m}_{t}" for m in ("ifs", "aifs", "wn2") for t in tags]:
        cases[c] = {"run_dir": "run_{slug}_" + c, "toml_out": "wflow_sbm_{slug}_" + c + ".toml",
                    "forcing_dir": "inmaps_" + c, "timestepsecs": 10800}
    return {"source_model": REPO_ROOT / "data" / "wflow" / slug / "model", "source_toml": "wflow_sbm.toml",
            # spin-up from 1 Oct 2025 since 18 Nov 2026 (was 1 Apr 2026, 2.5-3.4 months; spin-up test,
            # scripts/spinup_test.py: peaks and volumes within 2 %), as California from 1 Jul 2022
            "spinup_start": "2025-10-01T00:00:00", "sim_end": sim_end, "forcing_years": [2025, 2026], "cases": cases}


WFLOW_CONFIG["texas_2026"] = _aifs_era_setup("texas_2026", "2026-07-19T00:00:00", ["jul07", "jul12", "jul13"], "jul12")
WFLOW_CONFIG["mississippi_2026"] = _aifs_era_setup("mississippi_2026", "2026-06-22T00:00:00", ["jun10", "jun15", "jun16"], "jun15")

# Spin-up length test (18 Nov 2026; see wflow_prep_forcing.py SPIN_TEST): reference run of
# three events from an earlier start, in data/wflow/<slug>_spin (static files copied by hand).
for _ev, _start in (("mississippi_2026", "2025-10-01"), ("texas_2026", "2025-10-01"), ("california_2023", "2022-07-01")):
    _c = dict(WFLOW_CONFIG[_ev])
    _c.update(spinup_start=f"{_start}T00:00:00",
              forcing_years=list(range(int(_start[:4]), int(_c["sim_end"][:4]) + 1)),
              cases={"hist_3h": WFLOW_CONFIG[_ev]["cases"]["hist_3h"]})
    WFLOW_CONFIG[f"{_ev}_spin"] = _c

# Files/dirs to skip when copying static model content
SKIP_NAMES = {
    "inmaps", "inmaps_era5", "outstate", "instate",
    "hydromt.log", "hydromt.log.tmp",
}


def strip_gridded_output(content: str) -> str:
    """Remove the [output.netcdf_grid] sections so wflow writes no gridded
    output.nc (~1.4 GB/run). The discharge output.csv (tiny) is kept and drives
    all comparison plots; gridded output is only needed for spatial peak maps,
    which we generate from the reference case alone.
    """
    out, skip = [], False
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            sec  = stripped.strip("[]")
            skip = sec.startswith("output.netcdf_grid")
        if not skip:
            out.append(line)
    return "".join(out)


NETCDF_GRID_BLOCK = (
    '[output.netcdf_grid]\n'
    'path = "output.nc"\n'
    'compressionlevel = 1\n'
    '\n'
    '[output.netcdf_grid.variables]\n'
    'river_water__volume_flow_rate = "river_q"\n'
    '\n'
)


def ensure_gridded_output(content: str) -> str:
    """Inverse of strip_gridded_output: add the section if it is absent.

    Needed because models built from scratch by wflow_build_*.py never got an
    [output.netcdf_grid] section (hydromt only writes [output.csv]). Without
    this, asking for gridded_output on such a model silently produced no
    output.nc and the spatial peak-discharge map could not be made — the
    working TOMLs had been hand-patched instead.
    """
    if "[output.netcdf_grid]" in content:
        return content
    i = content.find("[output.csv]")
    if i < 0:
        return content.rstrip() + "\n\n" + NETCDF_GRID_BLOCK
    return content[:i] + NETCDF_GRID_BLOCK + content[i:]


def patch_toml(content: str, slug: str, case_cfg: dict, wflow_cfg: dict) -> str:
    """Patch specific TOML keys using regex line replacement."""
    run_dir     = case_cfg["run_dir"].format(slug=slug)
    forcing_dir = case_cfg["forcing_dir"]
    timestep    = case_cfg["timestepsecs"]

    patches = {
        r"^(dir_output\s*=\s*).*$":     f'\\g<1>"{run_dir}"',
        r"^(starttime\s*=\s*).*$":       f'\\g<1>{wflow_cfg["spinup_start"]}',
        r"^(endtime\s*=\s*).*$":         f'\\g<1>{wflow_cfg["sim_end"]}',
        r"^(timestepsecs\s*=\s*).*$":    f'\\g<1>{timestep}',
        r"^(path_forcing\s*=\s*).*$":    f'\\g<1>"{forcing_dir}/*.nc"',
    }
    for pattern, replacement in patches.items():
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # fews_run__flag = true (set in the operational Drammen model) makes Wflow start
    # one step AFTER `starttime`: a restart then skips a step, and the run no longer
    # starts where every other model starts. Always off for our runs.
    content = re.sub(r"^(fews_run__flag\s*=\s*).*$", r"\g<1>false", content, flags=re.MULTILINE)

    # Gridded output.nc only for cases that explicitly request it (reference
    # case used for spatial peak maps); all others write discharge CSV only.
    if case_cfg.get("gridded_output", False):
        content = ensure_gridded_output(content)
    else:
        content = strip_gridded_output(content)
    return content


def main():
    parser = argparse.ArgumentParser(description="Set up wflow model for AIFS event")
    parser.add_argument("event_slug", help="Event slug, e.g. dana_2024")
    parser.add_argument("case",       help="Case name, e.g. hist_3h")
    args = parser.parse_args()

    slug = args.event_slug
    case = args.case

    if slug not in WFLOW_CONFIG:
        sys.exit(
            f"No wflow config for event '{slug}'.\n"
            f"Available: {list(WFLOW_CONFIG)}"
        )

    wflow_cfg = WFLOW_CONFIG[slug]

    if case not in wflow_cfg["cases"]:
        sys.exit(
            f"Unknown case '{case}' for event '{slug}'.\n"
            f"Available: {list(wflow_cfg['cases'])}"
        )

    case_cfg = wflow_cfg["cases"][case]
    src = wflow_cfg["source_model"]
    dst = REPO_ROOT / "data" / "wflow" / slug / "model"

    if not src.exists():
        sys.exit(f"Source model not found:\n  {src}")

    print(f"Event  : {slug}")
    print(f"Case   : {case}")
    print(f"Source : {src}")
    print(f"Dest   : {dst}")

    # ── Copy static model files ───────────────────────────────────────────────
    if dst.exists() and any(dst.iterdir()):
        print("Model dir already populated — skipping static file copy.")
        # The TOML is ALWAYS regenerated from the source below, but staticmaps is
        # not — so a source model that gets recalibrated leaves the working copy
        # with a new TOML pointing at maps its old grid does not contain. That
        # exact mismatch broke texas_2025 (source gained an RF_250 ksathorfrac
        # map; the working staticmaps predated it) with an opaque Julia error:
        # "Collection has multiple elements, must contain exactly 1 element".
        src_sm, dst_sm = src / "staticmaps.nc", dst / "staticmaps.nc"
        if (src_sm.exists() and dst_sm.exists()
                and not src_sm.samefile(dst_sm)
                and src_sm.stat().st_mtime > dst_sm.stat().st_mtime + 1):
            print(f"\n  [WARN] {src_sm} is NEWER than the working copy\n"
                  f"         {dst_sm}\n"
                  f"         The regenerated TOML comes from the source but the grid\n"
                  f"         does not. If wflow fails on a missing parameter, refresh\n"
                  f"         the working staticmaps.nc from the source.\n")
    else:
        dst.mkdir(parents=True, exist_ok=True)
        print("Copying static model files...")
        for item in sorted(src.iterdir()):
            if item.name in SKIP_NAMES or item.name.startswith("run_"):
                continue
            dest_item = dst / item.name
            if item.is_dir():
                shutil.copytree(item, dest_item)
            else:
                shutil.copy2(item, dest_item)
            print(f"  {item.name}")

    # ── Copy relevant forcing (inmaps) files ─────────────────────────────────
    # For the daily 'hist' case: copy MSWEP inmaps from source archive.
    # For 3h cases: forcing is built by wflow_prep_forcing.py — just check it exists.
    forcing_dir = case_cfg["forcing_dir"]

    if forcing_dir == "inmaps":
        src_inmaps = src / "inmaps"
        dst_inmaps = dst / "inmaps"
        dst_inmaps.mkdir(exist_ok=True)

        forcing_years = set(wflow_cfg["forcing_years"])
        copied: list[str] = []

        for nc in sorted(src_inmaps.glob("*.nc")):
            date_part = nc.stem.split("_")[-1]
            try:
                year = int(date_part[:4])
            except (ValueError, IndexError):
                continue
            if year not in forcing_years:
                continue
            dest_nc = dst_inmaps / nc.name
            if dest_nc.exists():
                print(f"  Forcing already present : {nc.name}")
            else:
                print(f"  Copying forcing         : {nc.name}")
                shutil.copy2(nc, dest_nc)
            copied.append(nc.name)

        if not copied:
            sys.exit(
                f"No forcing files found for years {sorted(forcing_years)}\n"
                f"in {src_inmaps}"
            )
        print(f"  Forcing files ready: {len(copied)} files")

    else:
        # 3h case: forcing built externally by wflow_prep_forcing.py.
        # Also ensure the daily inmaps are present (needed for grid coords + temp/PET).
        dst_inmaps = dst / "inmaps"
        if not (dst_inmaps.exists() and any(dst_inmaps.glob("*.nc"))):
            print("  [INFO] Copying daily inmaps for grid coords + temp/PET…")
            src_inmaps = src / "inmaps"
            dst_inmaps.mkdir(exist_ok=True)
            forcing_years = set(wflow_cfg["forcing_years"])
            for nc in sorted(src_inmaps.glob("*.nc")):
                date_part = nc.stem.split("_")[-1]
                try:
                    year = int(date_part[:4])
                except (ValueError, IndexError):
                    continue
                if year not in forcing_years:
                    continue
                dest_nc = dst_inmaps / nc.name
                if not dest_nc.exists():
                    shutil.copy2(nc, dest_nc)
            nc_count = len(list(dst_inmaps.glob("*.nc")))
            print(f"  Daily inmaps ready: {nc_count} files")

        dst_forcing = dst / forcing_dir
        if dst_forcing.exists() and any(dst_forcing.glob("*.nc")):
            nc_count = len(list(dst_forcing.glob("*.nc")))
            print(f"  3h forcing dir ready: {forcing_dir}/ ({nc_count} files)")
        else:
            print(f"  [INFO] Forcing dir not yet populated: {dst_forcing}")
            print(f"         Run: python scripts/wflow_prep_forcing.py {slug} {case}")
            print(f"         Then re-run this setup script to write the TOML.")

    # ── Patch and write event TOML ────────────────────────────────────────────
    src_toml  = src / wflow_cfg.get("source_toml", "wflow_sbm_run.toml")
    toml_name = case_cfg["toml_out"].format(slug=slug)
    dst_toml  = dst / toml_name

    content = src_toml.read_text()
    content = patch_toml(content, slug, case_cfg, wflow_cfg)
    dst_toml.write_text(content)
    print(f"\nTOML  : {dst_toml}")
    print(f"\nSetup complete.")
    print(f"Next  : python scripts/wflow_run.py {slug} {case}")


if __name__ == "__main__":
    main()
