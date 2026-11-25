"""
Central event configuration for the precipitation forecast verification pipeline.

To switch events, change ACTIVE_EVENT to the desired event slug.
All download, preprocessing, and analysis scripts import `cfg` from here.

Usage:
    from event_config import cfg
    # Then use cfg["lat_min"], cfg["event_start"], etc.

To add a new event, add an entry to the EVENTS dict following the same schema.
"""

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
import os

# ── Select active event ─────────────────────────────────────────────────────
ACTIVE_EVENT = "boris_2024"   # Options: "kristin_2026" | "dana_2024" | "texas_2025" | "gezani_2026" | "boris_2024" | "hans_2023" | "california_ar_2023" | "bc_ar_2024" | "zermatt_2024" | "catalonia_2026"

# Optional per-process override (lets several events download in parallel without
# editing the line above). Set env var AIFS_EVENT=<slug> to override.
ACTIVE_EVENT = os.environ.get("AIFS_EVENT", ACTIVE_EVENT)

# ── Event registry ──────────────────────────────────────────────────────────
EVENTS = {

    "kristin_2026": {
        # Human-readable name and filesystem slug
        "name":  "Storm Kristin",
        "slug":  "kristin_2026",

        # Spatial bounding box (degrees, -180..180 longitude convention)
        "lat_min":  37.67,
        "lat_max":  47.41,
        "lon_min":  -9.65,
        "lon_max":   0.46,

        # Forecast initialisation dates (YYYY-MM-DD); 00Z and 12Z are both downloaded.
        # 20-21 Jan added 3 Nov 2026 for a -7 d start like the other events (the pilot
        # download began on 22 Jan, which gave Kristin -5 d and no >=5.5 d rain leads).
        "init_dates": [
            "2026-01-20", "2026-01-21",
            "2026-01-22", "2026-01-23", "2026-01-24",
            "2026-01-25", "2026-01-26",
        ],

        # Verification/event window (used for precipitation totals and skill metrics)
        "event_start": "2026-01-27T00:00",
        "event_end":   "2026-01-31T23:59:59",

        # Broader download window for ERA5 / MSWEP (extra context days)
        "ref_start": "2026-01-26",
        "ref_end":   "2026-02-07",

        # ERA5 CDS request date fields
        "era5_years":  ["2026"],
        "era5_months": ["01", "02"],
        "era5_days":   ["26", "27", "28", "29", "30", "31",
                        "01", "02", "03", "04", "05", "06", "07"],

        # MSWEP archive sub-directory: "NRT" for recent data, "Past" for older
        "mswep_subdir": "NRT",

        # Point of interest for point-based timeseries figure
        "poi_lat":  40.21,
        "poi_lon":  -8.43,
        "poi_name": "Coimbra",

        # Which forecast models are available
        "has_aifs": True,

        # Basin geometry for basin-averaged analysis
        "basin_file": _ROOT + "/data/basins/pfafstetter_Europe_12000uparea_295311.gpkg",
        "basin_name": "Tagus/Mondego Basin",

        # Maximum lead time for MARS step string (hours)
        "max_lead_hours": 360,

        # Init time used for single-init spatial figures (earliest = max lead time)
        "best_init": "2026-01-22T00:00",   # Maximum lead time init (~5 days before)
        "near_init": "2026-01-25T00:00",   # Near-event init (~2 days before)

        # Precipitation exceedance and FSS thresholds (mm event-total)
        "exceed_thresholds": [50, 100, 150],
        "fss_thresholds":    [50, 100, 150],
        "fss_scales_km":     [25, 50, 100, 200],
    },

    "dana_2024": {
        "name":  "Valencia DANA Floods",
        "slug":  "dana_2024",

        "lat_min":  38.191,
        "lat_max":  41.135,
        "lon_min":  -2.5371,
        "lon_max":   1.4314,

        "init_dates": [
            "2024-10-22", "2024-10-23", "2024-10-24",
            "2024-10-25", "2024-10-26", "2024-10-27", "2024-10-28",
        ],

        "event_start": "2024-10-29T00:00",
        "event_end":   "2024-10-31T23:59:59",

        # Extended to 11-10: with IFS/WN2 now pulled to the full 15 days the
        # forecasts reach 2024-11-09, and we do not want the ERA5 reference (a
        # skill-ceiling line, not the scored reference) to cap the whole event's
        # simulation window at 11-03.
        "ref_start": "2024-10-28",
        "ref_end":   "2024-11-10",

        "era5_years":  ["2024"],
        "era5_months": ["10", "11"],
        "era5_days":   ["28", "29", "30", "31", "01", "02", "03",
                        "04", "05", "06", "07", "08", "09", "10"],

        "mswep_subdir": "Past",   # Oct 2024 is in the historical archive

        "poi_lat":  39.47,
        "poi_lon":  -0.38,
        "poi_name": "Valencia",

        "has_aifs": False,   # AIFS (class=ai) not available for 2024

        # Basin geometry for basin-averaged analysis
        "basin_file": _ROOT + "/data/basins/pfafstetter_Europe_12000uparea_291741.gpkg",
        "basin_name": "Júcar Basin (Valencia)",

        "max_lead_hours": 360,   # 15 days: IFS/AIFS both archive it (see kristin_2026)

        "best_init": "2024-10-22T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2024-10-27T00:00",   # Near-event init (~2 days before)

        "exceed_thresholds": [25, 50, 100],
        "fss_thresholds":    [25, 50, 100],
        "fss_scales_km":     [10, 25, 50, 100],
    },

    "texas_2025": {
        "name":  "Central Texas Floods",
        "slug":  "texas_2025",

        # Bounding box covering the Hill Country + broader 20-county disaster area
        "lat_min":  29.5,
        "lat_max":  31.5,
        "lon_min": -100.5,
        "lon_max":  -97.5,

        # 00Z inits from 7 days before event onset (Jul 4) through day before
        "init_dates": [
            "2025-06-27", "2025-06-28", "2025-06-29",
            "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03",
        ],

        # Core event: Jul 4–7 (catastrophic rainfall; secondary event Jul 14 excluded)
        "event_start": "2025-07-04T00:00",
        "event_end":   "2025-07-07T23:59:59",

        # Broader download window for ERA5 / MSWEP
        "ref_start": "2025-07-03",
        "ref_end":   "2025-07-14",

        # ERA5 CDS request date fields
        "era5_years":  ["2025"],
        "era5_months": ["07"],
        "era5_days":   ["03", "04", "05", "06", "07", "08", "09",
                        "10", "11", "12", "13", "14"],

        # Jul 2025 is NRT
        "mswep_subdir": "NRT",

        # Kerrville, TX — epicentre of flooding (Kerr County, 117+ fatalities)
        "poi_lat":  30.047,
        "poi_lon":  -99.140,
        "poi_name": "Kerrville",

        "has_aifs": True,
        "aifs_available_from": "2025-07-01T12",  # First available AIFS init in MARS (00Z not archived)

        # Basin geometry — copy meta_basins_highres.geojson to this path manually
        "basin_file": _ROOT + "/data/basins/meta_basins_highres.geojson",
        "basin_name": "Guadalupe River Basin",

        "max_lead_hours": 360,   # 15 days — the full IFS/AIFS ENS horizon.
        # Was 240 h, which is shorter than what MARS actually archives and left
        # WeatherNext 2 (15 d) as the only product with data late in the window.

        "best_init": "2025-06-27T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2025-07-02T00:00",   # Near-event init (~2 days before)

        # Lower thresholds than Kristin — convective flash-flood regime
        "exceed_thresholds": [50, 100, 200],
        "fss_thresholds":    [50, 100, 200],
        "fss_scales_km":     [10, 25, 50, 100],
    },

    "gezani_2026": {
        "name":  "Cyclone Gezani",
        "slug":  "gezani_2026",

        # Bounding box focused on the broader Toamasina coastal-basin area of E.
        # Madagascar (Ivondro + Rianila + smaller coastal rivers), headwaters in
        # the eastern highlands (~47°E) plus ~0.7° buffer (incl. offshore/east to
        # capture the approaching cyclone)
        "lat_min": -19.8,
        "lat_max": -16.8,
        "lon_min":  46.8,
        "lon_max":  50.3,

        # 00Z inits 7 days before Madagascar landfall (Feb 10), through Feb 12
        "init_dates": [
            "2026-02-03", "2026-02-04", "2026-02-05",
            "2026-02-06", "2026-02-07", "2026-02-08", "2026-02-09",
        ],

        # Event window: Madagascar landfall through Mozambique passage
        "event_start": "2026-02-10T00:00",
        "event_end":   "2026-02-17T23:59:59",

        # Broader download window for ERA5 / MSWEP
        "ref_start": "2026-02-09",
        "ref_end":   "2026-02-20",

        # ERA5 CDS request date fields
        "era5_years":  ["2026"],
        "era5_months": ["02"],
        "era5_days":   ["09", "10", "11", "12", "13", "14",
                        "15", "16", "17", "18", "19", "20"],

        # Feb 2026 is NRT
        "mswep_subdir": "NRT",

        # Point of interest: Toamasina (Tamatave) — direct landfall city
        "poi_lat":  -18.15,
        "poi_lon":   49.40,
        "poi_name": "Toamasina",

        "has_aifs": True,   # 2026 data: AIFS available

        # Representative Toamasina catchment (Ivondro River @ Toamasina, ~3,470 km²)
        # delineated from MERIT Hydro by scripts/delineate_basin.py — used for the
        # basin-averaged figures (no wflow model built yet).
        "basin_file": _ROOT + "/data/basins/toamasina/ivondro.geojson",
        "basin_name": "Ivondro Basin (Toamasina)",

        "max_lead_hours": 360,   # 15 days — the full IFS/AIFS ENS horizon.
        # Was 240 h, which is shorter than what MARS actually archives and left
        # WeatherNext 2 (15 d) as the only product with data late in the window.

        "best_init": "2026-02-03T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2026-02-08T00:00",   # Near-event init (~2 days before)

        # Cyclone rainfall thresholds (mm event-total over ~7-day window)
        "exceed_thresholds": [50, 100, 200],
        "fss_thresholds":    [50, 100, 200],
        "fss_scales_km":     [25, 50, 100, 200],
    },

    "boris_2024": {
        "name":  "Storm Boris",
        "slug":  "boris_2024",

        # Bounding box: CzechRep9 (Upper Elbe) footprint, widened east so the
        # Jeseník rainfall epicentre (~17.2°E) sits inside the grid
        "lat_min":  48.4,
        "lat_max":  51.1,
        "lon_min":  11.5,
        "lon_max":  17.6,

        # 00Z + 12Z inits from 7 days before onset (Sep 12) through the day before
        "init_dates": [
            "2024-09-05", "2024-09-06", "2024-09-07",
            "2024-09-08", "2024-09-09", "2024-09-10", "2024-09-11",
        ],

        # Core event: the Sep 12–16 multi-day extreme rainfall over Central Europe
        "event_start": "2024-09-12T00:00",
        "event_end":   "2024-09-16T23:59:59",

        # Broader download window for ERA5 / MSWEP
        "ref_start": "2024-09-11",
        "ref_end":   "2024-09-22",

        # ERA5 CDS request date fields
        "era5_years":  ["2024"],
        "era5_months": ["09"],
        "era5_days":   ["11", "12", "13", "14", "15", "16",
                        "17", "18", "19", "20", "21", "22"],

        "mswep_subdir": "Past",   # Sep 2024 is in the historical archive

        # Jeseník — rainfall epicentre of Storm Boris (record multi-day totals)
        "poi_lat":  50.23,
        "poi_lon":  17.20,
        "poi_name": "Jeseník",

        "has_aifs": False,   # AIFS (class=ai) not archived for 2024

        # Basin geometry — existing CzechRep9 (Elbe) wflow model footprint
        "basin_file": _ROOT + "/data/basins/CzechRep9/staticgeoms/basins.geojson",
        "basin_name": "Elbe Basin (Czechia)",

        # 15 days, NOT 10. The Elbe peaks ~2 days after the rain stops (Děčín
        # crests 2024-09-17 18Z), so a 240 h ifs_sep05 forecast ends 2024-09-15 —
        # before the peak it is supposed to predict. Boris is the only event where
        # trimming the scored window to the forecast horizon would destroy the
        # result, so IFS is pulled to the full 15 days instead.
        "max_lead_hours": 360,

        "best_init": "2024-09-05T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2024-09-10T00:00",   # Near-event init (~2 days before)

        # Large-scale orographic multi-day totals (300–500 mm observed)
        "exceed_thresholds": [50, 100, 200],
        "fss_thresholds":    [50, 100, 200],
        "fss_scales_km":     [25, 50, 100, 200],
    },

    "hans_2023": {
        "name":  "Storm Hans",
        "slug":  "hans_2023",

        # Box broader than the Drammen (Drammensvassdraget) catchment
        # (catchment ~lon 7.3–10.7, lat 59.4–61.4) to capture the SE-Norway storm
        "lat_min":  58.5,
        "lat_max":  62.0,
        "lon_min":   6.0,
        "lon_max":  12.0,

        # 00Z + 12Z inits from 7 days before onset (Aug 7) through the day before
        "init_dates": [
            "2023-07-31", "2023-08-01", "2023-08-02",
            "2023-08-03", "2023-08-04", "2023-08-05", "2023-08-06",
        ],

        # Core event: the Aug 7–9 extreme rainfall over southern Norway
        "event_start": "2023-08-07T00:00",
        "event_end":   "2023-08-09T23:59:59",

        # Broader window for ERA5 / MSWEP (downstream flood crest ran to mid-Aug)
        "ref_start": "2023-08-06",
        "ref_end":   "2023-08-17",

        "era5_years":  ["2023"],
        "era5_months": ["08"],
        "era5_days":   ["06", "07", "08", "09", "10", "11",
                        "12", "13", "14", "15", "16", "17"],

        "mswep_subdir": "Past",   # Aug 2023 is in the historical archive

        # Nesbyen (Hallingdal) — severe Hans flooding, within the Drammen watercourse
        "poi_lat":  60.567,
        "poi_lon":   9.109,
        "poi_name": "Nesbyen",

        "has_aifs": False,   # AIFS (class=ai) not archived for 2023

        # Existing Drammen wflow model geometry
        "basin_file": _ROOT + "/data/basins/wflow_drammen/staticgeoms/basins.geojson",
        "basin_name": "Drammen Basin",

        "max_lead_hours": 360,   # 15 days — the full IFS/AIFS ENS horizon.
        # Was 240 h, which is shorter than what MARS actually archives and left
        # WeatherNext 2 (15 d) as the only product with data late in the window.

        "best_init": "2023-07-31T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2023-08-05T00:00",   # Near-event init (~2 days before)

        "exceed_thresholds": [50, 100, 200],
        "fss_thresholds":    [50, 100, 200],
        "fss_scales_km":     [25, 50, 100, 200],
    },

    "california_ar_2023": {
        "name":  "California AR Sequence (Jan 2023)",
        "slug":  "california_ar_2023",

        # Box: NorCal coast AR landfall + Russian River basin
        # (basin ~lon -123.5..-122.6, lat 38.3..39.45)
        "lat_min":  37.8,
        "lat_max":  39.9,
        "lon_min": -124.6,
        "lon_max": -122.0,

        # 00Z + 12Z inits from 7 days before the Jan 4 pulse (spans the year boundary)
        "init_dates": [
            "2022-12-28", "2022-12-29", "2022-12-30",
            "2022-12-31", "2023-01-01", "2023-01-02", "2023-01-03",
        ],

        # Core January cluster: the Jan 4–5 major AR through the Jan 9–11 pulse
        # that crested the Russian River (whole-sequence event total)
        "event_start": "2023-01-04T00:00",
        "event_end":   "2023-01-11T23:59:59",

        "ref_start": "2023-01-03",
        "ref_end":   "2023-01-17",

        "era5_years":  ["2023"],
        "era5_months": ["01"],
        "era5_days":   ["03", "04", "05", "06", "07", "08", "09", "10",
                        "11", "12", "13", "14", "15", "16", "17"],

        "mswep_subdir": "Past",   # Jan 2023 is in the historical archive

        # Guerneville, CA — iconic Russian River flood gauge (Sonoma County)
        "poi_lat":  38.502,
        "poi_lon": -122.997,
        "poi_name": "Guerneville",

        "has_aifs": False,   # AIFS (class=ai) not archived for 2022/2023

        # Russian River basin geometry — TBD: provide before basin/wflow steps
        "basin_file": _ROOT + "/data/basins/russian_river/russian_river.geojson",
        "basin_name": "Russian River Basin",

        # Earliest init (Dec 28) to event end (Jan 11) ~= 15 days
        "max_lead_hours": 360,

        "best_init": "2022-12-28T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2023-01-02T00:00",   # Near-event init (~2 days before)

        # AR-sequence multi-day totals are large (200–500+ mm in the coastal ranges)
        "exceed_thresholds": [100, 200, 300],
        "fss_thresholds":    [100, 200, 300],
        "fss_scales_km":     [25, 50, 100, 200],
    },

    # ── Same AR sequence, where it was actually extreme (scouted 2026-10-14) ──
    # The Russian River stayed at T <= 5 yr. The two impact catchments:
    #   Arroyo Seco nr Soledad (USGS 11152000, 632 km2): record of 120 yr,
    #     30,700 cfs on 2023-01-09 18:30 UTC, unregulated, 15-min IV
    #   Cosumnes R at Michigan Bar (USGS 11335000, 1,388 km2): 3rd of 119 yr,
    #     67,900 cfs on 2022-12-31 17:45 UTC, unregulated; 1-17 day rain
    #     40-110 yr (Atlas 14)
    # One download box and init set for both (the box also holds the Russian
    # River); per-catchment analysis entries alias this one via proc_slug.
    # Inits run from a week before the Cosumnes peak to the day before the
    # Arroyo Seco peak.
    "california_2023": {
        "name":  "California AR Sequence: Arroyo Seco + Cosumnes (Dec 2022 – Jan 2023)",
        "slug":  "california_2023",

        "lat_min":  35.0,
        "lat_max":  40.0,
        "lon_min": -124.6,
        "lon_max": -119.5,

        "init_dates": [
            "2022-12-24", "2022-12-25", "2022-12-26", "2022-12-27",
            "2022-12-28", "2022-12-29", "2022-12-30", "2022-12-31",
            "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04",
            "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08",
        ],

        # Whole sequence: Cosumnes AR (Dec 26 - Jan 1) through the Jan 9-11
        # pulse that set the Arroyo Seco record
        "event_start": "2022-12-26T00:00",
        "event_end":   "2023-01-11T23:59:59",

        "ref_start": "2022-12-20",
        "ref_end":   "2023-01-20",

        "era5_years":  ["2022", "2023"],
        "era5_months": ["12", "01"],
        "era5_days":   [f"{d:02d}" for d in range(1, 32)],

        "mswep_subdir": "Past",

        # Arroyo Seco gauge (the record flood)
        "poi_lat":  36.281,
        "poi_lon": -121.322,
        "poi_name": "Arroyo Seco nr Soledad",

        "has_aifs": False,   # AIFS (class=ai) not archived for 2022/2023

        "basin_file": _ROOT + "/data/basins/california_2023/arroyo_cosumnes.geojson",
        "basin_name": "Arroyo Seco + Cosumnes",

        "max_lead_hours": 360,

        "best_init": "2022-12-24T00:00",
        "near_init": "2023-01-07T00:00",   # ~2.5 days before the Arroyo Seco peak

        "exceed_thresholds": [100, 200, 300],
        "fss_thresholds":    [100, 200, 300],
        "fss_scales_km":     [25, 50, 100, 200],
    },

    "bc_ar_2024": {
        "name":  "BC Atmospheric River (Oct 2024)",
        "slug":  "bc_ar_2024",

        # Box: SW British Columbia + Vancouver Island landfall zone, wide enough to
        # hold the AR corridor offshore. Basin (Chilliwack R.) is ~lon -122.0..-121.2,
        # lat 48.95..49.35 and straddles the US border into Washington State.
        "lat_min":  48.3,
        "lat_max":  50.2,
        "lon_min": -124.0,
        "lon_max": -120.5,

        # 00Z + 12Z inits from 7 days before the Oct 18 landfall
        "init_dates": [
            "2024-10-11", "2024-10-12", "2024-10-13",
            "2024-10-14", "2024-10-15", "2024-10-16", "2024-10-17",
        ],

        # Core event: the Oct 18–20 AR (Burke Mtn 256 mm, Kennedy Lake 318 mm)
        "event_start": "2024-10-18T00:00",
        "event_end":   "2024-10-20T23:59:59",

        "ref_start": "2024-10-17",
        "ref_end":   "2024-10-26",

        "era5_years":  ["2024"],
        "era5_months": ["10"],
        "era5_days":   ["17", "18", "19", "20", "21",
                        "22", "23", "24", "25", "26"],

        "mswep_subdir": "Past",   # Oct 2024 is in the historical archive

        # Vedder Crossing — the Chilliwack River gauge site (WSC 08MH001)
        "poi_lat":  49.10,
        "poi_lon": -121.96,
        "poi_name": "Vedder Crossing",

        "has_aifs": False,   # AIFS (class=ai) archive starts 2025-07-01T12

        "basin_file": _ROOT + "/data/basins/chilliwack/chilliwack.geojson",
        "basin_name": "Chilliwack River Basin",

        # 15 days — the full IFS/WN2 ENS horizon. At 240 h the long-lead init ended
        # 2024-10-21, i.e. right at the peak with no recession left to score.
        "max_lead_hours": 360,

        "best_init": "2024-10-11T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2024-10-16T00:00",   # Near-event init (~2 days before)

        # Coastal-mountain AR totals over the 3-day window
        "exceed_thresholds": [50, 100, 200],
        "fss_thresholds":    [50, 100, 200],
        # Small steep basin -> finer scales than the large-basin events
        "fss_scales_km":     [10, 25, 50, 100],
    },

    "zermatt_2024": {
        "name":  "Valais / Vispa Floods (Jun 2024)",
        "slug":  "zermatt_2024",

        # Box: Valais/Wallis + the southerly-flow moisture source over N. Italy.
        # Basin (Vispa @ Visp) is ~lon 7.6..8.1, lat 45.95..46.30.
        "lat_min": 45.7,
        "lat_max": 46.9,
        "lon_min":  7.0,
        "lon_max":  8.7,

        # 00Z + 12Z inits from ~7 days before the Jun 21 thunderstorm front,
        # and (added 18 Nov 2026) from ~7 days before the Jun 29-30 Saas flood,
        # so the Saaser Vispa gets the same -1/-2/-7 d starts and short rain
        # leads as every other gauge (the Jun 14-20 inits reach it only at
        # 8.5-15 d). One init set for both floods, as for california_2023.
        "init_dates": [
            "2024-06-14", "2024-06-15", "2024-06-16",
            "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20",
            "2024-06-22", "2024-06-23", "2024-06-24", "2024-06-25",
            "2024-06-26", "2024-06-27", "2024-06-28",
        ],

        # Window spans BOTH episodes: Jun 21–22 (Mattervispa/Triftbach flood the
        # Zermatt town centre) and Jun 29–30 (Saas-Grund debris flow).
        # event_end is Jun 30 00Z, NOT 23:59 — that is exactly best_init (Jun 15
        # 00Z) + max_lead_hours (360). IFS/WN2 both cap at 15 days, so a later
        # event_end could not be covered by the long-lead init at all.
        "event_start": "2024-06-21T00:00",
        "event_end":   "2024-06-30T00:00:00",

        "ref_start": "2024-06-19",
        "ref_end":   "2024-07-04",

        "era5_years":  ["2024"],
        "era5_months": ["06", "07"],
        "era5_days":   ["19", "20", "21", "22", "23", "24",
                        "25", "26", "27", "28", "29", "30",
                        "01", "02", "03", "04"],

        "mswep_subdir": "Past",   # Jun 2024 is in the historical archive

        # Visp — Vispa/Rhône confluence, basin outlet (BAFU station 2611)
        "poi_lat":  46.294,
        "poi_lon":   7.882,
        "poi_name": "Visp",

        "has_aifs": False,   # AIFS (class=ai) archive starts 2025-07-01T12

        "basin_file": _ROOT + "/data/basins/vispa/vispa.geojson",
        "basin_name": "Vispa Basin (Valais)",

        # 15 days: needed to reach the Jun 29–30 episode from the long-lead init
        "max_lead_hours": 360,

        # NOT the usual 7-day init: IFS/WN2 cap at 15 days, and Jun 15 00Z + 360 h
        # = Jun 30 00Z, exactly covering the Saas-Grund episode. A Jun 14 init
        # would fall short of event_end.
        "best_init": "2024-06-15T00:00",
        "near_init": "2024-06-19T00:00",   # Near-event init (~2 days before)

        # Alpine convective/orographic totals (Binntal 158 mm in 1 h; Mesolcina
        # 125 mm/24 h) — lower thresholds than the multi-day AR events
        "exceed_thresholds": [50, 100, 150],
        "fss_thresholds":    [50, 100, 150],
        # 785 km2 basin -> the small scales are the informative ones
        "fss_scales_km":     [10, 25, 50, 100],
    },

    "helene_2024": {
        "name":  "Hurricane Helene, southern Appalachians (Sep 2024)",
        "slug":  "helene_2024",

        # Replaces BC AR 2024 (user, 2026-10-15). Box: western NC / east TN /
        # upstate SC, wide enough for the predecessor rain and the hurricane core.
        "lat_min":  33.5,
        "lat_max":  37.5,
        "lon_min": -85.5,
        "lon_max": -79.5,

        # 00Z + 12Z from 7 days before onset; last_init drops Sep 25 12Z (rain on).
        "init_dates": [
            "2024-09-18", "2024-09-19", "2024-09-20", "2024-09-21",
            "2024-09-22", "2024-09-23", "2024-09-24", "2024-09-25",
        ],
        "last_init": "2024-09-25T00",

        # Predecessor rain from ~25 Sep, hurricane rain 26-27 Sep (landfall Big Bend
        # FL 27 Sep ~03 UTC); river peaks 27 Sep 14:30-21:40 UTC (obs-scout).
        "event_start": "2024-09-25T00:00",
        "event_end":   "2024-09-28T23:59:59",

        "ref_start": "2024-09-20",
        "ref_end":   "2024-10-03",

        "era5_years":  ["2024"],
        "era5_months": ["09", "10"],
        "era5_days":   [f"{d:02d}" for d in range(1, 32)],

        "mswep_subdir": "Past",

        # French Broad at Asheville (USGS 03451500; record of 130 yr)
        "poi_lat":  35.6095,
        "poi_lon": -82.5793,
        "poi_name": "French Broad at Asheville",

        "has_aifs": False,   # AIFS ENS archive starts 2025-07

        "basin_file": _ROOT + "/data/basins/helene_2024/helene_basins.geojson",
        "basin_name": "French Broad + South Toe + Cataloochee + upper Catawba",

        "max_lead_hours": 360,

        "best_init": "2024-09-18T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2024-09-23T00:00",   # Near-event init (~2 days before)

        # 72 h totals up to ~31 in (Busick); the extreme part is the 24-72 h volume
        "exceed_thresholds": [100, 200, 300, 400],
        "fss_thresholds":    [100, 200, 300],
        "fss_scales_km":     [10, 25, 50, 100],
    },

    "texas_2026": {
        "name":  "Texas Hill Country floods (Jul 2026)",
        "slug":  "texas_2026",

        # AIFS-era event (AIFS archive starts Jul 2025), chosen 18 Oct 2026 from the
        # obs-scout candidates: Pedernales nr Fredericksburg (record of 41 yr), Cibolo
        # Ck nr St Hedwig (record of 19), Medina abv English Crossing (record of 8),
        # Guadalupe at Comfort (2nd of 90 behind Jul 2025). Box covers the four basins.
        "lat_min":  28.9,
        "lat_max":  31.0,
        "lon_min": -100.3,
        "lon_max":  -97.6,

        # 00Z + 12Z from 7 days before the rain (Jul 14) to the last init before it
        "init_dates": ["2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10",
                       "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14"],
        "last_init": "2026-07-14T00",

        # Rain 14-16 Jul (MRMS window max 425 mm on 0.25 deg), river peaks 15-16 Jul UTC
        "event_start": "2026-07-14T00:00",
        "event_end":   "2026-07-17T23:59:59",

        "ref_start": "2026-07-06",
        "ref_end":   "2026-07-20",

        "era5_years":  ["2026"],
        "era5_months": ["07"],
        "era5_days":   ["06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],

        "mswep_subdir": "NRT",

        "poi_lat":  30.2205,
        "poi_lon": -98.8698,
        "poi_name": "Pedernales nr Fredericksburg",

        "has_aifs": True,

        "basin_file": _ROOT + "/data/basins/texas_2026/texas_2026_basins.geojson",
        "basin_name": "Pedernales + Medina + Cibolo + Guadalupe to Comfort",

        "max_lead_hours": 360,

        "best_init": "2026-07-07T00:00",   # ~7 days before
        "near_init": "2026-07-12T00:00",   # ~2 days before

        "exceed_thresholds": [100, 200, 300, 400],
        "fss_thresholds":    [100, 200, 300],
        "fss_scales_km":     [10, 25, 50, 100],
    },

    "mississippi_2026": {
        "name":  "Mississippi Gulf Coast floods (Jun 2026)",
        "slug":  "mississippi_2026",

        # AIFS-era event chosen 18 Oct 2026: Biloxi at Wortham (record of 73 yr),
        # Red Creek at Vestry (record of 65), Wolf R nr Landon (2nd of 54), Okatoma Ck
        # at Sanford (2nd of 31). MRMS is ~40 percent low in the heaviest core (Wiggins).
        "lat_min":  29.9,
        "lat_max":  32.0,
        "lon_min": -90.5,
        "lon_max":  -88.0,

        "init_dates": ["2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13",
                       "2026-06-14", "2026-06-15", "2026-06-16", "2026-06-17"],
        "last_init": "2026-06-17T00",

        # Rain 16-20 Jun (MRMS window max 393 mm on 0.25 deg), peaks 19-20 Jun UTC
        "event_start": "2026-06-17T00:00",
        "event_end":   "2026-06-20T23:59:59",

        "ref_start": "2026-06-09",
        "ref_end":   "2026-06-23",

        "era5_years":  ["2026"],
        "era5_months": ["06"],
        "era5_days":   ["09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"],

        "mswep_subdir": "NRT",

        "poi_lat":  30.5586,
        "poi_lon": -89.1219,
        "poi_name": "Biloxi River at Wortham",

        "has_aifs": True,

        "basin_file": _ROOT + "/data/basins/mississippi_2026/mississippi_2026_basins.geojson",
        "basin_name": "Biloxi + Red Creek + Wolf + Okatoma",

        "max_lead_hours": 360,

        "best_init": "2026-06-10T00:00",
        "near_init": "2026-06-15T00:00",

        "exceed_thresholds": [100, 200, 300, 400],
        "fss_thresholds":    [100, 200, 300],
        "fss_scales_km":     [10, 25, 50, 100],
    },

    "catalonia_2026": {
        "name":  "Catalonia Mesoscale Storm (Oct 2026)",
        "slug":  "catalonia_2026",

        # Meteo box: Catalan coast + the Mediterranean where the convective
        # system organised, wide enough to show position errors. The hydrology
        # domain is much smaller: Besos (outlet Sant Adria) + Tordera (outlet
        # Blanes/Malgrat) + the Maresme rieres, ~lon 1.95..2.90, lat 41.38..41.95.
        "lat_min": 40.8,
        "lat_max": 42.6,
        "lon_min":  0.8,
        "lon_max":  3.4,

        # 00Z + 12Z from 7 days before onset (Sep 26 added 2026-10-14 so the
        # long lead matches the other events); last_init drops Oct 3 12Z, which
        # starts after the rain began (~10 UTC) -- Oct 3 00Z is the shortest lead.
        "init_dates": [
            "2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29",
            "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-03",
        ],
        "last_init": "2026-10-03T00",

        # Rain Sat Oct 3 ~10 UTC -> Sun Oct 4 evening, peak early Oct 4 (Besos
        # 936 m3/s at Santa Coloma on Sunday morning); window runs to Oct 6 00Z
        # for the recession.
        "event_start": "2026-10-03T00:00",
        "event_end":   "2026-10-05T23:59:59",

        "ref_start": "2026-10-01",
        "ref_end":   "2026-10-08",

        "era5_years":  ["2026"],
        "era5_months": ["10"],
        "era5_days":   ["01", "02", "03", "04", "05", "06", "07", "08"],

        "mswep_subdir": "NRT",

        # Reference data not yet complete at setup (2026-10-09): ERA5T reached
        # Oct 4 10 UTC (CDS lags ~5 days), MSWEP NRT on the share stopped Sep 30.
        # scripts/check_refetch.py lists these and re-pulls them once due.
        # ERA5 re-pulled 2026-10-14 (complete to Oct 8 23 UTC). MSWEP NRT on the
        # share still ends Sep 30 (2026273.03) -- waiting on its maintainers.
        "refetch": {"mswep": "2026-10-14"},

        # Besos at Santa Coloma de Gramenet (ACA gauge; 936 m3/s peak)
        "poi_lat":  41.452,
        "poi_lon":   2.208,
        "poi_name": "Santa Coloma de Gramenet",

        "has_aifs": True,
        "aifs_available_from": "2025-07-01T12",

        "basin_file": _ROOT + "/data/basins/catalonia/besos_tordera.geojson",
        "basin_name": "Besos + Tordera (Catalonia)",

        "max_lead_hours": 360,

        "best_init": "2026-09-26T00:00",   # Maximum lead time init (~7 days before)
        "near_init": "2026-10-01T00:00",   # Near-event init (~2 days before)

        # 288 mm Vilassar de Mar, 255 mm Mataro, 248 mm Granollers (68 mm/30 min)
        "exceed_thresholds": [50, 100, 200],
        "fss_thresholds":    [50, 100, 200],
        # ~1,900 km2 of small steep basins -> small scales are the informative ones
        "fss_scales_km":     [10, 25, 50, 100],
    },

}

# ── Clipped sub-basins of kristin_2026 ──────────────────────────────────────
# Same storm, same download bbox, same processed files — only the catchment
# polygon differs. They therefore inherit Kristin's entry wholesale and override
# just the geometry, with `proc_slug` pointing back at the parent because they
# have no data/processed/<slug>/ of their own.
#
# Only basin_analysis.py is meaningful for these: analysis.py is bbox-level and
# would emit byte-identical duplicates of Kristin's meteo figures. They stay out
# of wflow_ifs_vs_ai.EXCLUDE's counterpart so Kristin counts once in the tally.
for _sub, _name, _basin in (
    ("kristin_mondego", "Storm Kristin — Mondego @ Coimbra", "Mondego @ Coimbra"),
    ("kristin_lis",     "Storm Kristin — Lis @ Leiria",      "Lis @ Leiria"),
):
    EVENTS[_sub] = {
        **EVENTS["kristin_2026"],
        "name":       _name,
        "slug":       _sub,
        "proc_slug":  "kristin_2026",
        "basin_file": _ROOT + f"/data/wflow/{_sub}/model/staticgeoms/basins.geojson",
        "basin_name": _basin,
    }

# ── Jeseníky / Kłodzko headwaters of storm Boris ───────────────────────────
# A separately built model (wflow_build_sudetes.py) inside the SAME forecast box
# as boris_2024 (lat 48.4–51.1, lon 11.5–17.6), so the processed forecast and
# reference files are boris_2024's (`proc_slug`); only the catchment differs.
EVENTS["boris_sudetes"] = {
    **EVENTS["boris_2024"],
    "name":       "Storm Boris — Jeseníky / Kłodzko headwaters",
    "slug":       "boris_sudetes",
    "proc_slug":  "boris_2024",
    "basin_file": _ROOT + "/data/wflow/boris_sudetes/model/staticgeoms/basins.geojson",
    "basin_name": "Nysa Kłodzka, Bělá, Vidnavka, upper Opava",
}

# Forecast products left out of every analysis, figure and tally. WN3 is only
# available as its ensemble mean (members need GCP billing), so it cannot be
# scored like the 51/64-member ensembles and was clouding the comparison
# (user decision, 2026-10-09). Its data and runs are kept; empty this set to
# bring it back once the members are downloaded.
EXCLUDED_PRODUCTS = {"wn3"}


def init_pairs(c):
    """(YYYY-MM-DD, "00"|"12") for every init the downloaders and preprocess use:
    each init_date at 00Z and 12Z, minus any after the optional ``last_init``
    ("YYYY-MM-DDTHH"), e.g. a 12Z run that would start after the rain began."""
    last = c.get("last_init")
    return [(d, hh) for d in c["init_dates"] for hh in ("00", "12")
            if last is None or f"{d}T{hh}" <= last]


# ── Convenience: expose active config at module level ───────────────────────
cfg = EVENTS[ACTIVE_EVENT]
