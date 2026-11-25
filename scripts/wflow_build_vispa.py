#!/usr/bin/env python3
"""
Build an UNCALIBRATED wflow_sbm model of the Vispa @ Visp (Valais, Switzerland)
from scratch with hydromt_wflow (v1), for the June 2024 Valais floods.

Mirrors scripts/wflow_build_chilliwack.py (hydromt `deltares_data` catalog + MERIT
Hydro delineation, globcover LULC, MODIS LAI, SoilGrids, default SBM constants) but
adds GLACIERS, which the other from-scratch models do not need.

Basin notes:
  * ~785 km², 650–4,600 m, draining to the Rhône at Visp (BAFU 2611).
  * Contains BOTH June 2024 episodes: the Matter Vispa arm (Zermatt / Täsch /
    Randa, 21–22 Jun) and the Saaser Vispa arm (Saas-Grund, 29–30 Jun). Gauges are
    placed on each arm above the Stalden confluence because the outlet integrates
    the two and the peak metric takes a max over the window — it would latch onto
    whichever episode is larger and silently ignore the other.
  * ~324 km² of RGI glacier area intersects the bounding box (roughly 15–20 % of
    the basin). A late-June flood here is a rain + snowmelt + icemelt event, so
    glacier__flag must be on; leaving it off (as the other build scripts do) would
    remove the ice contribution entirely.
  * Saas-Grund was a DEBRIS FLOW. wflow_sbm models neither debris flow nor the
    sediment-driven channel blockage, so skill there scores the rainfall trigger,
    not the impact.

Usage:  conda activate hydromt-sfincs-test
        HDF5_USE_FILE_LOCKING=FALSE python scripts/wflow_build_vispa.py
"""
import os
import argparse
import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (.rio accessor)
import hydromt_wflow
from hydromt_wflow import WflowSbmModel

_ap = argparse.ArgumentParser()
_ap.add_argument("--start", default="2015-01-02")   # throwaway build-forcing year
_ap.add_argument("--end",   default="2015-12-31")
_args, _ = _ap.parse_known_args()
PERIOD = (_args.start, _args.end)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "data/wflow/zermatt_2024/model")
RES  = 1.0 / 120.0   # ~925 m (MERIT-native upscale)

# Vispa @ Visp: outlet snapped on MERIT Hydro (uparea ≈ 785 km²), the same point
# delineate_basin.py used for the meteo basin polygon.
REGION = {"subbasin": [7.8796, 46.2921], "bounds": [7.55, 45.90, 8.35, 46.35]}

GAUGE_CSV = os.path.join(REPO, "data/wflow/zermatt_2024/gauges.csv")

LULC_MAP = os.path.join(os.path.dirname(hydromt_wflow.__file__),
                        "data", "lulc", "v1.0", "globcover_mapping.csv")

# Default SBM constants (Wflow.jl v1 names) — uncalibrated, same set as other
# models. The three glacier_* entries are DIFFERENT Wflow variables from the two
# maps setup_glaciers writes (glacier_fraction, glacier_initial_leq_depth), so
# there is no clash between setup_constant_pars and setup_glaciers.
CONST = {
    "snowpack__degree_day_coefficient": 3.75,
    "soil_surface_water__infiltration_reduction_parameter": 0.038,
    "vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio": 0.11,
    "compacted_soil_surface_water__infiltration_capacity": 10,
    "soil_water_saturated_zone_bottom__max_leakage_volume_flux": 0,
    "soil_wet_root__sigmoid_function_shape_parameter": -500,
    "atmosphere_air__snowfall_temperature_threshold": 0,
    "atmosphere_air__snowfall_temperature_interval": 2,
    "snowpack__melting_temperature_threshold": 0,
    "snowpack__liquid_water_holding_capacity": 0.1,
    "glacier_ice__degree_day_coefficient": 3,
    "glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction": 0.001,
    "glacier_ice__melting_temperature_threshold": 0,
    "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio": 100,
}

# 0.25 km², not the hydromt default of 1.0: the default discards 12 % of the ice
# in this basin (285 of 324 km²), while 0.25 keeps 97 %. At 1/120° a model cell is
# ~0.6 km², so sub-km² glaciers are resolvable and worth keeping.
GLACIER_MIN_AREA = 0.25


def load_gauges():
    df = pd.read_csv(GAUGE_CSV)
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:4326")


def main():
    os.makedirs(ROOT, exist_ok=True)
    m = WflowSbmModel(root=ROOT, mode="w", data_libs=["deltares_data"])

    print("setup_basemaps (delineate Vispa subbasin + grid)…")
    m.setup_basemaps(region=REGION, res=RES, hydrography_fn="merit_hydro")
    print("setup_rivers…")
    m.setup_rivers(hydrography_fn="merit_hydro", river_geom_fn="hydro_rivers_lin", river_upa=30)
    m.setup_river_roughness()
    print("setup_lulcmaps (globcover)…")
    m.setup_lulcmaps(lulc_fn="globcover", lulc_mapping_fn=LULC_MAP)
    print("setup_laimaps (modis)…")
    m.setup_laimaps(lai_fn="modis_lai")
    print("setup_soilmaps (soilgrids)…")
    m.setup_soilmaps(soil_fn="soilgrids")

    print(f"setup_glaciers (RGI v6, min_area={GLACIER_MIN_AREA} km²)…")
    m.setup_glaciers(glaciers_fn="rgi", min_area=GLACIER_MIN_AREA, geom_name="glaciers")
    assert m.config.get_value("model.glacier__flag") is True, \
        "setup_glaciers did not enable model.glacier__flag"

    m.setup_constant_pars(**CONST)

    m.setup_config({
        "time.calendar":     "proleptic_gregorian",
        "time.starttime":    PERIOD[0] + "T00:00:00",
        "time.endtime":      PERIOD[1] + "T00:00:00",
        "time.timestepsecs": 86400,
        "model.cold_start__flag": True,
    })

    gauges = load_gauges()
    print(f"setup_gauges ({len(gauges)} point(s): outlet + both arms + Zermatt) + outlets…")
    m.setup_gauges(gauges_fn=gauges, index_col="id", snap_to_river=True,
                   derive_subcatch=True, basename="vispa",
                   gauge_toml_header=["river_q"],
                   gauge_toml_param=["river_water__volume_flow_rate"])
    # Distinct header: the outlet cell coincides with gauge id 1, and reusing
    # "river_q" would emit two columns literally named river_q_1, which pandas
    # silently mangles to river_q_1 / river_q_1.1 downstream.
    m.setup_outlets(river_only=True, gauge_toml_header=["river_q_outlet"])

    # Gridded output must be requested explicitly — hydromt writes only
    # [output.csv], so without this the spatial peak-discharge map cannot be made.
    m.setup_config({
        "output.netcdf_grid.path": "output.nc",
        "output.netcdf_grid.compressionlevel": 1,
        "output.netcdf_grid.variables.river_water__volume_flow_rate": "river_q",
    })

    # ── Throwaway build forcing (CHIRPS + ERA5), read directly (deltares netCDF
    #    read path errors on the /p mount) so hydromt writes a complete v1 TOML.
    grid = m.staticmaps.data
    ydim = "y" if "y" in grid.coords else "latitude"
    xdim = "x" if "x" in grid.coords else "longitude"
    glat, glon = grid[ydim].values, grid[xdim].values
    B = 0.6
    lat0, lat1 = float(glat.min()) - B, float(glat.max()) + B
    lon0, lon1 = float(glon.min()) - B, float(glon.max()) + B
    YEARS = range(int(PERIOD[0][:4]), int(PERIOD[1][:4]) + 1)

    def _clip(ds):
        ds = ds.sortby("latitude")
        # No-op for this basin (positive longitudes) but kept so the read path is
        # identical to the negative-longitude models.
        if float(ds["longitude"].max()) > 180:
            ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")
        return ds.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1),
                      time=slice(*PERIOD))

    CH = os.environ.get("HYDROMT_METEO", os.path.join(REPO, "data", "external", "hydromt_meteo")) + "/chirps_global_daily_v2.0/chirps-v2.0.%s.days_p05.nc"
    E5 = os.environ.get("HYDROMT_METEO", os.path.join(REPO, "data", "external", "hydromt_meteo")) + "/era5_daily/nc_merged/era5_%s_daily.nc"
    print("read CHIRPS + ERA5 (local, direct)…")
    pr = _clip(xr.open_mfdataset([CH % y for y in YEARS], engine="netcdf4", combine="by_coords"))["precip"]
    pr = pr.where(pr >= 0.0, 0.0).load(); pr.attrs["unit"] = "mm"; pr = pr.rio.write_crs("EPSG:4326")
    e5 = _clip(xr.open_mfdataset([E5 % y for y in YEARS], engine="netcdf4", combine="by_coords"))
    e5 = e5.rename({"t2m": "temp", "msl": "press_msl", "ssrd": "kin", "tisr": "kout"})
    e5["temp"] = e5["temp"] - 273.15
    e5["press_msl"] = e5["press_msl"] * 0.01
    e5["kin"] = e5["kin"] * 0.000277778
    e5["kout"] = e5["kout"] * 0.000277778
    e5 = e5[["temp", "press_msl", "kin", "kout"]].load().rio.write_crs("EPSG:4326")

    m.setup_precip_forcing(precip_fn=pr)
    m.setup_temp_pet_forcing(temp_pet_fn=e5, pet_method="debruin",
                             press_correction=False, temp_correction=False, wind_correction=False)

    def _sanitize(attrs):
        for k in list(attrs):
            if not isinstance(attrs[k], (str, int, float, list, tuple, bytes)):
                attrs[k] = str(attrs[k])[:200]
    fc = m.forcing.data
    _sanitize(fc.attrs)
    for v in fc.data_vars:
        _sanitize(fc[v].attrs)

    print("write model…")
    m.write()
    print("DONE:", ROOT)


if __name__ == "__main__":
    main()
