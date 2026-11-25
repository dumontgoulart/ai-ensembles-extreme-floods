#!/usr/bin/env python3
"""
Build an UNCALIBRATED wflow_sbm model of the southern Appalachian catchments
where Hurricane Helene (25-28 Sep 2024) set record floods. Same recipe as
wflow_build_mississippi_2026.py.

Four independent outlets (subbasins), USGS gauges in
data/wflow/mississippi_2026/gauges.csv (basename gauges_helene), all records:
  * French Broad R at Asheville 03451500 (2,448 km2)   130 yr record; contains
    Mills River 03446000 (173) and Swannanoa at Biltmore 03451000 (337, peak
    estimated, value only)
  * South Toe R nr Celo 03463300 (112)                 x2.2 previous record
  * Cataloochee Cr 03460000 (127)                      x2.5
  * Catawba R nr Pleasant Gardens 02137727 (326)       x1.84

Usage:  conda activate hydromt-sfincs-test
        HDF5_USE_FILE_LOCKING=FALSE python scripts/wflow_build_mississippi_2026.py
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
ROOT = os.path.join(REPO, "data/wflow/mississippi_2026/model")
RES  = 1.0 / 120.0   # ~925 m (MERIT-native upscale)

# Four outlets (x list, y list) at the gauges; snap to streams of >= 50 km2
OUTLETS = [(-89.1219444, 30.5586111), (-88.7811111, 30.7361111), (-89.2744444, 30.4836111), (-89.4333333, 31.48916667)]
REGION = {"subbasin": [[x for x, _ in OUTLETS], [y for _, y in OUTLETS]], "uparea": 50,
          "bounds": [-90.45, 30.2, -88.45, 31.95]}

GAUGE_CSV = os.path.join(REPO, "data/wflow/mississippi_2026/gauges.csv")

LULC_MAP = os.path.join(os.path.dirname(hydromt_wflow.__file__),
                        "data", "lulc", "v1.0", "globcover_mapping.csv")

# Default SBM constants (Wflow.jl v1 names) — uncalibrated, same set as other models.
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
    "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio": 100,
}



def load_gauges():
    df = pd.read_csv(GAUGE_CSV)
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:4326")


def main():
    os.makedirs(ROOT, exist_ok=True)
    m = WflowSbmModel(root=ROOT, mode="w", data_libs=["deltares_data"])

    print("setup_basemaps (four subbasins + grid)…")
    m.setup_basemaps(region=REGION, res=RES, hydrography_fn="merit_hydro")
    print("setup_rivers…")
    m.setup_rivers(hydrography_fn="merit_hydro", river_geom_fn="hydro_rivers_lin", river_upa=10)
    m.setup_river_roughness()
    print("setup_lulcmaps (globcover)…")
    m.setup_lulcmaps(lulc_fn="globcover", lulc_mapping_fn=LULC_MAP)
    print("setup_laimaps (modis)…")
    m.setup_laimaps(lai_fn="modis_lai")
    print("setup_soilmaps (soilgrids)…")
    m.setup_soilmaps(soil_fn="soilgrids")


    m.setup_constant_pars(**CONST)

    m.setup_config({
        "time.calendar":     "proleptic_gregorian",
        "time.starttime":    PERIOD[0] + "T00:00:00",
        "time.endtime":      PERIOD[1] + "T00:00:00",
        "time.timestepsecs": 86400,
        "model.cold_start__flag": True,
    })

    gauges = load_gauges()
    gauges["uparea"] = gauges["official_km2"]     # snap by upstream area (Catalonia lesson)
    print(f"setup_gauges ({len(gauges)} points) + outlets…")
    m.setup_gauges(gauges_fn=gauges, index_col="id", snap_to_river=True,
                   snap_uparea=True, wdw=3, rel_error=0.05,
                   derive_subcatch=True, basename="mississippi2026",
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
        if float(ds["longitude"].max()) > 180:
            ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")
        return ds.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1),
                      time=slice(*PERIOD))

    # Throwaway build precipitation from ERA5, as for the other new models (the
    # event forcing replaces it in any case).
    E5 = os.environ.get("HYDROMT_METEO", os.path.join(REPO, "data", "external", "hydromt_meteo")) + "/era5_daily/nc_merged/era5_%s_daily.nc"
    print("read ERA5 daily (local, direct)…")
    e5 = _clip(xr.open_mfdataset([E5 % y for y in YEARS], engine="netcdf4", combine="by_coords"))
    pr = e5["tp"]
    pr = pr.where(pr >= 0.0, 0.0).load(); pr.attrs["unit"] = "mm"; pr = pr.rio.write_crs("EPSG:4326")
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
