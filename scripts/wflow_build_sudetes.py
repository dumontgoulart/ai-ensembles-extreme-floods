#!/usr/bin/env python3
"""
Build an UNCALIBRATED wflow_sbm model of the Jeseníky / Kłodzko headwaters
(Eastern Sudetes, CZ–PL) from scratch with hydromt_wflow (v1), for Storm Boris
(Sept 2024) -- the rainfall epicentre (CHMI stations 470-704 mm, IMGW up to
~500 mm), which the Upper Elbe model of boris_2024 does not contain.

Same recipe as scripts/wflow_build_vispa.py (hydromt `deltares_data` catalog,
MERIT Hydro, globcover, MODIS LAI, SoilGrids, default SBM constants) without
glaciers, and with FOUR outlets, because the storm area drains to four separate
rivers that only meet far downstream (below the Nysa/Otmuchów reservoirs or in
the Oder):
  * Nysa Kłodzka @ Bardo (PL)               ~1,744 km²  (Kłodzko 1,082 km² nested)
  * Biała Głuchołaska / Bělá @ Głuchołazy   (Mikulovice 223, Jeseník 118 km² nested)
  * Vidnavka @ Vidnava (CZ)                  152 km²
  * Opava @ Karlovice (CZ)                   151 km²  (Krnov/Opava are east of the
    17.6°E edge of the Boris forecast box, so the Opava stops here)
Gauges (data/wflow/boris_sudetes/gauges.csv) are written as gauges_sudetes; a
river threshold of 10 km² keeps the smallest gauge (Zlatý potok, 22.7 km²) on
a river cell.

Usage:  conda activate hydromt-sfincs-test
        HDF5_USE_FILE_LOCKING=FALSE python scripts/wflow_build_sudetes.py
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
ROOT = os.path.join(REPO, "data/wflow/boris_sudetes/model")
RES  = 1.0 / 120.0   # ~925 m (MERIT-native upscale)

# Four outlets (x list, y list): Bardo, Głuchołazy, Vidnava, Karlovice
OUTLETS = [(16.7397, 50.5044), (17.3806, 50.3172), (17.1824, 50.3734), (17.4280, 50.1079)]
# uparea: snap each outlet to a stream of >= 100 km2 (all four are >= 150 km2);
# without it hydromt snapped Bardo onto a small side creek (~400 km2 domain).
REGION = {"subbasin": [[x for x, _ in OUTLETS], [y for _, y in OUTLETS]], "uparea": 100,
          "bounds": [16.15, 49.95, 17.65, 50.75]}

GAUGE_CSV = os.path.join(REPO, "data/wflow/boris_sudetes/gauges.csv")

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

    print("setup_basemaps (delineate the four subbasins + grid)…")
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
    print(f"setup_gauges ({len(gauges)} points) + outlets…")
    m.setup_gauges(gauges_fn=gauges, index_col="id", snap_to_river=True,
                   derive_subcatch=True, basename="sudetes",
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

    # CHIRPS stops at 50N and this domain is 50.0-50.7N, so the throwaway build
    # precipitation is ERA5's (the event forcing replaces it in any case).
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
