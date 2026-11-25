#!/usr/bin/env python3
"""
Build an UNCALIBRATED wflow_sbm model of the Catalan coast hit by the 3-5 Oct
2026 mesoscale storm: the Besos, the Tordera and the Maresme rieres in between
(Granollers 248 mm/24 h ~400 yr, Mataro 244 mm ~220 yr; Besos 937 m3/s at
Santa Coloma). Same recipe as wflow_build_sudetes.py (hydromt `deltares_data`,
MERIT Hydro, globcover, MODIS LAI, SoilGrids, default SBM constants).

Region: every basin whose mouth lies on the coast between the Besos (Sant
Adria) and the Tordera (Blanes/Malgrat) -- the rieres drain straight to the sea,
so listing outlets is impractical. The Llobregat (mouth south of 41.38N) and the
Costa Brava (east of the Tordera) stay out.
Gauges: the 10 ACA gauges in data/wflow/catalonia_2026/gauges.csv (basename
gauges_catalonia); river_upa 5 km2 keeps the smallest (riera d'Arenys,
11.8 km2) on a river cell.

Usage:  conda activate hydromt-sfincs-test
        HDF5_USE_FILE_LOCKING=FALSE python scripts/wflow_build_catalonia.py
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
ROOT = os.path.join(REPO, "data/wflow/catalonia_2026/model")
RES  = 1.0 / 120.0   # ~925 m (MERIT-native upscale)

# Every MERIT sea outlet (pit) of >= 5 km2 on the coast 1.95-2.81E, 41.38-41.95N:
# Besos (1,037 km2), Tordera (876), riera d'Argentona (73) and 16 Maresme rieres.
# Listed explicitly with a capped read window: the {"basin": bbox, "outlets":
# True} region made hydromt load continent-sized MERIT tiles and run out of 28 GB.
OUTLETS = [(2.2329, 41.4196), (2.7779, 41.6521), (2.4246, 41.5204), (2.6196, 41.5996),
           (2.6863, 41.6196), (2.5529, 41.5787), (2.7013, 41.6246), (2.2204, 41.4062),
           (2.3079, 41.4763), (2.5004, 41.5571), (2.7396, 41.6404), (2.3921, 41.5012),
           (2.5296, 41.5687), (2.1938, 41.3812), (2.7929, 41.6737), (2.6562, 41.6112),
           (2.2413, 41.4346), (2.4579, 41.5412), (2.2429, 41.4371)]
REGION = {"subbasin": [[x for x, _ in OUTLETS], [y for _, y in OUTLETS]], "uparea": 5,
          "bounds": [1.8, 41.25, 3.0, 42.1]}

GAUGE_CSV = os.path.join(REPO, "data/wflow/catalonia_2026/gauges.csv")

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

    print("setup_basemaps (19 coastal basins Besos..Tordera + grid)…")
    m.setup_basemaps(region=REGION, res=RES, hydrography_fn="merit_hydro")
    print("setup_rivers…")
    m.setup_rivers(hydrography_fn="merit_hydro", river_geom_fn="hydro_rivers_lin", river_upa=5)
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
    # Snap by upstream area to the ACA official areas: nearest-river snapping put
    # TO-15 (riera de Santa Coloma, 322 km2) on the Tordera just below the
    # confluence (762 km2). Window 3 cells, 5% tolerance.
    gauges["uparea"] = gauges["official_km2"]
    print(f"setup_gauges ({len(gauges)} points) + outlets…")
    m.setup_gauges(gauges_fn=gauges, index_col="id", snap_to_river=True,
                   snap_uparea=True, wdw=3, rel_error=0.05,
                   derive_subcatch=True, basename="catalonia",
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

    # Throwaway build precipitation from ERA5, as for the Sudetes model (the
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
