#!/usr/bin/env python3
"""
Full upstream area of every model gauge, from the wflow flow directions.

hydromt's subcatchment map is incremental (each cell belongs to the nearest
gauge downstream), so a main-river gauge would lose its nested tributaries.
Here every gauge gets its whole upstream area, as a mask on the model grid and
as an outline, for rain-over-the-catchment scoring (step 2 of the extreme-gauge
framework) and for maps.

Outputs, next to the model:
    data/wflow/<slug>/gauge_upstream.nc       mask(gauge, latitude, longitude)
    data/wflow/<slug>/gauge_upstream.geojson  one polygon per gauge, with the
                                              gauges.csv attributes and model_km2

    python scripts/gauge_upstream.py catalonia_2026 [--gauge-map gauges_catalonia]
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import rasterio.features
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import unary_union

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Models whose static maps use other names or a projected grid (the Drammen
# model: ETRS89 / UTM 33N, 1 km, wflow v0 names). Default: hydromt v1 lat/lon.
GRID = {
    "hans_2023": {"staticmaps": "staticmaps/staticmaps_v5_updated_reservoirs.nc", "ldd": "wflow_ldd",
                  "gauges": "wflow_gauges_drammen", "upa": "wflow_uparea"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slug")
    ap.add_argument("--gauge-map", default=None, help="staticmaps variable (default: the only gauges_* map)")
    a = ap.parse_args()
    root = os.path.join(REPO, "data", "wflow", a.slug)
    gc = GRID.get(a.slug, {})
    sm = xr.open_dataset(os.path.join(root, "model", gc.get("staticmaps", "staticmaps.nc")))
    gname = a.gauge_map or gc.get("gauges") or next(v for v in sm.data_vars if v.startswith("gauges_"))
    projected = "x" in sm.dims and "latitude" not in sm.dims
    ydim, xdim = ("y", "x") if projected else ("latitude", "longitude")
    lat, lon = sm[ydim].values, sm[xdim].values            # northing/easting when projected
    res = abs(float(lon[1] - lon[0]))
    north = lat[0] > lat[-1]
    if not north:
        raise ValueError("expected north-up staticmaps")
    transform = from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res)
    crs = (sm["spatial_ref"].attrs.get("crs_wkt") if projected and "spatial_ref" in sm else "EPSG:4326")

    ldd = np.nan_to_num(sm[gc.get("ldd", "local_drain_direction")].values.astype("float64"), nan=255).astype("uint8")
    fd = pyflwdir.from_array(ldd, ftype="ldd", transform=transform, latlon=not projected)
    gmap = sm[gname].values
    upa = sm[gc.get("upa", "meta_upstream_area")].values
    cell_km2 = (np.full(gmap.shape, res * res / 1e6) if projected
                else pyflwdir.gis_utils.reggrid_area(lat, lon) / 1e6)

    gauges = pd.read_csv(os.path.join(root, "gauges.csv"))
    masks, rows = [], []
    for _, g in gauges.iterrows():
        rr, cc = np.where(gmap == g["id"])
        if len(rr) != 1:
            raise ValueError(f"gauge {g['id']}: {len(rr)} cells in {gname}")
        idx = rr[0] * gmap.shape[1] + cc[0]
        m = fd.basins(idxs=np.array([idx]), ids=np.array([1], dtype=np.int32)) == 1
        poly = unary_union([shape(s) for s, v in rasterio.features.shapes(
            m.astype("uint8"), mask=m, transform=transform) if v == 1])
        masks.append(m)
        rows.append({**g.to_dict(), "model_km2": round(float(cell_km2[m].sum()), 1),
                     "upa_km2": round(float(upa[rr[0], cc[0]]), 1), "n_cells": int(m.sum()),
                     "geometry": poly})
    mask = xr.DataArray(np.stack(masks), dims=("gauge", ydim, xdim),
                        coords={"gauge": gauges["id"].values, ydim: lat, xdim: lon},
                        name="mask")
    mask.attrs["note"] = "full upstream area of each gauge (gauges.csv id), from local_drain_direction"
    mask.to_netcdf(os.path.join(root, "gauge_upstream.nc"), encoding={"mask": {"dtype": "uint8", "zlib": True}})
    gdf = gpd.GeoDataFrame(rows, crs=crs).to_crs("EPSG:4326")
    gdf.to_file(os.path.join(root, "gauge_upstream.geojson"), driver="GeoJSON")
    cols = [c for c in ("id", "station", "name", "official_km2", "model_km2", "upa_km2", "n_cells") if c in gdf]
    print(gdf[cols].to_string(index=False))


if __name__ == "__main__":
    main()
