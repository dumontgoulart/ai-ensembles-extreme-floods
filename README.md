# AI weather ensembles on recent extreme floods

Code and derived scores for the HESS Brief Communication *Evaluating AI-based global medium-range weather forecasts on recent extreme flood events* (Goulart, Imhoff, Moreno-Rodenas, Weerts and Schellekens, in preparation).

The study compares the precipitation of three ensembles over the catchments of 47 independent river gauges in eleven recent extreme floods in Europe and the USA:
- ECMWF's physics-based IFS ENS;
- ECMWF's AIFS ENS;
- Google DeepMind's WeatherNext 2 (WN2).

It scores the catchment-average rainfall totals against a gauge-checked reference rainfall. It also scores the discharge that each member produces in wflow_sbm against a reference run of the same model.

The repository starts from data that have already been downloaded. Download scripts, cluster job scripts and the raw forecast and observation data are not included (see [Data](#data)).

## Reproduce the paper's figures and tables from the included scores

The derived scores of every event are included (about 1 MB), so the analysis runs without any model or forecast data:

```bash
python scripts/backbone_analysis.py      # all tables in figures/backbone/ and Figs. 1, 2, S12-S15
python scripts/si_windows_table.py       # Table S2
python scripts/si_robustness.py          # numbers of Sect. S3, Tables S3-S5
```

`python scripts/backbone_analysis.py --figs-only` redraws the figures from the saved tables in a few seconds. The per-event figures (Figs. S1–S11) also need the simulated hydrographs, so they are skipped when those are absent.

| Paper item | File (in `figures/backbone/`) | Script |
|---|---|---|
| Fig. 1 | `fig2_rain_metrics_vs_lead.png` | `backbone_analysis.py` |
| Fig. 2 | `fig3b_peak_volume.png` | `backbone_analysis.py` |
| Table S1 (reference run / observed peak) | `../aggregate_2/synthesis/reference_run_vs_observed.csv` | `reference_run_vs_observed.py` |
| Table S2 | `windows_counts.md` | `si_windows_table.py` |
| Tables S3–S5, Sect. S3 | `si_robustness.txt` | `si_robustness.py` |
| Figs. S1–S11 | `event_<event>.png` | `backbone_analysis.py` (needs model outputs) |
| Fig. S12 | `fig4_trends.png` | `backbone_analysis.py` |
| Fig. S13 | `fig6_tail_hits.png` | `backbone_analysis.py` |
| Fig. S14 | `fig2alt_discharge_crpss.png` | `backbone_analysis.py` |
| Fig. S15 | `fig7_rain_to_discharge_wn2.png` | `backbone_analysis.py` |

The CRPSS values and tallies quoted in the text are in `meteo_crpss.csv`, `discharge_crpss.csv` and the `*_best_*.csv` tables.

## Included data

| Path | Content |
|---|---|
| `figures/<event>/upstream_rain/upstream_rain_scores.csv` | Rainfall scores per gauge, product and initialisation (fair CRPS, bias, rank, …) |
| `figures/<event>/upstream_rain/upstream_rain_members.csv.gz` | Window total of every member, per gauge, product and initialisation |
| `figures/<event>/gauge_discharge/discharge_scores.csv` | Discharge scores per gauge, product and start (peak and event volume) |
| `data/wflow/<event>/gauges.csv`, `gauge_upstream.geojson`, `gauge_upstream.nc` | Gauges, their upstream areas on the model grid, and the area masks |
| `data/forecast_starts.csv` | Actual initialisation time of every forecast case (read from the processed forecasts) |
| `figures/backbone/` | All tables behind the paper |

Event folders: `catalonia_2026`, `boris_sudetes` (Storm Boris), `texas_2025`, `texas_2026`, `mississippi_2026`, `helene_2024`, `hans_2023`, `zermatt_2024` (Valais/Vispa), `california_2023`, `kristin_mondego` and `kristin_lis` (Storm Kristin), and `dana_2024` (Valencia).

## Full pipeline, from downloaded data

Event settings (domains, initialisations, windows, gauges) are in `scripts/event_config.py`. The case settings of the hydrological runs are in `scripts/wflow_setup.py` and `scripts/wflow_prep_forcing.py`.

1. **Forecast preprocessing.** IFS ENS and AIFS ENS GRIB files (51 members, every 00 and 12 UTC initialisation) and WN2 NetCDF files (64 members) under `data/raw/<event>/` are de-accumulated and cropped to the event domain:
   ```bash
   AIFS_EVENT=<event> python scripts/preprocess.py     # -> data/processed/<event>/
   ```
2. **Reference rainfall** (details in Sect. S1 of the supplement):

   | Event | Reference | Script |
   |---|---|---|
   | Catalonia | Meteocat XEMA gauge grid, 30 min | `xema_grid.py` |
   | Storm Boris | CHMI and IMGW gauge grid, hourly | `gauge_grid.py` |
   | Valais/Vispa | MeteoSwiss station grid rescaled to RhiresD | `gauge_grid.py`, `rhiresd_rescale.py` |
   | DANA (Valencia) | AEMET, CHJ SAIH and partner gauges with sub-daily timing | `dana_reference_rain.py` |
   | US events | MRMS scaled to GHCN-D daily gauges | `mrms_gaugescaled.py`, `california_mrms_gaugescaled.py` |
   | Storm Kristin | ERA5 (and the gauge-scaled alternative of Table S1) | `extend_era5_for_wflow.py`, `kristin_reference_rain.py`, `kristin_era5_reference.py` |
   | Storm Hans | MSWEP V3.16 | `extract_mswep.py` |

3. **Hydrological models** (wflow_sbm, about 1 km, built with HydroMT-Wflow):
   - `wflow_build_<event>.py` builds the models for Catalonia, Storm Boris (`sudetes`), Valais/Vispa, Helene, California and the 2026 Texas and Mississippi events.
   - Storm Kristin and DANA start from a European wflow_sbm set-up, clipped with `wflow_clip.py`.
   - Storm Hans uses a calibrated Drammen model (`wflow_drammen_forcing.py`).
   - `wflow_add_gauges.py` places the gauges, and `gauge_upstream.py` derives their upstream areas.
   - ERA5 temperature and potential evaporation are regridded to the model grid by HydroMT-Wflow, without elevation correction; `wflow_build_meteo.py` covers the events whose models came without them.
4. **Reference run, spin-ups, member runs and scores** for one event:
   ```bash
   python scripts/run_event.py <event> --workers 4
   ```
   - The script builds the forcing and runs one spin-up per forecast start, with a check that a warm restart reproduces an uninterrupted run.
   - Every member runs warm-started from its start (`wflow_ensemble.py`), then the continuous cases (`wflow_run.py`).
   - Finally it scores rainfall and discharge (`upstream_rain_scores.py`, `gauge_discharge_scores.py`).
   - Finished member runs are skipped, so an interrupted run can be restarted.
5. **Analysis:** `reference_run_vs_observed.py`, then the three scripts of the first section.

### Paths to external data

| Variable | Default | Used for |
|---|---|---|
| `MSWEP_ROOT` | `data/external/MSWEP_V316` | MSWEP V3.16 3-hourly (`Past/3hourly`, `NRT/3hourly`), the spin-up rainfall |
| `HYDROMT_METEO` | `data/external/hydromt_meteo` | ERA5 daily and CHIRPS files read by the model builders |
| `WFLOW_EUROPE_MODELS` | `data/external/wflow_europe` | European wflow_sbm set-up (Storm Kristin, DANA) |
| `WFLOW_ENS_SCRATCH` | `$TMPDIR` | Temporary member forcing |
| `WFLOW_THREADS` | `1` in `run_event.py` | Julia threads per member run |

Some optional steps read credentials from `~/.netrc`: IMERG (NASA Earthdata) in the DANA and Kristin reference scripts, and ERA5 (Destination Earth Data Hub) in `wflow_build_meteo.py`.

## Data

The raw data are not redistributed here:
- IFS ENS and AIFS ENS forecasts are available from the ECMWF MARS archive. AIFS ENS is archived from 1 July 2025, and the 10 June 2026 AIFS ENS run comes from the ECMWF open data.
- WeatherNext 2 forecasts are available from Google Cloud Storage.
- ERA5 is available from the Copernicus Climate Data Store, and MSWEP from GloH2O.
- MRMS and GHCN-D come from NOAA, and IMERG from NASA.
- Rain-gauge and discharge data come from Meteocat (XEMA), ACA, CHMI, IMGW, USGS, NVE, MET Norway, MeteoSwiss (including RhiresD), BAFU, IPMA (via Meteostat), APA, AEMET and CHJ (SAIH).
- The AVAMET and SISRITEL networks for the Valencia storm were compiled by CSIC Climatoc-Lab (Beguería et al., 2024, https://doi.org/10.20350/digitalCSIC/16716).

## Software

The results were produced with:
- Python 3.11, with numpy 2.3, pandas 3.0, xarray 2026.2, scipy 1.17, matplotlib 3.10, geopandas 1.1, rasterio 1.4, HydroMT 1.3, HydroMT-Wflow 1.0.2 and pyflwdir 0.5 (`environment.yml`);
- cfgrib 0.9.15 and eccodes 2.44 for the GRIB files;
- Wflow.jl 1.0.2 in Julia, which `wflow_run.py` and `wflow_ensemble.py` call as `julia --project=@wflow1 -e "using Wflow; Wflow.run()"`.

## Licence

MIT (see `LICENSE`).
