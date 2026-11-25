"""
Canonical product styling — single source of truth for ALL figures
(analysis, ensemble_scan, basin_analysis, aggregate_analysis, wflow analyses).

Validation / reference datasets  → SOLID, DARK, a bit THICKER.
Forecast products                → distinct colours; each figure keeps its own
                                    linewidth and linestyle (only the colour is
                                    standardised here).
"""

# ── Reference / validation datasets (solid, dark, thicker) ──────────────────
REF_COLOR = {
    "era5":  "#000000",   # black
    "mswep": "#8B0000",   # dark red
    "imerg": "#006400",   # dark green
    "eobs":  "#404040",   # dark gray
}
REF_LS = "-"      # always solid for reference/validation data
REF_LW = 2.6      # a bit thicker than the forecast lines

# Convenience scalars (back-compat with existing *_COLOR constants)
ERA5_COLOR  = REF_COLOR["era5"]
MSWEP_COLOR = REF_COLOR["mswep"]
IMERG_COLOR = REF_COLOR["imerg"]
EOBS_COLOR  = REF_COLOR["eobs"]

# ── Forecast products (distinct colours; keep each figure's lw + linestyle) ──
FCST_COLOR = {
    "ifs":  "#1f77b4",   # blue
    "aifs": "#ff7f0e",   # orange
    "wn2":  "#9467bd",   # purple
    "wn3":  "#2ca02c",   # green
}
IFS_COLOR  = FCST_COLOR["ifs"]
AIFS_COLOR = FCST_COLOR["aifs"]
WN2_COLOR  = FCST_COLOR["wn2"]
WN3_COLOR  = FCST_COLOR["wn3"]
