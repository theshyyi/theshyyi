#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Performance diagram (POD vs 1-FAR) for multiple precipitation products.

This script:
  1) Treats all files matching "*.TIMEFIX.daily.CHINA.nc" under NC_DIR as
     precipitation products.
  2) Uses REF_NAME as the reference product.
  3) Computes POD and FAR relative to the reference at:
       - 4 seasons (MAM/JJA/SON/DJF)
       - 12 months (1-12)
  4) Draws performance diagrams:
       - 2x2 seasonal panels
       - 4x3 monthly panels

Notes
-----
- A China domain mask is built from SHP_PATH (auto reprojected to EPSG:4326).
- The code aims to avoid label/legend overlap by:
    * using smaller, consistent font sizes
    * showing axis labels only on outer panels
    * reserving bottom margin for a figure-level legend
    * reserving right margin for a dedicated colorbar axis
    * saving with bbox_inches="tight" to prevent cropping

Dependencies
------------
    pip install numpy xarray matplotlib geopandas regionmask
"""

from __future__ import annotations

import os
import glob
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask
from matplotlib import rcParams, font_manager


# ============================ User configuration ============================

NC_DIR = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish"  # folder containing nc files
REF_NAME = "CMFDV2"                                   # reference product prefix
SHP_PATH = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp"
THRESHOLD = 0.1                                        # event threshold (mm/day)

OUT_FIG_SEASON = "performance_seasons_01mm.png"
OUT_FIG_MONTH = "performance_months_01mm.png"

# ===========================================================================


def set_font() -> None:
    """Global font settings (Times New Roman primary; SimHei as fallback)."""
    font_path_tnr = "/home/ud202380664/Times_New_Roman.ttf"
    font_path_simhei = "/home/ud202380664/Ubuntu_18.04_SimHei.ttf"

    # Safely add fonts if they exist.
    if os.path.exists(font_path_tnr):
        font_manager.fontManager.addfont(font_path_tnr)
    if os.path.exists(font_path_simhei):
        font_manager.fontManager.addfont(font_path_simhei)

    rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],
        "font.size": 14,            # smaller than 16 to reduce overlap
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.dpi": 600,
    })


def find_precip_var(ds: xr.Dataset) -> str:
    """Find a precipitation variable containing (time, lat, lon) dimensions."""
    for v in ds.data_vars:
        da = ds[v]
        if {"time", "lat", "lon"}.issubset(set(da.dims)):
            return v
    raise ValueError(
        "No variable with dims including {time, lat, lon} was found. "
        "Please update find_precip_var() or standardize variable/dims first."
    )


def build_china_mask(ref_ds: xr.Dataset, shp_path: str) -> xr.DataArray | None:
    """Build a boolean mask (lat, lon) from a shapefile. True indicates inside."""
    if not shp_path or (not os.path.exists(shp_path)):
        print("No valid SHP provided, skip spatial mask.")
        return None

    print(f">>> Reading shapefile for mask: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)
    else:
        print("Warning: shapefile has no CRS; assuming EPSG:4326.")

    reg = regionmask.from_geopandas(gdf)
    mask = reg.mask(ref_ds, lon_name="lon", lat_name="lat")
    inside = mask.notnull()
    return inside


def subset_by_months(da: xr.DataArray, months: List[int]) -> xr.DataArray:
    """Subset a DataArray by month list."""
    return da.sel(time=da["time"].dt.month.isin(months))


def compute_pod_far(
    obs: xr.DataArray,
    sim: xr.DataArray,
    threshold: float,
    china_mask: xr.DataArray | None = None,
) -> Tuple[float, float]:
    """Compute POD and FAR on (time, lat, lon) grids.

    POD = hits / (hits + misses)
    FAR = false_alarm / (hits + false_alarm)
    """
    # align time/space
    obs, sim = xr.align(obs, sim, join="inner")

    if china_mask is not None:
        obs = obs.where(china_mask)
        sim = sim.where(china_mask)

    valid = np.isfinite(obs) & np.isfinite(sim)

    obs_evt = obs >= threshold
    sim_evt = sim >= threshold

    hits = float(((obs_evt & sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True))
    miss = float(((obs_evt & ~sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True))
    fa = float(((~obs_evt & sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True))

    pod = np.nan
    far = np.nan
    if (hits + miss) > 0:
        pod = hits / (hits + miss)
    if (hits + fa) > 0:
        far = fa / (hits + fa)

    return pod, far


def compute_all_stats(
    nc_dir: str,
    ref_name: str,
    shp_path: str | None,
    threshold: float,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], Dict[int, Dict[str, Tuple[float, float]]], List[str]]:
    """Compute seasonal and monthly POD/FAR for all products."""

    nc_paths = sorted(glob.glob(os.path.join(nc_dir, "*.TIMEFIX.daily.CHINA.nc")))
    if not nc_paths:
        raise FileNotFoundError(f"No *.TIMEFIX.daily.CHINA.nc found under: {nc_dir}")

    ref_path = None
    product_paths: Dict[str, str] = {}

    for p in nc_paths:
        prod = os.path.basename(p).split(".")[0]
        if prod == ref_name:
            ref_path = p
        else:
            product_paths[prod] = p

    if ref_path is None:
        raise FileNotFoundError(
            f"Reference file not found: {ref_name}.TIMEFIX.daily.CHINA.nc in {nc_dir}"
        )

    products = list(product_paths.keys())
    print(f"Reference: {ref_name} -> {ref_path}")
    print(f"Products ({len(products)}): {products}")

    # open reference
    ref_ds = xr.open_dataset(ref_path)
    ref_var = find_precip_var(ref_ds)
    ref_da = ref_ds[ref_var]

    # build mask on the reference grid
    china_mask = build_china_mask(ref_ds, shp_path) if shp_path else None
    if china_mask is not None:
        print("China mask built from shapefile.")

    seasons_def = {
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
        "Winter": [12, 1, 2],
    }
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    months = list(range(1, 13))

    season_stats: Dict[str, Dict[str, Tuple[float, float]]] = {s: {} for s in season_order}
    month_stats: Dict[int, Dict[str, Tuple[float, float]]] = {m: {} for m in months}

    for prod, path in product_paths.items():
        print(f"\nProcessing: {prod} -> {path}")
        sim_ds = xr.open_dataset(path)
        sim_var = find_precip_var(sim_ds)
        sim_da = sim_ds[sim_var]

        for sname in season_order:
            mon_list = seasons_def[sname]
            obs_s = subset_by_months(ref_da, mon_list)
            sim_s = subset_by_months(sim_da, mon_list)
            pod, far = compute_pod_far(obs_s, sim_s, threshold, china_mask)
            season_stats[sname][prod] = (pod, far)

        for m in months:
            obs_m = subset_by_months(ref_da, [m])
            sim_m = subset_by_months(sim_da, [m])
            pod, far = compute_pod_far(obs_m, sim_m, threshold, china_mask)
            month_stats[m][prod] = (pod, far)

        sim_ds.close()

    ref_ds.close()
    print("\nAll POD/FAR computed.")
    return season_stats, month_stats, products


# =============================== Plotting ===============================


def _get_color_marker_maps(products: List[str]):
    """Assign each product a deterministic color and marker."""
    n_prod = len(products)

    # Use a larger categorical colormap when products > 10.
    if n_prod <= 10:
        cmap = plt.cm.get_cmap("tab10", n_prod)
    elif n_prod <= 20:
        cmap = plt.cm.get_cmap("tab20", n_prod)
    else:
        cmap = plt.cm.get_cmap("gist_ncar", n_prod)

    markers = [
        "o", "s", "^", "v", "<", ">", "D", "P", "X", "*",
        "h", "H", "p", "8", "+", "x", "1", "2", "3", "4",
    ]

    color_map = {p: cmap(i) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}
    return color_map, marker_map


def draw_performance_background(
    ax,
    csi_levels=np.arange(0.1, 1.0, 0.1),
):
    """Draw CSI shading/contours and bias lines; return contourf for colorbar."""

    # SR (=1-FAR) and POD grid
    sr = np.linspace(0.01, 1.0, 200)
    pod = np.linspace(0.01, 1.0, 200)
    SR, POD = np.meshgrid(sr, pod)

    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    CSI[(CSI < 0) | ~np.isfinite(CSI)] = np.nan

    cf = ax.contourf(
        SR,
        POD,
        CSI,
        levels=np.linspace(0.1, 0.9, 9),
        cmap="Blues",
        extend="both",
    )

    cs = ax.contour(
        SR,
        POD,
        CSI,
        levels=csi_levels,
        colors="k",
        linewidths=0.6,
        linestyles="dashed",
    )
    ax.clabel(cs, inline=True, fontsize=10, fmt="%.1f")

    # bias lines: bias = POD / SR
    bias_values = [0.5, 0.7, 1.0, 1.5, 2.0]
    sr_line = np.linspace(0.01, 1.0, 200)
    for b in bias_values:
        pod_line = b * sr_line
        pod_line[pod_line > 1.0] = np.nan
        ax.plot(sr_line, pod_line, "k--", linewidth=0.8)

        # place label near the visible end
        if np.all(np.isnan(pod_line)):
            continue
        idx = np.nanargmax(pod_line)
        ax.text(
            sr_line[idx],
            pod_line[idx] + 0.02,
            f"{b:.1f}",
            fontsize=10,
            ha="center",
            va="bottom",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)

    return cf


def plot_season_figure(season_stats, products: List[str], out_png: str) -> None:
    """2x2 seasonal performance diagrams with figure-level legend and colorbar."""

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4), dpi=600, sharex=True, sharey=True)
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    color_map, marker_map = _get_color_marker_maps(products)

    cf_for_cbar = None
    for i, (season, label) in enumerate(zip(season_order, panel_labels)):
        ax = axes.flat[i]
        cf = draw_performance_background(ax)
        if cf_for_cbar is None:
            cf_for_cbar = cf

        for p in products:
            pod, far = season_stats[season][p]
            if np.isnan(pod) or np.isnan(far):
                continue
            sr = 1.0 - far
            ax.scatter(
                sr,
                pod,
                marker=marker_map[p],
                color=color_map[p],
                s=55,
                edgecolors="k",
                linewidths=0.6,
                zorder=5,
            )

        ax.text(
            0.02,
            0.95,
            f"{label} {season}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2),
        )
        ax.tick_params(labelsize=12)

    # show axis labels only on outer panels
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    axes[1, 0].set_ylabel("Probability of Detection (POD)", fontsize=14)
    axes[1, 1].set_xlabel("Success Ratio (1 - FAR)", fontsize=14)

    # legend handles
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker_map[p],
            markersize=6,
            markerfacecolor=color_map[p],
            markeredgecolor="k",
            label=p,
        )
        for p in products
    ]

    # layout: reserve bottom for legend, right for colorbar
    fig.subplots_adjust(left=0.08, right=0.88, top=0.96, bottom=0.26, wspace=0.22, hspace=0.22)

    fig.legend(
        handles,
        products,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.08),
        ncol=min(len(products), 7),
        frameon=False,
        fontsize=12,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    # colorbar axis
    cax = fig.add_axes([0.90, 0.30, 0.02, 0.60])
    cbar = fig.colorbar(cf_for_cbar, cax=cax)
    cbar.set_label("Critical Success Index (CSI)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved seasonal figure: {out_png}")


def plot_month_figure(month_stats, products: List[str], out_png: str) -> None:
    """4x3 monthly performance diagrams with figure-level legend and colorbar."""

    fig, axes = plt.subplots(4, 3, figsize=(11.0, 12.2), dpi=600, sharex=True, sharey=True)
    months = list(range(1, 13))
    panel_labels = [f"({chr(97 + i)})" for i in range(12)]  # (a)~(l)

    color_map, marker_map = _get_color_marker_maps(products)

    cf_for_cbar = None
    for i, (m, label) in enumerate(zip(months, panel_labels)):
        ax = axes.flat[i]
        cf = draw_performance_background(ax)
        if cf_for_cbar is None:
            cf_for_cbar = cf

        for p in products:
            pod, far = month_stats[m][p]
            if np.isnan(pod) or np.isnan(far):
                continue
            sr = 1.0 - far
            ax.scatter(
                sr,
                pod,
                marker=marker_map[p],
                color=color_map[p],
                s=40,
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
            )

        ax.text(
            0.02,
            0.95,
            f"{label} M{m:02d}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
        )
        ax.tick_params(labelsize=10)

    # outer labels only
    for ax in axes[:-1, :].ravel():
        ax.set_xlabel("")
    for ax in axes[:, 1:].ravel():
        ax.set_ylabel("")

    axes[-1, 1].set_xlabel("Success Ratio (1 - FAR)", fontsize=14)
    axes[1, 0].set_ylabel("Probability of Detection (POD)", fontsize=14)

    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker_map[p],
            markersize=5.5,
            markerfacecolor=color_map[p],
            markeredgecolor="k",
            label=p,
        )
        for p in products
    ]

    # layout: reserve bottom for legend (bigger for monthly), right for colorbar
    fig.subplots_adjust(left=0.07, right=0.88, top=0.97, bottom=0.22, wspace=0.18, hspace=0.18)

    fig.legend(
        handles,
        products,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.08),
        ncol=min(len(products), 9),
        frameon=False,
        fontsize=10,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    cax = fig.add_axes([0.90, 0.26, 0.02, 0.66])
    cbar = fig.colorbar(cf_for_cbar, cax=cax)
    cbar.set_label("Critical Success Index (CSI)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved monthly figure: {out_png}")


def main() -> None:
    set_font()

    season_stats, month_stats, products = compute_all_stats(
        NC_DIR, REF_NAME, SHP_PATH, THRESHOLD
    )

    plot_season_figure(season_stats, products, OUT_FIG_SEASON)
    plot_month_figure(month_stats, products, OUT_FIG_MONTH)


if __name__ == "__main__":
    main()
