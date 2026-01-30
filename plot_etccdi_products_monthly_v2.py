#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ETCCDI monthly multi-product plotting & evaluation (v2, STREAMING, journal-ready)
================================================================================

Key goals:
- Memory-safe: do NOT keep all xarray Datasets in memory; stream by index/product
- China-only: mask outside China using 7-region union mask; set extent to China boundary
- Lat/Lon ticks: only left column (lat) and bottom row (lon)
- Overlays: China outline + 7-region boundaries + South China Sea nine-dash-line inset
- Publication style: larger fonts, robust color scales, consistent colormaps
"""

import os
import re
import json
import math
import glob
import gc
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# cartopy optional
try:
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
except Exception:
    ccrs = None


# -----------------------------
# Basic I/O
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def product_name_from_filename(fn: str) -> str:
    base = os.path.basename(fn)
    m = re.match(r"^(.+?)\.TIMEFIX\.daily\.CHINA-ETCCDI-yearly\.nc$", base)
    if m:
        return m.group(1)
    if ".TIMEFIX" in base:
        return base.split(".TIMEFIX")[0]
    return os.path.splitext(base)[0]


# -----------------------------
# Fonts & Style
# -----------------------------
def setup_fonts(font_path_tnr: str, font_path_simhei: str) -> Tuple[str, str]:
    if os.path.isfile(font_path_tnr):
        font_manager.fontManager.addfont(font_path_tnr)
        tnr_name = font_manager.FontProperties(fname=font_path_tnr).get_name()
    else:
        tnr_name = "Times New Roman"

    if os.path.isfile(font_path_simhei):
        font_manager.fontManager.addfont(font_path_simhei)
        simhei_name = font_manager.FontProperties(fname=font_path_simhei).get_name()
    else:
        simhei_name = "SimHei"

    return tnr_name, simhei_name


def apply_rcparams(style_cfg: dict, tnr_name: str):
    rcParams.update({
        "font.family": tnr_name,
        "axes.unicode_minus": False,

        "axes.titlesize": style_cfg.get("axes_title_size", 13),
        "axes.labelsize": style_cfg.get("axes_label_size", 12),
        "xtick.labelsize": style_cfg.get("tick_label_size", 11),
        "ytick.labelsize": style_cfg.get("tick_label_size", 11),
        "legend.fontsize": style_cfg.get("legend_size", 11),

        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.8,
        "ytick.major.size": 3.8,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03
    })


# -----------------------------
# Data utilities
# -----------------------------
def to_float64(da: xr.DataArray) -> xr.DataArray:
    if str(da.dtype).startswith("int") or str(da.dtype).startswith("uint"):
        return da.astype("float64")
    return da


def ensure_latlon_order(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure 2D da is ordered as (lat, lon).
    """
    if ("lat" in da.dims) and ("lon" in da.dims):
        if da.dims == ("lat", "lon"):
            return da
        if da.dims == ("lon", "lat"):
            return da.transpose("lat", "lon")
        # fallback
        return da.transpose("lat", "lon")
    return da


def sample_values(arr: np.ndarray, max_n: int) -> np.ndarray:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return v
    if v.size <= max_n:
        return v
    return np.random.choice(v, size=max_n, replace=False)


def robust_minmax(values: np.ndarray, p_low: float, p_high: float) -> Tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 1.0
    vmin = np.nanpercentile(v, p_low)
    vmax = np.nanpercentile(v, p_high)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
        vmin = float(np.nanmin(v))
        vmax = float(np.nanmax(v))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax == vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def symmetric_minmax(values: np.ndarray, center: float = 0.0, p_high: float = 98) -> Tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return -1.0, 1.0
    hi = np.nanpercentile(np.abs(v - center), p_high)
    if (not np.isfinite(hi)) or hi == 0:
        hi = float(np.nanmax(np.abs(v - center)))
        if (not np.isfinite(hi)) or hi == 0:
            hi = 1.0
    return center - hi, center + hi


# -----------------------------
# Regions
# -----------------------------
def build_regions_from_shp(
    shp_path: str,
    climate_field: str,
    climate_name_map: dict,
    climate_abbr_map: dict
):
    gdf = gpd.read_file(shp_path)

    if climate_field not in gdf.columns:
        raise ValueError(f"Shapefile missing field '{climate_field}': {shp_path}")

    if gdf.crs is None:
        warnings.warn("Shapefile CRS is None. Assuming EPSG:4326.")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf_reg = gdf.dissolve(by=climate_field, as_index=False)
    gdf_reg = gdf_reg.sort_values(by=climate_field).reset_index(drop=True)

    region_names_cn = [str(v) for v in gdf_reg[climate_field].tolist()]
    region_names_en = [climate_name_map.get(x, x) for x in region_names_cn]
    region_names_abbr = [climate_abbr_map.get(x, x) for x in region_names_cn]

    regions = regionmask.Regions(
        outlines=list(gdf_reg.geometry),
        names=region_names_en,
        numbers=list(range(len(region_names_en))),
        name="ChinaClimate7"
    )

    gdf_outline = gdf.dissolve()
    return gdf, gdf_reg, gdf_outline, regions, region_names_cn, region_names_en, region_names_abbr


def regions_mask(regions, lon: np.ndarray, lat: np.ndarray) -> xr.DataArray:
    try:
        m = regions.mask(lon=lon, lat=lat)
    except TypeError:
        m = regions.mask(lon, lat)
    return m


def get_china_mask_and_extent(ds_ref: xr.Dataset, regions_obj, gdf_outline):
    lon = ds_ref["lon"].values
    lat = ds_ref["lat"].values
    reg_mask = regions_mask(regions_obj, lon=lon, lat=lat)  # (lat, lon)
    china_mask = xr.where(np.isfinite(reg_mask), 1.0, np.nan)

    minx, miny, maxx, maxy = gdf_outline.total_bounds
    pad = 0.5
    extent = (minx - pad, maxx + pad, miny - pad, maxy + pad)
    return china_mask, extent, reg_mask


# -----------------------------
# Nine-dash-line
# -----------------------------
def load_nine_dash_shp(nine_shp_path: str) -> gpd.GeoDataFrame:
    gdf9 = gpd.read_file(nine_shp_path)
    if gdf9.crs is None:
        warnings.warn("Nine-dash-line shp CRS is None; assuming EPSG:4326.")
    else:
        gdf9 = gdf9.to_crs("EPSG:4326")
    return gdf9

def add_nine_dash_inset(ax, da2d_full, gdf_outline, gdf_nine, nine_cfg: dict,
                        use_cartopy: bool, cmap: str, vmin: float, vmax: float):
    """
    在子图右下角画南海 inset：
    - inset 内绘制与主图同一指标的空间分布（用未 mask 的 da2d_full）
    - 叠加中国边界与九段线（黑色虚线）
    """
    if (not nine_cfg) or (not nine_cfg.get("enabled", False)) or (gdf_nine is None):
        return
    if (not use_cartopy) or (ccrs is None):
        return

    extent = nine_cfg.get("extent", [105, 125, 3, 25])

    iax = inset_axes(
        ax,
        width=nine_cfg.get("width", "22%"),
        height=nine_cfg.get("height", "22%"),
        loc=nine_cfg.get("loc", "lower right"),
        borderpad=float(nine_cfg.get("borderpad", 0.6)),
        axes_class=type(ax),
        axes_kwargs={"projection": ccrs.PlateCarree()}
    )
    iax.set_extent(extent, crs=ccrs.PlateCarree())

    # 1) 先画栅格（关键：用未 mask 的 da2d_full）
    da2d_full = ensure_latlon_order(da2d_full)
    lon = da2d_full["lon"].values
    lat = da2d_full["lat"].values
    data = da2d_full.values

    iax.pcolormesh(
        lon, lat, data,
        shading="auto",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    # 2) 中国边界
    try:
        gdf_outline.boundary.plot(
            ax=iax,
            linewidth=float(nine_cfg.get("outline_width", 0.5)),
            color=nine_cfg.get("outline_color", "k"),
            zorder=3
        )
    except Exception:
        pass

    # 3) 九段线：不要 boundary.plot，用 plot + 虚线
    try:
        gdf_nine.plot(
            ax=iax,
            linewidth=float(nine_cfg.get("line_width", 0.6)),
            color=nine_cfg.get("line_color", "k"),
            linestyle="--",
            zorder=4
        )
    except Exception:
        pass

    iax.set_xticks([])
    iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_linewidth(0.7)



# -----------------------------
# Plot helpers
# -----------------------------
def get_fig_axes(n_panels: int, ncols: int, use_cartopy: bool, style: dict):
    nrows = int(math.ceil(n_panels / ncols))
    fig_w = float(style.get("fig_w_per_col", 5.2))
    fig_h = float(style.get("fig_h_per_row", 4.6))
    figsize = (fig_w * ncols, fig_h * nrows)

    if use_cartopy and (ccrs is not None):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            dpi=150,
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True
        )
    else:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            dpi=150,
            constrained_layout=True
        )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)
    return fig, axes


def draw_boundaries(ax, gdf_outline, gdf_regions, lw_outline: float, lw_region: float):
    try:
        gdf_outline.boundary.plot(ax=ax, linewidth=lw_outline, color="k", zorder=6)
    except Exception:
        pass
    try:
        gdf_regions.boundary.plot(ax=ax, linewidth=lw_region, color="dimgray", zorder=7)
    except Exception:
        pass


def set_lonlat_ticks_for_grid(ax, is_left_col: bool, is_bottom_row: bool, extent: tuple, tick_fs: int):
    if ccrs is None:
        return

    ax.tick_params(labelsize=tick_fs)

    if is_left_col:
        y0, y1 = extent[2], extent[3]
        y_ticks = np.arange(math.floor(y0/5)*5, math.ceil(y1/5)*5 + 0.1, 10)
        ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    else:
        ax.set_yticks([])

    if is_bottom_row:
        x0, x1 = extent[0], extent[1]
        x_ticks = np.arange(math.floor(x0/10)*10, math.ceil(x1/10)*10 + 0.1, 20)
        ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
    else:
        ax.set_xticks([])


def plot_map_panel(ax, da2d: xr.DataArray, title: str,
                   vmin, vmax, cmap: str, use_cartopy: bool, coastline_lw: float,
                   china_mask: xr.DataArray, extent: tuple):
    da2d = ensure_latlon_order(da2d)
    da2d = to_float64(da2d).astype("float32")

    da2d_cn = da2d.where(np.isfinite(china_mask))
    lon = da2d_cn["lon"].values
    lat = da2d_cn["lat"].values
    data = da2d_cn.values  # (lat, lon)

    transform = ccrs.PlateCarree() if (use_cartopy and ccrs is not None) else None

    im = ax.pcolormesh(
        lon, lat, data,
        shading="auto",
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        transform=transform
    )
    ax.set_title(title)

    if use_cartopy and ccrs is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution="50m", linewidth=coastline_lw)
        except Exception:
            pass
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    return im

def add_region_boxplot_inset(ax, data_by_region: List[np.ndarray], labels: List[str], style: dict):
    """
    Absolute-positioned inset boxplot (bbox_to_anchor) to avoid overlapping lon/lat tick labels.
    Coordinates are in axes fraction (0–1).
    """
    # inset 的绝对位置与大小（轴坐标系 0~1）
    # 默认：左下角往里缩一些，避开经纬度刻度
    x0 = float(style.get("box_inset_x0", 0.10))   # 向右移动（越大越靠右）
    y0 = float(style.get("box_inset_y0", 0.12))   # 向上移动（越大越靠上）
    w  = float(style.get("box_inset_w", 0.36))    # 宽度（轴比例）
    h  = float(style.get("box_inset_h", 0.26))    # 高度（轴比例）

    iax = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(x0, y0, w, h),
        bbox_transform=ax.transAxes,
        borderpad=0.0
    )

    cleaned = []
    for v in data_by_region:
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        if v.size == 0:
            v = np.array([np.nan])
        cleaned.append(v)

    bp = iax.boxplot(
        cleaned,
        vert=True,
        showfliers=False,
        widths=0.55,
        patch_artist=False
    )

    # 线条更细（期刊风格）
    for k in ["boxes", "whiskers", "caps", "medians"]:
        for obj in bp.get(k, []):
            try:
                obj.set_linewidth(0.6)
            except Exception:
                pass

    # x 轴标签更紧凑
    iax.set_xticks(range(1, len(labels) + 1))
    iax.set_xticklabels(
        labels,
        rotation=int(style.get("box_inset_xtick_rotation", 45)),
        fontsize=int(style.get("box_inset_tick_size", 9)),
        ha="right"
    )
    iax.tick_params(axis="y", labelsize=int(style.get("box_inset_tick_size", 9)))
    iax.tick_params(axis="x", pad=0.2)
    iax.grid(alpha=0.18, linewidth=0.3)

    for spine in iax.spines.values():
        spine.set_linewidth(0.6)


# -----------------------------
# Metrics (streaming-friendly)
# -----------------------------
def calc_metrics_map(sim: xr.DataArray, obs: xr.DataArray) -> Dict[str, xr.DataArray]:
    sim2, obs2 = xr.align(sim, obs, join="inner")

    sim2 = to_float64(sim2)
    obs2 = to_float64(obs2)

    diff = sim2 - obs2
    rmse = np.sqrt((diff ** 2).mean("time", skipna=True))
    mae  = np.abs(diff).mean("time", skipna=True)
    bias = diff.mean("time", skipna=True)

    obs_mean = obs2.mean("time", skipna=True)
    denom = ((obs2 - obs_mean) ** 2).sum("time", skipna=True)
    numer = ((sim2 - obs2) ** 2).sum("time", skipna=True)
    nse = 1.0 - (numer / denom)
    nse = nse.where(np.isfinite(nse))

    rmse = ensure_latlon_order(rmse).astype("float32")
    mae  = ensure_latlon_order(mae).astype("float32")
    bias = ensure_latlon_order(bias).astype("float32")
    nse  = ensure_latlon_order(nse).astype("float32")

    return {"RMSE": rmse, "MAE": mae, "NSE": nse, "BIAS": bias}


def extract_region_arrays(da2d: xr.DataArray, reg_mask: xr.DataArray, n_regions: int, max_samples: int) -> List[np.ndarray]:
    """
    Return list of sampled arrays (one per region) for boxplot.
    """
    da2d = ensure_latlon_order(da2d)
    datav = da2d.values
    maskv = reg_mask.values
    out = []
    for i in range(n_regions):
        vv = datav[maskv == i]
        vv = vv[np.isfinite(vv)]
        if vv.size > max_samples:
            vv = np.random.choice(vv, size=max_samples, replace=False)
        out.append(vv)
    return out


# -----------------------------
# Annual aggregation for trends
# -----------------------------
def annual_aggregate(da: xr.DataArray, varname: str) -> xr.DataArray:
    da = to_float64(da)

    count_like = {"r10mm", "r20mm", "rr1", "cdd", "cwd"}
    amount_like = {"prcptot", "r95p", "r99p", "r95ptot", "r99ptot"}
    intensity_like = {"rx1day", "rx5day", "sdii"}

    if varname in count_like or varname in amount_like:
        return da.groupby("time.year").sum("time", skipna=True)
    elif varname in intensity_like:
        return da.groupby("time.year").mean("time", skipna=True)
    else:
        return da.groupby("time.year").mean("time", skipna=True)


def lin_trend(years: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(years) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    a, b = np.polyfit(years[mask], y[mask], 1)
    return float(a), float(b)


# -----------------------------
# Plot 1) Mean maps (STREAMING)
# -----------------------------
def plot_mean_maps_stream(
    product_paths: Dict[str, str],
    indices: List[str],
    ref_name: str,
    regions_obj,
    gdf_outline,
    gdf_regions,
    out_dir: str,
    ncols: int,
    dpi: int,
    style: dict,
    use_cartopy: bool,
    nine_cfg: dict,
    gdf_nine
):
    """
    Multi-year mean maps (China-only in main panels) with South China Sea inset showing
    the SAME raster field (unmasked), plus nine-dash dashed line.
    Streaming/memory-safe.
    """
    outp = os.path.join(out_dir, "01_INDEX_MEAN_MAPS")
    ensure_dir(outp)

    cmap_mean = style.get("cmap_index_mean", "cividis")
    p_low = style.get("robust_percentile_low", 2)
    p_high = style.get("robust_percentile_high", 98)
    sample_cbar = int(style.get("sample_for_colorbar", 20000))

    lw_outline = style.get("outline_linewidth", 0.8)
    lw_region = style.get("region_linewidth", 0.8)
    coastline_lw = style.get("coastline_linewidth", 0.4)
    tick_fs = int(style.get("tick_label_size", 11))

    # Build china_mask/extent on ref grid
    with xr.open_dataset(product_paths[ref_name], decode_times=True) as ds_ref:
        china_mask, extent, reg_mask = get_china_mask_and_extent(ds_ref, regions_obj, gdf_outline)

    prod_names = list(product_paths.keys())

    for idx in indices:
        # store both: masked for main panel, full for SCS inset
        means_cn: Dict[str, xr.DataArray] = {}
        means_full: Dict[str, xr.DataArray] = {}
        samples = []

        for p in prod_names:
            nc_path = product_paths[p]
            try:
                with xr.open_dataset(nc_path, decode_times=True) as ds:
                    if idx not in ds:
                        continue

                    da = ds[idx]

                    # full (unmasked) mean for inset
                    da2d_full = to_float64(da).mean("time", skipna=True)
                    da2d_full = ensure_latlon_order(da2d_full).astype("float32")

                    # China-only for main panel
                    da2d_cn = da2d_full.where(np.isfinite(china_mask))

                    means_full[p] = da2d_full
                    means_cn[p] = da2d_cn

                    vv = sample_values(da2d_cn.values, sample_cbar)
                    if vv.size > 0:
                        samples.append(vv)

            except Exception as e:
                print(f"[WARN] mean map failed: {p} {idx}: {e}")
                continue

        if not means_cn:
            print(f"[WARN] index '{idx}' not found, skip.")
            continue

        all_vals = np.concatenate(samples) if len(samples) else np.array([0.0], dtype="float32")
        vmin, vmax = robust_minmax(all_vals, p_low, p_high)

        fig, axes = get_fig_axes(len(means_cn), ncols=ncols, use_cartopy=use_cartopy, style=style)
        axes_flat = axes.ravel()
        last_im = None

        # preserve consistent ordering
        for i, p in enumerate(means_cn.keys()):
            ax = axes_flat[i]
            da2d_cn = means_cn[p]
            da2d_full = means_full[p]

            last_im = plot_map_panel(
                ax, da2d_cn, title=p,
                vmin=vmin, vmax=vmax, cmap=cmap_mean,
                use_cartopy=use_cartopy, coastline_lw=coastline_lw,
                china_mask=china_mask, extent=extent
            )

            draw_boundaries(ax, gdf_outline, gdf_regions, lw_outline, lw_region)

            row = i // ncols
            col = i % ncols
            is_left = (col == 0)
            is_bottom = (row == (axes.shape[0] - 1))
            set_lonlat_ticks_for_grid(ax, is_left, is_bottom, extent=extent, tick_fs=tick_fs)

            # South China Sea inset uses FULL raster (unmasked)
            add_nine_dash_inset(
                ax,
                da2d_full=da2d_full,
                gdf_outline=gdf_outline,
                gdf_nine=gdf_nine,
                nine_cfg=nine_cfg,
                use_cartopy=use_cartopy,
                cmap=cmap_mean,
                vmin=vmin, vmax=vmax
            )

        for j in range(len(means_cn), len(axes_flat)):
            axes_flat[j].axis("off")

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=[a for a in axes_flat if a.has_data()], fraction=0.03, pad=0.02)
            cbar.set_label(idx, fontsize=int(style.get("colorbar_label_size", 12)))

        fig.suptitle(f"{idx} Multi-year Mean (China only)", fontsize=int(style.get("main_title_size", 16)))

        out_png = os.path.join(outp, f"MEANMAP_{idx}.png")
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)

        means_cn.clear()
        means_full.clear()
        del all_vals, samples
        gc.collect()

        print(f"[OK] saved: {out_png}")


# -----------------------------
# Plot 2) Metrics maps + boxplot inset (STREAMING)
# -----------------------------
def plot_metrics_maps_stream(
    product_paths: Dict[str, str],
    indices: List[str],
    metrics: List[str],
    ref_name: str,
    regions_obj,
    gdf_outline,
    gdf_regions,
    region_labels_inset: List[str],
    out_dir: str,
    ncols: int,
    dpi: int,
    style: dict,
    use_cartopy: bool,
    nine_cfg: dict,
    gdf_nine
):
    """
    Metrics maps vs reference (China-only main panels) + region boxplot inset.
    South China Sea inset shows the SAME metric raster (UNMASKED) so it won't be blank.
    Streaming/memory-safe.
    """
    outp = os.path.join(out_dir, "02_METRICS_MAPS_WITH_REGION_BOXPLOTS")
    ensure_dir(outp)

    p_low = style.get("robust_percentile_low", 2)
    p_high = style.get("robust_percentile_high", 98)
    sample_cbar = int(style.get("sample_for_colorbar", 20000))
    sample_box = int(style.get("sample_for_boxplot", 30000))

    lw_outline = style.get("outline_linewidth", 0.8)
    lw_region = style.get("region_linewidth", 0.8)
    coastline_lw = style.get("coastline_linewidth", 0.4)
    tick_fs = int(style.get("tick_label_size", 11))

    cmap_err = style.get("cmap_error", "magma")
    cmap_div = style.get("cmap_diverging", "RdBu_r")
    div_center = float(style.get("diverging_center", 0.0))
    nse_vmin_fixed = float(style.get("nse_vmin", -1.0))
    nse_vmax_fixed = float(style.get("nse_vmax", 1.0))

    # build china_mask/extent/reg_mask on ref grid
    with xr.open_dataset(product_paths[ref_name], decode_times=True) as ds_ref:
        china_mask, extent, reg_mask = get_china_mask_and_extent(ds_ref, regions_obj, gdf_outline)
        n_regions = len(region_labels_inset)

        for idx in indices:
            if idx not in ds_ref:
                print(f"[WARN] ref missing index '{idx}', skip.")
                continue

            obs = ds_ref[idx]  # time, lat, lon

            # masked maps for main panels
            metric_maps_cn: Dict[str, Dict[str, xr.DataArray]] = {m: {} for m in metrics}
            # full maps for SCS inset (unmasked)
            metric_maps_full: Dict[str, Dict[str, xr.DataArray]] = {m: {} for m in metrics}
            metric_samples: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}

            for p, nc_path in product_paths.items():
                if p == ref_name:
                    continue
                try:
                    with xr.open_dataset(nc_path, decode_times=True) as ds:
                        if idx not in ds:
                            continue
                        sim = ds[idx]
                        mets = calc_metrics_map(sim=sim, obs=obs)

                        for m in metrics:
                            if m not in mets:
                                continue

                            da2d_full = mets[m]  # (lat, lon) float32
                            da2d_cn = da2d_full.where(np.isfinite(china_mask))

                            metric_maps_full[m][p] = da2d_full
                            metric_maps_cn[m][p] = da2d_cn

                            vv = sample_values(da2d_cn.values, sample_cbar)
                            if vv.size > 0:
                                metric_samples[m].append(vv)

                except Exception as e:
                    print(f"[WARN] metric failed: {p} {idx}: {e}")
                    continue

            # plot per metric
            for m in metrics:
                if not metric_maps_cn[m]:
                    continue

                all_vals = np.concatenate(metric_samples[m]) if len(metric_samples[m]) else np.array([0.0], dtype="float32")

                if m.upper() == "NSE":
                    cmap = cmap_div
                    vmin, vmax = nse_vmin_fixed, nse_vmax_fixed
                elif m.upper() == "BIAS":
                    cmap = cmap_div
                    vmin, vmax = symmetric_minmax(all_vals, center=div_center, p_high=p_high)
                else:
                    cmap = cmap_err
                    vmin, vmax = robust_minmax(all_vals, p_low, p_high)

                fig, axes = get_fig_axes(len(metric_maps_cn[m]), ncols=ncols, use_cartopy=use_cartopy, style=style)
                axes_flat = axes.ravel()
                last_im = None

                # consistent ordering
                for i, p in enumerate(metric_maps_cn[m].keys()):
                    ax = axes_flat[i]
                    da2d_cn = metric_maps_cn[m][p]
                    da2d_full = metric_maps_full[m][p]

                    last_im = plot_map_panel(
                        ax, da2d_cn, title=p,
                        vmin=vmin, vmax=vmax, cmap=cmap,
                        use_cartopy=use_cartopy, coastline_lw=coastline_lw,
                        china_mask=china_mask, extent=extent
                    )
                    draw_boundaries(ax, gdf_outline, gdf_regions, lw_outline, lw_region)

                    # region boxplot inset uses CHINA-masked da2d (avoid ocean/outside)
                    region_data = extract_region_arrays(da2d_cn, reg_mask, n_regions=n_regions, max_samples=sample_box)
                    add_region_boxplot_inset(ax, region_data, region_labels_inset, style)

                    row = i // ncols
                    col = i % ncols
                    is_left = (col == 0)
                    is_bottom = (row == (axes.shape[0] - 1))
                    set_lonlat_ticks_for_grid(ax, is_left, is_bottom, extent=extent, tick_fs=tick_fs)

                    # South China Sea inset uses FULL metric raster (unmasked)
                    add_nine_dash_inset(
                        ax,
                        da2d_full=da2d_full,
                        gdf_outline=gdf_outline,
                        gdf_nine=gdf_nine,
                        nine_cfg=nine_cfg,
                        use_cartopy=use_cartopy,
                        cmap=cmap,
                        vmin=vmin, vmax=vmax
                    )

                for j in range(len(metric_maps_cn[m]), len(axes_flat)):
                    axes_flat[j].axis("off")

                if last_im is not None:
                    cbar = fig.colorbar(last_im, ax=[a for a in axes_flat if a.has_data()], fraction=0.03, pad=0.02)
                    cbar.set_label(f"{idx} - {m} (vs {ref_name})", fontsize=int(style.get("colorbar_label_size", 12)))

                fig.suptitle(f"{idx} {m} vs {ref_name} (China only)", fontsize=int(style.get("main_title_size", 16)))

                out_png = os.path.join(outp, f"METRIC_{idx}_{m}_vs_{ref_name}.png")
                fig.savefig(out_png, dpi=dpi)
                plt.close(fig)

                gc.collect()
                print(f"[OK] saved: {out_png}")

            metric_maps_cn.clear()
            metric_maps_full.clear()
            metric_samples.clear()
            gc.collect()


# -----------------------------
# Plot 3) Trend (STREAMING; China mean)
# -----------------------------
def plot_trends_stream(
    product_paths: Dict[str, str],
    ref_name: str,
    wet_indices: List[str],
    dry_indices: List[str],
    regions_obj,
    gdf_outline,
    out_dir: str,
    dpi: int,
    style: dict,
    trend_cfg: dict
):
    outp = os.path.join(out_dir, "03_TRENDS")
    ensure_dir(outp)

    plot_all = bool(trend_cfg.get("plot_all_products", True))
    highlight_products = trend_cfg.get("highlight_products", [ref_name])
    add_trend = bool(trend_cfg.get("add_trend_line_for_highlight", True))

    # get china_mask on ref grid
    with xr.open_dataset(product_paths[ref_name], decode_times=True) as ds_ref:
        china_mask, extent, reg_mask = get_china_mask_and_extent(ds_ref, regions_obj, gdf_outline)

    def china_mean_annual_series(nc_path: str, var: str) -> Optional[pd.Series]:
        try:
            with xr.open_dataset(nc_path, decode_times=True) as ds:
                if var not in ds:
                    return None
                da = ds[var]  # time, lat, lon
                ann = annual_aggregate(da, var)  # year, lat, lon

                # apply china mask (broadcast ok)
                ann_cn = ann.where(np.isfinite(china_mask))
                years = ann_cn["year"].values.astype(int)
                vals = ann_cn.mean(("lat", "lon"), skipna=True).values.astype("float64")
                return pd.Series(vals, index=years, name=var)
        except Exception:
            return None

    def plot_one(var: str, tag: str):
        series_dict = {}
        for p, path in product_paths.items():
            s = china_mean_annual_series(path, var)
            if s is not None:
                series_dict[p] = s

        if not series_dict:
            print(f"[WARN] no series for {var}")
            return

        common = None
        for s in series_dict.values():
            common = set(s.index) if common is None else (common & set(s.index))
        common_years = np.array(sorted(list(common)), dtype=int)
        if common_years.size < 5:
            print(f"[WARN] too few common years for {var}")
            return

        fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150, constrained_layout=True)

        if plot_all:
            for p, s in series_dict.items():
                if p in highlight_products:
                    continue
                y = s.loc[common_years].values.astype(float)
                ax.plot(common_years, y, linewidth=0.9, alpha=0.55)

        for hp in highlight_products:
            if hp not in series_dict:
                continue
            y = series_dict[hp].loc[common_years].values.astype(float)
            ax.plot(common_years, y, linewidth=2.4, label=f"{hp} (highlight)")
            if add_trend:
                a, b = lin_trend(common_years.astype(float), y)
                if np.isfinite(a):
                    ax.plot(common_years, a * common_years + b, linestyle="--", linewidth=2.0,
                            label=f"{hp} trend: {a:.3g}/yr")

        ax.set_title(f"{var} Annual Evolution ({tag}, China mean)", fontsize=int(style.get("axes_title_size", 13)))
        ax.set_xlabel("Year")
        ax.set_ylabel(var)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.legend(frameon=True, fontsize=int(style.get("legend_size", 11)))

        out_png = os.path.join(outp, f"TREND_{tag}_{var}_china.png")
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        gc.collect()
        print(f"[OK] saved: {out_png}")

    for v in wet_indices:
        plot_one(v, "Wet")
    for v in dry_indices:
        plot_one(v, "Dry")


# -----------------------------
# Main
# -----------------------------
def main(cfg_path: str):
    cfg = load_config(cfg_path)

    nc_dir = cfg["nc_dir"]
    nc_glob = cfg.get("nc_glob", "*.nc")
    ref_name = cfg["ref_name"]

    shp_path = cfg["shp_path"]
    climate_field = cfg["shp_climate_field"]

    out_dir = cfg["out_dir"]
    dpi = int(cfg.get("dpi", 600))
    ncols = int(cfg.get("ncols", 4))

    style = cfg.get("figure_style", {})
    nine_cfg = cfg.get("nine_dash", {})

    # fonts + rcParams
    tnr_name, simhei_name = setup_fonts(cfg.get("font_path_tnr", ""), cfg.get("font_path_simhei", ""))
    apply_rcparams(style, tnr_name)

    use_cartopy = bool(style.get("use_cartopy", True)) and (ccrs is not None)
    if bool(style.get("use_cartopy", True)) and (ccrs is None):
        print("[WARN] cartopy not available -> fallback without projection.")

    ensure_dir(out_dir)

    # discover NetCDFs (paths only)
    nc_paths = sorted(glob.glob(os.path.join(nc_dir, nc_glob)))
    if not nc_paths:
        raise FileNotFoundError(f"No NetCDF files found: {os.path.join(nc_dir, nc_glob)}")

    product_paths: Dict[str, str] = {}
    for pth in nc_paths:
        pname = product_name_from_filename(pth)
        product_paths[pname] = pth

    print(f"[INFO] discovered products ({len(product_paths)}): {list(product_paths.keys())}")
    if ref_name not in product_paths:
        raise ValueError(f"Reference '{ref_name}' not found in products: {list(product_paths.keys())}")

    # regions
    climate_name_map = cfg.get("climate_name_map", {})
    climate_abbr_map = cfg.get("climate_abbr_map", {})
    gdf_all, gdf_regions, gdf_outline, regions_obj, region_cn, region_en, region_abbr = build_regions_from_shp(
        shp_path=shp_path,
        climate_field=climate_field,
        climate_name_map=climate_name_map,
        climate_abbr_map=climate_abbr_map
    )

    # nine-dash
    gdf_nine = None
    if nine_cfg.get("enabled", False):
        gdf_nine = load_nine_dash_shp(nine_cfg["shp_path"])

    indices = cfg.get("indices_to_plot", [])
    metrics = cfg.get("metrics_to_plot", ["RMSE", "MAE", "NSE", "BIAS"])

    # 1) mean maps (streaming)
    plot_mean_maps_stream(
        product_paths=product_paths,
        indices=indices,
        ref_name=ref_name,
        regions_obj=regions_obj,
        gdf_outline=gdf_outline,
        gdf_regions=gdf_regions,
        out_dir=out_dir,
        ncols=ncols,
        dpi=dpi,
        style=style,
        use_cartopy=use_cartopy,
        nine_cfg=nine_cfg,
        gdf_nine=gdf_nine
    )

    # 2) metrics maps (streaming)
    plot_metrics_maps_stream(
        product_paths=product_paths,
        indices=indices,
        metrics=metrics,
        ref_name=ref_name,
        regions_obj=regions_obj,
        gdf_outline=gdf_outline,
        gdf_regions=gdf_regions,
        region_labels_inset=region_abbr,   # inset 用缩写，避免拥挤
        out_dir=out_dir,
        ncols=ncols,
        dpi=dpi,
        style=style,
        use_cartopy=use_cartopy,
        nine_cfg=nine_cfg,
        gdf_nine=gdf_nine
    )

    # 3) trends (streaming)
    trend_cfg = cfg.get("trend", {})
    if trend_cfg.get("enabled", True):
        plot_trends_stream(
            product_paths=product_paths,
            ref_name=ref_name,
            wet_indices=trend_cfg.get("wet_indices", []),
            dry_indices=trend_cfg.get("dry_indices", []),
            regions_obj=regions_obj,
            gdf_outline=gdf_outline,
            out_dir=out_dir,
            dpi=dpi,
            style=style,
            trend_cfg=trend_cfg
        )

    print("[DONE] All figures generated (streaming).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config_etccdi_plot.json")
    args = parser.parse_args()
    main(args.config)
