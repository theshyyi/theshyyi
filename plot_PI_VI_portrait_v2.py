#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portrait diagram for PI & VI of ETCCDI monthly indices
======================================================

- Panels: 3x3, 8 subplots for (China + 7 regions) and the 9th axis for colorbar.
- Indices: ALL ETCCDI variables in the reference CMFDV2 file with dims (time,lat,lon).
- Missing index in a product => keep the column and fill NaN for that product.
- Fonts: Chinese -> SimHei, English -> Times New Roman (from given TTF paths).
- Default metric mode: "spatial"
    * PI: RMSE of climatology field (time-mean) over region (spatial RMSE)
    * VI: mean over region of (sigma - 1/sigma)^2, sigma = STD_prod / STD_ref (STD over time)
  Optional: "ts" mode (regional-mean time series RMSE/STD)

Run:
  python plot_PI_VI_portrait.py
or:
  python plot_PI_VI_portrait.py --config /path/to/config_PI_VI.json

Dependencies:
  pip/conda install numpy pandas xarray netcdf4 geopandas regionmask matplotlib dask
"""

import os
import glob
import re
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =========================
# 0) Config (default)
# =========================
DEFAULT_CONFIG_NAME = "config_PI_VI.json"


def default_config_template() -> dict:
    return {
        "nc_dir": "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/ETCCDI_manual/yearly",
        "shp": "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp",
        "climate_field": "climate",
        "climate_name_map": {
            "暖温带半湿润地区": "Warm Temperate (Semi-humid)",
            "中温带干旱地区": "Mid-Temperate (Arid)",
            "北亚热带湿润地区": "North Subtropical (Humid)",
            "中温带半湿润地区": "Mid-Temperate (Semi-humid)",
            "中温带半干旱地区": "Mid-Temperate (Semi-arid)",
            "高原温带半干旱地区": "Plateau Temperate (Semi-arid)",
            "边缘热带湿润地区": "Marginal Tropical (Humid)"
        },
        "ref": "CMFDV2",
        "out_dir": "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/ETCCDI_manual/yearly/PI_VI_portrait",
        "dpi": 600,

        # ----- compute mode -----
        # "spatial": climatology spatial RMSE + variability spatial term (recommended for portrait)
        # "ts": regional-mean time series RMSE + std ratio term
        "mode": "spatial",

        # ----- dask chunks (set null to disable dask) -----
        "chunks": {"time": 2000, "lat": 100, "lon": 100},

        # ----- fonts -----
        "font_path_tnr": "/home/ud202380664/Times_New_Roman.ttf",
        "font_path_simhei": "/home/ud202380664/Ubuntu_18.04_SimHei.ttf",

        # ----- plotting -----
        "figsize": [20, 12],
        "wspace": 0.25,
        "hspace": 0.35,
        "left": 0.06,
        "right": 0.98,
        "top": 0.92,
        "bottom": 0.08,

        # tick font sizes
        "title_fs": 14,
        "tick_fs": 14,
        "suptitle_fs": 16,

        # colorbar settings
        "cbar_height": "25%",  # inset_axes height
        "cbar_width": "95%",
        "cbar_loc": "lower center",

        # PI range fixed like examples
        "pi_range": {"vmin": -1.0, "vmax": 1.0},

        # VI range: vmin fixed; vmax = percentile of all VI values
        "vi_range": {"vmin": 0.0, "vmax_percentile": 5}
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path: str, cfg: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def has_chinese(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(s))


def safe_filename(name: str) -> str:
    """
    Make a filesystem-friendly filename from a region label.
    Keep ASCII letters/numbers/._- and replace other chars with underscore.
    """
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "region"


# =========================
# 1) IO helpers
# =========================
def safe_open_dataset(nc_path: str, chunks: dict | None):
    """
    Open dataset; if time decode fails, fallback to decode_times=False.
    """
    try:
        return xr.open_dataset(nc_path, chunks=chunks)
    except Exception:
        return xr.open_dataset(nc_path, decode_times=False, chunks=chunks)


def product_from_path(nc_path: str) -> str:
    base = os.path.basename(nc_path)
    return base.split(".TIMEFIX")[0]


def find_lat_lon_names(ds: xr.Dataset):
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]
    lat_name = next((c for c in lat_candidates if c in ds.coords or c in ds.dims), None)
    lon_name = next((c for c in lon_candidates if c in ds.coords or c in ds.dims), None)
    if lat_name is None or lon_name is None:
        raise ValueError("Cannot find lat/lon in dataset. Please standardize to lat/lon.")
    return lat_name, lon_name


def find_time_dim(da: xr.DataArray, lat_name: str, lon_name: str):
    for d in da.dims:
        if d not in (lat_name, lon_name):
            return d
    return None


def list_all_indices_in_ref(ds_ref: xr.Dataset, lat_name: str, lon_name: str):
    """
    List ALL variables with dims containing (time, lat, lon) from reference file.
    Keep order as stored in file.
    """
    indices = []
    for v in ds_ref.data_vars:  # preserves nc order
        da = ds_ref[v]
        if (lat_name in da.dims) and (lon_name in da.dims):
            tdim = find_time_dim(da, lat_name, lon_name)
            if tdim is not None:
                indices.append(v)
    if not indices:
        raise ValueError("No (time,lat,lon) ETCCDI variables found in reference file.")
    return indices


# =========================
# 2) Region mask
# =========================
def build_region_mask(shp_path: str, climate_field: str, ds_ref: xr.Dataset, lat_name: str, lon_name: str):
    gdf = gpd.read_file(shp_path)
    if climate_field not in gdf.columns:
        raise ValueError(f"Field '{climate_field}' not found in shapefile attributes.")
    if gdf.crs is None:
        raise ValueError("Shapefile CRS is missing. Please define CRS before use.")
    gdf = gdf.to_crs(epsg=4326)

    # region names keep shp order
    names = gdf[climate_field].astype(str).tolist()
    numbers = list(range(len(names)))

    regions = regionmask.Regions(
        outlines=gdf.geometry.tolist(),
        names=names,
        numbers=numbers
    )

    lats = ds_ref[lat_name].values
    lons = ds_ref[lon_name].values
    mask = regions.mask(lons, lats)  # (lat, lon) with region ids
    return mask, names


# =========================
# 3) Metrics (dask-safe scalar output)
# =========================
def scalar_float(da0):
    """
    Convert 0-d DataArray to python float safely (supports dask).
    """
    return float(da0.compute().values)


def rmse_spatial(diff2_2d: xr.DataArray, mask: xr.DataArray, region_id: int | None,
                 lat_name: str, lon_name: str):
    """
    diff2_2d: (lat,lon) squared difference
    region_id None => China union (mask finite)
    """
    if region_id is None:
        m = diff2_2d.where(np.isfinite(mask)).mean(dim=(lat_name, lon_name), skipna=True)
    else:
        m = diff2_2d.where(mask == region_id).mean(dim=(lat_name, lon_name), skipna=True)
    return scalar_float(np.sqrt(m))


def mean_term_spatial(term_2d: xr.DataArray, mask: xr.DataArray, region_id: int | None,
                      lat_name: str, lon_name: str):
    if region_id is None:
        m = term_2d.where(np.isfinite(mask)).mean(dim=(lat_name, lon_name), skipna=True)
    else:
        m = term_2d.where(mask == region_id).mean(dim=(lat_name, lon_name), skipna=True)
    return scalar_float(m)


def rmse_ts(prod_ts: xr.DataArray, ref_ts: xr.DataArray, time_dim: str):
    prod_al, ref_al = xr.align(prod_ts, ref_ts, join="inner")
    if prod_al.sizes.get(time_dim, 0) == 0:
        return np.nan
    da = np.sqrt(((prod_al - ref_al) ** 2).mean(dim=time_dim, skipna=True))
    return scalar_float(da)


def std_ts(ts: xr.DataArray, time_dim: str):
    da = ts.std(dim=time_dim, skipna=True)
    return scalar_float(da)


# =========================
# 4) Compute PI & VI
# =========================
def compute_pi_vi(cfg: dict):
    nc_dir = cfg["nc_dir"]
    ref_name = cfg["ref"]
    shp_path = cfg["shp"]
    climate_field = cfg.get("climate_field", "climate")
    mode = cfg.get("mode", "spatial").lower()

    chunks = cfg.get("chunks", None)
    if chunks in ("null", "None"):
        chunks = None

    nc_paths = sorted(glob.glob(os.path.join(nc_dir, "*-ETCCDI-yearly.nc")))
    if not nc_paths:
        raise FileNotFoundError(f"No '*-ETCCDI-monthly.nc' found in: {nc_dir}")

    prod2path = {product_from_path(p): p for p in nc_paths}
    if ref_name not in prod2path:
        raise FileNotFoundError(f"Reference '{ref_name}' not found. Available: {list(prod2path.keys())}")

    # open ref
    ds_ref = safe_open_dataset(prod2path[ref_name], chunks=chunks)
    lat_name, lon_name = find_lat_lon_names(ds_ref)

    # build mask and region list
    mask, climate_names = build_region_mask(shp_path, climate_field, ds_ref, lat_name, lon_name)
    # map Chinese region names to English for plotting/output (optional)
    climate_names = [str(x).strip() for x in climate_names]
    climate_name_map = cfg.get("climate_name_map", {}) or {}
    climate_names = [climate_name_map.get(x, x) for x in climate_names]
    if len(climate_names) != 7:
        raise ValueError(f"Your shp has {len(climate_names)} regions, but expected 7 (China + 7 => 8 panels).")

    region_labels = ["China"] + climate_names
    region_ids = [None] + list(range(len(climate_names)))  # None => China union

    # indices: ALL from reference file (no intersection!)
    indices = list_all_indices_in_ref(ds_ref, lat_name, lon_name)

    # products excluding ref
    products = sorted([p for p in prod2path.keys() if p != ref_name])

    # allocate rmse and vi matrices per region
    rmse_by_region = {r: pd.DataFrame(index=products, columns=indices, dtype=float) for r in region_labels}
    vi_by_region = {r: pd.DataFrame(index=products, columns=indices, dtype=float) for r in region_labels}

    # precompute ref fields (mean/std or ts) per index
    ref_cache = {}
    for idx in indices:
        if idx not in ds_ref.variables:
            continue
        da_ref = ds_ref[idx].astype("float32")
        time_dim = find_time_dim(da_ref, lat_name, lon_name)
        if time_dim is None:
            continue

        if mode == "spatial":
            ref_mean = da_ref.mean(dim=time_dim, skipna=True)  # (lat,lon)
            ref_std = da_ref.std(dim=time_dim, skipna=True)    # (lat,lon)
            ref_cache[idx] = {"time_dim": time_dim, "mean": ref_mean, "std": ref_std}
        elif mode == "ts":
            # build regional mean time series for all regions
            ref_china_ts = da_ref.where(np.isfinite(mask)).mean(dim=(lat_name, lon_name), skipna=True)
            ref_group = da_ref.groupby(mask).mean(dim=(lat_name, lon_name), skipna=True)  # dim 'mask'
            ref_cache[idx] = {"time_dim": time_dim, "china_ts": ref_china_ts, "group_ts": ref_group}
        else:
            raise ValueError("cfg['mode'] must be 'spatial' or 'ts'.")

    # loop products
    for p in products:
        ds_p = safe_open_dataset(prod2path[p], chunks=chunks)

        # compute per index
        for idx in indices:
            if idx not in ds_p.variables or idx not in ref_cache:
                # keep indicator column, fill NaN for this product across all regions
                for r in region_labels:
                    rmse_by_region[r].loc[p, idx] = np.nan
                    vi_by_region[r].loc[p, idx] = np.nan
                continue

            da_p = ds_p[idx].astype("float32")
            time_dim = ref_cache[idx]["time_dim"]

            if mode == "spatial":
                # 2D fields
                p_mean = da_p.mean(dim=time_dim, skipna=True)
                p_std = da_p.std(dim=time_dim, skipna=True)

                ref_mean = ref_cache[idx]["mean"]
                ref_std = ref_cache[idx]["std"]

                # RMSE over space within region
                diff2 = (p_mean - ref_mean) ** 2

                # VI term over space within region
                # sigma = std_p / std_ref, avoid zeros
                sigma = p_std / ref_std
                term = (sigma - 1.0 / sigma) ** 2
                term = term.where(np.isfinite(term) & (sigma > 0) & np.isfinite(ref_std) & (ref_std > 0))

                # compute scalars for China + 7 regions
                for rlab, rid in zip(region_labels, region_ids):
                    rmse_val = rmse_spatial(diff2, mask, rid, lat_name, lon_name)
                    vi_val = mean_term_spatial(term, mask, rid, lat_name, lon_name)
                    rmse_by_region[rlab].loc[p, idx] = rmse_val
                    vi_by_region[rlab].loc[p, idx] = vi_val

            else:  # mode == "ts"
                ref_china_ts = ref_cache[idx]["china_ts"]
                ref_group_ts = ref_cache[idx]["group_ts"]  # dim mask

                p_china_ts = da_p.where(np.isfinite(mask)).mean(dim=(lat_name, lon_name), skipna=True)
                p_group_ts = da_p.groupby(mask).mean(dim=(lat_name, lon_name), skipna=True)

                # China
                rmse_val = rmse_ts(p_china_ts, ref_china_ts, time_dim)
                std_ref = std_ts(ref_china_ts, time_dim)
                std_pv = std_ts(p_china_ts, time_dim)
                if np.isfinite(std_ref) and std_ref > 0 and np.isfinite(std_pv) and std_pv > 0:
                    sigma = std_pv / std_ref
                    vi_val = (sigma - 1.0 / sigma) ** 2
                else:
                    vi_val = np.nan
                rmse_by_region["China"].loc[p, idx] = rmse_val
                vi_by_region["China"].loc[p, idx] = vi_val

                # 7 regions
                for rid, rlab in zip(region_ids[1:], region_labels[1:]):
                    if "mask" in p_group_ts.dims:
                        p_ts = p_group_ts.sel(mask=rid) if rid in p_group_ts["mask"].values else None
                    else:
                        p_ts = None
                    if "mask" in ref_group_ts.dims:
                        r_ts = ref_group_ts.sel(mask=rid) if rid in ref_group_ts["mask"].values else None
                    else:
                        r_ts = None

                    if p_ts is None or r_ts is None:
                        rmse_by_region[rlab].loc[p, idx] = np.nan
                        vi_by_region[rlab].loc[p, idx] = np.nan
                        continue

                    rmse_val = rmse_ts(p_ts, r_ts, time_dim)
                    std_ref = std_ts(r_ts, time_dim)
                    std_pv = std_ts(p_ts, time_dim)
                    if np.isfinite(std_ref) and std_ref > 0 and np.isfinite(std_pv) and std_pv > 0:
                        sigma = std_pv / std_ref
                        vi_val = (sigma - 1.0 / sigma) ** 2
                    else:
                        vi_val = np.nan

                    rmse_by_region[rlab].loc[p, idx] = rmse_val
                    vi_by_region[rlab].loc[p, idx] = vi_val

        ds_p.close()

    ds_ref.close()

    # Convert RMSE -> PI for each region (per index median across products)
    PI = {}
    VI = {}
    for r in region_labels:
        rmse_mat = rmse_by_region[r]
        vi_mat = vi_by_region[r]

        pi_mat = rmse_mat.copy()
        for idx in indices:
            vals = rmse_mat[idx].values.astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                pi_mat[idx] = np.nan
                continue
            med = np.median(vals)
            if not np.isfinite(med) or med == 0:
                pi_mat[idx] = np.nan
            else:
                pi_mat[idx] = (rmse_mat[idx].astype(float) - med) / med

        PI[r] = pi_mat
        VI[r] = vi_mat

    return region_labels, products, indices, PI, VI


# =========================
# 5) Plot: 3x3 (8 panels + colorbar axis)
# =========================
def plot_portrait_3x3(cfg: dict, region_labels, products, indices, mats_by_region,
                      title_prefix, out_png, out_pdf=None,
                      cmap="RdBu_r", vmin=None, vmax=None, cbar_label=""):

    # fonts
    font_path_tnr = cfg["font_path_tnr"]
    font_path_simhei = cfg["font_path_simhei"]
    fp_en = FontProperties(fname=font_path_tnr)
    fp_zh = FontProperties(fname=font_path_simhei)

    # global rc
    rcParams["axes.unicode_minus"] = False

    # layout
    figsize = tuple(cfg.get("figsize", [26, 18]))
    wspace = float(cfg.get("wspace", 0.25))
    hspace = float(cfg.get("hspace", 0.35))
    left = float(cfg.get("left", 0.06))
    right = float(cfg.get("right", 0.98))
    top = float(cfg.get("top", 0.92))
    bottom = float(cfg.get("bottom", 0.08))

    title_fs = int(cfg.get("title_fs", 14))
    tick_fs = int(cfg.get("tick_fs", 14))
    suptitle_fs = int(cfg.get("suptitle_fs", 16))

    if len(region_labels) != 8:
        raise ValueError(f"Expect 8 panels (China + 7 regions), got {len(region_labels)}")

    fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=False)
    axes = axes.ravel()

    im_last = None

    for i, region in enumerate(region_labels):
        ax = axes[i]
        mat = mats_by_region[region].loc[products, indices].astype(float).values

        im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        im_last = im

        # title: Chinese -> SimHei, otherwise Times
        title_fp = fp_zh if has_chinese(region) else fp_en
        ax.set_title(f"({chr(97+i)}) {region}", fontproperties=title_fp, fontsize=title_fs)

        # y ticks: product names (English)
        ax.set_yticks(np.arange(len(products)))
        if (i % 3) == 0:  # 左列
            ax.set_yticklabels(products, fontproperties=fp_en, fontsize=tick_fs)
            ax.tick_params(axis="y", pad=1)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)


        # x ticks: indices (English)
        ax.set_xticks(np.arange(len(indices)))
        if i >= 6:  # 底行
            ax.set_xticklabels(indices, rotation=45, ha="right",
                               fontproperties=fp_en, fontsize=tick_fs)
            ax.tick_params(axis="x", pad=2)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)


        # grid look
        ax.set_xticks(np.arange(-.5, len(indices), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(products), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    # colorbar in the 9th axis (bottom-right)
    cax_host = axes[8]
    cax_host.axis("off")

    if im_last is not None:
        cbax = inset_axes(
            cax_host,
            width=cfg.get("cbar_width", "95%"),
            height=cfg.get("cbar_height", "25%"),
            loc=cfg.get("cbar_loc", "lower center"),
            borderpad=1.2
        )
        cbar = fig.colorbar(im_last, cax=cbax, orientation="horizontal")
        cbar.set_label(cbar_label, fontproperties=fp_en, fontsize=16)
        for t in cbar.ax.get_xticklabels():
            t.set_fontproperties(fp_en)
            t.set_fontsize(10)

    # spacing to avoid overlap
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

    # suptitle
    fig.suptitle(title_prefix, fontproperties=fp_en, fontsize=suptitle_fs)

    dpi = int(cfg.get("dpi", 600))
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================
# 6) Main (JSON config)
# =========================
def main():
    parser = argparse.ArgumentParser(description="Plot PI/VI portrait diagrams (China + 7 regions).")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_NAME,
                        help=f"Path to config json (default: {DEFAULT_CONFIG_NAME})")
    args = parser.parse_args()

    cfg_path = args.config

    # If config missing, create a template and exit
    if not os.path.exists(cfg_path):
        tpl = default_config_template()
        save_config(cfg_path, tpl)
        print(f"[Created] Config template: {cfg_path}")
        print("Please edit paths (nc_dir, shp, out_dir, font paths) then run again.")
        return

    cfg = load_config(cfg_path)

    # Ensure output dir exists
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Compute
    region_labels, products, indices, PI, VI = compute_pi_vi(cfg)

    # Save CSV per region
    for region in region_labels:
        reg_fn = safe_filename(region)
        PI[region].to_csv(os.path.join(out_dir, f"PI_{reg_fn}.csv"), float_format="%.6f")
        VI[region].to_csv(os.path.join(out_dir, f"VI_{reg_fn}.csv"), float_format="%.6f")

    # Plot PI (fixed range)
    pi_vmin = float(cfg.get("pi_range", {}).get("vmin", -1.0))
    pi_vmax = float(cfg.get("pi_range", {}).get("vmax", 1.0))

    plot_portrait_3x3(
        cfg=cfg,
        region_labels=region_labels,
        products=products,
        indices=indices,
        mats_by_region=PI,
        title_prefix=f"Portrait diagram of relative PIs (ref: {cfg['ref']})",
        out_png=os.path.join(out_dir, "PI_portrait.png"),
        out_pdf=os.path.join(out_dir, "PI_portrait.pdf"),
        cmap="RdBu_r",
        vmin=pi_vmin, vmax=pi_vmax,
        cbar_label="PI"
    )

    # Plot VI (vmax by percentile)
    vi_vmin = float(cfg.get("vi_range", {}).get("vmin", 0.0))
    vi_pct = float(cfg.get("vi_range", {}).get("vmax_percentile", 5))

    all_vi = np.concatenate([np.ravel(VI[r].values.astype(float)) for r in region_labels])
    vi_vmax = np.nanpercentile(all_vi, vi_pct)
    if not np.isfinite(vi_vmax) or vi_vmax <= 0:
        vi_vmax = 2.0

    plot_portrait_3x3(
        cfg=cfg,
        region_labels=region_labels,
        products=products,
        indices=indices,
        mats_by_region=VI,
        title_prefix=f"Portrait diagram of relative VIs (ref: {cfg['ref']})",
        out_png=os.path.join(out_dir, "VI_portrait.png"),
        out_pdf=os.path.join(out_dir, "VI_portrait.pdf"),
        cmap="RdBu_r",
        vmin=vi_vmin, vmax=vi_vmax,
        cbar_label="VI"
    )

    print("Done.")
    print(f"Output directory: {out_dir}")
    print(f"Products (excluding ref): {len(products)}")
    print(f"Indices from ref (ALL): {len(indices)} -> {indices}")


if __name__ == "__main__":
    main()
