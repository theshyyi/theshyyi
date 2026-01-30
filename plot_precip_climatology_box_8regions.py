#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ============================================================
# 0) 配置
# ============================================================
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_product_name(nc_path):
    # ChinaMet.TIMEFIX.daily.CHINA.nc -> ChinaMet
    base = os.path.basename(nc_path)
    return base.split(".TIMEFIX.daily.CHINA.nc")[0].split(".")[0]

def clean_values(x):
    x = np.asarray(x, dtype="float64").ravel()
    x = x[np.isfinite(x)]
    # 剔除填充值/溢出值（保险；不影响正常值）
    x = x[np.abs(x) < 1e20]
    return x

def sample_values(x, max_n, rng):
    if x.size <= max_n:
        return x
    idx = rng.choice(x.size, size=max_n, replace=False)
    return x[idx]

def regionmask_mask_compat(regions, lon, lat):
    try:
        return regions.mask(lon=lon, lat=lat)
    except TypeError:
        try:
            return regions.mask(lon, lat)
        except TypeError:
            return regions.mask(x=lon, y=lat)


def build_regionmask_from_shp(shp_path, region_field, lon, lat):
    gdf = gpd.read_file(shp_path)
    if region_field not in gdf.columns:
        raise ValueError(f"SHP 缺少字段 {region_field}")
    if gdf.crs is None:
        raise ValueError("SHP 缺少 CRS（请确保 1980_albert CRS 信息完整）。")

    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    # 转为经纬度（regionmask 以 lon/lat 网格做 mask）
    gdf_ll = gdf.to_crs("EPSG:4326").copy()
    gdf_ll[region_field] = gdf_ll[region_field].astype(str)

    # 固定顺序（按分区名排序）
    reg_cn = sorted(pd.unique(gdf_ll[region_field]))

    # 重新排序并创建编号列 rid（0..6）
    gdf_ll = gdf_ll.set_index(region_field).loc[reg_cn].reset_index()
    gdf_ll["rid"] = np.arange(len(gdf_ll), dtype=int)

    # 关键修复点：
    # - names 传列名（字符串），不要传 list
    # - numbers 传编号列名
    regs = regionmask.from_geopandas(
        gdf_ll,
        names=region_field,   # <-- 这里必须是列名
        numbers="rid",
        name="China_Climate_7"
    )

    mask = regionmask_mask_compat(regs, lon, lat)
    return mask, reg_cn



def extract_8regions(field2d, mask7, reg_cn):
    out = {}
    out["China"] = clean_values(field2d.values)
    for idx, cn in enumerate(reg_cn):
        arr = field2d.where(mask7 == idx)
        out[cn] = clean_values(arr.values)
    return out

def robust_xlim(data_list):
    allv = np.concatenate([d for d in data_list if d.size > 0], axis=0)
    if allv.size == 0:
        return None
    lo = np.nanpercentile(allv, 2)
    hi = np.nanpercentile(allv, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(allv)
        hi = np.nanmax(allv)
    span = hi - lo
    pad = 0.08 * span if span > 0 else 1.0
    return (max(0.0, lo - pad), hi + pad)

def plot_8panel_box(
    out_png, row_label, x_unit_label,
    region_names_en, region_keys,
    products, values_by_region_product,
    color_map,
    fig_w, fig_h, dpi, wspace, hspace,
    title_fs=13, ylabel_fs=13
):
    fig, axes = plt.subplots(2, 4, figsize=(fig_w, fig_h), dpi=dpi)
    axes = axes.flatten()

    for i, reg in enumerate(region_keys):
        ax = axes[i]
        series_list, labels, colors = [], [], []

        for p in products:
            arr = values_by_region_product.get((reg, p), None)
            if arr is None or arr.size == 0:
                continue
            series_list.append(arr)
            labels.append(p)
            colors.append(color_map[p])

        if not series_list:
            ax.set_axis_off()
            continue

        bp = ax.boxplot(
            series_list,
            vert=False,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            widths=0.62,
            medianprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=0.9),
            capprops=dict(linewidth=0.9),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.35)
            patch.set_linewidth(1.0)

        means = [np.mean(v) for v in series_list]
        y = np.arange(1, len(series_list) + 1)
        ax.scatter(means, y, s=12, c="k", zorder=3)

        ax.grid(True, axis="both", linestyle="-", alpha=0.18)
        ax.invert_yaxis()

        ax.set_title(region_names_en.get(reg, str(reg)), fontsize=title_fs, fontweight="bold")

        xl = robust_xlim(series_list)
        if xl is not None:
            ax.set_xlim(xl)

        # 只在第一列显示产品名
        if (i % 4) != 0:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=7, pad = 1)

        # 每一行第一列写 row label
        if i % 4 == 0:
            ax.set_ylabel(row_label, fontsize=ylabel_fs, fontweight="bold")

        # 底行加 x label
        if i >= 4:
            ax.set_xlabel(x_unit_label, fontsize=10)

    fig.subplots_adjust(left=0.18, right=0.995, top=0.94, bottom=0.08,
                        wspace=wspace, hspace=hspace)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


# ============================================================
# 1) 主流程
# ============================================================
def main(cfg_path):
    cfg = load_config(cfg_path)

    nc_dir = cfg["nc_dir"]
    nc_glob = cfg.get("nc_glob", "*.TIMEFIX.daily.CHINA.nc")
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    shp_path = cfg["shp_path"]
    region_field = cfg.get("region_field", "climate")
    climate_name_map = cfg.get("climate_name_map", {})

    font_path_tnr = cfg.get("font_path_tnr", "")
    if font_path_tnr and os.path.exists(font_path_tnr):
        font_manager.fontManager.addfont(font_path_tnr)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    dpi = int(cfg.get("dpi", 600))
    fig_w = float(cfg.get("fig_w", 22))
    fig_h = float(cfg.get("fig_h", 14))
    wspace = float(cfg.get("wspace", 0.22))
    hspace = float(cfg.get("hspace", 0.32))

    season_order = cfg.get("season_order", ["DJF", "MAM", "JJA", "SON"])
    month_order = cfg.get("month_order", list(range(1, 13)))
    sample_max = int(cfg.get("sample_max_per_product", 20000))

    # 文件列表
    nc_paths = sorted(glob.glob(os.path.join(nc_dir, nc_glob)))
    if not nc_paths:
        raise FileNotFoundError(f"No nc files matched: {os.path.join(nc_dir, nc_glob)}")

    products = [infer_product_name(p) for p in nc_paths]
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(products))
    color_map = {p: cmap(i) for i, p in enumerate(products)}
    rng = np.random.default_rng(2025)
    n_prod = len(products)

    # 每行子图需要的高度（英寸）：产品越多，越需要高
    inch_per_product = 0.22   # 0.20~0.26 可调
    row_h_min = 1.2 + n_prod * inch_per_product

    # 2 行（2×4）+ 上下边距
    fig_h = max(fig_h, 2 * row_h_min + 2.0)

    # 用第一个产品建立区域 mask
    ds0 = xr.open_dataset(nc_paths[0])
    if "pr" not in ds0.data_vars:
        raise ValueError(f"{nc_paths[0]} 中找不到变量 pr")
    da0 = ds0["pr"]
    lon = ds0["lon"].values
    lat = ds0["lat"].values

    mask7, reg_cn = build_regionmask_from_shp(shp_path, region_field, lon, lat)
    region_keys = ["China"] + reg_cn
    region_names_en = {"China": "China"}
    for cn in reg_cn:
        region_names_en[cn] = climate_name_map.get(cn, cn)

    # 存储：避免一次性把全部产品都变成巨大数组
    STORE = {p: {"annual": None, "daily_mean": None, "seasonal": {}, "monthly": {}} for p in products}

    # 逐产品计算
    for nc_path, prod in zip(nc_paths, products):
        ds = xr.open_dataset(nc_path)
        da = ds["pr"]

        # daily mean (mm/day)
        daily_mean_2d = da.mean("time", skipna=True)

        # monthly total (mm/month)
        monthly_sum = da.resample(time="MS").sum("time", skipna=True)

        # annual total (mm/yr) then climatological mean
        annual_sum = da.resample(time="YS").sum("time", skipna=True)
        annual_mean_2d = annual_sum.mean("time", skipna=True)

        # seasonal total: 用月累计 + season_year 修正（DJF 跨年）
        # seasonal total (mm/season): 兼容旧 xarray（不使用链式 groupby）
        monthly = monthly_sum

        year = monthly["time.year"]
        mon  = monthly["time.month"]
        sea  = monthly["time.season"]  # DJF/MAM/JJA/SON

        # season_year：把 12 月算到下一年的 DJF
        season_year = year.where(mon != 12, year + 1)

        # 组合分组键：例如 "2001_DJF"
        season_key = xr.DataArray(
            season_year.astype(str).values + "_" + sea.astype(str).values,
            coords={"time": monthly["time"]},
            dims=("time",),
            name="season_key"
        )

        # 先按 season_key 对月累计求和 -> 每个 season_year+season 的总量
        seasonal_sum_by_key = monthly.groupby(season_key).sum("time", skipna=True)
        # dims: season_key, lat, lon

        # 把 season_key 拆成 season_year 与 season 两个坐标
        key_index = seasonal_sum_by_key["season_key"].values.astype(str)
        key_year = np.array([int(k.split("_")[0]) for k in key_index], dtype=int)
        key_season = np.array([k.split("_")[1] for k in key_index], dtype=object)

        seasonal_sum_by_key = seasonal_sum_by_key.assign_coords(
            season_year=("season_key", key_year),
            season=("season_key", key_season),
        )

        # 多年平均到季节：对同一 season 的不同 season_year 取 mean
        seasonal_mean_by_season = seasonal_sum_by_key.groupby("season").mean("season_key", skipna=True)
        # dims: season, lat, lon

        STORE[prod]["seasonal"] = {}
        for s in season_order:
            if s not in seasonal_mean_by_season["season"].values:
                continue
            s2d = seasonal_mean_by_season.sel(season=s)
            STORE[prod]["seasonal"][s] = {
                reg: sample_values(v, sample_max, rng)
                for reg, v in extract_8regions(s2d, mask7, reg_cn).items()
            }


    # 组装绘图输入
    def build_values(timescale, key=None):
        values = {}
        for prod in products:
            if timescale == "annual":
                dct = STORE[prod]["annual"]
            elif timescale == "daily_mean":
                dct = STORE[prod]["daily_mean"]
            elif timescale == "monthly":
                dct = STORE[prod]["monthly"][key]
            elif timescale == "seasonal":
                dct = STORE[prod]["seasonal"][key]
            else:
                raise ValueError(timescale)

            for reg in region_keys:
                values[(reg, prod)] = dct.get(reg, np.array([], dtype=float))
        return values

    # 输出：年
    values = build_values("annual")
    plot_8panel_box(
        out_png=os.path.join(out_dir, "BOX_AnnualMean_8regions.png"),
        row_label="Annual mean",
        x_unit_label="mm/year",
        region_names_en=region_names_en,
        region_keys=region_keys,
        products=products,
        values_by_region_product=values,
        color_map=color_map,
        fig_w=fig_w, fig_h=fig_h, dpi=dpi, wspace=wspace, hspace=hspace,
        title_fs=13, ylabel_fs=13
    )

    # 输出：日
    values = build_values("daily_mean")
    plot_8panel_box(
        out_png=os.path.join(out_dir, "BOX_DailyMean_8regions.png"),
        row_label="Daily mean",
        x_unit_label="mm/day",
        region_names_en=region_names_en,
        region_keys=region_keys,
        products=products,
        values_by_region_product=values,
        color_map=color_map,
        fig_w=fig_w, fig_h=fig_h, dpi=dpi, wspace=wspace, hspace=hspace,
        title_fs=13, ylabel_fs=13
    )

    # 输出：季（4 张）
    for s in season_order:
        # 若某季缺失则跳过
        ok = all(s in STORE[p]["seasonal"] for p in products)
        if not ok:
            warnings.warn(f"Skip season {s}: not available in all products")
            continue
        values = build_values("seasonal", s)
        plot_8panel_box(
            out_png=os.path.join(out_dir, f"BOX_SeasonMean_{s}_8regions.png"),
            row_label=f"{s} mean",
            x_unit_label="mm/season",
            region_names_en=region_names_en,
            region_keys=region_keys,
            products=products,
            values_by_region_product=values,
            color_map=color_map,
            fig_w=fig_w, fig_h=fig_h, dpi=dpi, wspace=wspace, hspace=hspace,
            title_fs=13, ylabel_fs=13
        )

    # 输出：月（12 张）
    for m in month_order:
        values = build_values("monthly", m)
        plot_8panel_box(
            out_png=os.path.join(out_dir, f"BOX_MonthMean_M{m:02d}_8regions.png"),
            row_label=f"M{m:02d} mean",
            x_unit_label="mm/month",
            region_names_en=region_names_en,
            region_keys=region_keys,
            products=products,
            values_by_region_product=values,
            color_map=color_map,
            fig_w=fig_w, fig_h=fig_h, dpi=dpi, wspace=wspace, hspace=hspace,
            title_fs=13, ylabel_fs=13
        )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to json config")
    args = ap.parse_args()
    main(args.config)
