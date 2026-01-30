#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""POD–(1-FAR) performance diagram by elevation bands
==================================================

- 目录 NC_DIR 下包含若干降水产品：
    {PRODUCT}.TIMEFIX.daily.CHINA.nc
  例如:
    CMFDV2.TIMEFIX.daily.CHINA.nc (参考)
    CHIRPSV2.TIMEFIX.daily.CHINA.nc
    MSWEP.TIMEFIX.daily.CHINA.nc
    ...

- REF_NAME 指定参考产品（OBS）

- DEM_NC: 海拔 nc 文件，含 lat, lon 和高程变量
  （若网格与降水不完全一致，会插值到参考产品网格）

- SHP_PATH: 中国 7 大分区 shp，用来生成“中国陆地区域”掩膜

- 在每个海拔带内 (ELEV_BINS)，计算：
    POD = hits / (hits + misses)
    FAR = false_alarm / (hits + false_alarm)
  并绘制多面板 POD vs (1-FAR) 图（每个面板一个海拔带）

依赖：
    pip install numpy xarray matplotlib geopandas regionmask
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask
from matplotlib import rcParams, font_manager

# ================= 用户需要修改的参数 =================

NC_DIR = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish"  # 存放所有 *.TIMEFIX.daily.CHINA.nc 的目录
REF_NAME = "CMFDV2"  # 参考产品名（文件前缀）

# 海拔 nc 文件
DEM_NC = "/home/ud202380664/CHINA/ObeservationData/etopo2_new.nc"

# 中国 shp 文件（k_1980_albert）
SHP_PATH = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp"
CLIMATE_FIELD = "climate"  # shp 中分区字段名

YEAR_START = 2000  # 统计起止年份
YEAR_END = 2022

THRESHOLD = 0.1  # 检测事件阈值（mm/day），按需要修改

# Chunking for large NetCDF (requires dask). Set to None to disable.
CHUNKS = {"time": 4000, "lat": 200, "lon": 200}

# 海拔分段（单位 m）[lower, upper)，可以按需要改动
ELEV_BINS = [
    (0, 200),
    (200, 500),
    (500, 1000),
    (1000, 1500),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 6000),
    (6000, 10000),
]

OUT_FIG_ELEV = "POD_FAR_by_elevation_01mm.png"

# ====================================================


def set_font():
    """全局字体设置：Times New Roman 为主"""
    font_path_tnr = "/home/ud202380664/Times_New_Roman.ttf"
    font_manager.fontManager.addfont(font_path_tnr)



def open_dataset_maybe_chunked(path: str, chunks: dict | None = None) -> xr.Dataset:
    """Open a NetCDF dataset with optional dask chunking.

    If dask is unavailable (or chunks is None), this falls back to a normal open_dataset.
    """
    if chunks is None:
        return xr.open_dataset(path)
    try:
        import dask  # noqa: F401
    except Exception:
        return xr.open_dataset(path)
    return xr.open_dataset(path, chunks=chunks)

    # 中文备用字体（不作为默认）
    font_path_simhei = "/home/ud202380664/Ubuntu_18.04_SimHei.ttf"
    font_manager.fontManager.addfont(font_path_simhei)

    rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,  # 原脚本 12；为适配多面板 + 长图例，这里更偏“紧凑”
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.dpi": 600,
        }
    )


# ---------------- 工具函数 ----------------


def find_precip_var(ds: xr.Dataset) -> str:
    """自动识别 time-lat-lon 的降水变量名"""
    for v in ds.data_vars:
        da = ds[v]
        if {"time", "lat", "lon"}.issubset(set(da.dims)):
            return v
    raise ValueError("未找到同时包含 time/lat/lon 维度的降水变量，请检查数据。")


def load_dem_on_ref_grid(dem_nc: str, ref_ds: xr.Dataset) -> xr.DataArray:
    """读取 DEM，并插值到参考降水网格（若需要）。

    etopo2_new.nc 常见维度：x/y + 变量 z(y, x)。这里统一重命名为 lon/lat。

    返回：dem_da(lat, lon)
    """
    print(f"读取 DEM: {dem_nc}")
    dem_ds = xr.open_dataset(dem_nc)

    if "x" in dem_ds.dims and "y" in dem_ds.dims:
        dem_ds = dem_ds.rename({"x": "lon", "y": "lat"})

    if "z" not in dem_ds.data_vars:
        # 兜底：自动找一个 lat/lon 的变量
        cand = None
        for v in dem_ds.data_vars:
            da = dem_ds[v]
            if {"lat", "lon"}.issubset(set(da.dims)):
                cand = v
                break
        if cand is None:
            raise ValueError("DEM 文件中未找到可用的 lat/lon 高程变量（常见变量名 z）。")
        dem_da = dem_ds[cand].astype("float32")
    else:
        dem_da = dem_ds["z"].astype("float32")

    if "_FillValue" in dem_da.attrs:
        fv = dem_da.attrs["_FillValue"]
        dem_da = dem_da.where(dem_da != fv)

    # 插值到参考网格
    need_interp = (
        (not np.array_equal(dem_da["lat"].values, ref_ds["lat"].values))
        or (not np.array_equal(dem_da["lon"].values, ref_ds["lon"].values))
    )

    if need_interp:
        print("DEM 网格与降水网格不同，正在插值到参考网格...")
        dem_da = dem_da.interp(lat=ref_ds["lat"], lon=ref_ds["lon"], method="linear")

    return dem_da


def build_region_mask(example_nc_path: str, shp_path: str, climate_field: str = "climate"):
    """用示例 nc 的 lon/lat + shp 生成气候分区掩膜。"""
    print(f"读取示例 NC（用于经纬度）：{example_nc_path}")
    ds = xr.open_dataset(example_nc_path)
    lat = ds["lat"]
    lon = ds["lon"]

    print(f"读取气候分区 shp: {shp_path}")
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        raise ValueError("Shapefile 没有 CRS，请先在外部软件中指定 CRS。")

    gdf = gdf.to_crs(epsg=4326)
    region_names = list(gdf[climate_field].astype(str).values)

    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        names=region_names,
        abbrevs=region_names,
        name="China7Climate",
    )

    print("生成气候分区掩膜(mask)...")
    mask = regions.mask(lon, lat)  # (lat, lon)

    ds.close()
    return mask, region_names


def build_china_mask(example_nc_path: str, shp_path: str, climate_field: str = "climate"):
    """基于 7 大分区掩膜生成中国陆地区域 bool 掩膜"""
    mask_region, _ = build_region_mask(example_nc_path, shp_path, climate_field)
    return mask_region.notnull()


def compute_pod_far(obs: xr.DataArray, sim: xr.DataArray, threshold: float, mask: xr.DataArray | None):
    """计算 POD 和 FAR。

    POD = hits / (hits + misses)
    FAR = false_alarm / (hits + false_alarm)

    obs, sim: (time, lat, lon)
    mask: (lat, lon) bool，True 表示参与统计的位置
    """
    obs, sim = xr.align(obs, sim, join="inner")

    if mask is not None:
        obs = obs.where(mask)
        sim = sim.where(mask)

    valid = np.isfinite(obs) & np.isfinite(sim)

    obs_evt = obs >= threshold
    sim_evt = sim >= threshold

    hits_da = ((obs_evt & sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)
    miss_da = ((obs_evt & ~sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)
    fa_da = ((~obs_evt & sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)

    hits = float(hits_da)
    miss = float(miss_da)
    fa = float(fa_da)

    pod = np.nan
    far = np.nan
    if hits + miss > 0:
        pod = hits / (hits + miss)
    if hits + fa > 0:
        far = fa / (hits + fa)

    return pod, far


# ---------------- 计算按海拔分段的 POD/FAR ----------------


def compute_stats_by_elevation(
    nc_dir: str,
    ref_name: str,
    shp_path: str,
    dem_nc: str,
    elev_bins,
    threshold: float,
    year_start: int,
    year_end: int,
    climate_field: str = "climate",
):
    """按海拔分段计算每个产品的 POD/FAR。"""

    nc_paths = sorted(glob.glob(os.path.join(nc_dir, "*.TIMEFIX.daily.CHINA.nc")))
    if not nc_paths:
        raise FileNotFoundError(f"目录 {nc_dir} 中未找到 *.TIMEFIX.daily.CHINA.nc")

    ref_path = None
    product_paths = {}
    for p in nc_paths:
        prod = os.path.basename(p).split(".")[0]
        if prod == ref_name:
            ref_path = p
        else:
            product_paths[prod] = p

    if ref_path is None:
        raise FileNotFoundError(f"未找到参考产品 {ref_name}.TIMEFIX.daily.CHINA.nc")

    print(f"参考产品: {ref_name} -> {ref_path}")
    print(f"待评估产品: {list(product_paths.keys())}")

    # 参考数据
    ref_ds = open_dataset_maybe_chunked(ref_path, CHUNKS)
    ref_var = find_precip_var(ref_ds)
    ref_da = ref_ds[ref_var].sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))

    # 中国陆地掩膜
    china_mask = build_china_mask(ref_path, shp_path, climate_field)
    print("已生成中国陆地区域掩膜。")

    # DEM -> ref grid
    dem_da = load_dem_on_ref_grid(dem_nc, ref_ds)

    # 每个海拔带掩膜
    elev_masks = {}
    elev_labels = []
    for (lo, hi) in elev_bins:
        label = f"{lo}-{hi} m"
        elev_labels.append(label)
        m = (dem_da >= lo) & (dem_da < hi)
        m = m & china_mask
        elev_masks[label] = m

    elev_stats = {lab: {} for lab in elev_labels}

    for prod, p in product_paths.items():
        print(f"\n处理产品: {prod} -> {p}")
        sim_ds = open_dataset_maybe_chunked(p, CHUNKS)
        sim_var = find_precip_var(sim_ds)
        sim_da = sim_ds[sim_var].sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))

        for lab in elev_labels:
            pod, far = compute_pod_far(ref_da, sim_da, threshold, elev_masks[lab])
            elev_stats[lab][prod] = (pod, far)

        sim_ds.close()

    ref_ds.close()
    print("\n按海拔带 POD/FAR 计算完成。")
    return elev_labels, elev_stats, list(product_paths.keys())


# ---------------- 绘制 performance diagram ----------------


def draw_performance_background(ax, csi_levels=np.arange(0.1, 1.0, 0.1)):
    """在给定 Axes 上画 CSI 背景 + bias 线。返回 contourf 用于 colorbar。"""

    sr = np.linspace(0.01, 1.0, 220)
    pod = np.linspace(0.01, 1.0, 220)
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

    cs = ax.contour(SR, POD, CSI, levels=csi_levels, colors="k", linewidths=0.6, linestyles="dashed")
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    bias_values = [0.5, 0.7, 1.0, 1.5, 2.0]
    for b in bias_values:
        sr_line = np.linspace(0.01, 1.0, 220)
        pod_line = b * sr_line
        pod_line[pod_line > 1.0] = np.nan
        ax.plot(sr_line, pod_line, "k--", linewidth=0.8)
        idx = np.nanargmax(pod_line)
        if not np.isnan(pod_line[idx]):
            ax.text(sr_line[idx], pod_line[idx] + 0.02, f"{b:.1f}", fontsize=8, ha="center", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=9)
    return cf


def plot_elevation_figure(elev_labels, elev_stats, products, out_png):
    """多面板 POD-(1-FAR) 图，每个面板 = 一个海拔带。"""

    n_band = len(elev_labels)
    ncol = 3
    nrow = int(np.ceil(n_band / ncol))

    # 画布加宽：给底部 legend 留空间；同时右侧留 colorbar
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.8 * nrow), dpi=600)
    axes = np.array(axes).reshape(nrow, ncol)

    # 为产品分配颜色和 marker
    n_prod = len(products)
    cmap = plt.cm.get_cmap("tab20", max(n_prod, 1))
    markers = ["o", "s", "^", "P", "X", "D", "*", "v", "<", ">", "h", "H", "p", "d"]
    color_map = {p: cmap(i % 20) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}

    panel_labels = [f"({chr(97 + i)})" for i in range(n_band)]
    cf_for_cbar = None

    for idx, (lab, plab) in enumerate(zip(elev_labels, panel_labels)):
        r = idx // ncol
        c = idx % ncol
        ax = axes[r, c]

        cf = draw_performance_background(ax)
        if cf_for_cbar is None:
            cf_for_cbar = cf

        for p in products:
            pod, far = elev_stats[lab].get(p, (np.nan, np.nan))
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
                linewidths=0.5,
                zorder=5,
            )

        # 面板标题：加白底避免与等值线文字撞车
        ax.text(
            0.02,
            0.95,
            f"{plab} {lab}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

    # 多余的 Axes 删除
    if n_band < nrow * ncol:
        for k in range(n_band, nrow * ncol):
            r = k // ncol
            c = k % ncol
            fig.delaxes(axes[r, c])

    # 只保留外圈坐标标签（减少重叠）
    for r in range(nrow):
        for c in range(ncol):
            if r * ncol + c >= n_band:
                continue
            ax = axes[r, c]
            if r != nrow - 1:
                ax.set_xlabel("")
            if c != 0:
                ax.set_ylabel("")

    # 公共坐标轴标签（放在 figure margin 中）
    fig.text(0.43, 0.18, "Success Ratio (1 - FAR)", ha="center", va="center", fontsize=12)
    fig.text(0.03, 0.58, "Probability of Detection (POD)", rotation="vertical", ha="center", va="center", fontsize=12)

    # legend：放底部预留区，使用 bbox 约束其换行范围
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

    # 产品多时，适当增加列数并缩小字号
    if len(products) <= 10:
        ncol_leg = len(products)
        leg_fs = 9
    elif len(products) <= 18:
        ncol_leg = 6
        leg_fs = 8.5
    else:
        ncol_leg = 8
        leg_fs = 8

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower left",
        bbox_to_anchor=(0.07, 0.04, 0.78, 0.15),
        ncol=ncol_leg,
        frameon=False,
        fontsize=leg_fs,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    # colorbar：右侧独立轴
    cax = fig.add_axes([0.89, 0.30, 0.02, 0.60])
    cbar = fig.colorbar(cf_for_cbar, cax=cax)
    cbar.set_label("Critical Success Index (CSI)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # 布局：明确预留底部 legend 区域；右侧为 colorbar
    fig.subplots_adjust(left=0.07, right=0.86, top=0.95, bottom=0.23, wspace=0.20, hspace=0.20)

    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"已保存按海拔分段的 POD-(1-FAR) 图：{out_png}")


# ---------------- 主函数 ----------------


def main():
    set_font()
    elev_labels, elev_stats, products = compute_stats_by_elevation(
        NC_DIR,
        REF_NAME,
        SHP_PATH,
        DEM_NC,
        ELEV_BINS,
        THRESHOLD,
        YEAR_START,
        YEAR_END,
        CLIMATE_FIELD,
    )
    plot_elevation_figure(elev_labels, elev_stats, products, OUT_FIG_ELEV)


if __name__ == "__main__":
    main()
