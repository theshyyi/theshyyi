#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
POD–(1-FAR) performance diagram by elevation bands
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
  （若网格与降水不完全一致，会用插值到参考产品网格）

- SHP_PATH: 中国 7 大分区 shp，用来生成“中国陆地区域”掩膜

- 在每个海拔带内 (ELEV_BINS)，计算
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
import matplotlib.dates as mdates

# ================= 用户需要修改的参数 =================

NC_DIR   = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish"          # 存放所有 *.TIMEFIX.daily.CHINA.nc 的目录
REF_NAME = "CMFDV2"                          # 参考产品名（文件前缀）

      # 海拔 nc 文件
DEM_NC = "/home/ud202380664/CHINA/ObeservationData/etopo2_new.nc"

SHP_PATH = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp" # 中国 shp 文件（k_1980_albert）
CLIMATE_FIELD = "climate"                      # shp 中分区字段名

YEAR_START = 2000                             # 统计起止年份
YEAR_END   = 2022

THRESHOLD = 1                              # 检测事件阈值（mm/day），按需要修改

# 海拔分段（单位 m）[lower, upper)，可以按需要改动
ELEV_BINS = [
    (0,   200),
    (0,   500),
    (500, 1000),
    (1000,1500),
    (1500, 3000),
    (3000, 10000),
]

OUT_FIG_ELEV = "POD_FAR_by_elevation_10mm.png"

# ====================================================

def set_font():
    """全局字体设置：Times New Roman 为主"""
    # 1. 注册 Times New Roman 字体（你系统里的路径）
    font_path_tnr = "/home/ud202380664/Times_New_Roman.ttf"
    font_manager.fontManager.addfont(font_path_tnr)

    # 2. 注册 SimHei（可选，仅作为中文备用，不会被设为默认）
    font_path_simhei = "/home/ud202380664/Ubuntu_18.04_SimHei.ttf"
    font_manager.fontManager.addfont(font_path_simhei)
    # 3. 创建一个 FontProperties，用于标题
    SIMHEI_FP = font_manager.FontProperties(fname=font_path_simhei)
    # 3. 更新 Matplotlib 全局配置：默认使用 Times New Roman
    config = {
        "font.family": "Times New Roman",  # 全局默认字体
        "font.size": 16,                   # 全局字号
        "mathtext.fontset": "stix"         # 数学公式用 STIX 系列，风格接近 Times
    }
    rcParams.update(config)

    # 一些通用设置
    rcParams["axes.unicode_minus"] = False   # 避免负号乱码
    rcParams["figure.dpi"] = 600


# ---------------- 工具函数 ----------------

def find_precip_var(ds: xr.Dataset) -> str:
    """自动识别 time-lat-lon 的降水变量名"""
    for v in ds.data_vars:
        da = ds[v]
        if {"time", "lat", "lon"}.issubset(set(da.dims)):
            return v
    raise ValueError("未找到同时包含 time/lat/lon 维度的降水变量，请检查数据。")


def find_elev_var(ds: xr.Dataset) -> str:
    """自动识别 lat-lon 的海拔变量名"""
    for v in ds.data_vars:
        da = ds[v]
        if {"lat", "lon"}.issubset(set(da.dims)):
            return v
    raise ValueError("未找到 lat/lon 维度的高程变量，请检查 DEM 文件。")


def load_dem_on_ref_grid(dem_nc: str, ref_ds: xr.Dataset) -> xr.DataArray:
    """
    读取 etopo2_new.nc，并插值到参考降水数据 ref_ds 的网格上。

    返回：
        dem_da_interp(lat, lon)  —— 高程 (m)
    """
    print(f"读取 DEM: {dem_nc}")
    dem_ds = xr.open_dataset(dem_nc)

    # 你的 DEM 维度是 x(y轴经度), y(纬度)，变量名是 z(y, x)
    # 先把 x/y 重命名为 lon/lat，方便后续统一处理
    dem_ds = dem_ds.rename({"x": "lon", "y": "lat"})
    dem_da = dem_ds["z"].astype("float32")

    # 将填充值转为 NaN（xarray 一般已经自动处理，这里再保险一次）
    if "_FillValue" in dem_da.attrs:
        fv = dem_da.attrs["_FillValue"]
        dem_da = dem_da.where(dem_da != fv)

    # 如果网格与参考数据不同，则插值到参考网格
    if (not np.array_equal(dem_da["lat"].values, ref_ds["lat"].values)) or \
       (not np.array_equal(dem_da["lon"].values, ref_ds["lon"].values)):
        print("DEM 网格与降水网格不同，正在插值到参考网格...")
        dem_da = dem_da.interp(
            lat=ref_ds["lat"],
            lon=ref_ds["lon"],
            method="linear",
        )

    return dem_da


def build_region_mask(example_nc_path, shp_path, climate_field="climate"):
    """
    用一个示例 nc 的 lon/lat + shp 生成 7 大气候分区掩膜。
    返回：
      mask_2d: DataArray(lat, lon) -> 分区索引(0..n-1)，海洋/无区为 NaN
      region_names: list[str]
    """
    print(f"读取示例 NC（用于经纬度）：{example_nc_path}")
    ds = xr.open_dataset(example_nc_path)
    lat = ds["lat"]
    lon = ds["lon"]

    print(f"读取气候分区 shp: {shp_path}")
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        raise ValueError("Shapefile 没有 CRS，请先在外部软件中指定为 Krasovsky_1940_Albers。")

    print("Original CRS of shp:", gdf.crs)
    gdf = gdf.to_crs(epsg=4326)
    print("Reprojected CRS:", gdf.crs)

    region_names = list(gdf[climate_field].astype(str).values)

    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        names=region_names,
        abbrevs=region_names,
        name="China7Climate",
    )

    print("生成气候分区掩膜(mask)...")
    mask = regions.mask(lon, lat)   # (lat, lon)

    ds.close()
    return mask, region_names


def build_china_mask(example_nc_path, shp_path, climate_field="climate"):
    """基于 7 大分区掩膜生成中国陆地区域 bool 掩膜"""
    mask_region, _ = build_region_mask(example_nc_path, shp_path, climate_field)
    china_mask = mask_region.notnull()   # True = 在任一区域
    return china_mask


def compute_pod_far(obs: xr.DataArray,
                    sim: xr.DataArray,
                    threshold: float,
                    mask: xr.DataArray):
    """
    计算 POD 和 FAR：
      POD = hits / (hits + misses)
      FAR = false_alarm / (hits + false_alarm)

    obs, sim: (time, lat, lon)
    mask: (lat, lon) bool，True 表示参与统计的位置
    """
    # 时间对齐
    obs, sim = xr.align(obs, sim, join="inner")

    if mask is not None:
        obs = obs.where(mask)
        sim = sim.where(mask)

    valid = np.isfinite(obs) & np.isfinite(sim)

    obs_evt = obs >= threshold
    sim_evt = sim >= threshold

    hits_da = ((obs_evt & sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)
    miss_da = ((obs_evt & ~sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)
    fa_da   = ((~obs_evt & sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)

    hits = float(hits_da)
    miss = float(miss_da)
    fa   = float(fa_da)

    pod = np.nan
    far = np.nan
    if hits + miss > 0:
        pod = hits / (hits + miss)
    if hits + fa > 0:
        far = fa / (hits + fa)

    return pod, far


# ---------------- 计算按海拔分段的 POD/FAR ----------------

def compute_stats_by_elevation(nc_dir, ref_name,
                               shp_path, dem_nc,
                               elev_bins, threshold,
                               year_start, year_end,
                               climate_field="climate"):
    """
    返回:
      elev_labels: list[str]
      elev_stats: dict[label][product] = (POD, FAR)
      products: list[str]（不含参考数据名）
    """
    # 所有降水产品文件
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

    # 打开参考数据
    ref_ds = xr.open_dataset(ref_path, chunks={"time": 2000})
    ref_var = find_precip_var(ref_ds)
    ref_da = ref_ds[ref_var].sel(
        time=slice(f"{year_start}-01-01", f"{year_end}-12-31")
    )

    # 中国掩膜
    china_mask = build_china_mask(ref_path, shp_path, climate_field)
    print("已生成中国陆地区域掩膜。")

    # 读 DEM，并插值到参考网格（如果需要）
    
        # 读 DEM 并插值到参考网格（适配 etopo2_new.nc）
    dem_da = load_dem_on_ref_grid(dem_nc, ref_ds)

    
    
    # dem_ds = xr.open_dataset(dem_nc)
    # elev_var = find_elev_var(dem_ds)
    # dem_da = dem_ds[elev_var]

    # # 若经纬度名不同，可在这里统一
    # # 假设 DEM 也叫 lat/lon；否则手动 rename
    # if not np.array_equal(dem_da["lat"], ref_ds["lat"]) or not np.array_equal(
        # dem_da["lon"], ref_ds["lon"]
    # ):
        # print("DEM 网格与降水网格不同，正在插值到参考网格...")
        # dem_da = dem_da.interp(
            # lat=ref_ds["lat"], lon=ref_ds["lon"], method="linear"
        # )

    # 构造每个海拔带掩膜
    elev_masks = {}
    elev_labels = []
    for (lo, hi) in elev_bins:
        label = f"{lo}-{hi} m"
        elev_labels.append(label)
        m = (dem_da >= lo) & (dem_da < hi)
        # 加上中国陆地掩膜
        if china_mask is not None:
            m = m & china_mask
        elev_masks[label] = m

    # 结果容器
    elev_stats = {lab: {} for lab in elev_labels}

    # 遍历每个产品
    for prod, p in product_paths.items():
        print(f"\n处理产品: {prod} -> {p}")
        sim_ds = xr.open_dataset(p, chunks={"time": 2000})
        sim_var = find_precip_var(sim_ds)
        sim_da = sim_ds[sim_var].sel(
            time=slice(f"{year_start}-01-01", f"{year_end}-12-31")
        )

        # 各海拔带 POD/FAR
        for lab in elev_labels:
            mask_band = elev_masks[lab]
            pod, far = compute_pod_far(ref_da, sim_da, threshold, mask_band)
            elev_stats[lab][prod] = (pod, far)

        sim_ds.close()

    ref_ds.close()
    # dem_ds.close()
    print("\n按海拔带 POD/FAR 计算完成。")
    return elev_labels, elev_stats, list(product_paths.keys())


# ---------------- 绘制 performance diagram ----------------

def draw_performance_background(ax, csi_levels=np.arange(0.1, 1.0, 0.1)):
    """
    在给定 Axes 上画 CSI 背景 + bias 线
    返回 contourf 用于 colorbar
    """
    sr = np.linspace(0.01, 1.0, 200)
    pod = np.linspace(0.01, 1.0, 200)
    SR, POD = np.meshgrid(sr, pod)

    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    CSI[(CSI < 0) | ~np.isfinite(CSI)] = np.nan

    cf = ax.contourf(SR, POD, CSI,
                     levels=np.linspace(0.1, 0.9, 9),
                     cmap="Blues",
                     extend="both")

    # CSI 等值线
    cs = ax.contour(SR, POD, CSI,
                    levels=csi_levels,
                    colors="k",
                    linewidths=0.6,
                    linestyles="dashed")
    ax.clabel(cs, inline=True, fontsize=14, fmt="%.1f")

    # bias 线
    bias_values = [0.5, 0.7, 1.0, 1.5, 2.0]
    for b in bias_values:
        sr_line = np.linspace(0.01, 1.0, 200)
        pod_line = b * sr_line
        pod_line[pod_line > 1.0] = np.nan
        ax.plot(sr_line, pod_line, "k--", linewidth=0.8)
        idx = np.nanargmax(pod_line)
        if not np.isnan(pod_line[idx]):
            ax.text(sr_line[idx], pod_line[idx] + 0.02, f"{b:.1f}",
                    fontsize=14, ha="center", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Success Ratio (1 - FAR)", fontsize=16)
    ax.set_ylabel("Probability of Detection (POD)", fontsize=16)
    ax.tick_params(labelsize=14)
    return cf


def plot_elevation_figure(elev_labels, elev_stats, products, out_png):
    """
    多面板 POD-(1-FAR) 图，每个面板 = 一个海拔带
    """
    n_band = len(elev_labels)
    ncol = 2
    nrow = int(np.ceil(n_band / ncol))

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(9, 4 * nrow),
                             dpi=600)
    axes = np.array(axes).reshape(nrow, ncol)

    # 为产品分配颜色和 marker
    n_prod = len(products)
    cmap = plt.cm.get_cmap("tab10", n_prod)
    markers = ["o", "s", "^", "P", "X", "D", "*", "v", "<", ">"]
    color_map = {p: cmap(i) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}

    panel_labels = [f"({chr(97+i)})" for i in range(n_band)]

    cf_for_cbar = None

    for idx, (lab, plab) in enumerate(zip(elev_labels, panel_labels)):
        r = idx // ncol
        c = idx % ncol
        ax = axes[r, c]

        cf = draw_performance_background(ax)
        if cf_for_cbar is None:
            cf_for_cbar = cf

        for p in products:
            pod, far = elev_stats[lab][p]
            if np.isnan(pod) or np.isnan(far):
                continue
            sr = 1.0 - far
            ax.scatter(sr, pod,
                       marker=marker_map[p],
                       color=color_map[p],
                       s=50,
                       edgecolors="k",
                       linewidths=0.6,
                       zorder=5)

        ax.text(0.02, 0.95, f"{plab} {lab}",
                transform=ax.transAxes,
                fontsize=16, fontweight="bold",
                ha="left", va="top")

    # 如果面板数为奇数，最后一个多余的 Axes 去掉
    if n_band < nrow * ncol:
        for k in range(n_band, nrow * ncol):
            r = k // ncol
            c = k % ncol
            fig.delaxes(axes[r, c])

    # 统一图例：底部
    handles = []
    labels = []
    for p in products:
        h = plt.Line2D([], [], linestyle="none",
                       marker=marker_map[p],
                       markersize=6,
                       markerfacecolor=color_map[p],
                       markeredgecolor="k",
                       label=p)
        handles.append(h)
        labels.append(p)

    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.02),
               ncol=min(len(products), 5),
               frameon=False,
               fontsize=14,
               columnspacing=0.8,
               handletextpad=0.4)

    # 统一 colorbar：右侧独立轴
    cax = fig.add_axes([0.92, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(cf_for_cbar, cax=cax)
    cbar.set_label("Critical Success Index (CSI)", fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout(rect=[0.05, 0.08, 0.90, 0.96])
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"已保存按海拔分段的 POD-(1-FAR) 图：{out_png}")


# ---------------- 主函数 ----------------

def main():
    set_font()
    elev_labels, elev_stats, products = compute_stats_by_elevation(
        NC_DIR, REF_NAME,
        SHP_PATH, DEM_NC,
        ELEV_BINS, THRESHOLD,
        YEAR_START, YEAR_END,
        CLIMATE_FIELD,
    )
    plot_elevation_figure(elev_labels, elev_stats, products, OUT_FIG_ELEV)


if __name__ == "__main__":
    main()
