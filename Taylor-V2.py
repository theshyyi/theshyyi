#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Taylor diagram (STD, CC, RMSE) for multi-precipitation products
================================================================

- 目录下所有 *.TIMEFIX.daily.CHINA.nc 视为降水产品
  * 文件名格式: {PRODUCT}.TIMEFIX.daily.CHINA.nc
  * 例如: CMFDV2.TIMEFIX.daily.CHINA.nc, CHIRPSV2.TIMEFIX.daily.CHINA.nc
- REF_NAME 指定哪个产品为参考（OBS）
- 与参考产品相比，计算：
  * 四季 (Spring/Summer/Autumn/Winter, MAM/JJA/SON/DJF)
  * 12 个月 (M01–M12)
  的：
    - 标准差 STD
    - 相关系数 CC
    - 中心化均方根误差 CRMSD (用于泰勒图上的 RMSE 圆弧)
- 用中国 7 大气候分区 shp 生成掩膜（只保留中国区域格点）

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

# ============ 用户需要修改的参数 ============

NC_DIR   = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish"          # 存放所有 *.TIMEFIX.daily.CHINA.nc 的目录
REF_NAME = "CMFDV2"                        # 参考产品前缀名（文件名开头部分）
SHP_PATH = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp" # 中国 shp 文件（k_1980_albert）
CLIMATE_FIELD = "climate"
OUT_TAYLOR_SEASON = "taylor_seasons.png"
OUT_TAYLOR_MONTH  = "taylor_months.png"

CLIMATE_FIELD = "climate"   # shp 中分区名称字段，如果不同请修改

# ==========================================
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
        "font.size": 14,                   # 全局字号
        "mathtext.fontset": "stix"         # 数学公式用 STIX 系列，风格接近 Times
    }
    rcParams.update(config)

    # 一些通用设置
    rcParams["axes.unicode_minus"] = False   # 避免负号乱码
    rcParams["figure.dpi"] = 600



# ---------- 利用你原来的函数，生成 7 大分区 mask ----------

def build_region_mask(example_nc_path, shp_path, climate_field="climate"):
    """
    用一个示例 nc 的 lon/lat + shp 生成 7 大气候分区掩膜。

    返回：
      mask_2d: DataArray(lat, lon) -> 分区索引(0..n-1)，海洋/无区为 NaN
      region_names: list[str] 分区名（与索引 0..n-1 对应，中文）
    """
    print(f"读取示例 NC（用于经纬度）：{example_nc_path}")
    ds = xr.open_dataset(example_nc_path)

    lat = ds["lat"]   # 1D
    lon = ds["lon"]   # 1D

    print(f"读取气候分区 shp: {shp_path}")
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        raise ValueError("Shapefile 没有 CRS，请先在外部软件中指定为 Krasovsky_1940_Albers。")

    print("Original CRS of shp:", gdf.crs)

    # 转为 WGS84，经纬度坐标（与 nc 一致）
    gdf = gdf.to_crs(epsg=4326)
    print("Reprojected CRS:", gdf.crs)

    # 分区名称列表（中文）
    region_names = list(gdf[climate_field].astype(str).values)

    # 使用老接口构造 Regions
    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        names=region_names,
        abbrevs=region_names,
        name="China7Climate"
    )

    print("生成气候分区掩膜(mask)...")
    mask = regions.mask(lon, lat)   # dims: (lat, lon)

    ds.close()
    return mask, region_names


def build_china_mask(example_nc_path, shp_path, climate_field="climate"):
    """
    基于 7 大分区 mask 合成一个 "中国陆地" 的 bool 掩膜。
    """
    mask_region, _ = build_region_mask(example_nc_path, shp_path, climate_field)
    china_mask = mask_region.notnull()   # True = 在任意一个分区内
    return china_mask


# ---------- 辅助函数 ----------

def find_precip_var(ds: xr.Dataset) -> str:
    """自动识别 time-lat-lon 的降水变量名"""
    for v in ds.data_vars:
        da = ds[v]
        if {"time", "lat", "lon"}.issubset(set(da.dims)):
            return v
    raise ValueError("未能在数据集中找到同时含有 time/lat/lon 的降水变量，请检查变量名。")


def subset_by_months(da: xr.DataArray, months: list[int]) -> xr.DataArray:
    """按月份列表子集数据（可以跨年：比如 [12,1,2] 表示冬季）"""
    return da.sel(time=da["time"].dt.month.isin(months))


def calc_stats_pair(obs: xr.DataArray,
                    sim: xr.DataArray,
                    china_mask: xr.DataArray | None = None):
    """
    使用 xarray/dask 直接计算:
      - obs_std
      - sim_std
      - corr (Pearson)
      - crmsd (中心化均方根误差)
    维度: (time, lat, lon)
    """
    # 时间对齐
    obs, sim = xr.align(obs, sim, join="inner")

    if china_mask is not None:
        obs = obs.where(china_mask)
        sim = sim.where(china_mask)

    # 去掉全为 NaN 的格点，下面 mean/std 会自动 skipna
    dims = ("time", "lat", "lon")

    obs_mean = obs.mean(dim=dims, skipna=True)
    sim_mean = sim.mean(dim=dims, skipna=True)

    obs_anom = obs - obs_mean
    sim_anom = sim - sim_mean

    obs_std = np.sqrt((obs_anom ** 2).mean(dim=dims, skipna=True))
    sim_std = np.sqrt((sim_anom ** 2).mean(dim=dims, skipna=True))
    
     # 检查并替换 NaN 或 Inf 值为一个最差的值（这里用9999表示最差的标准差）
    if np.isnan(obs_std) or np.isinf(obs_std):
        print(f"Warning: Invalid observation std detected. Replacing with a poor value.")
        obs_std = 10.0  # 可以选择任何合适的最差值

    if np.isnan(sim_std) or np.isinf(sim_std):
        print(f"Warning: Invalid simulation std detected. Replacing with a poor value.")
        sim_std = 10.0  # 可以选择任何合适的最差值

    cov = (obs_anom * sim_anom).mean(dim=dims, skipna=True)
    corr = cov / (obs_std * sim_std)

    crmsd = np.sqrt(((sim_anom - obs_anom) ** 2).mean(dim=dims, skipna=True))
    
       # 处理相关系数为 NaN 或 Inf 的情况
    if np.isnan(corr) or np.isinf(corr):
        print(f"Warning: Invalid correlation detected. Replacing with a poor value.")
        corr = -1  # 相关系数设为 -1，表示极差的匹配

    # 处理均方根误差为 NaN 或 Inf 的情况
    if np.isnan(crmsd) or np.isinf(crmsd):
        print(f"Warning: Invalid CRMSD detected. Replacing with a poor value.")
        crmsd = 10.0  # 设置一个极差的 CRMSD 值

    # ---- 统一转换为 float 标量，兼容 DataArray / numpy 标量 / float ----
    def to_scalar(v):
        # xarray.DataArray 或 numpy 数组
        if hasattr(v, "values"):
            v = v.values
        v = np.asarray(v)
        # 空数组就返回 NaN
        if v.size == 0:
            return np.nan
        return float(v)

    return (to_scalar(obs_std),
            to_scalar(sim_std),
            to_scalar(corr),
            to_scalar(crmsd))



# ---------- 计算季节 & 月尺度统计量 ----------

def compute_all_stats_taylor(nc_dir: str,
                             ref_name: str,
                             shp_path: str,
                             climate_field: str):

    # 1. 收集所有产品路径
    nc_paths = sorted(glob.glob(os.path.join(nc_dir, "*.TIMEFIX.daily.CHINA.nc")))
    if not nc_paths:
        raise FileNotFoundError(f"目录 {nc_dir} 中未找到 *.TIMEFIX.daily.CHINA.nc 文件")

    ref_path = None
    product_paths = {}
    for p in nc_paths:
        fname = os.path.basename(p)
        prod = fname.split(".")[0]
        if prod == ref_name:
            ref_path = p
        else:
            product_paths[prod] = p

    if ref_path is None:
        raise FileNotFoundError(f"未在 {nc_dir} 中找到参考产品 {ref_name}.TIMEFIX.daily.CHINA.nc")

    print(f"参考数据: {ref_name} -> {ref_path}")
    print(f"待评估产品: {list(product_paths.keys())}")

    # 2. 打开参考数据、生成中国掩膜
    ref_ds = xr.open_dataset(ref_path, chunks={"time": 2000})
    ref_var = find_precip_var(ref_ds)
    ref_da = ref_ds[ref_var]

    china_mask = build_china_mask(ref_path, shp_path, climate_field)
    print("已生成中国区域掩膜。")

    # 3. 定义季节和月份
    seasons_def = {
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
        "Winter": [12, 1, 2],
    }
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    months = list(range(1, 13))

    # 4. 结果结构：
    #   season_stats["Spring"] = {"obs_std": float, "models": {prod: (std, cc, crmsd)}}
    season_stats = {s: {"obs_std": None, "models": {}} for s in season_order}
    month_stats = {m: {"obs_std": None, "models": {}} for m in months}

    # 5. 遍历每个产品
    for prod, path in product_paths.items():
        print(f"\n处理产品: {prod} -> {path}")
        sim_ds = xr.open_dataset(path, chunks={"time": 2000})
        sim_var = find_precip_var(sim_ds)
        sim_da = sim_ds[sim_var]

        # ---- 四季 ----
        for sname in season_order:
            mon_list = seasons_def[sname]
            obs_s = subset_by_months(ref_da, mon_list)
            sim_s = subset_by_months(sim_da, mon_list)

            obs_std, sim_std, corr, crmsd = calc_stats_pair(
                obs_s, sim_s, china_mask
            )

            if season_stats[sname]["obs_std"] is None:
                season_stats[sname]["obs_std"] = obs_std
            season_stats[sname]["models"][prod] = (sim_std, corr, crmsd)

        # ---- 逐月 ----
        for m in months:
            obs_m = subset_by_months(ref_da, [m])
            sim_m = subset_by_months(sim_da, [m])

            obs_std, sim_std, corr, crmsd = calc_stats_pair(
                obs_m, sim_m, china_mask
            )

            if month_stats[m]["obs_std"] is None:
                month_stats[m]["obs_std"] = obs_std
            month_stats[m]["models"][prod] = (sim_std, corr, crmsd)

        sim_ds.close()

    ref_ds.close()
    print("\nSTD / CC / RMSE 计算完成。")
    return season_stats, month_stats, list(product_paths.keys())


# ---------- 泰勒图绘图工具 ----------

def draw_taylor_axes(ax, obs_std, r_max):
    """
    在 ax 上画泰勒图背景：
      - 外边界 & 内部 std 圆
      - CC 放射线
      - RMSE 圆弧
    """
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, r_max * 1.05)
    ax.set_ylim(0, r_max * 1.05)

    theta = np.linspace(0, np.pi / 2, 200)

    # 外边界
    ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), "k", lw=1.2, zorder=1)
    # 坐标轴
    ax.plot([0, r_max], [0, 0], "k", lw=1.2, zorder=1)
    ax.plot([0, 0], [0, r_max], "k", lw=1.2, zorder=1)

    # 若干 std 圆
    for r in np.linspace(r_max / 4, r_max, 3, endpoint=False):
        ax.plot(r * np.cos(theta), r * np.sin(theta), "k:", lw=0.6, zorder=1)

    # OBS 的 std 圆
    ax.plot(obs_std * np.cos(theta), obs_std * np.sin(theta),
            color="red", lw=1.4, zorder=1)

    # 相关系数 (0.1~0.99) 放射线
    cc_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for cc in cc_levels:
        t = np.arccos(cc)
        x = np.array([0, r_max * np.cos(t)])
        y = np.array([0, r_max * np.sin(t)])
        ax.plot(x, y, color="b", linestyle="dashdot", lw=0.6, zorder=2)

        # 在边界稍外侧标注相关系数值
        ax.text((r_max * 1.02) * np.cos(t),
                (r_max * 1.02) * np.sin(t),
                f"{cc:.2f}" if cc < 0.99 else "0.99",
                fontsize=12, color="b",
                ha="center", va="center", zorder=4)

    # RMSE 圆弧（以 OBS 点为圆心）
    rmse_levels = [obs_std * 0.5, obs_std * 1.0, obs_std * 1.5]
    for rm in rmse_levels:
        r_list = []
        x_list = []
        y_list = []
        for t in theta:
            # 解: d^2 = r^2 - 2 r obs_std cos(t) + obs_std^2 = rm^2
            A = 1.0
            B = -2.0 * obs_std * np.cos(t)
            C = obs_std**2 - rm**2
            disc = B**2 - 4 * A * C
            if disc < 0:
                r = np.nan
            else:
                r = (-B + np.sqrt(disc)) / (2 * A)
            r_list.append(r)
            x_list.append(r * np.cos(t))
            y_list.append(r * np.sin(t))
        ax.plot(x_list, y_list, "g--", lw=0.8, zorder=3)

    # RMSE 标识
    ax.text(0.02, 0.78, "RMSE",
            transform=ax.transAxes,
            color="g", fontsize=10,
            ha="left", va="top", zorder=4)

    # 标注 OBS
    ax.plot(obs_std, 0, "o", color="red", zorder=5)
    ax.text(obs_std, -0.04 * r_max, "OBS",
            color="red", fontsize=12,
            ha="center", va="top", zorder=5)

    # 轴标签
    ax.set_xlabel("Standard Deviation", fontsize=12, zorder=4)
    ax.set_ylabel("Standard Deviation", fontsize=12, zorder=4)
    ax.tick_params(labelsize=12, zorder=4)


def plot_taylor_panel(ax, stats_one_period, products, panel_title, panel_label,
                      color_map, marker_map):
    """
    在给定 ax 上绘制一个时期（一个季节或一个月份）的泰勒图。
    """
    obs_std = stats_one_period["obs_std"]
    model_stats = stats_one_period["models"]

    # 找到该时期所有模型 std 的最大值，作为 r_max
    std_max = obs_std
    for p in products:
        std_p, _, _ = model_stats[p]
        std_max = max(std_max, std_p)
    r_max = std_max * 1.25

    draw_taylor_axes(ax, obs_std, r_max)

    # 各产品点
    for p in products:
        std_p, corr_p, _ = model_stats[p]
        if np.isnan(std_p) or np.isnan(corr_p):
            continue
        theta = np.arccos(np.clip(corr_p, -1.0, 1.0))
        x = std_p * np.cos(theta)
        y = std_p * np.sin(theta)
        ax.scatter(x, y,
                   marker=marker_map[p],
                   color=color_map[p],
                   s=50,
                   edgecolors="k",
                   linewidths=0.6,
                   zorder=5)

    # 左上角标注 (a) Spring 之类
    ax.text(0.95, 0.96, f"{panel_label} {panel_title}",
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            ha="right", va="top")


# ---------- 季节 / 月份 泰勒图 ----------
def plot_taylor_seasons(season_stats, products, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=600)
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    n_prod = len(products)
    cmap = plt.cm.get_cmap("tab10", n_prod)
    markers = ["o", "s", "^", "P", "X", "D", "*", "v", "<", ">"]
    color_map = {p: cmap(i) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}

    for i, (s, lab) in enumerate(zip(season_order, panel_labels)):
        ax = axes.flat[i]
        plot_taylor_panel(ax, season_stats[s], products, s, lab,
                          color_map, marker_map)

    # --- 构造 legend ---
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

    # 先给底部留足空间（关键：别让 tight_layout 抢占底部）
    # bottom 越大，底部留白越多；0.18~0.25 一般够容纳 1~2 行图例
    fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.25,
                        wspace=0.18, hspace=0.20)

    # 图例放在画布外底部居中，避免被裁切
    ncol = min(len(products), 6)
    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.04),   # 0.00~0.06 之间都可微调
               ncol=ncol,
               frameon=False,
               fontsize=10,
               columnspacing=0.8,
               handletextpad=0.4)

    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"已保存季节泰勒图：{out_png}")


def plot_taylor_months(month_stats, products, out_png):
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), dpi=600)
    months = list(range(1, 13))
    panel_labels = [f"({chr(97+i)})" for i in range(12)]  # (a) ~ (l)

    n_prod = len(products)
    cmap = plt.cm.get_cmap("tab10", n_prod)
    markers = ["o", "s", "^", "P", "X", "D", "*", "v", "<", ">"]
    color_map = {p: cmap(i) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}

    for i, (m, lab) in enumerate(zip(months, panel_labels)):
        ax = axes.flat[i]
        title = f"M{m:02d}"
        plot_taylor_panel(ax, month_stats[m], products, title, lab,
                          color_map, marker_map)

    # 把 label 简化：只在最左列 / 最下行画轴标签
    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
    for row in range(4):
        axes[row, 0].set_ylabel("Standard Deviation", fontsize=16)
    for col in range(3):
        axes[3, col].set_xlabel("Standard Deviation", fontsize=16)

    # --- 构造 legend ---
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

    # 月度图一般产品多、图例容易占两行：给底部留更大空间
    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.20,
                        wspace=0.16, hspace=0.18)

    ncol = min(len(products), 6)
    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.04),   # 这里不要用 0.03/0.08 那类“靠上”的值
               ncol=ncol,
               frameon=False,
               fontsize=10,
               columnspacing=0.8,
               handletextpad=0.4)

    # 用 bbox_inches="tight" 可避免图例被裁切；pad_inches 控制边缘留白
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"已保存月份泰勒图：{out_png}")



# =================== 主程序 ===================

def main():
    set_font()
    season_stats, month_stats, products = compute_all_stats_taylor(
        NC_DIR, REF_NAME, SHP_PATH, CLIMATE_FIELD
    )
    plot_taylor_seasons(season_stats, products, OUT_TAYLOR_SEASON)
    plot_taylor_months(month_stats, products, OUT_TAYLOR_MONTH)


if __name__ == "__main__":
    main()
