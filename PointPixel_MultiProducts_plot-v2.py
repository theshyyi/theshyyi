#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import string
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ===================== 字体 =====================
def set_font():
    font_path_tnr = "/home/ud202380664/Times_New_Roman.ttf"
    font_path_simhei = "/home/ud202380664/Ubuntu_18.04_SimHei.ttf"
    font_manager.fontManager.addfont(font_path_tnr)
    font_manager.fontManager.addfont(font_path_simhei)
    SIMHEI_FP = font_manager.FontProperties(fname=font_path_simhei)
    rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],
        "font.size": 14,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.dpi": 600
    })
    return SIMHEI_FP

SIMHEI_FP = set_font()

# ===================== 路径 =====================
shapefile_path = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp"
base_dir = "/home/ud202380664/CHINA/ObeservationData/PointPixel/"
nine_dash_path = "/home/ud202380664/CHINA/ObeservationData/NINELINE/NINE.shp"

out_dir = os.path.join(base_dir, "_ALL_PRODUCTS_METRIC_MAPS")
os.makedirs(out_dir, exist_ok=True)

# ===================== 画图参数 =====================
N_COLS = 4
POINT_SIZE = 10
POINT_EDGE = 0.35
POINT_ALPHA = 0.95

# inset 内点可以更小一点
INSET_POINT_SIZE = 6
INSET_POINT_EDGE = 0.25
INSET_POINT_ALPHA = 0.95

AXIS_OFF = True
CHINA_OUTLINE_LW = 0.8
DRAW_CLIMATE_BOUNDARY = False
CLIMATE_BOUNDARY_LW = 0.35

# inset 的范围（南海）
SCS_XLIM = (105, 125)
SCS_YLIM = (3, 25)

# 每个子图 inset 的大小和位置（轴坐标，0~1）
# 你如果仍觉得挡住主图，就调 x/y；想更大就调 w/h
INSET_RECT = (0.72, -0.1, 0.4, 0.4)


# 极端不合理值阈值：用于“估计色标范围”时剔除
HARD_ABS_MAX_FOR_RANGE = 1e20

# 用分位数确定范围（避免极端值拉爆，但仍是“实际值驱动”）
Q_LOW = 0.02
Q_HIGH = 0.98

# ===================== 指标“最好值最深”的模式设置 =====================
# mode:
#   - "high": 越大越好（最优=最大值最深）
#   - "low" : 越小越好（最优=最小值最深）
#   - "center": 最优在 center（最深），向两侧变浅
METRIC_MODE = {
    "bias":  ("center", 0.0),
    "fbias": ("center", 1.0),

    "mae":   ("low", None),
    "rmse":  ("low", None),
    "far":   ("low", None),

    "corr":  ("high", None),
    "kge":   ("high", None),
    "pod":   ("high", None),
    "csi":   ("high", None),
    "hss":   ("high", None),

    "std_o": ("high", None),
    "std_s": ("high", None),
}

# 已知天然范围的指标（直接固定，不用分位数）
FIXED_RANGES = {
    "corr": (0.0, 1.0),
    "pod":  (0.0, 1.0),
    "far":  (0.0, 1.0),
    "csi":  (0.0, 1.0),
    # kge/hss 通常 [-1,1]，如果你希望严格固定也可打开：
    "kge":  (-1.0, 1.0),
    "hss":  (-1.0, 1.0),
}

# ===================== 工具函数 =====================
def safe_to_crs(gdf, epsg=4326):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg, allow_override=True)
    return gdf.to_crs(epsg=4326)

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def panel_label(i: int) -> str:
    letters = string.ascii_lowercase
    s = ""
    n = i
    while True:
        n, r = divmod(n, 26)
        s = letters[r] + s
        if n == 0:
            break
        n -= 1  # 关键：处理“进位”（保证 26 -> aa，而不是 ba）
    return f"({s})"


def make_center_dark_cmap(base="YlGn"):
    base_cmap = plt.get_cmap(base)
    light = base_cmap(0.12)
    dark  = base_cmap(0.95)
    return LinearSegmentedColormap.from_list("center_dark", [light, dark, light], N=256)

def sanitize_for_range(arr):
    """用于估计色标范围：剔除 NaN/inf 及明显离谱的极端值（不影响绘图时 clip 显示全部点）"""
    x = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    x[np.abs(x) >= HARD_ABS_MAX_FOR_RANGE] = np.nan
    return x

def clip_for_plot(arr, vmin, vmax):
    """用于绘图：所有点都显示，超范围贴边"""
    x = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    # 这里不把极端值变 NaN，而是直接贴边
    return np.clip(x, vmin, vmax)

def compute_vmin_vmax(metric, values_all):
    """按“实际值”确定色标范围（全产品合并），并具备鲁棒性"""
    m = metric.lower().strip()

    # 固定范围优先
    if m in FIXED_RANGES:
        return FIXED_RANGES[m]

    # 用分位数范围
    x = sanitize_for_range(values_all)
    x = x[np.isfinite(x)]
    if x.size == 0:
        # 兜底
        return (0.0, 1.0)

    mode, center = METRIC_MODE.get(m, ("high", None))

    if mode == "center":
        # 以 center 为中心，对称取“偏离程度”的分位数
        dev = np.abs(x - float(center))
        mdev = np.nanquantile(dev, Q_HIGH)
        if not np.isfinite(mdev) or mdev == 0:
            mdev = np.nanmax(dev)
            if not np.isfinite(mdev) or mdev == 0:
                mdev = 1.0
        vmin = float(center) - float(mdev)
        vmax = float(center) + float(mdev)

        # fbias 这种有天然下界 0，可做个保护（可选）
        if m == "fbias":
            vmin = max(0.0, vmin)

        # 确保 vmin<vmax
        if vmin == vmax:
            vmax = vmin + 1e-6
        return (vmin, vmax)

    else:
        vmin = np.nanquantile(x, Q_LOW)
        vmax = np.nanquantile(x, Q_HIGH)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(x))
            vmax = float(np.nanmax(x))
            if vmin == vmax:
                vmax = vmin + 1e-6

        # 对误差类通常不希望 vmin<0
        if mode == "low" and vmin < 0:
            vmin = 0.0
        return (float(vmin), float(vmax))

def get_norm_cmap(metric, vmin, vmax):
    """根据 mode 返回 norm/cmap（不改变数值，仅决定颜色映射方向/发散）"""
    m = metric.lower().strip()
    mode, center = METRIC_MODE.get(m, ("high", None))

    if mode == "high":
        cmap = "YlGn"          # 越大越深
        norm = Normalize(vmin=vmin, vmax=vmax)

    elif mode == "low":
        cmap = "YlGn_r"        # 越小越深
        norm = Normalize(vmin=vmin, vmax=vmax)

    else:  # center：用发散色带，确保 <center 与 >center 颜色明显不同
        c = float(center)

        # 关键：建议对称范围，保证 ± 同幅度颜色强度一致
        max_abs = max(abs(vmin - c), abs(vmax - c))
        vmin2, vmax2 = c - max_abs, c + max_abs

        cmap = "RdBu_r"        # 也可用 "coolwarm", "PuOr", "BrBG" 等
        norm = TwoSlopeNorm(vmin=vmin2, vcenter=c, vmax=vmax2)

    return norm, cmap

def draw_base(ax, gdf_outline, gdf_climate, gdf_nine, XLIM, YLIM):
    gdf_outline.plot(ax=ax, facecolor="white", edgecolor="black",
                     linewidth=CHINA_OUTLINE_LW, zorder=1)
    if DRAW_CLIMATE_BOUNDARY:
        gdf_climate.boundary.plot(ax=ax, color="black", linewidth=CLIMATE_BOUNDARY_LW,
                                  alpha=0.6, zorder=1)
    if gdf_nine is not None:
        gdf_nine.plot(ax=ax, color="black", linewidth=0.6, zorder=2)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    if AXIS_OFF:
        ax.set_axis_off()

def add_inset_with_points(ax, gdf_outline, gdf_nine, lons, lats, vals, norm, cmap):
    """每个子图 inset：画九段线 + 该范围内站点散点"""
    if gdf_nine is None:
        return

    x0, y0, w, h = INSET_RECT
    axins = ax.inset_axes([x0, y0, w, h])  # 轴坐标矩形，位置稳定
    axins.set_facecolor("white")

    # 底图
    gdf_outline.plot(ax=axins, facecolor="white", edgecolor="black", linewidth=0.55, zorder=1)
    gdf_nine.plot(ax=axins, color="black", linewidth=0.75, zorder=2)

    # inset 内站点
    in_mask = (np.isfinite(lons) & np.isfinite(lats) & np.isfinite(vals) &
               (lons >= SCS_XLIM[0]) & (lons <= SCS_XLIM[1]) &
               (lats >= SCS_YLIM[0]) & (lats <= SCS_YLIM[1]))

    if np.any(in_mask):
        axins.scatter(
            lons[in_mask], lats[in_mask],
            c=vals[in_mask],
            s=INSET_POINT_SIZE,
            cmap=cmap, norm=norm,
            edgecolors="0.35", linewidths=INSET_POINT_EDGE,
            alpha=INSET_POINT_ALPHA,
            zorder=3
        )

    axins.set_xlim(*SCS_XLIM)
    axins.set_ylim(*SCS_YLIM)
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_aspect("equal", adjustable="box")

# ===================== 读取 shp =====================
gdf_climate = safe_to_crs(gpd.read_file(shapefile_path), 4326)
gdf_outline = gdf_climate.dissolve()

gdf_nine = None
if os.path.exists(nine_dash_path):
    gdf_nine = safe_to_crs(gpd.read_file(nine_dash_path), 4326)
else:
    print(f"[Warning] 九段线 shp 不存在：{nine_dash_path}（将不绘制）")

# 中国范围
xmin, ymin, xmax, ymax = gdf_outline.total_bounds
pad_x = (xmax - xmin) * 0.03
pad_y = (ymax - ymin) * 0.03
XLIM = (xmin - pad_x, xmax + pad_x)
YLIM = (ymin - pad_y, ymax + pad_y)

# ===================== 读取产品 CSV =====================
product_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

product_data = {}
for product in product_folders:
    csv_path = os.path.join(base_dir, product, f"PointPixel_metrics_{product}.csv")
    if not os.path.exists(csv_path) or csv_path == os.path.join(base_dir, product, f"PointPixel_metrics_CMFDV2.csv"):
        continue
    df = pd.read_csv(csv_path)

    col_station = find_col(df, ["station", "id", "site", "stid"])
    col_lat = find_col(df, ["lat", "latitude"])
    col_lon = find_col(df, ["lon", "longitude", "long"])

    if col_lat is None or col_lon is None:
        print(f"[Skip] {product}: 找不到 lat/lon 列")
        continue

    if col_station is not None:
        df = df.drop_duplicates(subset=[col_station], keep="first").copy()

    df = df.rename(columns={col_lat: "lat", col_lon: "lon"})
    if col_station is not None and col_station != "station":
        df = df.rename(columns={col_station: "station"})

    product_data[product] = df

if len(product_data) == 0:
    raise FileNotFoundError("未找到任何 PointPixel_metrics_{PRODUCT}.csv")

products = list(product_data.keys())
n_prod = len(products)
n_rows = math.ceil(n_prod / N_COLS)

def metrics_of(df):
    return [c for c in df.columns if c not in {"station", "lat", "lon"}]

metrics_sets = [set(metrics_of(df)) for df in product_data.values()]
metrics = sorted(list(set.intersection(*metrics_sets))) if metrics_sets else []
if len(metrics) == 0:
    metrics = sorted(list(set.union(*metrics_sets)))

# ===================== 主循环：逐指标出图 =====================
for metric in metrics:
    mkey = metric.lower().strip()
    if mkey not in METRIC_MODE:
        # 你也可以选择 continue；这里直接跳过未定义模式的指标
        print(f"[Skip metric] 未在 METRIC_MODE 中定义：{metric}")
        continue

    # 1) 全产品合并，用于确定该指标的色标范围（基于实际值 + 分位数）
    all_for_range = []
    for p in products:
        dfp = product_data[p]
        if metric not in dfp.columns:
            continue
        all_for_range.append(dfp[metric].to_numpy())
    all_for_range = np.concatenate(all_for_range) if len(all_for_range) else np.array([])

    vmin, vmax = compute_vmin_vmax(metric, all_for_range)
    norm, cmap = get_norm_cmap(metric, vmin, vmax)

    # 2) 画布与 GridSpec（最后一行专门放 colorbar）
    fig_w = N_COLS * 3.2
    fig_h = n_rows * 2.6 + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=n_rows + 1, ncols=N_COLS,
        height_ratios=[1.0] * n_rows + [0.10],
        hspace=0.10, wspace=0.05
    )

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    # 3) 子图
    for i, p in enumerate(products):
        r = i // N_COLS
        c = i % N_COLS
        ax = fig.add_subplot(gs[r, c])

        draw_base(ax, gdf_outline, gdf_climate, gdf_nine, XLIM, YLIM)

        dfp = product_data[p]
        if metric not in dfp.columns:
            ax.set_title(f"{panel_label(i)} {p} {metric.upper()}", fontsize=11, loc="left")
            continue

        lons = dfp["lon"].to_numpy()
        lats = dfp["lat"].to_numpy()

        # 关键：绘图值全部贴边（不做归一化；仅替换不合理值到边界）
        vals = clip_for_plot(dfp[metric].to_numpy(), vmin, vmax)

        mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(vals)

        ax.scatter(
            lons[mask], lats[mask],
            c=vals[mask],
            s=POINT_SIZE,
            cmap=cmap, norm=norm,
            edgecolors="0.35", linewidths=POINT_EDGE,
            alpha=POINT_ALPHA,
            zorder=3
        )

        # 每个子图都加 inset（并绘制 inset 内站点）
        add_inset_with_points(ax, gdf_outline, gdf_nine, lons, lats, vals, norm, cmap)

        ax.set_title(f"{panel_label(i)} {p} {metric.upper()}", fontsize=11, loc="left")

    # 多余子图关闭
    for j in range(n_prod, n_rows * N_COLS):
        rr = j // N_COLS
        cc = j % N_COLS
        ax_empty = fig.add_subplot(gs[rr, cc])
        ax_empty.set_axis_off()

    # 4) 底部 colorbar（用实际值范围 vmin/vmax）
    cax = fig.add_subplot(gs[-1, :])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(metric.upper())

    out_png = os.path.join(out_dir, f"{metric.upper()}_ALL_PRODUCTS.png")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_png}")

print("全部指标绘制完成。")
