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

# ===================== 字体（按你给的 set_font） =====================
def set_font():
    """全局字体设置：Times New Roman 为主；中文备用 SimHei"""
    font_path_tnr = "/home/ud202380664/Times_New_Roman.ttf"
    font_path_simhei = "/home/ud202380664/Ubuntu_18.04_SimHei.ttf"

    font_manager.fontManager.addfont(font_path_tnr)
    font_manager.fontManager.addfont(font_path_simhei)

    SIMHEI_FP = font_manager.FontProperties(fname=font_path_simhei)

    rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],
        "font.size": 16,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.dpi": 600
    })
    return SIMHEI_FP

SIMHEI_FP = set_font()

# ===================== 路径参数 =====================
shapefile_path = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp"
base_dir = "/home/ud202380664/CHINA/ObeservationData/PointPixel/"
out_dir = os.path.join(base_dir, "_ALL_PRODUCTS_ZONE_BOXPLOTS")
os.makedirs(out_dir, exist_ok=True)

# 产品子图排版
N_COLS = 4

# 你要的四个指标（注意：列名要与 CSV 一致，通常是小写）
TARGET_METRICS = ["pod", "far", "rmse", "kge"]

# 明显离谱值阈值（用于剔除/防止污染）
HARD_ABS_MAX = 1e20

# RMSE 上限用于“贴边”（可选）：None 表示用全体 98% 分位数自动设定
RMSE_CAP = None  # 例如你想固定：RMSE_CAP = 50.0



# 中文 climate -> 英文缩写
climate_abbr_map = {
    "暖温带半湿润地区": "WT-SH",
    "中温带干旱地区": "MT-A",
    "北亚热带湿润地区": "NST-H",
    "中温带半湿润地区": "MT-SH",
    "中温带半干旱地区": "MT-SA",
    "高原温带半干旱地区": "PT-SA",
    "边缘热带湿润地区": "MTr-H"
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

def clean_metric_values(metric, arr, rmse_cap=None):
    """
    清洗/截断：
      - POD/FAR: clip 到 [0,1]
      - KGE: clip 到 [-1,1]
      - RMSE: 负值设 NaN；>rmse_cap 的贴边（如给定）；再剔除极端离谱值
    """
    x = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan

    # 剔除极端离谱值（例如 1e35）
    x[np.abs(x) >= HARD_ABS_MAX] = np.nan

    m = metric.lower().strip()
    if m in ["pod", "far"]:
        x = np.clip(x, 0.0, 1.0)
    elif m == "kge":
        x = np.clip(x, -1.0, 1.0)
    elif m == "rmse":
        x[x < 0] = np.nan
        if rmse_cap is not None and np.isfinite(rmse_cap):
            x = np.clip(x, 0.0, float(rmse_cap))
    return x

def zone_stats(df_long):
    """
    输入 long 表：columns = [product, climate, metric, value]
    输出每个 product × climate × metric 的统计
    """
    g = df_long.groupby(["product", "climate", "metric"])["value"]
    out = g.agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        q25=lambda s: np.nanquantile(s, 0.25),
        q75=lambda s: np.nanquantile(s, 0.75),
    ).reset_index()
    return out

# ===================== 1) 读取分区 shp =====================
gdf_climate = safe_to_crs(gpd.read_file(shapefile_path), 4326)
if "climate" not in gdf_climate.columns:
    raise ValueError("你的 shp 中没有 climate 字段，请检查字段名。")

# 分区顺序（用 shp 中出现的顺序；你也可以改成自定义顺序列表）
# climate_order = list(pd.unique(gdf_climate["climate"]))
climate_cn_order = list(pd.unique(gdf_climate["climate"]))
climate_order = [climate_abbr_map.get(z, None) for z in climate_cn_order]
climate_order = [z for z in climate_order if z is not None]

# dissolve 用于绘图边界（可选），这里主要用于 join 不需要 dissolve
gdf_climate_for_join = gdf_climate[["climate", "geometry"]].copy()

# ===================== 2) 读取所有产品 CSV，并做空间匹配 =====================
product_folders = sorted([
    f for f in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, f))
])

product_data = {}  # product -> DataFrame(含 climate)
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

    # 去重：每站只取一次（与你之前一致）
    if col_station is not None:
        df = df.drop_duplicates(subset=[col_station], keep="first").copy()

    df = df.rename(columns={col_lat: "lat", col_lon: "lon"})
    if col_station is not None and col_station != "station":
        df = df.rename(columns={col_station: "station"})

    # 必要指标列检查
    cols_lower = {c.lower(): c for c in df.columns}
    missing = [m for m in TARGET_METRICS if m not in cols_lower]
    if missing:
        print(f"[Skip] {product}: 缺少指标列 {missing}")
        continue

    # points -> spatial join 到 climate
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    # spatial join: within（若有站点落在边界线，可改 predicate="intersects"）
    joined = gpd.sjoin(
        gdf_points,
        gdf_climate_for_join,
        how="left",
        predicate="within"
    ).drop(columns=["index_right"], errors="ignore")

    # 没匹配到分区的站点丢弃
    joined = pd.DataFrame(joined)
    joined = joined.dropna(subset=["climate"]).copy()
    
    # --- 新增：中文 climate -> 英文缩写 ---
    joined["climate"] = joined["climate"].map(climate_abbr_map)

    # 未映射到的（例如 shp 里出现了字典外的中文分区）直接丢弃，避免后续出现 NaN 分区
    joined = joined.dropna(subset=["climate"]).copy()

    product_data[product] = joined

if len(product_data) == 0:
    raise FileNotFoundError("未找到任何可用的 PointPixel_metrics_{PRODUCT}.csv 或指标列不齐全。")

products = list(product_data.keys())
n_prod = len(products)
n_rows = math.ceil(n_prod / N_COLS)

print(f"Products found: {n_prod}")
print("Climate zones:", climate_order)

# ===================== 3) 组织 long 表 + 统计输出 =====================
# 先为 RMSE 自动估计一个 cap（如果你没手动给）
if RMSE_CAP is None:
    all_rmse = []
    for p in products:
        col_rmse = {c.lower(): c for c in product_data[p].columns}.get("rmse")
        x = clean_metric_values("rmse", product_data[p][col_rmse].to_numpy(), rmse_cap=None)
        x = x[np.isfinite(x)]
        if x.size:
            all_rmse.append(x)
    if all_rmse:
        all_rmse = np.concatenate(all_rmse)
        RMSE_CAP = float(np.nanquantile(all_rmse, 0.98))
        if not np.isfinite(RMSE_CAP) or RMSE_CAP <= 0:
            RMSE_CAP = None
    print(f"[RMSE_CAP] auto = {RMSE_CAP}")

records = []
for p in products:
    dfp = product_data[p]

    # 统一小写映射到原列名
    cols_map = {c.lower(): c for c in dfp.columns}

    for metric in TARGET_METRICS:
        col = cols_map[metric]
        if metric == "rmse":
            vals = clean_metric_values(metric, dfp[col].to_numpy(), rmse_cap=RMSE_CAP)
        else:
            vals = clean_metric_values(metric, dfp[col].to_numpy(), rmse_cap=None)

        tmp = pd.DataFrame({
            "product": p,
            "climate": dfp["climate"].values,
            "metric": metric,
            "value": vals
        })
        tmp = tmp.dropna(subset=["value"])
        records.append(tmp)

df_long = pd.concat(records, ignore_index=True)

# 统计表输出
df_stats = zone_stats(df_long)
stats_csv = os.path.join(out_dir, "ZoneStats_POD_FAR_RMSE_KGE_byProduct.csv")
df_stats.to_csv(stats_csv, index=False, encoding="utf-8-sig")
print(f"[Saved] {stats_csv}")

# ===================== 4) 逐指标绘制：每张图多个产品子图，每子图=7分区箱线图 =====================
def ylims_for_metric(metric, df_long, rmse_cap):
    m = metric.lower()
    if m in ["pod", "far"]:
        return (0.0, 1.0)
    if m == "kge":
        return (-1.0, 1.0)
    if m == "rmse":
        # 用 cap 或 98% 分位数
        if rmse_cap is not None and np.isfinite(rmse_cap):
            return (0.0, rmse_cap)
        vals = df_long.loc[df_long["metric"] == "rmse", "value"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            return (0.0, float(np.nanquantile(vals, 0.98)))
        return (0.0, 1.0)
    return (None, None)

for metric in TARGET_METRICS:
    # 当前指标数据
    dfi = df_long[df_long["metric"] == metric].copy()

    # y 轴范围统一（全产品一致）
    # y0, y1 = ylims_for_metric(metric, df_long, RMSE_CAP)

    # 画布
    fig_w = N_COLS * 4.2
    fig_h = n_rows * 3.0 + 0.6
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(n_rows, N_COLS)

    for i, p in enumerate(products):
        r = i // N_COLS
        c = i % N_COLS
        ax = axes[r, c]

        dfp = dfi[dfi["product"] == p]

        # 准备每个分区的数据列表（按 climate_order 固定顺序）
        data = []
        for z in climate_order:
            v = dfp.loc[dfp["climate"] == z, "value"].to_numpy()
            data.append(v)

        # 箱线图
        ax.boxplot(
            data,
            patch_artist=False,
            showfliers=False,          # 不显示离群点，图更干净（需要可改 True）
            widths=0.55
        )

        ax.set_title(f"{panel_label(i)} {p} {metric.upper()}",
                     loc="left", fontsize=16)

        ax.set_xticks(range(1, len(climate_order) + 1))
        ax.set_xticklabels(climate_order, rotation=25, ha="right")

        # if y0 is not None and y1 is not None:
            # ax.set_ylim(y0, y1)

        ax.grid(True, axis="y", alpha=0.25)

    # 关掉多余子图
    for j in range(n_prod, n_rows * N_COLS):
        rr = j // N_COLS
        cc = j % N_COLS
        axes[rr, cc].axis("off")

    # 总标签（可选）
    fig.suptitle(f"{metric.upper()} by Climate Zones", y=0.995, fontsize=24)

    out_png = os.path.join(out_dir, f"{metric.upper()}_Boxplots_7Zones_AllProducts.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_png}")

print("全部完成。")
