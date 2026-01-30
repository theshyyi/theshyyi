#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) 参数
# =========================
BASE_DIR = "/home/ud202380664/CHINA/ObeservationData/PointPixel/"
OUT_PNG  = "/home/ud202380664/CHINA/ObeservationData/PointPixel_metrics_boxplot_3x3_pretty.png"
DPI = 600

# 你的 9 个指标（固定顺序，3x3）
METRICS = ["bias", "mae", "rmse", "corr", "kge", "pod", "far", "csi", "hss"]

# 不要的列
DROP_COLS = {"station", "lat", "lon", "n", "fbias"}

# 每个指标最佳值虚线
BEST_VALUE = {
    "bias": 0.0,
    "mae": 0.0,
    "rmse": 0.0,
    "corr": 1.0,
    "kge": 1.0,
    "pod": 1.0,
    "far": 0.0,
    "csi": 1.0,
    "hss": 1.0,
}

# 0-1 有界指标：强制 xlim=[0,1]
BOUNDED_01 = {"corr", "pod", "far", "csi", "hss"}

CSV_PATTERN = "PointPixel_metrics_*.csv"
SKIP_NAME = "PointPixel_metrics_CMFDV2.csv"

def find_metrics_csvs(base_dir: str):
    pairs = []
    for prod_dir in sorted(glob.glob(os.path.join(base_dir, "*"))):
        if not os.path.isdir(prod_dir):
            continue

        product = os.path.basename(prod_dir)

        # 找到所有匹配的csv，并跳过 CMFDV2 那个文件
        cands = sorted(
            p for p in glob.glob(os.path.join(prod_dir, CSV_PATTERN))
            if os.path.basename(p) != SKIP_NAME
        )

        if cands:
            pairs.append((product, cands[0]))

    return pairs


# =========================
# 2) 数据清洗与轴范围
# =========================
def clean_values(arr: np.ndarray):
    """剔除 NaN/inf，并强力剔除疑似填充值/溢出值（1e35/1e36 这类）"""
    x = np.asarray(arr, dtype="float64").ravel()
    x = x[np.isfinite(x)]
    # 关键：剔除极端异常值（通常是 _FillValue 或溢出）
    x = x[np.abs(x) < 1e20]
    return x

def robust_xlim(metric: str, data_list: list):
    """
    为每个指标给一个稳健的 x 轴范围，避免少量异常点拉爆。
    - 0-1 指标：固定 [0,1]
    - bias/kge 等：用 2%~98% 分位，并留一点边距
    - mae/rmse：>=0，用 2%~98% 分位并从 0 起
    """
    if metric in BOUNDED_01:
        return (0.0, 1.0)

    allv = np.concatenate([d for d in data_list if d.size > 0], axis=0)
    if allv.size == 0:
        return None

    lo = np.nanpercentile(allv, 2)
    hi = np.nanpercentile(allv, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(allv)
        hi = np.nanmax(allv)

    # 加边距
    span = hi - lo
    pad = 0.08 * span if span > 0 else 1.0

    if metric in {"mae", "rmse"}:
        lo2 = max(0.0, lo - pad)
        hi2 = hi + pad
        return (lo2, hi2)

    if metric == "bias":
        # bias 通常希望 0 在中间更直观：用对称范围（基于分位）
        m = max(abs(lo), abs(hi))
        m = m + 0.08 * m if m > 0 else 1.0
        return (-m, m)

    # kge 等无严格界：直接用分位范围
    return (lo - pad, hi + pad)


# =========================
# 3) 主绘图
# =========================
def plot_pretty_3x3(base_dir: str, out_png: str, dpi: int = 600):
    pairs = find_metrics_csvs(base_dir)
    if not pairs:
        raise FileNotFoundError(f"在 {base_dir} 下未找到任何 {CSV_PATTERN}")

    # 读入所有产品
    data_by_prod = {}
    for product, csv_path in pairs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            warnings.warn(f"读取失败，跳过: {csv_path} -> {e}")
            continue

        df.columns = [str(c).strip().lower() for c in df.columns]
        # 去掉不需要列
        keep_cols = [c for c in df.columns if c not in DROP_COLS]
        df = df[keep_cols]
        data_by_prod[product] = df

    if not data_by_prod:
        raise RuntimeError("所有 CSV 都读取失败，无法绘图。")

    # 校验指标列
    first_prod = next(iter(data_by_prod))
    missing = [m for m in METRICS if m not in data_by_prod[first_prod].columns]
    if missing:
        raise ValueError(f"CSV 缺少指标列: {missing}。请确认列名（脚本已转为小写）。")

    products = sorted(data_by_prod.keys())
    
    n_prod = len(products)

    # 每个产品大概需要的纵向空间（英寸/产品），你可以在 0.18~0.30 之间调
    inch_per_product = 0.26   # 0.24~0.30 都可，产品越多取越大
    fig_w = 28

    row_h = 1.4 + n_prod * inch_per_product     # 单行子图高度（英寸）
    fig_h = 3 * row_h + 2.2                      # 3 行 + 上下边距


    # 颜色：每个产品固定颜色
    cmap = plt.cm.get_cmap("tab20", len(products))
    color_map = {p: cmap(i) for i, p in enumerate(products)}

    # 全局风格（尽量贴近你参考图）
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 0.9,
    })

    # fig, axes = plt.subplots(3, 3, figsize=(14, 8), dpi=dpi)
    fig, axes = plt.subplots(3, 3, figsize=(fig_w, fig_h), dpi=dpi)

    axes = axes.flatten()

    for i, metric in enumerate(METRICS):
        ax = axes[i]

        # 组装每个产品的清洗后数据
        series_list = []
        labels = []
        colors = []

        for p in products:
            vals = pd.to_numeric(data_by_prod[p][metric], errors="coerce").to_numpy()
            vals = clean_values(vals)
            if vals.size == 0:
                continue
            series_list.append(vals)
            labels.append(p)
            colors.append(color_map[p])

        if not series_list:
            ax.set_title(metric)
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        # 箱线图（水平），不画离群点（避免黑点糊一片）
        bp = ax.boxplot(
            series_list,
            vert=False,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),          # 稳健须：5–95%
            widths=0.62,
            medianprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=0.9),
            capprops=dict(linewidth=0.9),
        )

        # 给每个箱子上色（半透明）
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_linewidth(1.0)

        # 均值点：黑色实心点（和参考图“点”类似）
        means = [np.mean(v) for v in series_list]
        y = np.arange(1, len(series_list) + 1)
        ax.scatter(means, y, s=14, c="k", zorder=3)

        # 最佳值虚线（紫色）
        ref = BEST_VALUE.get(metric, None)
        if ref is not None:
            ax.axvline(ref, linestyle="--", linewidth=1.4, color="#7a3df0")

        # 网格与标题
        ax.set_title(metric)
        ax.grid(True, axis="both", linestyle="-", alpha=0.18)

        # x 轴范围：稳健处理，避免极端值拉爆
        xl = robust_xlim(metric, series_list)
        if xl is not None:
            ax.set_xlim(xl)

        # 产品从上到下（与你参考图一致）
        ax.invert_yaxis()

        # 子图边框更干净
        for spine in ax.spines.values():
            spine.set_alpha(0.9)

    # 防止标题/标签挤在一起：手动留边距（比 constrained_layout 更可控）
    # fig.subplots_adjust(left=0.12, right=0.985, top=0.95, bottom=0.06, wspace=0.16, hspace=0.22)
    
    fig.subplots_adjust(
        left=0.08,   # 左边距加大：容纳大量产品名
        right=0.99,
        top=0.96,
        bottom=0.05,
        wspace=0.35, # 子图列间距加大
        hspace=0.28  # 子图行间距加大
    )


    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


if __name__ == "__main__":
    plot_pretty_3x3(BASE_DIR, OUT_PNG, dpi=DPI)
