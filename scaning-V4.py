#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# ================== 全局字体设置 ==================
rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})

# ================== 基本配置 ==================
DATA_DIR  = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/"
REF_FILE  = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/CMFDV2.TIMEFIX.daily.CHINA.nc"
VAR_NAME  = "pr"

# True: 先对 lat/lon 平均，得到 1D(time)；False: 保留网格（time,lat,lon）或（lat,lon）
SPATIAL_MEAN = False

TIME_START = None  # "2000-01-01"
TIME_END   = None  # "2022-12-31"

OUT_DIR = "./scatter_daily2"

SKIP_IF_PNG_EXISTS = True
MIN_PNG_SIZE = 10 * 1024  # 10KB
os.makedirs(OUT_DIR, exist_ok=True)

# ================== 关键：统计/绘图口径 ==================
# MODE="climatology": 先对 time 求平均 -> (lat,lon) 或标量（若 SPATIAL_MEAN=True）
# MODE="daily":       保留 time 维度 -> (time,lat,lon) 或 (time)（若 SPATIAL_MEAN=True）
MODE = "climatology"   # "daily" 或 "climatology"

# 可选：湿日阈值（mm/day）。例如 0.1；None 表示不过滤
WET_THRESHOLD = None

# 若点数太大，抽样用于绘图与统计（保证“统计量=图上点的统计量”）
MAX_PLOT_POINTS = 250_000
RANDOM_SEED = 42

# 拟合线：是否强制过原点（你原来是 y ≈ kx）
FORCE_ZERO_INTERCEPT = True

# 坐标上限：None 表示自动；例如 20 就固定到 20
AXIS_UPPER = 40  # 你截图里就是 0-20；若想自动可设 None


MAX_MAJOR_TICKS = 6   # 主刻度最多显示多少个（含 0 和上限）


# ================== 工具函数 ==================
def _standardize_latlon(da: xr.DataArray) -> xr.DataArray:
    """统一维度命名为 lat/lon，并保证 lat 从小到大（可对齐）"""
    rename = {}
    if "latitude" in da.dims and "lat" not in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims and "lon" not in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    # 有些数据 lat 可能是降序；对齐前统一成升序更稳
    if "lat" in da.coords:
        try:
            if da["lat"].values[0] > da["lat"].values[-1]:
                da = da.sortby("lat")
        except Exception:
            pass
    return da

def nice_tick_step(vmin, vmax, max_ticks=6):
    """
    根据 [vmin, vmax] 自动给出“漂亮”的主刻度间隔，使主刻度数量 <= max_ticks
    返回值为 1,2,5 * 10^k 系列
    """
    span = float(vmax - vmin)
    if span <= 0:
        return 1.0

    # 目标：max_ticks 个刻度 -> 大约 (max_ticks-1) 个间隔
    raw = span / max(1, (max_ticks - 1))

    # 取 1/2/5 * 10^k 里最接近且不小于 raw 的
    exp = np.floor(np.log10(raw))
    base = 10 ** exp
    candidates = np.array([1, 2, 5, 10], dtype=float) * base
    step = candidates[candidates >= raw].min() if np.any(candidates >= raw) else 10 * base
    return float(step)


def load_da(nc_path, var_name, time_start=None, time_end=None, spatial_mean=False, mode="climatology"):
    ds = xr.open_dataset(nc_path)

    if var_name not in ds:
        raise KeyError(f"{nc_path} 中找不到变量 {var_name}")

    da = ds[var_name]
    da = _standardize_latlon(da)

    if "time" not in da.dims:
        raise ValueError(f"{nc_path} 中变量 {var_name} 不含 time 维度")

    if time_start is not None or time_end is not None:
        da = da.sel(time=slice(time_start, time_end))

    # 空间平均（若需要）
    if spatial_mean:
        if "lat" not in da.dims or "lon" not in da.dims:
            raise ValueError(f"{nc_path} 变量 {var_name} 缺少 lat/lon 维度，无法空间平均")
        da = da.mean(dim=("lat", "lon"), skipna=True)

    # mode 控制是否 time 平均
    if mode.lower() == "climatology":
        da = da.mean(dim="time", skipna=True)

    return da


def _prepare_xy_for_plot(ref_da: xr.DataArray, prod_da: xr.DataArray):
    """对齐后展平为 x,y，并完成 NaN/inf 过滤、湿日过滤、抽样（统计量=绘图点）"""
    x = np.asarray(ref_da.values).ravel()
    y = np.asarray(prod_da.values).ravel()

    # 1) NaN/inf 过滤
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    # 2) 可选湿日阈值
    if WET_THRESHOLD is not None:
        wet = (x >= WET_THRESHOLD) | (y >= WET_THRESHOLD)
        x = x[wet]
        y = y[wet]

    if x.size == 0:
        return x, y

    # 3) 抽样（保证统计量与图上点一致）
    if x.size > MAX_PLOT_POINTS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(x.size, size=MAX_PLOT_POINTS, replace=False)
        x = x[idx]
        y = y[idx]

    return x, y


def scatter_density_plot(x, y, title, out_png, x_label, y_label, mode_label=""):
    """密度散点 + 拟合线 + 1:1 线；统计量严格基于 x,y（即图上点）"""
    if x.size == 0:
        print(f"Warning: 有效数据为空，跳过：{title}")
        return

    # 轴范围
    min_val = min(np.min(x), np.min(y))
    min_val = min(0.0, float(min_val))

    if AXIS_UPPER is None:
        max_val = max(np.max(x), np.max(y))
        upper = float(np.ceil(max_val / 20.0) * 20.0) if max_val > 0 else 10.0
    else:
        upper = float(AXIS_UPPER)

    # 若你固定 upper=20，但数据里有>20 的点，不想“统计包含但图看不到”，就裁掉（统计量=图上点）
    # 这一步会让“图上点”和“统计量”彻底一致
    in_range = (x >= min_val) & (x <= upper) & (y >= min_val) & (y <= upper)
    x = x[in_range]
    y = y[in_range]

    if x.size == 0:
        print(f"Warning: 裁剪到坐标范围后无点，跳过：{title}")
        return

    # 点密度（KDE）
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # 统计量（产品-参考）
    bias = float(np.mean(y - x))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    r    = float(np.corrcoef(x, y)[0, 1])

    # 拟合线
    if FORCE_ZERO_INTERCEPT:
        denom = np.sum(x ** 2)
        k = float(np.sum(x * y) / denom) if denom != 0 else 0.0
        b = 0.0
        fit_label = f"Fit: y = {k:.3f}x"
    else:
        k, b = np.polyfit(x, y, 1)
        k = float(k); b = float(b)
        fit_label = f"Fit: y = {k:.3f}x + {b:.3f}"

    x_line = np.linspace(min_val, upper, 200)
    y_line = k * x_line + b

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    sc = ax.scatter(
        x, y,
        c=z * 100,
        s=5,
        marker="o",
        edgecolors="none",
        cmap="gist_rainbow",
        label=mode_label if mode_label else "Data"
    )

    cbar = plt.colorbar(sc, shrink=1, orientation="vertical", extend="both", pad=0.015, aspect=30)
    cbar.set_label("Point density")

    ax.plot(x_line, y_line, c="r", lw=1.5, label=fit_label)
    ax.plot([min_val, upper], [min_val, upper], "k--", lw=1.5, label="1:1 line")

    for spine in ["bottom", "top", "left", "right"]:
        ax.spines[spine].set_linewidth(2.5)

    ax.tick_params(which="major", width=2.5, length=5)
    # ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.yaxis.set_major_locator(MultipleLocator(5))
    step = nice_tick_step(min_val, upper, max_ticks=MAX_MAJOR_TICKS)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))


    # 文字
    text_x = min_val + 0.05 * (upper - min_val)
    text_y = min_val + 0.95 * (upper - min_val)

    ax.text(text_x, text_y,            f"$N={x.size:.0f}$")
    ax.text(text_x + 0.30*(upper-min_val), text_y, f"$R={r:.2f}$")
    ax.text(text_x, text_y - 0.05*(upper-min_val), f"$BIAS={bias:.2f}$")
    ax.text(text_x, text_y - 0.10*(upper-min_val), f"$RMSE={rmse:.2f}$")

    # 标题/坐标轴（保持你原来中文字体写法）
    font_zh = {"family": "SimHei", "size": 24, "color": "k"}
    plt.title(title, fontdict=font_zh)
    plt.xlabel(x_label, fontdict=font_zh)
    plt.ylabel(y_label, fontdict=font_zh)

    ax.set_xlim(min_val, upper)
    ax.set_ylim(min_val, upper)
    # ax.legend(frameon=False)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.52, 0.50),   # x越大越往右，y越小越往下
        borderaxespad=0.0
    )


    plt.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_png}")


# ================== 主程序 ==================
def main():
    print(f"读取参考数据: {REF_FILE}")
    ref_da = load_da(
        REF_FILE, VAR_NAME,
        time_start=TIME_START, time_end=TIME_END,
        spatial_mean=SPATIAL_MEAN, mode=MODE
    )

    nc_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.TIMEFIX.daily.CHINA.nc")))
    ref_abs = os.path.abspath(REF_FILE)

    for nc_path in nc_files:
        if os.path.abspath(nc_path) == ref_abs:
            continue

        fname = os.path.basename(nc_path)
        prod_name = fname.split(".")[0]
        out_png = os.path.join(OUT_DIR, f"scatter_{MODE}_{prod_name}.png")

        if SKIP_IF_PNG_EXISTS and os.path.exists(out_png) and os.path.getsize(out_png) >= MIN_PNG_SIZE:
            print(f"  -> 已存在输出图，跳过: {out_png}")
            continue

        print(f"处理产品: {prod_name} ({nc_path})")

        try:
            prod_da = load_da(
                nc_path, VAR_NAME,
                time_start=TIME_START, time_end=TIME_END,
                spatial_mean=SPATIAL_MEAN, mode=MODE
            )
        except Exception as e:
            print(f"  读取产品 {prod_name} 出错，跳过：{e}")
            continue

        # 对齐（不仅 time，对 lat/lon 也对齐；join=inner 保证是配对点）
        ref_aligned, prod_aligned = xr.align(ref_da, prod_da, join="inner")

        if ref_aligned.size == 0:
            print(f"  {prod_name} 与参考数据交集为空，跳过。")
            continue

        # 展平 + 过滤 + 抽样（统计量=图上点）
        x, y = _prepare_xy_for_plot(ref_aligned, prod_aligned)

        # 图例标签自动匹配 MODE
        if MODE.lower() == "daily":
            mode_label = "Daily data"
        else:
            mode_label = "Climatology"

        title   = f"{prod_name} vs CMFD-V2"
        x_label = "CMFD-V2 (mm)"
        y_label = f"{prod_name} (mm)"

        scatter_density_plot(x, y, title, out_png, x_label, y_label, mode_label=mode_label)


if __name__ == "__main__":
    main()
