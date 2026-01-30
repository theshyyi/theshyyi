#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAL diagram for daily precipitation products over China
=======================================================

- 读取 /home/ud202380664/PRE_MERGE/TIMEFIX 下的 *.TIMEFIX.daily.CHINA.nc
- 使用指定的参考产品 (OBS_PRODUCT) 作为观测
- 从观测中选出域平均降水最大的 TOP_N_EVENTS 天
- 对每个产品与观测计算 SAL (Structure, Amplitude, Location)
- 绘制 3×3 SAL 面板图，点颜色为 L
"""

import os
import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

import geopandas as gpd
import regionmask
from scipy import ndimage  # conda install scipy

# ================== 路径 & 参数 ==================

# 降水 nc 文件目录
NC_DIR = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish"

# 7 大气候分区 shp，包含 climate 字段（用于生成 China mask）
SHP_PATH = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp"

# 变量名，如不确定设为 None，会自动用第一个 data_var
VAR_NAME = None

# 参考产品（文件前缀），例如 CMFDV2.TIMEFIX.daily.CHINA.nc
OBS_PRODUCT = "CMFDV2"

# 选取的事件个数（如：域平均降水最大的 30 天）
TOP_N_EVENTS = 60

# SAL 中对象阈值：取 max 的几倍 (0.1 表示 10% * max)
THRESHOLD_FACTOR = 0.1

# 点大小
POINT_SIZE = 25

# 颜色条 L 范围（可按需要调整）
L_VMIN = 0.0
L_VMAX = 0.5

# ================== 字体设置（Times New Roman） ==================

font_path_tnr = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
if os.path.exists(font_path_tnr):
    font_manager.fontManager.addfont(font_path_tnr)

config = {
    "font.family": "Times New Roman",
    "font.size": 16,
    "mathtext.fontset": "stix",
}
rcParams.update(config)
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 600


# ================== 工具函数：生成 China 掩膜 ==================

def build_china_mask(example_nc_path, shp_path, climate_field="climate"):
    """
    用一个示例 nc 的 lon/lat 和 7 大分区 shp 生成 China 掩膜
    返回：
        china_mask : DataArray (lat, lon) 布尔型
    """
    ds = xr.open_dataset(example_nc_path)
    lat = ds["lat"]
    lon = ds["lon"]

    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError("Shapefile 没有 CRS，请先在 GIS 软件中指定。")
    gdf = gdf.to_crs(epsg=4326)

    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        names=list(gdf[climate_field].values),
        abbrevs=list(gdf[climate_field].values),
        name="China7Climate",
    )

    mask = regions.mask(lon, lat)  # (lat, lon)
    china_mask = mask.notnull()
    ds.close()
    return china_mask


# ================== SAL 公式实现 ==================

def amplitude_component(F, R):
    """Amplitude A 分量"""
    mean_F = F.mean()
    mean_R = R.mean()
    denom = 0.5 * (mean_F + mean_R)
    if denom == 0:
        return np.nan
    return (mean_F - mean_R) / denom


def structure_measure(X, thresh_factor=THRESHOLD_FACTOR):
    """
    计算单个场的结构特征 S_X
    X : 2D ndarray (>=0)
    """
    X_max = X.max()
    if X_max <= 0:
        return np.nan

    thresh = thresh_factor * X_max
    mask = X >= thresh
    if not mask.any():
        return np.nan

    labeled, num = ndimage.label(mask)
    if num == 0:
        return np.nan

    S_sum = 0.0
    O_sum = 0.0

    for label in range(1, num + 1):
        idx = labeled == label
        Xi = X[idx]
        O_i = Xi.sum()
        if O_i <= 0:
            continue
        V_i = O_i / (Xi.max() * idx.sum())
        S_sum += O_i * V_i
        O_sum += O_i

    if O_sum == 0:
        return np.nan
    return S_sum / O_sum


def structure_component(F, R):
    """Structure S 分量"""
    S_F = structure_measure(F)
    S_R = structure_measure(R)
    if np.isnan(S_F) or np.isnan(S_R):
        return np.nan
    denom = 0.5 * (S_F + S_R)
    if denom == 0:
        return np.nan
    return (S_F - S_R) / denom


def location_component(F, R):
    """
    Location L 分量 = L1 + L2
    使用格点索引作为坐标，整体缩放对 L 无影响
    """
    ny, nx = F.shape
    y_idx, x_idx = np.indices((ny, nx))  # y:0..ny-1, x:0..nx-1
    d = np.sqrt((nx - 1) ** 2 + (ny - 1) ** 2)  # 域对角线距离

    def center_of_mass(X):
        tot = X.sum()
        if tot <= 0:
            return np.array([np.nan, np.nan])
        cx = (X * x_idx).sum() / tot
        cy = (X * y_idx).sum() / tot
        return np.array([cx, cy])

    def object_centers(X, thresh_factor=THRESHOLD_FACTOR):
        X_max = X.max()
        if X_max <= 0:
            return None, None
        thresh = thresh_factor * X_max
        mask = X >= thresh
        if not mask.any():
            return None, None
        labeled, num = ndimage.label(mask)
        if num == 0:
            return None, None
        centers = []
        masses = []
        for label in range(1, num + 1):
            idx = labeled == label
            Xi = X[idx]
            O_i = Xi.sum()
            if O_i <= 0:
                continue
            cx = (Xi * x_idx[idx]).sum() / O_i
            cy = (Xi * y_idx[idx]).sum() / O_i
            centers.append(np.array([cx, cy]))
            masses.append(O_i)
        if not centers:
            return None, None
        return np.stack(centers, axis=0), np.array(masses)

    # ----- L1 -----
    cF = center_of_mass(F)
    cR = center_of_mass(R)
    if np.any(np.isnan(cF)) or np.any(np.isnan(cR)):
        return np.nan

    L1 = np.linalg.norm(cF - cR) / d

    # ----- L2 -----
    centers_F, masses_F = object_centers(F)
    centers_R, masses_R = object_centers(R)
    if centers_F is None or centers_R is None:
        return np.nan

    def r_X(centers, masses, cX):
        dist = np.linalg.norm(centers - cX, axis=1)
        return np.sum(masses * dist) / np.sum(masses)

    rF = r_X(centers_F, masses_F, cF)
    rR = r_X(centers_R, masses_R, cR)

    L2 = 2.0 * (rF - rR) / d

    return L1 + L2


def compute_sal(F_da, R_da, china_mask=None):
    """
    计算单个时次的 SAL 指标
    F_da, R_da : xarray.DataArray，包含空间维度
    china_mask : (lat, lon) bool，可选
    """

    # --- 1) 统一为 (lat, lon) 的维度顺序 ---
    def to_latlon(da: xr.DataArray) -> xr.DataArray:
        """
        确保 da 的空间维度顺序是 (lat, lon)。
        假定文件里存在 lat / lon 或 latitude / longitude。
        """
        # 先把 latitude/longitude 这种名字统一改成 lat/lon
        dim_rename = {}
        if "latitude" in da.dims and "lat" not in da.dims:
            dim_rename["latitude"] = "lat"
        if "longitude" in da.dims and "lon" not in da.dims:
            dim_rename["longitude"] = "lon"
        if dim_rename:
            da = da.rename(dim_rename)

        # 如果包含 lat/lon，就显式转置
        if ("lat" in da.dims) and ("lon" in da.dims):
            return da.transpose("lat", "lon")

        # 兜底：只取最后两个维度当空间维度（比如某些奇怪命名）
        spatial_dims = da.dims[-2:]
        return da.transpose(*spatial_dims)

    F_da = to_latlon(F_da)
    R_da = to_latlon(R_da)

    # --- 2) 应用 China 掩膜（已经是 lat/lon 顺序） ---
    if china_mask is not None:
        # 保证掩膜和数据经纬度对齐
        F_da = F_da.where(china_mask)
        R_da = R_da.where(china_mask)

    # --- 3) 转为 numpy 数组，并统一处理缺失值 ---
    F = F_da.values
    R = R_da.values

    # 有些产品可能仍然尺寸不一致（比如分辨率没完全统一），直接跳过
    if F.shape != R.shape:
        # 返回 NaN，后面绘图时会自动丢掉这个点
        return np.nan, np.nan, np.nan

    # 转为非负数组，NaN 和负值置为 0
    F = np.where(np.isfinite(F) & (F > 0), F, 0.0)
    R = np.where(np.isfinite(R) & (R > 0), R, 0.0)

    if F.sum() == 0 and R.sum() == 0:
        return np.nan, np.nan, np.nan

    # --- 4) 分别计算 A / S / L ---
    A = amplitude_component(F, R)
    S = structure_component(F, R)
    L = location_component(F, R)

    return S, A, L



# ================== 选择事件（top N 日平均最大） ==================

def select_top_events(obs_ds, var_name, china_mask=None, top_n=TOP_N_EVENTS):
    """
    从观测数据中选出域平均降水最大的 top_n 天
    返回对应的 time 索引（numpy.datetime64 数组）
    """
    da = obs_ds[var_name]
    if china_mask is not None:
        da = da.where(china_mask)

    # 域平均：lazy 计算即可
    daily_mean = da.mean(dim=("lat", "lon"), skipna=True)

    vals = daily_mean.values
    idx = np.argsort(vals)[-top_n:]  # 越后面越大
    idx = idx[::-1]  # 从大到小

    times = daily_mean.time.values[idx]
    return times


# ================== 主流程：计算并绘图 ==================

def main():
    # ----------- 找到所有产品文件 -----------
    nc_paths = sorted(glob.glob(os.path.join(NC_DIR, "*.TIMEFIX.daily.CHINA.nc")))
    if not nc_paths:
        raise FileNotFoundError(f"No *.TIMEFIX.daily.CHINA.nc found in {NC_DIR}")

    # 示例 nc，用于生成 China 掩膜
    china_mask = build_china_mask(nc_paths[0], SHP_PATH)

    # 观测产品路径
    obs_path = None
    for p in nc_paths:
        if os.path.basename(p).startswith(OBS_PRODUCT):
            obs_path = p
            break
    if obs_path is None:
        raise FileNotFoundError(f"Cannot find reference product {OBS_PRODUCT} in {NC_DIR}")

    print(f"Reference product (obs): {obs_path}")

    # 打开观测数据
    obs_ds = xr.open_dataset(obs_path, chunks={"time": 365})
    if VAR_NAME is None:
        vname = list(obs_ds.data_vars)[0]
        print(f"Auto-detected variable name: {vname}")
    else:
        vname = VAR_NAME

    # 选出 top N 事件（仅基于观测）
    top_times = select_top_events(obs_ds, vname, china_mask, top_n=TOP_N_EVENTS)
    print(f"Selected {len(top_times)} events (top daily mean precipitation).")

    # 产品列表（除观测外）
    products = []
    for p in nc_paths:
        prefix = os.path.basename(p).split(".")[0]
        if prefix == OBS_PRODUCT:
            continue
        products.append(prefix)

    print("Products to plot in SAL diagram (before time check):", products)

    # 预先打开所有产品数据集（包含观测 + 所有候选产品）
    ds_dict = {}
    for prefix in [OBS_PRODUCT] + products:
        path = None
        for p in nc_paths:
            if os.path.basename(p).startswith(prefix):
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"Cannot find file for product {prefix}")
        ds_dict[prefix] = xr.open_dataset(path, chunks={"time": 365})

    # ----------- 过滤掉在这些事件日期中有缺失的产品 -----------
    valid_products = []
    for prod in products:
        # 该产品的全部时间坐标
        tcoord = ds_dict[prod]["time"].values  # numpy.datetime64 数组
        # 检查每个 top_times 是否都在该产品的时间轴中
        mask = np.isin(top_times, tcoord)
        if np.all(mask):
            valid_products.append(prod)
        else:
            missing_times = top_times[~mask]
            if missing_times.size > 0:
                # 打印前几个缺失日期示例
                sample_str = ", ".join(
                    np.datetime_as_string(missing_times[:5], unit="D")
                )
            else:
                sample_str = "None"
            print(
                f"[SAL] Skip product {prod}: "
                f"{missing_times.size}/{len(top_times)} selected days missing. "
                f"Examples: {sample_str}"
            )

    products = valid_products
    print("Products actually used in SAL:", products)

    # 如果所有产品都被剔除了，直接结束
    if len(products) == 0:
        print("No product has complete data on selected days. Abort SAL plotting.")
        for ds in ds_dict.values():
            ds.close()
        return

    # ----------- 计算 SAL -----------
    SAL_results = {prod: {"S": [], "A": [], "L": []} for prod in products}

    for t in top_times:
        # 观测在这些时间肯定有数据（因为 top_times 就是从它选出来的）
        R_da = ds_dict[OBS_PRODUCT][vname].sel(time=t)
        for prod in products:
            F_da = ds_dict[prod][vname].sel(time=t)

            S, A, L = compute_sal(F_da, R_da, china_mask=china_mask)
            SAL_results[prod]["S"].append(S)
            SAL_results[prod]["A"].append(A)
            SAL_results[prod]["L"].append(L)

    # 关闭数据集
    for ds in ds_dict.values():
        ds.close()

    # 转为 numpy 数组
    for prod in products:
        SAL_results[prod]["S"] = np.array(SAL_results[prod]["S"])
        SAL_results[prod]["A"] = np.array(SAL_results[prod]["A"])
        SAL_results[prod]["L"] = np.array(SAL_results[prod]["L"])

    # ----------- 绘制 SAL 图 -----------
    n_prod = len(products)
    # 每行 4 个子图，行数自动
    ncols = 4
    nrows = int(np.ceil(n_prod / ncols))

    fig_width = 10
    fig_height = 3.0 * nrows   # 每行大约 3 英寸高度，可按需要调
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)

    # axes 统一展平成 1D，方便索引
    axes = np.array(axes).ravel()

    cmap = plt.get_cmap("RdBu_r")
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)",
               "(g)", "(h)", "(i)", "(j)", "(k)", "(l)",
               "(m)", "(n)", "(o)", "(p)", "(q)", "(r)",
               "(s)", "(t)", "(u)", "(v)", "(w)", "(x)",
               "(y)", "(z)", "(aa)"]
    # 字母最多给到 r，产品再多也只是标题没有字母，不影响使用

    last_scatter = None

    for i, ax in enumerate(axes):
        if i < n_prod:
            prod = products[i]
            S = SAL_results[prod]["S"]
            A = SAL_results[prod]["A"]
            L = SAL_results[prod]["L"]

            # 去掉 NaN
            mask = np.isfinite(S) & np.isfinite(A) & np.isfinite(L)
            S = S[mask]
            A = A[mask]
            L = L[mask]

            if S.size > 0:
                sc = ax.scatter(
                    S, A,
                    c=L,
                    cmap=cmap,
                    vmin=L_VMIN,
                    vmax=L_VMAX,
                    s=POINT_SIZE,
                    alpha=0.8,
                    edgecolors="none",
                )
                last_scatter = sc

            # 标题
            letter = letters[i] if i < len(letters) else ""
            ax.set_title(f"{letter} {prod}", fontsize=12)
        else:
            # 超出产品数量的多余子图，直接关闭
            ax.axis("off")
            continue

        # 坐标系 & 网格
        ax.axhline(0, color="black", linewidth=1.0)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    # 只在最左一列加 y 轴标签，在最后一行加 x 轴标签
    for r in range(nrows):
        idx = r * ncols
        if idx < len(axes):
            axes[idx].set_ylabel("A", fontsize=14)

    start_last_row = (nrows - 1) * ncols
    for idx in range(start_last_row, start_last_row + ncols):
        if idx < len(axes):
            axes[idx].set_xlabel("S", fontsize=14)

    # 统一 colorbar，放在底部横着
    # 总标题
    fig.suptitle(
        "SAL diagram for daily precipitation (top {} days)".format(TOP_N_EVENTS),
        fontsize=14,
    )

    # 先对子图做紧凑布局，预留底部空间给 colorbar
    plt.tight_layout(rect=[0.03, 0.10, 0.97, 0.95])

    # 统一 colorbar：在底部单独开一个小轴，横向放
    if last_scatter is not None:
        # [left, bottom, width, height] 都是 0–1 的 figure 坐标，可以按需要微调
        cax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = fig.colorbar(
            last_scatter,
            cax=cax,
            orientation="horizontal",
        )
        cbar.set_label("L", fontsize=12)

    out_png = os.path.join(
        NC_DIR,
        "SAL_diagram_top{}_days.png".format(TOP_N_EVENTS)
    )
    print("Save SAL figure to:", out_png)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)





if __name__ == "__main__":
    main()
