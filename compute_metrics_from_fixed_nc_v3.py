# -*- coding: utf-8 -*-
"""
脚本 B（增强版-分区版）：基于 TIMEFIX 目录中修复后的 nc 文件，逐个产品、逐个分区计算指标：
  - 与参考产品对比（参考也用 TIMEFIX 后的日尺度 nc）
  - 指标全部是“标量”，但变成“每个分区一套标量”，包括：
      - RMSE, MAE, Bias, CC, NSE, KGE（基于【分区平均】日序列）
      - POD, FAR, CSI, HSS（基于【分区内所有格点+时间】的雨日检测）
      - ETCCDI 极端指数误差（基于【分区平均】日序列，按年计算后取多年平均）：
          * R10mm, R20mm, RR1, PRCPTOT, SDII
          * RX1day, RX5day
          * R95p, R99p, R95pTOT, R99pTOT
          * CDD, CWD
        说明：
          1) R95p/R99p 的阈值（95/99 分位）统一用【参考数据-该分区】湿日（pr>=1mm）的全时段阈值
          2) 所有指数按“年”计算，最终输出多年平均；误差定义为 (prod - ref)。
      - N_Days_Overlap（该分区均值序列的有效重叠天数）

  - 输出 CSV：Final_MCDM_Input_Ranking_7regions.csv （长表：Product×Region）

依赖：geopandas, regionmask, shapely, xarray, numpy, pandas
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ====================== 用户配置 ======================

DATA_FOLDER = "/home/ud202380664/PRE_MERGE/TIMEFIX"
TIMEFIX_FOLDER = os.path.join(DATA_FOLDER, "Finish")

REF_NAME_PREFIX = "CMFDV2"
VAR_NAME = "pr"

# 7 分区 shp（按你的实际路径改）
REGION_SHP = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp"
REGION_FIELD = "climate"  # shp 里分区字段名（你之前就是 climate）

# 雨日检测（用于 POD/FAR/CSI/HSS）：常用 0.1 mm/day 作为“降水发生”
RAIN_THRESHOLD = 0.1

# ETCCDI 湿日阈值（用于 RR1/PRCPTOT/SDII/CDD/CWD 等）：按 ETCCDI 定义一般取 1 mm/day
WETDAY_THRESHOLD = 1.0

# ETCCDI 分位阈值（对 R95p/R99p）：
Q95 = 0.95
Q99 = 0.99

OUTPUT_CSV = os.path.join(TIMEFIX_FOLDER, "Final_MCDM_Input_Ranking_7regions.csv")

# 如 climate 字段为中文，可选：映射英文/缩写（不需要可删掉）
CLIMATE_NAME_MAP = {
    "暖温带半湿润地区": "Warm Temperate (Semi-humid)",
    "中温带干旱地区": "Mid-Temperate (Arid)",
    "北亚热带湿润地区": "North Subtropical (Humid)",
    "中温带半湿润地区": "Mid-Temperate (Semi-humid)",
    "中温带半干旱地区": "Mid-Temperate (Semi-arid)",
    "高原温带半干旱地区": "Plateau Temperate (Semi-arid)",
    "边缘热带湿润地区": "Marginal Tropical (Humid)",
}
CLIMATE_ABBR_MAP = {
    "暖温带半湿润地区": "WT-SH",
    "中温带干旱地区": "MT-A",
    "北亚热带湿润地区": "NST-H",
    "中温带半湿润地区": "MT-SH",
    "中温带半干旱地区": "MT-SA",
    "高原温带半干旱地区": "PT-SA",
    "边缘热带湿润地区": "MTr-H",
}

# =====================================================


def open_fixed_var(path, var_name):
    """懒加载 TIMEFIX 文件中的变量"""
    ds = xr.open_dataset(path)
    if var_name not in ds:
        ds.close()
        raise KeyError(f"{path} 中未找到变量 {var_name}")
    da = ds[var_name]
    return ds, da


def get_timefix_files(folder, ref_prefix):
    """列出 TIMEFIX 文件：参考 + 产品"""
    all_files = sorted(f for f in os.listdir(folder) if f.endswith(".nc"))
    if not all_files:
        raise FileNotFoundError(f"{folder} 下没有 TIMEFIX nc 文件")

    ref_path = None
    prod_paths = {}
    for fname in all_files:
        if not fname.endswith(".TIMEFIX.daily.CHINA.nc"):
            continue
        prefix = fname.split(".")[0]  # 例如 CMFDV2, IMERG-F
        full = os.path.join(folder, fname)
        if prefix == ref_prefix:
            ref_path = full
        else:
            prod_paths[prefix] = full

    if ref_path is None:
        raise FileNotFoundError(f"TIMEFIX 目录中未找到参考前缀 {ref_prefix} 对应文件")

    return ref_path, prod_paths


def get_latlon_names(da):
    lat_name = "lat" if "lat" in da.dims else ("latitude" if "latitude" in da.dims else None)
    lon_name = "lon" if "lon" in da.dims else ("longitude" if "longitude" in da.dims else None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"无法识别经纬度维度名，当前 dims={da.dims}")
    return lat_name, lon_name


def build_region_mask3d(da_grid, shp_path, field):
    """
    从 shp 构建 mask_3D(region, lat, lon)，True 表示该格点属于该 region
    使用 mask_3D(lons, lats) 的位置参数，避免 regionmask 版本差异导致的 lon=/lat= 报错
    """
    import geopandas as gpd
    import regionmask

    lat_name, lon_name = get_latlon_names(da_grid)
    lats = da_grid[lat_name].values
    lons = da_grid[lon_name].values

    gdf = gpd.read_file(shp_path)
    if field not in gdf.columns:
        raise KeyError(f"SHP 中未找到字段 {field}，可用字段：{list(gdf.columns)}")

    # 尽量转到经纬度 CRS（若已是 EPSG:4326 则无影响）
    try:
        gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    names_raw = gdf[field].astype(str).tolist()
    names_en = [CLIMATE_NAME_MAP.get(n, n) for n in names_raw]
    abbrevs = [CLIMATE_ABBR_MAP.get(n, n) for n in names_raw]

    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        names=names_en,
        abbrevs=abbrevs,
        numbers=list(range(len(gdf)))
    )

    mask3d = regions.mask_3D(lons, lats).astype(bool)  # dims: region, lat, lon（默认命名）

    # 对齐维度名到 da_grid（如果 da_grid 用的是 latitude/longitude）
    if "lat" in mask3d.dims and lat_name != "lat":
        mask3d = mask3d.rename({"lat": lat_name})
    if "lon" in mask3d.dims and lon_name != "lon":
        mask3d = mask3d.rename({"lon": lon_name})

    region_info = pd.DataFrame({
        "Region_ID": list(range(len(gdf))),
        "Region_Raw": names_raw,
        "Region": names_en,
        "Region_Abbr": abbrevs
    })
    return regions, mask3d, region_info


def compute_skill_metrics_from_ts(ref_ts, prod_ts):
    """
    基于【分区平均】日序列 (1D time) 计算 RMSE, MAE, Bias, CC, NSE, KGE
    """
    eps = 1e-12
    ref = ref_ts.astype(float)
    prod = prod_ts.astype(float)

    mask = np.isfinite(ref) & np.isfinite(prod)
    ref = ref[mask]
    prod = prod[mask]

    if ref.size == 0:
        return {k: np.nan for k in ["RMSE", "MAE", "Bias", "CC", "NSE", "KGE"]}

    diff = prod - ref
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)

    cc = np.corrcoef(ref, prod)[0, 1]

    ref_mean = np.mean(ref)
    num = np.sum((prod - ref) ** 2)
    denom = np.sum((ref - ref_mean) ** 2) + eps
    nse = 1 - num / denom

    std_ref = np.std(ref)
    std_prod = np.std(prod)
    alpha = std_prod / (std_ref + eps)
    beta = np.mean(prod) / (ref_mean + eps)
    kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return {"RMSE": rmse, "MAE": mae, "Bias": bias, "CC": cc, "NSE": nse, "KGE": kge}


def compute_categorical_metrics_by_regions(ref_da, prod_da, mask3d, threshold):
    """
    基于阈值的 POD, FAR, CSI, HSS（按分区统计）：
      - ref_da, prod_da: (time, lat, lon)
      - mask3d: (region, lat, lon) bool
      - 所有格点+时间一起算，但只在各分区 mask 内累计
    返回：dict(key -> np.ndarray shape(n_region,))
    """
    eps = 1e-12
    lat_name, lon_name = get_latlon_names(ref_da)

    valid = np.isfinite(ref_da) & np.isfinite(prod_da)
    ref = ref_da.where(valid)
    prod = prod_da.where(valid)

    obs_hit = ref >= threshold
    obs_miss = ref < threshold
    pred_hit = prod >= threshold
    pred_miss = prod < threshold

    m = mask3d.astype("int8")

    H = ((obs_hit & pred_hit).astype("int8") * m).sum(dim=("time", lat_name, lon_name)).values
    M = ((obs_hit & pred_miss).astype("int8") * m).sum(dim=("time", lat_name, lon_name)).values
    F = ((obs_miss & pred_hit).astype("int8") * m).sum(dim=("time", lat_name, lon_name)).values
    CN = ((obs_miss & pred_miss).astype("int8") * m).sum(dim=("time", lat_name, lon_name)).values

    pod = H / (H + M + eps)
    far = F / (H + F + eps)
    csi = H / (H + M + F + eps)

    num = 2 * (H * CN - M * F)
    den = (H + M) * (M + CN) + (H + F) * (F + CN) + eps
    hss = num / den

    return {"POD": pod, "FAR": far, "CSI": csi, "HSS": hss}


def _max_consecutive_true(flags_1d: np.ndarray) -> int:
    maxlen = 0
    curlen = 0
    for v in flags_1d:
        if bool(v):
            curlen += 1
        else:
            if curlen > maxlen:
                maxlen = curlen
            curlen = 0
    if curlen > maxlen:
        maxlen = curlen
    return int(maxlen)


def _annual_etccdi_from_series(s: pd.Series, wet_th: float, q95_base: float, q99_base: float) -> pd.DataFrame:
    def one_year_metrics(y: pd.Series) -> dict:
        wet = y >= wet_th
        dry = y < wet_th

        rr1 = int(wet.sum())
        prcptot = float(y.where(wet).sum(skipna=True))

        sdii = prcptot / rr1 if rr1 > 0 else np.nan

        r10 = int((y >= 10.0).sum())
        r20 = int((y >= 20.0).sum())

        rx1 = float(y.max(skipna=True)) if y.notna().any() else np.nan

        if y.notna().sum() >= 5:
            rx5 = float(y.rolling(window=5, min_periods=5).sum().max(skipna=True))
        else:
            rx5 = np.nan

        r95 = float(y.where(wet & (y > q95_base)).sum(skipna=True))
        r99 = float(y.where(wet & (y > q99_base)).sum(skipna=True))

        r95tot = (r95 / prcptot * 100.0) if prcptot > 0 else np.nan
        r99tot = (r99 / prcptot * 100.0) if prcptot > 0 else np.nan

        cdd = _max_consecutive_true(dry.to_numpy(dtype=bool))
        cwd = _max_consecutive_true(wet.to_numpy(dtype=bool))

        return {
            "R10mm": r10, "R20mm": r20, "RR1": rr1, "PRCPTOT": prcptot, "SDII": sdii,
            "RX1day": rx1, "RX5day": rx5,
            "R95p": r95, "R99p": r99, "R95pTOT": r95tot, "R99pTOT": r99tot,
            "CDD": cdd, "CWD": cwd,
        }

    years = sorted(s.index.year.unique())
    out = []
    for yr in years:
        y = s[s.index.year == yr]
        out.append(pd.Series(one_year_metrics(y), name=int(yr)))
    df = pd.DataFrame(out)
    df.index.name = "year"
    return df


def compute_etccdi_errors_from_ts(
    ref_ts: np.ndarray,
    prod_ts: np.ndarray,
    time_index: pd.DatetimeIndex,
    wet_th: float = 1.0,
    q95: float = 0.95,
    q99: float = 0.99,
):
    """
    分区版 ETCCDI 极端指数误差（标量）：
      1) 用参考序列的全时段“湿日”(ref>=wet_th) 计算 q95/q99 阈值；
      2) 逐年计算各指数；输出多年平均；
      3) 误差 = prod_mean - ref_mean。
    """
    ref_s = pd.Series(ref_ts.astype(float), index=time_index)
    prod_s = pd.Series(prod_ts.astype(float), index=time_index)

    ok = ref_s.notna() & prod_s.notna()
    ref_s = ref_s.where(ok)
    prod_s = prod_s.where(ok)

    ref_wet = ref_s[ref_s >= wet_th].dropna()
    keys = ["R10mm","R20mm","RR1","PRCPTOT","SDII","RX1day","RX5day",
            "R95p","R99p","R95pTOT","R99pTOT","CDD","CWD"]
    if ref_wet.empty:
        return {f"Err_{k}": np.nan for k in keys}

    q95_base = float(ref_wet.quantile(q95))
    q99_base = float(ref_wet.quantile(q99))

    ref_y = _annual_etccdi_from_series(ref_s, wet_th, q95_base, q99_base)
    prod_y = _annual_etccdi_from_series(prod_s, wet_th, q95_base, q99_base)

    ref_mean = ref_y.mean(axis=0, skipna=True)
    prod_mean = prod_y.mean(axis=0, skipna=True)

    err = (prod_mean - ref_mean).to_dict()
    return {f"Err_{k}": float(v) if np.isfinite(v) else np.nan for k, v in err.items()}


def region_mean_ts_all_regions(da, mask3d):
    """一次性计算所有分区的区域平均日序列：返回 DataArray (region, time)"""
    lat_name, lon_name = get_latlon_names(da)
    num = (da * mask3d).sum(dim=(lat_name, lon_name))
    den = mask3d.sum(dim=(lat_name, lon_name))
    return num / den


def main():
    ref_path, prod_paths = get_timefix_files(TIMEFIX_FOLDER, REF_NAME_PREFIX)
    print(">>> TIMEFIX 参考文件:", os.path.basename(ref_path))
    print(">>> TIMEFIX 产品数量:", len(prod_paths))

    ds_ref, da_ref = open_fixed_var(ref_path, VAR_NAME)
    time_index = pd.to_datetime(da_ref["time"].values)
    print(f"    参考时间步数: {len(time_index)}")

    # 构建 7 分区 mask
    print(">>> 构建分区 mask_3D ...")
    regions, mask3d, region_info = build_region_mask3d(da_ref, REGION_SHP, REGION_FIELD)
    n_region = mask3d.sizes["region"]
    print(f"    分区数: {n_region}")
    print(region_info)

    # 参考：所有分区的均值序列 (region, time)
    ref_reg_ts_da = region_mean_ts_all_regions(da_ref, mask3d)
    ref_reg_ts = ref_reg_ts_da.transpose("region", "time").values

    rows = []

    for prod_name, prod_path in prod_paths.items():
        print(f"\n--- 计算产品指标（分区）: {prod_name} ---")
        try:
            ds_p, da_p = open_fixed_var(prod_path, VAR_NAME)
            da_p = da_p.sel(time=da_ref["time"])  # 对齐时间

            # 产品：所有分区的均值序列
            prod_reg_ts_da = region_mean_ts_all_regions(da_p, mask3d)
            prod_reg_ts = prod_reg_ts_da.transpose("region", "time").values

            # 分类指标（分区内所有格点+时间）
            metrics_cat = compute_categorical_metrics_by_regions(
                da_ref, da_p, mask3d, threshold=RAIN_THRESHOLD
            )

            # 每个分区一行
            for rid in range(n_region):
                ref_ts = ref_reg_ts[rid, :]
                prod_ts = prod_reg_ts[rid, :]

                mask_ts = np.isfinite(ref_ts) & np.isfinite(prod_ts)
                n_overlap = int(mask_ts.sum())

                metrics_skill = compute_skill_metrics_from_ts(ref_ts, prod_ts)

                metrics_etccdi = compute_etccdi_errors_from_ts(
                    ref_ts, prod_ts, time_index,
                    wet_th=WETDAY_THRESHOLD, q95=Q95, q99=Q99
                )

                row = {
                    "Product": prod_name,
                    "Region_ID": int(region_info.loc[rid, "Region_ID"]),
                    "Region": str(region_info.loc[rid, "Region"]),
                    "Region_Abbr": str(region_info.loc[rid, "Region_Abbr"]),
                    "N_Days_Overlap": n_overlap,
                    "POD": float(metrics_cat["POD"][rid]),
                    "FAR": float(metrics_cat["FAR"][rid]),
                    "CSI": float(metrics_cat["CSI"][rid]),
                    "HSS": float(metrics_cat["HSS"][rid]),
                }
                row.update(metrics_skill)
                row.update(metrics_etccdi)

                rows.append(row)

            ds_p.close()

        except Exception as e:
            print(f"    [错误] {prod_name} 计算失败: {e}")
            continue

    ds_ref.close()

    df = pd.DataFrame(rows)
    df = df.sort_values(["Region_ID", "Product"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n>>> 分区 MCDM 指标矩阵已保存: {OUTPUT_CSV}")
    print(df.head(30))


if __name__ == "__main__":
    main()
