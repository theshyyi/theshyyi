# -*- coding: utf-8 -*-
"""
脚本 B（增强版）：基于 TIMEFIX 目录中修复后的 nc 文件，逐个产品计算指标：
  - 与参考产品对比（参考也用 TIMEFIX 后的日尺度 nc）
  - 指标全部是“标量”，包括：
      - RMSE, MAE, Bias, CC, NSE, KGE（基于中国域平均日序列）
      - POD, FAR, CSI, HSS（基于所有格点+时间的雨日检测）
      - ETCCDI 降水极端指数的“误差项”（基于中国域平均日序列，按年计算后取多年平均）：
          * R10mm, R20mm, RR1, PRCPTOT, SDII
          * RX1day, RX5day
          * R95p, R99p, R95pTOT, R99pTOT
          * CDD, CWD
        说明：
          1) R95p/R99p 的阈值（95/99 分位）统一用“参考数据”湿日（pr>=1mm）的全时段阈值，
             然后对参考/产品都用同一个阈值进行累积，避免“各算各的分位阈值”导致不可比。
          2) 所有指数按“年”计算（groupby year），最终输出多年平均；误差定义为 (prod - ref)。
      - N_Days_Overlap（时间重叠天数，应当 = 参考日数）
  - 输出一个 CSV：Final_MCDM_Input_Ranking.csv

使用前确认：
  - TIMEFIX 目录中已有 *.TIMEFIX.daily.CHINA.nc（由脚本 A 生成）
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

# 雨日检测（用于 POD/FAR/CSI/HSS）：常用 0.1 mm/day 作为“降水发生”
RAIN_THRESHOLD = 0.1

# ETCCDI 湿日阈值（用于 RR1/PRCPTOT/SDII/CDD/CWD 等）：按 ETCCDI 定义一般取 1 mm/day
WETDAY_THRESHOLD = 1.0

# ETCCDI 分位阈值（对 R95p/R99p）：
Q95 = 0.95
Q99 = 0.99

OUTPUT_CSV = os.path.join(TIMEFIX_FOLDER, "Final_MCDM_Input_Ranking.csv")

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


def compute_skill_metrics_from_ts(ref_ts, prod_ts):
    """
    基于中国域平均日序列 (1D time) 计算 RMSE, MAE, Bias, CC, NSE, KGE
    """
    eps = 1e-12
    ref = ref_ts.astype(float)
    prod = prod_ts.astype(float)

    # 去除 NaN 对
    mask = np.isfinite(ref) & np.isfinite(prod)
    ref = ref[mask]
    prod = prod[mask]

    if ref.size == 0:
        return {k: np.nan for k in ["RMSE", "MAE", "Bias", "CC", "NSE", "KGE"]}

    diff = prod - ref

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)

    # CC
    cc = np.corrcoef(ref, prod)[0, 1]

    # NSE
    ref_mean = np.mean(ref)
    num = np.sum((prod - ref) ** 2)
    denom = np.sum((ref - ref_mean) ** 2) + eps
    nse = 1 - num / denom

    # KGE
    std_ref = np.std(ref)
    std_prod = np.std(prod)
    alpha = std_prod / (std_ref + eps)
    beta = np.mean(prod) / (ref_mean + eps)
    kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "CC": cc,
        "NSE": nse,
        "KGE": kge,
    }


def compute_categorical_metrics(ref_da, prod_da, threshold):
    """
    基于阈值的 POD, FAR, CSI, HSS：
      - ref_da, prod_da: (time, lat, lon)
      - 所有格点+时间一起算
    """
    eps = 1e-12

    # 对 NaN 做联合掩膜（避免把缺测当成 miss/false）
    mask = np.isfinite(ref_da) & np.isfinite(prod_da)
    ref = ref_da.where(mask)
    prod = prod_da.where(mask)

    obs_hit = ref >= threshold
    obs_miss = ref < threshold
    pred_hit = prod >= threshold
    pred_miss = prod < threshold

    H = (obs_hit & pred_hit).sum().item()
    M = (obs_hit & pred_miss).sum().item()
    F = (obs_miss & pred_hit).sum().item()
    CN = (obs_miss & pred_miss).sum().item()

    pod = H / (H + M + eps)
    far = F / (H + F + eps)
    csi = H / (H + M + F + eps)

    num = 2 * (H * CN - M * F)
    den = (H + M) * (M + CN) + (H + F) * (F + CN) + eps
    hss = num / den

    return {
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "HSS": hss,
    }


def _max_consecutive_true(flags_1d: np.ndarray) -> int:
    """返回一维 bool 序列中连续 True 的最大长度；NaN 对应的比较结果一般是 False，会自然断开。"""
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


def _annual_etccdi_from_series(
    s: pd.Series,
    wet_th: float,
    q95_base: float,
    q99_base: float,
) -> pd.DataFrame:
    """
    输入：日尺度序列 s（DatetimeIndex），输出：逐年的 ETCCDI 降水极端指数。
    """
    def one_year_metrics(y: pd.Series) -> dict:
        # y 是某一年的日序列（可能含 NaN）
        wet = y >= wet_th
        dry = y < wet_th

        rr1 = int(wet.sum())
        prcptot = float(y.where(wet).sum(skipna=True))

        sdii = prcptot / rr1 if rr1 > 0 else np.nan

        r10 = int((y >= 10.0).sum())
        r20 = int((y >= 20.0).sum())

        rx1 = float(y.max(skipna=True)) if y.notna().any() else np.nan

        # Rx5day：年内 5 日滑动和的最大值（不足 5 天则 NaN）
        if y.notna().sum() >= 5:
            rx5 = float(y.rolling(window=5, min_periods=5).sum().max(skipna=True))
        else:
            rx5 = np.nan

        # R95p/R99p：使用统一阈值（来自参考全时段湿日分位）
        r95 = float(y.where(wet & (y > q95_base)).sum(skipna=True))
        r99 = float(y.where(wet & (y > q99_base)).sum(skipna=True))

        r95tot = (r95 / prcptot * 100.0) if prcptot > 0 else np.nan
        r99tot = (r99 / prcptot * 100.0) if prcptot > 0 else np.nan

        cdd = _max_consecutive_true(dry.to_numpy(dtype=bool))
        cwd = _max_consecutive_true(wet.to_numpy(dtype=bool))

        return {
            "R10mm": r10,
            "R20mm": r20,
            "RR1": rr1,
            "PRCPTOT": prcptot,
            "SDII": sdii,
            "RX1day": rx1,
            "RX5day": rx5,
            "R95p": r95,
            "R99p": r99,
            "R95pTOT": r95tot,
            "R99pTOT": r99tot,
            "CDD": cdd,
            "CWD": cwd,
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
    计算 ETCCDI 降水极端指数误差（标量）：
      1) 用参考序列的全时段“湿日”(ref>=wet_th) 计算 q95/q99 阈值；
      2) 逐年计算各指数；输出多年平均；
      3) 误差 = prod_mean - ref_mean。
    """
    # 构建序列（保留 NaN；NaN 会在 sum/max 中被跳过，在 run length 中被当作断开）
    ref_s = pd.Series(ref_ts.astype(float), index=time_index)
    prod_s = pd.Series(prod_ts.astype(float), index=time_index)

    # 统一有效时间：若某天任一序列缺测，则两者都置为 NaN（避免可比性问题）
    ok = ref_s.notna() & prod_s.notna()
    ref_s = ref_s.where(ok)
    prod_s = prod_s.where(ok)

    # 基准分位阈值（来自参考全时段湿日）
    ref_wet = ref_s[ref_s >= wet_th].dropna()
    if ref_wet.empty:
        # 没有湿日时，所有极端指数都无法定义
        keys = ["R10mm","R20mm","RR1","PRCPTOT","SDII","RX1day","RX5day",
                "R95p","R99p","R95pTOT","R99pTOT","CDD","CWD"]
        return {f"Err_{k}": np.nan for k in keys}

    q95_base = float(ref_wet.quantile(q95))
    q99_base = float(ref_wet.quantile(q99))

    # 年度指数
    ref_y = _annual_etccdi_from_series(ref_s, wet_th, q95_base, q99_base)
    prod_y = _annual_etccdi_from_series(prod_s, wet_th, q95_base, q99_base)

    # 多年平均（标量）
    ref_mean = ref_y.mean(axis=0, skipna=True)
    prod_mean = prod_y.mean(axis=0, skipna=True)

    err = (prod_mean - ref_mean).to_dict()
    return {f"Err_{k}": float(v) if np.isfinite(v) else np.nan for k, v in err.items()}


def main():
    # 1. TIMEFIX 目录中找到参考和产品
    ref_path, prod_paths = get_timefix_files(TIMEFIX_FOLDER, REF_NAME_PREFIX)
    print(">>> TIMEFIX 参考文件:", os.path.basename(ref_path))
    print(">>> TIMEFIX 产品数量:", len(prod_paths))

    # 2. 读取参考 TIMEFIX 变量（懒加载）
    ds_ref, da_ref = open_fixed_var(ref_path, VAR_NAME)
    time_index = pd.to_datetime(da_ref["time"].values)
    n_days_ref = len(time_index)
    print(f"    参考时间步数: {n_days_ref}")

    rows = []

    for name, path in prod_paths.items():
        print(f"\n--- 计算产品指标: {name} ---")
        try:
            ds_p, da_p = open_fixed_var(path, VAR_NAME)

            # 时间对齐（应该已经对齐了，这里再保险一次）
            da_p = da_p.sel(time=da_ref["time"])

            # 计算域平均时间序列 (1D time)
            ref_ts_da = da_ref.mean(dim=("lat", "lon"))
            prod_ts_da = da_p.mean(dim=("lat", "lon"))

            ref_ts = ref_ts_da.values
            prod_ts = prod_ts_da.values

            # 重叠有效天数
            mask_ts = np.isfinite(ref_ts) & np.isfinite(prod_ts)
            n_overlap = int(mask_ts.sum())
            print(f"    有效叠加天数: {n_overlap}")

            # 技能指标（基于域平均 TS）
            metrics_skill = compute_skill_metrics_from_ts(ref_ts, prod_ts)

            # 分类指标（全空间+时间）
            metrics_cat = compute_categorical_metrics(
                da_ref, da_p, threshold=RAIN_THRESHOLD
            )

            # ETCCDI 极端指数误差（域平均 TS，按年）
            metrics_etccdi = compute_etccdi_errors_from_ts(
                ref_ts, prod_ts, time_index,
                wet_th=WETDAY_THRESHOLD, q95=Q95, q99=Q99
            )

            metrics_all = {}
            metrics_all.update(metrics_skill)
            metrics_all.update(metrics_cat)
            metrics_all.update(metrics_etccdi)
            metrics_all["N_Days_Overlap"] = n_overlap

            row = {"Product": name}
            row.update(metrics_all)
            rows.append(row)

            ds_p.close()

        except Exception as e:
            print(f"    [错误] {name} 计算失败: {e}")
            continue

    ds_ref.close()

    df = pd.DataFrame(rows).set_index("Product")
    df.to_csv(OUTPUT_CSV)
    print(f"\n>>> MCDM 指标矩阵已保存: {OUTPUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()
