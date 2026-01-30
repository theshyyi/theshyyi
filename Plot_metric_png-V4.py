import os
import glob
import math
import warnings
import gc
import json

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")  # 必须在 import pyplot 前设置
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter

warnings.filterwarnings("ignore")

# ====================== 运行模式 ======================
#   "scan" : 只扫描所有产品的颜色条范围 + 泰勒统计，不画任何图
#   "plot" : 使用上一步保存的颜色条范围，逐个产品绘图 + 输出区域/季节统计
MODE = "scan"  # 第一次运行先设为 "scan"，跑完后改成 "plot"

# ====================== 用户配置区域 ======================
DATA_FOLDER = r"/home/ud202380664/PRE_MERGE/TIMEFIX/"
REF_NAME_PREFIX = "CMFDV2"
VAR_NAME = "pr"

RAIN_THRESHOLD = 0.1
EXTREME_Q = 0.95

SCATTER_MAX_POINTS = 50000

# <<< NEW: 中国 7 个自然区 shapefile 配置 >>>
# 这里写你的 7 区划分 shp 路径（支持 .shp 或 .gpkg 等）
# <<< NEW: 中国 7 个自然区 shapefile 配置 >>>
SHP_FILE = r"/home/ud202380664/PRE_MERGE/TIMEFIX/SHP/Chinese_climate.shp"
SHP_REGION_FIELD = "climate"  # 分区字段名


# 是否启用 shapefile 区域掩膜（如果 False，则退回全域）
USE_REGION_MASK = True

# 是否用 icclim 计算极端指数（只提供模板，需按你 icclim 版本微调）
USE_ICCLIM = True
# ICCLIM_INDICES = ["rx1day", "r10mm", "sdii", "r95p"]  # 可以按需增删

# =========================================================

plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["mathtext.default"] = "regular"
RANGE_JSON = os.path.join(DATA_FOLDER, "metric_ranges.json")
TAYLOR_CSV = os.path.join(DATA_FOLDER, "taylor_stats.csv")
REGION_METRIC_CSV = os.path.join(DATA_FOLDER, "region_metrics_by_season.csv")
ETCCDI_REGION_CSV = os.path.join(DATA_FOLDER, "etccdi_region_season.csv")


# 要用 icclim 计算的降水 ETCCDI 指标（和你之前代码一致）
# ICCLIM_INDICES = [
    # "r10mm", "r20mm",
    # "rx1day", "rx5day",
    # "r95ptot", "r99ptot",
    # "r95p", "r99p",
    # "rr1", "cdd", "cwd",
    # "sdii", "prcptot",
# ]


# ================= dask 自动检测 =================
try:
    import dask  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


def _normalize_to_day(time_values):
    return pd.to_datetime(time_values).normalize()


class PreAlignedEvaluator:
    def __init__(self, folder, ref_prefix, var_name):
        self.folder = folder
        self.ref_prefix = ref_prefix
        self.var_name = var_name

        self.ref_da: xr.DataArray | None = None
        self.ref_days = None
        self.prod_files: dict[str, str] = {}

        self.time_report_rows: list[dict] = []
        self._time_scanned: set[str] = set()

        self.chunks = {"time": 365} if HAS_DASK else None

        # ⭐ 全部产品的泰勒图统计：[{name, std, cc}, ...]
        self.taylor_stats: list[dict] = []

        # ⭐ 统一色标范围：{metric_name: {"vmin":..., "vmax":...}}
        self.metric_ranges: dict[str, dict] = {}

        # <<< NEW: shapefile 区域掩膜相关 >>>
        self.china_mask2d: xr.DataArray | None = None  # True=中国境内
        self.region_mask_3d: xr.DataArray | None = None  # (region, lat, lon) bool
        self.region_names: list[str] = []
        self.region_stats_rows: list[dict] = []  # 区域 + 季节 指标表

        # <<< NEW: icclim 参考极端指数缓存 >>>
        # self.ref_etccdi: dict[str, xr.DataArray] = {}
        
                # # <<< NEW: icclim 参考极端指数缓存 >>>
        # # 年尺度：多年平均（用于整体 Bias）
        # self.ref_etccdi_annual: dict[str, xr.DataArray] = {}
        # # 月尺度：逐月指数（用于分季节 Bias）
        # self.ref_etccdi_monthly: dict[str, xr.DataArray] = {}


    # ---------- 通用 open_dataset ----------
    def _open_ds(self, path: str):
        if self.chunks is not None:
            return xr.open_dataset(path, chunks=self.chunks)
        else:
            return xr.open_dataset(path)

    # ---------- 空间对齐 ----------
    def _align_prod(self, da: xr.DataArray) -> xr.DataArray:
        """
        调整维度顺序 & 替换空间坐标为参考数据坐标
        """
        da = da.transpose(*self.ref_da.dims)
        new_coords = {dim: self.ref_da[dim] for dim in self.ref_da.dims if dim in da.dims}
        da = da.assign_coords(new_coords)
        return da

    # ---------- 1. 加载参考数据 + 记录产品文件 ----------
    def load_data(self):
        """加载数据，参考数据在这里就转换为日尺度"""
        print(">>> 1. 正在加载数据并处理参考数据为日尺度...")

        all_files = glob.glob(os.path.join(self.folder, "*.nc"))
        if not all_files:
            raise FileNotFoundError(f"在文件夹 {self.folder} 下未找到任何 .nc 文件！")

        # 1) 找参考文件
        ref_file = None
        for f in all_files:
            fname = os.path.basename(f)
            prefix = fname.split(".")[0]
            if prefix == self.ref_prefix:
                ref_file = f
                break

        if ref_file is None:
            raise FileNotFoundError(f"未找到前缀为 '{self.ref_prefix}' 的参考文件！")

        print(f"   参考文件: {os.path.basename(ref_file)}")

        # 2) 打开参考文件并转换为日尺度
        ds_ref = self._open_ds(ref_file)

        if self.var_name not in ds_ref:
            raise KeyError(f"参考文件中未找到变量 {self.var_name}！")

        da_ref = ds_ref[self.var_name]
        if "time" not in da_ref.dims:
            raise KeyError("参考数据中未找到 time 维度")

        time_raw = da_ref["time"].values
        ref_days = _normalize_to_day(time_raw)

        # 同一天多条记录 → groupby 平均
        day_df = pd.DataFrame({"day": ref_days})
        if day_df["day"].duplicated().any():
            print("   [提示] 参考数据中同一天有多条记录，执行按日平均。")
            ds_ref = ds_ref.assign_coords(time=("time", ref_days))
            ds_ref = ds_ref.groupby("time").mean()
            self.ref_da = ds_ref[self.var_name]
        else:
            da_ref = da_ref.assign_coords(time=("time", ref_days))
            self.ref_da = da_ref

        self.ref_days = pd.to_datetime(self.ref_da["time"].values).normalize().values
        ref_len = len(self.ref_days)
        print(f"   参考（日尺度）时间长度: {ref_len}")

        # 3) 记录各产品文件路径（只检查基本形状）
        for f in all_files:
            if f == ref_file:
                continue
            fname = os.path.basename(f)
            prefix = fname.split(".")[0]

            with self._open_ds(f) as ds:
                if self.var_name not in ds:
                    print(f"   [警告] {fname} 中不含变量 {self.var_name}，跳过！")
                    continue
                da = ds[self.var_name]
                if da.ndim != self.ref_da.ndim:
                    print(
                        f"   [警告] {prefix} 维度数 {da.ndim} 与参考 {self.ref_da.ndim} 不一致，跳过！"
                    )
                    continue

            self.prod_files[prefix] = f
            print(f"   记录产品文件: {prefix} -> {fname}")

        ds_ref.close()
        print(f">>> 数据加载完成，共 {len(self.prod_files)} 个待评估产品。\n")

        # <<< NEW: 加载 shapefile，构建区域掩膜 >>>
        if USE_REGION_MASK:
            self._build_region_masks()

        # <<< NEW: 如启用 icclim，预计算参考极端指数（可选，只运行一次） >>>
        if USE_ICCLIM:
            try:
                self._compute_ref_etccdi_icclim()
            except Exception as e:
                print(f"[警告] 计算参考 icclim 极端指数失败：{e}")

    # ---------- 2. 从产品文件读取日尺度并对齐 ----------
    def _load_prod_daily(self, name: str, filepath: str) -> xr.DataArray:
        ds = self._open_ds(filepath)
        if self.var_name not in ds:
            ds.close()
            raise KeyError(f"{filepath} 中未找到变量 {self.var_name}")

        da = ds[self.var_name]
        if "time" not in da.dims:
            ds.close()
            raise KeyError(f"{filepath} 中变量 {self.var_name} 无 time 维度")

        time_raw = pd.to_datetime(da["time"].values)
        prod_days = time_raw.normalize()

        unique_days, counts = np.unique(prod_days, return_counts=True)
        if np.any(counts > 1):
            da_with_day = da.assign_coords(day=("time", prod_days))
            da_day = da_with_day.groupby("day").mean(dim="time")
            da_day = da_day.rename({"day": "time"})
        else:
            da_day = da.assign_coords(time=("time", prod_days))

        prod_days_unique = pd.to_datetime(da_day["time"].values).normalize().values

        if name not in self._time_scanned:
            missing = self.ref_days[~np.isin(self.ref_days, prod_days_unique)]
            extra = prod_days_unique[~np.isin(prod_days_unique, self.ref_days)]

            for t in missing:
                self.time_report_rows.append(
                    {
                        "product": name,
                        "status": "missing",
                        "date": str(pd.to_datetime(t).date()),
                    }
                )
            for t in extra:
                self.time_report_rows.append(
                    {
                        "product": name,
                        "status": "extra",
                        "date": str(pd.to_datetime(t).date()),
                    }
                )

            print(
                f"   [时间检查] {name}: 缺失日期={len(missing)}, 多余日期={len(extra)}"
            )
            self._time_scanned.add(name)

        da_fixed = da_day.reindex(time=self.ref_days)
        da_fixed = self._align_prod(da_fixed)

        ds.close()
        return da_fixed

    # ---------- 统一色标：更新范围 ----------
    def _update_metric_range(
        self,
        metric_name: str,
        da: xr.DataArray,
        symmetric: bool = False,
        p_low: float = 1.0,
        p_high: float = 99.0,
    ):
        """
        用该 2D 指标场更新 metric 的全局 vmin/vmax。
        默认只使用分位数范围，避免极端点影响。
        """
        vals = da.values.astype(float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return

        if symmetric:
            limit = np.nanpercentile(np.abs(vals), p_high)
            cur = self.metric_ranges.get(metric_name)
            if cur is None:
                self.metric_ranges[metric_name] = {"vmin": -limit, "vmax": limit}
            else:
                cur_limit = max(abs(cur["vmin"]), abs(cur["vmax"]))
                new_limit = max(cur_limit, limit)
                cur["vmin"], cur["vmax"] = -new_limit, new_limit
        else:
            lo = np.nanpercentile(vals, p_low)
            hi = np.nanpercentile(vals, p_high)
            cur = self.metric_ranges.get(metric_name)
            if cur is None:
                self.metric_ranges[metric_name] = {"vmin": lo, "vmax": hi}
            else:
                cur["vmin"] = min(cur["vmin"], lo)
                cur["vmax"] = max(cur["vmax"], hi)

    def save_metric_ranges(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metric_ranges, f, ensure_ascii=False, indent=2)
        print(f">>> 指标颜色条范围已保存到: {path}")

    def load_metric_ranges(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.metric_ranges = json.load(f)
        print(f">>> 已从 {path} 读取指标颜色条范围。")

    # ---------- 通用空间绘图 ----------
    def _plot_spatial(
        self,
        metric_name,
        da_dict,
        cmap,
        vmin=None,
        vmax=None,
        symmetric=False,
        suffix: str | None = None,
    ):
        if not da_dict:
            print(f"[警告] {metric_name} 没有任何产品结果可画，跳过。")
            return

        n_prods = len(da_dict)
        ncols = min(4, n_prods)
        nrows = math.ceil(n_prods / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols, 4 * nrows),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        if isinstance(axes, np.ndarray):
            axes_flat = axes.ravel()
        else:
            axes_flat = [axes]

        lon_min, lon_max = 70, 140
        lat_min, lat_max = 15, 55
        lon_ticks = np.arange(70, 141, 30)
        lat_ticks = np.arange(15, 56, 30)

        im = None
        for idx, (name, da) in enumerate(da_dict.items()):
            ax = axes_flat[idx]
            row = idx // ncols
            col = idx % ncols

            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

            im = da.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            ax.set_title(f"{name}", fontsize=12, fontweight="bold")

            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                xlocs=lon_ticks,
                ylocs=lat_ticks,
                draw_labels=True,
                linewidth=0.3,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (col == 0)
            gl.bottom_labels = (row == nrows - 1)
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}

        for j in range(n_prods, nrows * ncols):
            axes_flat[j].set_visible(False)

        if im is not None:
            cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04])
            cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
            cb.set_label(metric_name, fontsize=12)

        plt.suptitle(f"Spatial Distribution of {metric_name}", y=0.98, fontsize=16)
        plt.subplots_adjust(bottom=0.2)

        suffix_str = f"_{suffix}" if suffix else ""
        outname = f"Map_{metric_name}{suffix_str}.png"
        plt.savefig(outname, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"      已保存: {outname}")

    # ====================== 阶段 1：扫描指标范围 & 泰勒 ======================
    def scan_ranges_and_taylor(self):
        """
        第一阶段：只扫描所有产品的颜色条范围 + 泰勒图统计。
        不画任何图，只更新 self.metric_ranges 和 self.taylor_stats，
        最后写到 RANGE_JSON 和 TAYLOR_CSV。
        """
        if self.ref_da is None:
            raise RuntimeError("请先调用 load_data()！")

        print(">>> [阶段1] 扫描各指标颜色条范围 + 泰勒统计（不绘图）...")

        ref = self.ref_da

        # 参考量
        ref_mean = ref.mean(dim="time")
        ref_std_t = ref.std(dim="time")
        denom_nse = ((ref - ref_mean) ** 2).sum(dim="time")

        # 极端指标参考（这里仍用简单定义；icclim 单独处理）
        ref_rx1 = ref.max(dim="time")
        ref_r10 = (ref >= 10).sum(dim="time")
        ref_sdii = ref.where(ref >= 1).mean(dim="time")
        ref_q95 = ref.quantile(EXTREME_Q, dim="time")
        ref_r95p = ref.where(ref > ref_q95).sum(dim="time")

        # 参考区域平均时间序列（泰勒）
        ref_ts = ref.mean(dim=["lat", "lon"])
        std_ref_ts = ref_ts.std(dim="time").load().item()

        # 清空旧的范围和泰勒统计
        self.metric_ranges = {}
        self.taylor_stats = []

        for name, f in self.prod_files.items():
            print(f"   [scan] 产品: {name}")
            prod = self._load_prod_daily(name, f)

            diff = prod - ref

            # ===== 连续型指标范围 =====
            bias = diff.mean(dim="time")
            if self.china_mask2d is not None:
                bias = bias.where(self.china_mask2d)  # <<< NEW: 中国境内掩膜
            self._update_metric_range("Bias", bias, symmetric=True)
            del bias

            rmse = np.sqrt((diff ** 2).mean(dim="time"))
            if self.china_mask2d is not None:
                rmse = rmse.where(self.china_mask2d)
            self._update_metric_range("RMSE", rmse, symmetric=False)
            del rmse

            mae = np.abs(diff).mean(dim="time")
            if self.china_mask2d is not None:
                mae = mae.where(self.china_mask2d)
            self._update_metric_range("MAE", mae, symmetric=False)
            del mae

            # # ===== 极端指标 Bias 范围（简单定义）=====
            # rx1_bias = prod.max(dim="time") - ref_rx1
            # if self.china_mask2d is not None:
                # rx1_bias = rx1_bias.where(self.china_mask2d)
            # self._update_metric_range("Bias_Rx1day", rx1_bias, symmetric=True)
            # del rx1_bias

            # r10_bias = (prod >= 10).sum(dim="time") - ref_r10
            # if self.china_mask2d is not None:
                # r10_bias = r10_bias.where(self.china_mask2d)
            # self._update_metric_range("Bias_R10mm", r10_bias, symmetric=True)
            # del r10_bias

            # sdii_bias = prod.where(prod >= 1).mean(dim="time") - ref_sdii
            # if self.china_mask2d is not None:
                # sdii_bias = sdii_bias.where(self.china_mask2d)
            # self._update_metric_range("Bias_SDII", sdii_bias, symmetric=True)
            # del sdii_bias

            # prod_q95 = prod.quantile(EXTREME_Q, dim="time")
            # prod_r95p = prod.where(prod > prod_q95).sum(dim="time")
            # r95p_bias = prod_r95p - ref_r95p
            # if self.china_mask2d is not None:
                # r95p_bias = r95p_bias.where(self.china_mask2d)
            # self._update_metric_range("Bias_R95p", r95p_bias, symmetric=True)
            # del r95p_bias, prod_q95, prod_r95p

            # ===== 泰勒图统计 =====
            prod_ts = prod.mean(dim=["lat", "lon"])
            std_prod_ts = prod_ts.std(dim="time").load().item()
            cc_ts = xr.corr(ref_ts, prod_ts, dim="time").load().item()

            # 时间序列 RMSE
            ts_diff = prod_ts - ref_ts
            rmse_ts = np.sqrt((ts_diff ** 2).mean(dim="time")).load().item()

            # 用于泰勒图的 STD 一般是标准差比（归一化标准差）
            std_norm_ts = std_prod_ts / (std_ref_ts + 1e-12)

            self.taylor_stats.append(
                {
                    "name": name,
                    "STD": std_norm_ts,  # 泰勒图半径
                    "CC": cc_ts,         # 泰勒图角度
                    "RMSE": rmse_ts,     # 额外输出
                }
            )

            del prod, diff, prod_ts, ts_diff
            gc.collect()

        # 扫描结束，输出范围 & 泰勒表
        self.save_metric_ranges(RANGE_JSON)

        if self.taylor_stats:
            df_t = pd.DataFrame(self.taylor_stats)
            df_t.to_csv(TAYLOR_CSV, index=False)
            print(f">>> 泰勒图统计已保存到: {TAYLOR_CSV}")

        print(">>> [阶段1] 扫描完成。\n")

    # ====================== 阶段 2：固定色标逐产品绘图 ======================
    def run_metrics_per_product_with_fixed_ranges(self):
        """
        第二阶段：基于 self.metric_ranges 中的颜色条范围，逐产品计算指标并绘图。
        同时按 7 个自然区 & 季节汇总区域平均指标（输出 CSV）。
        """
        if self.ref_da is None:
            raise RuntimeError("请先调用 load_data()！")
        if not self.metric_ranges:
            raise RuntimeError("self.metric_ranges 为空，请先运行 scan_ranges_and_taylor() 并加载 JSON！")

        print(">>> [阶段2] 使用固定颜色条范围逐产品绘图，并按区域/季节输出统计...")

        ref = self.ref_da

        ref_mean = ref.mean(dim="time")
        ref_std_t = ref.std(dim="time")
        denom_nse = ((ref - ref_mean) ** 2).sum(dim="time")

        obs_hit = ref >= RAIN_THRESHOLD
        obs_miss = ref < RAIN_THRESHOLD
        eps = 1e-6

        # 极端指标参考（简单定义；icclim 另算）
        # ref_rx1 = ref.max(dim="time")
        # ref_r10 = (ref >= 10).sum(dim="time")
        # ref_sdii = ref.where(ref >= 1).mean(dim="time")
        # ref_q95 = ref.quantile(EXTREME_Q, dim="time")
        # ref_r95p = ref.where(ref > ref_q95).sum(dim="time")

        self.region_stats_rows = []  # 清空旧表

        for name, f in self.prod_files.items():
            print(f"\n=== 产品 {name} ===")
            prod = self._load_prod_daily(name, f)

            diff = prod - ref

            # ---------- 连续指标（全年） ----------
            bias = diff.mean(dim="time").load()
            if self.china_mask2d is not None:
                bias = bias.where(self.china_mask2d)
            r = self.metric_ranges.get("Bias", None)
            self._plot_spatial(
                "Bias", {name: bias}, "RdBu",
                vmin=r["vmin"] if r else None,
                vmax=r["vmax"] if r else None,
                suffix=name,
            )
            self._aggregate_regions(name, "Bias", "ALL", bias)  # <<< NEW: 区域平均
            del bias

            rmse = np.sqrt((diff ** 2).mean(dim="time")).load()
            if self.china_mask2d is not None:
                rmse = rmse.where(self.china_mask2d)
            r = self.metric_ranges.get("RMSE", None)
            self._plot_spatial(
                "RMSE", {name: rmse}, "magma_r",
                vmin=r["vmin"] if r else 0,
                vmax=r["vmax"] if r else None,
                suffix=name,
            )
            self._aggregate_regions(name, "RMSE", "ALL", rmse)
            del rmse

            mae = np.abs(diff).mean(dim="time").load()
            if self.china_mask2d is not None:
                mae = mae.where(self.china_mask2d)
            r = self.metric_ranges.get("MAE", None)
            self._plot_spatial(
                "MAE", {name: mae}, "magma_r",
                vmin=r["vmin"] if r else 0,
                vmax=r["vmax"] if r else None,
                suffix=name,
            )
            self._aggregate_regions(name, "MAE", "ALL", mae)
            del mae

            cc = xr.corr(ref, prod, dim="time").load()
            if self.china_mask2d is not None:
                cc = cc.where(self.china_mask2d)
            self._plot_spatial("CC", {name: cc}, "Spectral_r",
                               vmin=0, vmax=1, suffix=name)
            self._aggregate_regions(name, "CC", "ALL", cc)
            del cc

            num_nse = ((diff) ** 2).sum(dim="time")
            nse = (1 - (num_nse / denom_nse)).where(denom_nse > 1e-4).load()
            if self.china_mask2d is not None:
                nse = nse.where(self.china_mask2d)
            self._plot_spatial("NSE", {name: nse}, "RdYlGn",
                               vmin=-1, vmax=1, suffix=name)
            self._aggregate_regions(name, "NSE", "ALL", nse)
            del nse, num_nse

            r_corr = xr.corr(ref, prod, dim="time")
            alpha = prod.std(dim="time") / ref_std_t
            beta = prod.mean(dim="time") / ref_mean
            kge = (1 - np.sqrt((r_corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)).load()
            if self.china_mask2d is not None:
                kge = kge.where(self.china_mask2d)
            self._plot_spatial("KGE", {name: kge}, "RdYlGn",
                               vmin=-1, vmax=1, suffix=name)
            self._aggregate_regions(name, "KGE", "ALL", kge)
            del kge, r_corr, alpha, beta, diff

            gc.collect()

            # ---------- 分类指标（全年） ----------
            pred_hit = prod >= RAIN_THRESHOLD
            pred_miss = prod < RAIN_THRESHOLD

            H = (obs_hit & pred_hit).sum(dim="time")
            M = (obs_hit & pred_miss).sum(dim="time")
            F = (obs_miss & pred_hit).sum(dim="time")
            CN = (obs_miss & pred_miss).sum(dim="time")

            pod = (H / (H + M + eps)).load()
            if self.china_mask2d is not None:
                pod = pod.where(self.china_mask2d)
            self._plot_spatial("POD", {name: pod}, "Blues",
                               vmin=0, vmax=1, suffix=name)
            self._aggregate_regions(name, "POD", "ALL", pod)
            del pod

            far = (F / (H + F + eps)).load()
            if self.china_mask2d is not None:
                far = far.where(self.china_mask2d)
            self._plot_spatial("FAR", {name: far}, "Reds",
                               vmin=0, vmax=1, suffix=name)
            self._aggregate_regions(name, "FAR", "ALL", far)
            del far

            csi = (H / (H + M + F + eps)).load()
            if self.china_mask2d is not None:
                csi = csi.where(self.china_mask2d)
            self._plot_spatial("CSI", {name: csi}, "Greens",
                               vmin=0, vmax=1, suffix=name)
            self._aggregate_regions(name, "CSI", "ALL", csi)
            del csi

            fbias = ((H + F) / (H + M + eps)).load()
            if self.china_mask2d is not None:
                fbias = fbias.where(self.china_mask2d)
            self._plot_spatial("FBIAS", {name: fbias}, "BrBG",
                               vmin=0, vmax=2, suffix=name)
            self._aggregate_regions(name, "FBIAS", "ALL", fbias)
            del fbias

            num_h = 2 * (H * CN - M * F)
            den_h = (H + M) * (M + CN) + (H + F) * (F + CN)
            hss = (num_h / (den_h + eps)).load()
            if self.china_mask2d is not None:
                hss = hss.where(self.china_mask2d)
            self._plot_spatial("HSS", {name: hss}, "RdYlBu_r",
                               vmin=-1, vmax=1, suffix=name)
            self._aggregate_regions(name, "HSS", "ALL", hss)
            del hss, num_h, den_h, H, M, F, CN, pred_hit, pred_miss

            gc.collect()

            # # ---------- 极端指标 Bias（简单定义，非 icclim） ----------
            # rx1_bias = (prod.max(dim="time") - ref_rx1).load()
            # if self.china_mask2d is not None:
                # rx1_bias = rx1_bias.where(self.china_mask2d)
            # r = self.metric_ranges.get("Bias_Rx1day", None)
            # self._plot_spatial(
                # "Bias_Rx1day", {name: rx1_bias}, "RdBu",
                # vmin=r["vmin"] if r else None,
                # vmax=r["vmax"] if r else None,
                # suffix=name,
            # )
            # self._aggregate_regions(name, "Bias_Rx1day", "ALL", rx1_bias)
            # del rx1_bias

            # r10_bias = ((prod >= 10).sum(dim="time") - ref_r10).load()
            # if self.china_mask2d is not None:
                # r10_bias = r10_bias.where(self.china_mask2d)
            # r = self.metric_ranges.get("Bias_R10mm", None)
            # self._plot_spatial(
                # "Bias_R10mm", {name: r10_bias}, "RdBu",
                # vmin=r["vmin"] if r else None,
                # vmax=r["vmax"] if r else None,
                # suffix=name,
            # )
            # self._aggregate_regions(name, "Bias_R10mm", "ALL", r10_bias)
            # del r10_bias

            # sdii_bias = (prod.where(prod >= 1).mean(dim="time") - ref_sdii).load()
            # if self.china_mask2d is not None:
                # sdii_bias = sdii_bias.where(self.china_mask2d)
            # r = self.metric_ranges.get("Bias_SDII", None)
            # self._plot_spatial(
                # "Bias_SDII", {name: sdii_bias}, "RdBu",
                # vmin=r["vmin"] if r else None,
                # vmax=r["vmax"] if r else None,
                # suffix=name,
            # )
            # self._aggregate_regions(name, "Bias_SDII", "ALL", sdii_bias)
            # del sdii_bias

            # prod_q95 = prod.quantile(EXTREME_Q, dim="time")
            # prod_r95p = prod.where(prod > prod_q95).sum(dim="time")
            # r95p_bias = (prod_r95p - ref_r95p).load()
            # if self.china_mask2d is not None:
                # r95p_bias = r95p_bias.where(self.china_mask2d)
            # r = self.metric_ranges.get("Bias_R95p", None)
            # self._plot_spatial(
                # "Bias_R95p", {name: r95p_bias}, "RdBu",
                # vmin=r["vmin"] if r else None,
                # vmax=r["vmax"] if r else None,
                # suffix=name,
            # )
            # self._aggregate_regions(name, "Bias_R95p", "ALL", r95p_bias)
            # del r95p_bias, prod_q95, prod_r95p

            # gc.collect()

            # ---------- 分季节的区域平均指标（不画季节空间图，只算区域均值） ----------
            self._compute_seasonal_region_metrics(name, ref, prod)

            # # ---------- 可选：用 icclim 计算极端指数并按区域/季节汇总 ----------
            # if USE_ICCLIM:
                # try:
                    # self._compute_etccdi_by_region_and_season_for_product(name, prod)
                # except Exception as e:
                    # print(f"[警告] 产品 {name} 计算 icclim 极端指数失败：{e}")

        # 时间不匹配报告
        if self.time_report_rows:
            df = pd.DataFrame(self.time_report_rows)
            df = df.sort_values(by=["product", "status", "date"])
            report_path = os.path.join(self.folder, "time_mismatch_report_daily.csv")
            df.to_csv(report_path, index=False)
            print(f"\n>>> 日尺度时间不匹配报告已输出: {report_path}")
        else:
            print("\n>>> 所有产品在日尺度时间轴上与参考完全一致，无缺失/多余日期。")

        # 输出区域/季节统计表
        if self.region_stats_rows:
            df_reg = pd.DataFrame(self.region_stats_rows)
            df_reg.to_csv(REGION_METRIC_CSV, index=False)
            print(f">>> 区域/季节指标统计已输出: {REGION_METRIC_CSV}")

        print("\n>>> [阶段2] 所有产品指标绘图 & 区域/季节统计完成。")

    # ---------- 区域平均（全年 / 季节） ----------
    def _aggregate_regions(self, product_name: str, metric_name: str,
                           season: str, da2d: xr.DataArray):
        """
        对给定 2D 场，在 7 个自然区上求区域平均，记录到 self.region_stats_rows。
        """
        if self.region_mask_3d is None or not self.region_names:
            return

        for i, reg_name in enumerate(self.region_names):
            mask_reg = self.region_mask_3d.isel(region=i)
            reg_vals = da2d.where(mask_reg)
            mean_val = float(reg_vals.mean().values) if np.isfinite(reg_vals.mean().values) else np.nan
            self.region_stats_rows.append(
                {
                    "product": product_name,
                    "region": reg_name,
                    "season": season,
                    "metric": metric_name,
                    "value": mean_val,
                }
            )

    def _compute_seasonal_region_metrics(self, product_name: str,
                                         ref: xr.DataArray, prod: xr.DataArray):
        """
        分季节（xarray 自带的 time.season: DJF/MAM/JJA/SON）计算
        Bias / RMSE / CC / KGE 等的区域平均（不画图）。
        """
        seasons = ["DJF", "MAM", "JJA", "SON"]
        eps = 1e-6

        for s in seasons:
            sel = ref["time"].dt.season == s
            if sel.sum().item() == 0:
                continue

            ref_s = ref.sel(time=sel)
            prod_s = prod.sel(time=sel)

            diff_s = prod_s - ref_s

            bias_s = diff_s.mean(dim="time")
            if self.china_mask2d is not None:
                bias_s = bias_s.where(self.china_mask2d)
            self._aggregate_regions(product_name, "Bias", s, bias_s)

            rmse_s = np.sqrt((diff_s ** 2).mean(dim="time"))
            if self.china_mask2d is not None:
                rmse_s = rmse_s.where(self.china_mask2d)
            self._aggregate_regions(product_name, "RMSE", s, rmse_s)

            cc_s = xr.corr(ref_s, prod_s, dim="time")
            if self.china_mask2d is not None:
                cc_s = cc_s.where(self.china_mask2d)
            self._aggregate_regions(product_name, "CC", s, cc_s)

            ref_mean_s = ref_s.mean(dim="time")
            ref_std_s = ref_s.std(dim="time")
            r_corr_s = xr.corr(ref_s, prod_s, dim="time")
            alpha_s = prod_s.std(dim="time") / ref_std_s
            beta_s = prod_s.mean(dim="time") / ref_mean_s
            kge_s = (1 - np.sqrt((r_corr_s - 1) ** 2 + (alpha_s - 1) ** 2 + (beta_s - 1) ** 2))
            if self.china_mask2d is not None:
                kge_s = kge_s.where(self.china_mask2d)
            self._aggregate_regions(product_name, "KGE", s, kge_s)

            # 也可以在这里按季节算 POD/CSI/FBIAS/HSS，有需要可以再补

    # ====================== shapefile 区域掩膜 ======================
        # ====================== shapefile 区域掩膜 ======================
    def _build_region_masks(self):
        """
        读取 shapefile (7 个自然区)，构建：
        - china_mask2d: (lat, lon) bool, 中国境内
        - region_mask_3d: (region, lat, lon) bool
        """
        if not os.path.exists(SHP_FILE):
            print(f"[警告] shp 文件 {SHP_FILE} 不存在，跳过区域掩膜，使用全域网格。")
            return

        try:
            import geopandas as gpd
            import regionmask
        except ImportError:
            print("[警告] 需要安装 geopandas 和 regionmask 才能使用 shapefile 掩膜。")
            return

        print(f">>> 正在读取 shapefile: {SHP_FILE}")
        gdf = gpd.read_file(SHP_FILE)

        print(f"   原始 CRS: {gdf.crs}")

        if SHP_REGION_FIELD not in gdf.columns:
            raise KeyError(f"shp 文件中未找到字段 {SHP_REGION_FIELD}！")

        # 你的 shp 是 Krasovsky_1940_Albers（投影坐标），要重投影到经纬度 WGS84
        # 这样才能和 NetCDF 的 lon/lat 对上
        gdf = gdf.to_crs(epsg=4326)
        print(f"   已重投影到 WGS84: {gdf.crs}")

        outlines = [geom for geom in gdf.geometry]
        names = list(gdf[SHP_REGION_FIELD].astype(str))
        numbers = list(range(len(names)))

        regions = regionmask.Regions(
            outlines=outlines,
            numbers=numbers,
            names=names,
            abbrevs=names,
            name="China7",
        )

        lon = self.ref_da["lon"].values
        lat = self.ref_da["lat"].values

        # 2D mask: value = region number 或 NaN
        mask2d = regions.mask(lon, lat)  # dims: lat, lon
        china_mask = mask2d.notnull()

        # 3D mask: (region, lat, lon) bool
        mask3d = regions.mask_3D(lon, lat)  # True/False

        self.china_mask2d = china_mask
        self.region_mask_3d = mask3d.astype(bool)
        self.region_names = names

        print(f">>> 区域掩膜构建完成，共 {len(names)} 个区域：{names}")


    def _compute_ref_etccdi_icclim(self):
        """
        用 icclim 对参考数据计算若干 ETCCDI 指数：
        - 年尺度（slice_mode="year"），再做多年平均，得到 2D climatology 场
        - 月尺度（slice_mode="month"），保留 time 维，后面按季节聚合
        """
        import icclim

        print(">>> [icclim] 正在计算参考数据的 ETCCDI 指数（年 + 月）...")

        self.ref_etccdi_annual = {}
        self.ref_etccdi_monthly = {}

        da = self.ref_da  # (time, lat, lon)

        for idx_name in ICCLIM_INDICES:
            print(f"   [icclim] 参考 {idx_name} ...")

            # 1) 年尺度（每年一个值），再多年平均
            func_year = getattr(icclim, idx_name)
            idx_year = func_year(
                in_files=da,
                var_name=self.var_name,
                slice_mode="year",   # 默认就是 year，这里写清楚
                out_file=None,
            )
            if "time" in idx_year.dims:
                idx_clim = idx_year.mean(dim="time")
            else:
                idx_clim = idx_year
            self.ref_etccdi_annual[idx_name] = idx_clim

            # 2) 月尺度（每月一个值），保留 time 维；后面按季节聚合
            idx_month = func_year(
                in_files=da,
                var_name=self.var_name,
                slice_mode="month",
                out_file=None,
            )
            self.ref_etccdi_monthly[idx_name] = idx_month

        print(">>> [icclim] 参考 ETCCDI 指数计算完成。")


    def _compute_etccdi_by_region_and_season_for_product(self, product_name: str,
                                                         prod: xr.DataArray):
        """
        用 icclim 对单个产品 prod 计算 ETCCDI 指标，并：
        - 年尺度：与参考 annual climatology 做 Bias，按 7 区平均（season="ALL_ETCCDI"）
        - 月尺度：按季节（DJF/MAM/JJA/SON）聚合后，与参考月尺度指数做 Bias，再按 7 区平均
        结果直接写入 self.region_stats_rows（和其他指标共用一个 CSV）。
        """
        import icclim

        if not self.ref_etccdi_annual or not self.ref_etccdi_monthly:
            print("   [icclim] 参考 ETCCDI 尚未计算，跳过产品 ETCCDI 评估。")
            return
        if self.region_mask_3d is None:
            print("   [icclim] 未构建区域掩膜，跳过产品 ETCCDI 评估。")
            return

        for idx_name in ICCLIM_INDICES:
            print(f"   [icclim] 产品 {product_name} 计算 {idx_name} ...")

            func = getattr(icclim, idx_name)

            # -------- 1) 年尺度：多年平均 Bias（季节标记为 ALL_ETCCDI） --------
            idx_year_prod = func(
                in_files=prod,
                var_name=self.var_name,
                slice_mode="year",
                out_file=None,
            )
            if "time" in idx_year_prod.dims:
                idx_clim_prod = idx_year_prod.mean(dim="time")
            else:
                idx_clim_prod = idx_year_prod

            idx_ref_clim = self.ref_etccdi_annual.get(idx_name)
            if idx_ref_clim is None:
                continue

            bias_annual = (idx_clim_prod - idx_ref_clim)
            if self.china_mask2d is not None:
                bias_annual = bias_annual.where(self.china_mask2d)

            for i, reg_name in enumerate(self.region_names):
                mask_reg = self.region_mask_3d.isel(region=i)
                reg_vals = bias_annual.where(mask_reg)
                mean_val = float(reg_vals.mean().values) if np.isfinite(reg_vals.mean().values) else np.nan
                self.region_stats_rows.append(
                    {
                        "product": product_name,
                        "region": reg_name,
                        "season": "ALL_ETCCDI",          # 年尺度
                        "metric": f"Bias_{idx_name}",     # 比如 Bias_r10mm
                        "value": mean_val,
                    }
                )

            # -------- 2) 月尺度：按季节聚合 Bias --------
            idx_month_prod = func(
                in_files=prod,
                var_name=self.var_name,
                slice_mode="month",
                out_file=None,
            )
            idx_month_ref = self.ref_etccdi_monthly.get(idx_name)
            if idx_month_ref is None or "time" not in idx_month_prod.dims:
                continue

            # xarray 的 dt.season 可以直接得到 DJF/MAM/JJA/SON
            season_prod = idx_month_prod.groupby("time.season").mean(dim="time")
            season_ref = idx_month_ref.groupby("time.season").mean(dim="time")

            # 两边对齐季节标签
            common_seasons = np.intersect1d(season_prod["season"].values,
                                            season_ref["season"].values)

            for s in common_seasons:
                s = str(s)
                bias_s = season_prod.sel(season=s) - season_ref.sel(season=s)
                if self.china_mask2d is not None:
                    bias_s = bias_s.where(self.china_mask2d)

                for i, reg_name in enumerate(self.region_names):
                    mask_reg = self.region_mask_3d.isel(region=i)
                    reg_vals = bias_s.where(mask_reg)
                    mean_val = float(reg_vals.mean().values) if np.isfinite(reg_vals.mean().values) else np.nan
                    self.region_stats_rows.append(
                        {
                            "product": product_name,
                            "region": reg_name,
                            "season": s,                  # DJF / MAM / JJA / SON
                            "metric": f"Bias_{idx_name}", # 同样是 Bias_r10mm，只是 season 不同
                            "value": mean_val,
                        }
                    )


    # ====================== 散点 & 季节箱线（原有） ======================
    def _scatter_density_single(self, name: str, prod: xr.DataArray, n_sample=SCATTER_MAX_POINTS):
        print(f"   [散点] {name}")
        ref = self.ref_da

        time_dim = "time"
        spatial_dims = [d for d in ref.dims if d != time_dim]

        ref_stack = ref.stack(points=[time_dim] + spatial_dims)
        n_points = ref_stack.sizes["points"]

        if n_points > n_sample:
            idx = np.random.choice(n_points, n_sample, replace=False)
        else:
            idx = np.arange(n_points)

        ref_sample = ref_stack.isel(points=idx).values
        prod_stack = prod.stack(points=[time_dim] + spatial_dims)
        prod_sample = prod_stack.isel(points=idx).values

        mask = np.isfinite(ref_sample) & np.isfinite(prod_sample)
        x = ref_sample[mask]
        y = prod_sample[mask]

        if x.size == 0:
            print(f"      [警告] {name} 有效样本数为 0，跳过散点图。")
            return

        fig, ax = plt.subplots(figsize=(5, 5))
        hb = ax.hexbin(x, y, gridsize=50, cmap="Spectral_r", mincnt=1, bins="log")

        lim = max(x.max(), y.max())
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{name} vs {self.ref_prefix}")
        ax.set_xlabel(f"{self.ref_prefix} (mm)")
        ax.set_ylabel(f"{name} (mm)")

        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Count (log)")

        outname = f"Scatter_Density_{name}.png"
        plt.tight_layout()
        plt.savefig(outname, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"      已保存: {outname}")

    def _seasonal_box_single(self, name: str, prod: xr.DataArray):
        print(f"   [季节箱线图] {name}")
        ts_ref = self.ref_da.mean(dim=["lat", "lon"]).to_dataframe(name="Precip")
        ts_ref["Product"] = self.ref_prefix
        ts_ref["Season"] = [self._get_season(m) for m in ts_ref.index.month]

        ts = prod.mean(dim=["lat", "lon"]).to_dataframe(name="Precip")
        ts["Product"] = name
        ts["Season"] = [self._get_season(m) for m in ts.index.month]

        df_all = pd.concat([ts_ref, ts])

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_all,
            x="Season",
            y="Precip",
            hue="Product",
            order=["Spring", "Summer", "Autumn", "Winter"],
            palette="Set2",
        )
        plt.title(f"Seasonal Distribution of Area-Averaged Precipitation - {name}")
        plt.ylabel("Precipitation (mm/day)")

        outname = f"Seasonal_Boxplot_{name}.png"
        plt.savefig(outname, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"      已保存: {outname}")

    # ====================== 综合泰勒图 ======================
    def plot_taylor_all(self):
        print(">>> 4. 绘制综合泰勒图...")

        if not self.taylor_stats:
            print("   [警告] 没有泰勒统计数据，请先运行 scan_ranges_and_taylor()")
            return

        fig = plt.figure(figsize=(9, 9))
        dia = TaylorDiagram(1.0, fig=fig, label=self.ref_prefix)

        colors = plt.cm.tab20(np.linspace(0, 1, len(self.taylor_stats)))
        markers = ["o", "s", "D", "^", "v", "P", "X", "h", "*", "+", "x"]

        for idx, item in enumerate(self.taylor_stats):
            dia.add_sample(
                item["STD"],
                item["CC"],
                marker=markers[idx % len(markers)],
                color=colors[idx],
                label=item["name"],
                ms=8,
                ls="",
            )

        plt.legend(bbox_to_anchor=(1.1, 1.0), loc="upper left")
        plt.title("Taylor Diagram (Normalized, All Products)", y=1.05)

        outname = "Taylor_Diagram_All.png"
        plt.savefig(outname, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"   已保存: {outname}")

    @staticmethod
    def _get_season(month: int) -> str:
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Winter"


# ================= 泰勒图辅助类 =================
class TaylorDiagram(object):
    def __init__(self, refstd, fig=None, rect=111, label="_"):
        self.refstd = refstd

        tr = PolarAxes.PolarTransform()

        r_locs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
        t_locs = np.arccos(r_locs)
        gl1 = FixedLocator(t_locs)
        tf1 = DictFormatter(dict(zip(t_locs, map(str, r_locs))))

        self.smax = 1.5 * self.refstd
        g2 = FixedLocator(np.linspace(0, self.smax, 5))

        grid_helper = floating_axes.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, 0, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
            grid_locator2=g2,
        )

        if fig is None:
            fig = plt.figure()

        ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
        fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation (Normalized)")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)

        ax.grid(True, linestyle="--", alpha=0.5)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        self.ax.plot([0], [self.refstd], "k*", ls="", ms=10, label=label)

        rs, ts = np.meshgrid(
            np.linspace(0, self.smax, 100),
            np.linspace(0, np.pi / 2, 100),
        )
        rms = np.sqrt(self.refstd**2 + rs**2 - 2 * self.refstd * rs * np.cos(ts))
        self.ax.contour(ts, rs, rms, 5, colors="k", alpha=0.4, linestyles="--")

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        theta = np.arccos(np.clip(corrcoef, -1, 1))
        self.ax.plot(theta, stddev, *args, **kwargs)


# ================= 主程序 =================
if __name__ == "__main__":
    app = PreAlignedEvaluator(DATA_FOLDER, REF_NAME_PREFIX, VAR_NAME)
    try:
        app.load_data()

        if MODE == "scan":
            # 阶段 1：只扫描范围 + 泰勒统计
            app.scan_ranges_and_taylor()

        elif MODE == "plot":
            # 阶段 2：读 JSON，按固定色带画图
            app.load_metric_ranges(RANGE_JSON)
            app.run_metrics_per_product_with_fixed_ranges()
            # 可选：综合泰勒图（如果你想在 plot 阶段画）
            # app.plot_taylor_all()

        else:
            # MODE 只能是 "scan" 或 "plot"
            raise ValueError(f"未知 MODE = {MODE}")

        print("\n>>> 全部任务完成。")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"发生错误: {e}")

