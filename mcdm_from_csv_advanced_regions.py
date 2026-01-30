# -*- coding: utf-8 -*-
"""
脚本 C（分区版）：基于“分区×产品”长表 CSV 执行 MCDM 2.0
  - 输入：Final_MCDM_Input_Ranking_7regions.csv（每行=Product+Region）
  - 输出：对 7 个分区分别输出
      * MCDM_Weights.csv
      * Ensemble_Raw_Scores.csv
      * Final_Ensemble_Ranking.csv
      * Ensemble_Ranking_Heatmap.png
      * Sensitivity_Rank_Stats.csv
    以及总汇总：
      * Summary_TopN_ByRegion.csv

兼容：多指标（含 Err_* 极端误差指标）
注意：本脚本按分区独立排序（不跨分区混合），避免不同气候区“可比性”问题。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn 仅用于热力图；若环境无 seaborn，可把 plot_heatmap 里 seaborn 去掉
import seaborn as sns

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 用户配置 ======================

DATA_FOLDER = "/home/ud202380664/PRE_MERGE/TIMEFIX"
TIMEFIX_FOLDER = os.path.join(DATA_FOLDER, "Finish")

# 新的“分区×产品”长表（由你改好的脚本 B(v2-分区版) 生成）
INPUT_CSV = os.path.join(TIMEFIX_FOLDER, "Final_MCDM_Input_Ranking_7regions.csv")

OUTPUT_DIR = os.path.join(TIMEFIX_FOLDER, "MCDM_RESULTS_7regions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 方法组合权重
METHOD_WEIGHTS = {"TOPSIS": 0.4, "VIKOR": 0.3, "GRA": 0.3}

# 极端指标非可补偿设置（在“归一化后分值”上做门槛）
# 这里的 pat 会匹配列名，Err_R95p / Err_RX1day / Err_SDII 都会命中 r95p/rx1day/sdii
EXTREME_PATTERNS = ["r95p", "rx1day", "sdii"]
EXTREME_MIN_SCORE = 0.2
EXTREME_PENALTY = 1.0

# 高相关去冗余阈值
CORR_PRUNE_THRESHOLD = 0.98

# 敏感性分析参数
SENS_N_RUNS = 200
SENS_WEIGHT_NOISE = 0.1
SENS_METHOD_WEIGHT_NOISE = 0.1

# 是否把某些“非性能指标/元数据列”排除出 MCDM
DROP_META_COLS = True
META_COLS = ["N_Days_Overlap", "Region_ID"]  # 可按需增减

# 汇总输出 TopN
SUMMARY_TOPN = 5

# =====================================================


class EnsembleMCDMAdvanced:
    """
    升级版 MCDM：支持
      1) 排名归一化
      2) 指标方向 & 分组自动识别
      3) 高相关指标去冗余
      4) 极端指标非可补偿门槛
      5) 多方法加权组合
      6) 敏感性分析
    """

    def __init__(
        self,
        df_raw: pd.DataFrame,
        output_dir=".",
        method_weights=None,
        corr_prune_threshold=0.98,
        extreme_patterns=None,
        extreme_min_score=0.2,
        extreme_penalty=1.0,
    ):
        self.df_raw = df_raw.copy()
        self.products = self.df_raw.index
        self.indicators = self.df_raw.columns
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(
            f"载入 MCDM 原始指标: {len(self.products)} 个产品, {len(self.indicators)} 个指标"
        )

        if method_weights is None:
            method_weights = {"TOPSIS": 0.4, "VIKOR": 0.3, "GRA": 0.3}
        mw = np.array(
            [
                method_weights.get("TOPSIS", 0.0),
                method_weights.get("VIKOR", 0.0),
                method_weights.get("GRA", 0.0),
            ]
        )
        mw = mw / (mw.sum() + 1e-12)
        self.method_weights = mw

        self.extreme_patterns = (
            [p.lower() for p in extreme_patterns]
            if extreme_patterns is not None
            else ["r95p", "rx1day", "sdii"]
        )
        self.extreme_min_score = extreme_min_score
        self.extreme_penalty = extreme_penalty

        self.corr_prune_threshold = corr_prune_threshold

        self.direction_map, self.group_map = self._default_indicator_meta()
        self.directions = None
        self.groups = None

        self.df_norm = None
        self.X = None
        self.w_entropy = None
        self.w_critic = None
        self.final_weights = None
        self.results = None
        self.extreme_flags = None

    @staticmethod
    def _default_indicator_meta():
        # 方向：+1 越大越好；-1 越小越好（随后会统一为“越大越好”）
        direction_map = {
            "rmse": -1,
            "mae": -1,
            "bias": -1,
            "err_": -1,  # 关键：Err_* 作为误差项，越小越好
            "cc": 1,
            "corr": 1,
            "kge": 1,
            "nse": 1,
            "pod": 1,
            "csi": 1,
            "hss": 1,
            "far": -1,
            "r95p": 1,     # 兼容你旧表里若存在非 Err_ 的极端指标
            "rx1day": 1,
            "sdii": 1,
        }

        group_map = {
            "rmse": "Error",
            "mae": "Error",
            "bias": "Error",
            "err_": "Extremes",   # Err_* 统一归到 Extremes
            "cc": "Skill",
            "corr": "Skill",
            "kge": "Skill",
            "nse": "Skill",
            "pod": "Detect",
            "csi": "Detect",
            "hss": "Detect",
            "far": "Detect",
            "r95p": "Extremes",
            "rx1day": "Extremes",
            "sdii": "Extremes",
            "n_days_overlap": "Meta",
            "region_id": "Meta",
        }

        return direction_map, group_map

    def _infer_directions_and_groups(self):
        dirs, groups = [], []
        for col in self.indicators:
            name = col.lower()
            d = 1
            g = "Other"
            for pat, val in self.direction_map.items():
                if pat in name:
                    d = val
                    break
            for pat, grp in self.group_map.items():
                if pat in name:
                    g = grp
                    break
            dirs.append(d)
            groups.append(g)

        print("\n指标方向与分组推断:")
        for col, d, g in zip(self.indicators, dirs, groups):
            print(f"  {col:22s} -> dir={d:+d}, group={g}")

        self.directions = np.array(dirs, dtype=float)
        self.groups = np.array(groups)

    def preprocess(
        self,
        abs_bias=True,
        fillna="median",
        drop_constant=True,
        prune_high_corr=True,
    ):
        df = self.df_raw.copy()

        # 误差型指标：默认取绝对值（Bias、Err_*），目标是“越接近 0 越好”
        if abs_bias:
            for col in df.columns:
                lc = col.lower()
                if "bias" in lc or "err_" in lc:
                    df[col] = df[col].abs()

        if fillna is not None:
            for col in df.columns:
                s = df[col]
                if fillna == "median":
                    val = s.median()
                elif fillna == "mean":
                    val = s.mean()
                else:
                    val = None
                if val is not None:
                    df[col] = s.fillna(val)

        valid_cols = []
        for col in df.columns:
            s = df[col]
            if s.isna().all():
                print(f"[警告] 指标 {col} 全为 NaN，已删除。")
                continue
            if drop_constant and np.isclose(s.max(), s.min()):
                print(f"[警告] 指标 {col} 为常数列，已删除。")
                continue
            valid_cols.append(col)

        df = df[valid_cols]
        self.indicators = df.columns

        self._infer_directions_and_groups()

        # 方向统一 -> 全部转换为“越大越好”
        df_oriented = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for i, col in enumerate(df.columns):
            x = df[col].values.astype(float)
            if self.directions[i] == 1:
                df_oriented[col] = x
            else:
                df_oriented[col] = -x

        # 排名归一化到 [0,1]
        df_norm = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        n_prod = len(df.index)
        for col in df.columns:
            s = pd.Series(df_oriented[col].values, index=df.index)
            ranks = s.rank(method="average")  # ascending=True：最小=1，最大=n
            if n_prod > 1:
                x_norm = (ranks - 1) / (n_prod - 1)  # 最差=0，最好=1
            else:
                x_norm = np.ones_like(ranks.values)
            df_norm[col] = x_norm.values

        self.df_norm = df_norm
        self.X = df_norm.values.astype(float)

        print("\n>>> 预处理完成：所有指标已统一为“越大越好”，并用排名归一化到 [0,1].")

        # 高相关去冗余
        if prune_high_corr and df_norm.shape[1] > 1:
            corr = df_norm.corr()
            to_drop = set()
            cols = corr.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if (
                        abs(corr.iloc[i, j]) >= self.corr_prune_threshold
                        and cols[j] not in to_drop
                    ):
                        print(
                            f"[去冗余] {cols[i]} 与 {cols[j]} 相关系数={corr.iloc[i,j]:.3f}, 删除 {cols[j]}"
                        )
                        to_drop.add(cols[j])

            if to_drop:
                self.df_norm = self.df_norm.drop(columns=list(to_drop))
                self.indicators = self.df_norm.columns
                self.X = self.df_norm.values.astype(float)

        return self.df_norm

    def calc_game_theory_weights(self):
        print("\n>>> 计算博弈论组合权重 (Entropy + CRITIC)...")

        X = self.X
        m, n = X.shape

        col_sum = X.sum(axis=0) + 1e-12
        P = X / col_sum
        e = -1 / np.log(m) * np.nansum(P * np.log(P + 1e-12), axis=0)
        d_entropy = 1 - e
        w_entropy = d_entropy / (d_entropy.sum() + 1e-12)

        std_dev = np.nanstd(X, axis=0)
        df_norm = pd.DataFrame(X, columns=self.indicators)
        corr = df_norm.corr().values
        conflict = np.sum(1 - np.abs(corr), axis=0)
        c_critic = std_dev * conflict
        w_critic = c_critic / (c_critic.sum() + 1e-12)

        W = np.vstack([w_entropy, w_critic])
        A = W @ W.T
        B = np.diag(A)
        try:
            alphas = np.linalg.solve(A, B)
            alphas = alphas / (alphas.sum() + 1e-12)
        except Exception:
            print("[警告] 权重矩阵奇异，退化为 Entropy/CRITIC 各 0.5")
            alphas = np.array([0.5, 0.5])

        print(f"    组合系数: Entropy={alphas[0]:.3f}, CRITIC={alphas[1]:.3f}")

        final_w = alphas[0] * w_entropy + alphas[1] * w_critic
        final_w = final_w / (final_w.sum() + 1e-12)

        self.w_entropy = w_entropy
        self.w_critic = w_critic
        self.final_weights = final_w

        df_w = pd.DataFrame(
            {
                "Indicator": self.indicators,
                "w_Entropy": self.w_entropy,
                "w_CRITIC": self.w_critic,
                "w_Final": self.final_weights,
            }
        )
        out_w = os.path.join(self.output_dir, "MCDM_Weights.csv")
        df_w.to_csv(out_w, index=False)
        print(f"    指标权重已导出: {out_w}")

        return self.final_weights

    # --------- 三个方法 ---------

    def method_topsis(self, w=None):
        if w is None:
            w = self.final_weights
        Z = self.X * w
        z_plus = Z.max(axis=0)
        z_minus = Z.min(axis=0)
        d_plus = np.sqrt(((Z - z_plus) ** 2).sum(axis=1))
        d_minus = np.sqrt(((Z - z_minus) ** 2).sum(axis=1))
        score = d_minus / (d_plus + d_minus + 1e-12)
        return score

    def method_vikor(self, w=None, v=0.5):
        if w is None:
            w = self.final_weights
        S = np.sum(w * (1 - self.X), axis=1)
        R = np.max(w * (1 - self.X), axis=1)
        S_star, S_minus = S.min(), S.max()
        R_star, R_minus = R.min(), R.max()
        denom_S = (S_minus - S_star) if not np.isclose(S_minus, S_star) else 1.0
        denom_R = (R_minus - R_star) if not np.isclose(R_minus, R_star) else 1.0
        Q = v * (S - S_star) / denom_S + (1 - v) * (R - R_star) / denom_R
        return 1 - Q

    def method_gra(self, w=None, rho=0.5):
        if w is None:
            w = self.final_weights
        ref_seq = np.ones(self.X.shape[1])
        diff = np.abs(self.X - ref_seq)
        min_diff = diff.min()
        max_diff = diff.max()
        xi = (min_diff + rho * max_diff) / (diff + rho * max_diff + 1e-12)
        r_gra = np.dot(xi, w)
        return r_gra

    # --------- 非可补偿约束 ---------

    def _apply_noncompensatory(self, df_res_scores):
        X = self.X
        cols = self.indicators
        m = X.shape[0]

        extreme_cols_idx = []
        for i, col in enumerate(cols):
            lc = col.lower()
            if any(pat in lc for pat in self.extreme_patterns):
                extreme_cols_idx.append(i)

        if not extreme_cols_idx:
            print("    [提示] 未识别到极端指标列，跳过非可补偿约束。")
            flags = np.zeros(m, dtype=bool)
        else:
            X_ext = X[:, extreme_cols_idx]
            min_scores = X_ext.min(axis=1)
            flags = min_scores < self.extreme_min_score
            print(
                f"    非可补偿检查：{flags.sum()} 个产品在极端指标上未达标 (阈值={self.extreme_min_score})"
            )

        self.extreme_flags = pd.Series(flags, index=self.products, name="ExtremeFail")
        df_res_scores["Flag_ExtremeFail"] = flags

        if self.extreme_penalty > 0 and "Weighted_Rank" in df_res_scores.columns:
            df_res_scores["Final_Rank"] = df_res_scores["Weighted_Rank"] + flags.astype(
                float
            ) * self.extreme_penalty
        else:
            df_res_scores["Final_Rank"] = df_res_scores.get(
                "Weighted_Rank", df_res_scores["Mean_Rank"]
            )

        return df_res_scores

    # --------- 集成排序 ---------

    def run_ensemble(self):
        print("\n>>> 执行多方法 Ensemble 耦合评估...")

        w = self.final_weights
        s1 = self.method_topsis(w=w)
        s2 = self.method_vikor(w=w)
        s3 = self.method_gra(w=w)

        df_res = pd.DataFrame(index=self.products)
        df_res["TOPSIS_Score"] = s1
        df_res["VIKOR_Score"] = s2
        df_res["GRA_Score"] = s3

        df_res["Rank_TOPSIS"] = df_res["TOPSIS_Score"].rank(
            ascending=False, method="min"
        )
        df_res["Rank_VIKOR"] = df_res["VIKOR_Score"].rank(
            ascending=False, method="min"
        )
        df_res["Rank_GRA"] = df_res["GRA_Score"].rank(
            ascending=False, method="min"
        )

        df_res["Mean_Rank"] = df_res[["Rank_TOPSIS", "Rank_VIKOR", "Rank_GRA"]].mean(
            axis=1
        )

        w_T, w_V, w_G = self.method_weights
        df_res["Weighted_Rank"] = (
            w_T * df_res["Rank_TOPSIS"]
            + w_V * df_res["Rank_VIKOR"]
            + w_G * df_res["Rank_GRA"]
        )

        df_res = self._apply_noncompensatory(df_res)

        df_res = df_res.sort_values("Final_Rank")
        self.results = df_res

        raw_csv = os.path.join(self.output_dir, "Ensemble_Raw_Scores.csv")
        df_res.to_csv(raw_csv)
        print(f"    各方法得分+排名已保存: {raw_csv}")

        rank_csv = os.path.join(self.output_dir, "Final_Ensemble_Ranking.csv")
        df_rank_only = df_res[
            [
                "Rank_TOPSIS",
                "Rank_VIKOR",
                "Rank_GRA",
                "Mean_Rank",
                "Weighted_Rank",
                "Final_Rank",
                "Flag_ExtremeFail",
            ]
        ]
        df_rank_only.to_csv(rank_csv)
        print(f"    最终综合排名表已保存: {rank_csv}")

        return df_rank_only

    def plot_heatmap(self):
        if self.results is None:
            raise RuntimeError("请先调用 run_ensemble() 生成 self.results")

        df = self.results.copy()
        cols = [
            "Rank_TOPSIS",
            "Rank_VIKOR",
            "Rank_GRA",
            "Mean_Rank",
            "Weighted_Rank",
            "Final_Rank",
        ]
        data = df[cols]

        plt.figure(figsize=(8, 0.4 * len(df) + 2))
        sns.heatmap(
            data,
            annot=True,
            cmap="viridis_r",
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={"label": "Rank"},
        )
        plt.title("Multi-Method Ensemble Ranking Results", fontsize=14)
        plt.ylabel("Product")
        plt.tight_layout()
        out_png = os.path.join(self.output_dir, "Ensemble_Ranking_Heatmap.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"    排名热力图已保存: {out_png}")

    def run_sensitivity(
        self,
        n_runs=200,
        weight_noise=0.1,
        method_weight_noise=0.1,
        random_seed=42,
    ):
        print("\n>>> 进行权重敏感性分析...")
        rng = np.random.default_rng(random_seed)

        m = len(self.products)
        ranks_runs = np.zeros((m, n_runs))

        base_w = self.final_weights.copy()
        base_mw = self.method_weights.copy()

        for k in range(n_runs):
            noise_w = rng.normal(loc=0.0, scale=weight_noise, size=base_w.shape)
            w_tmp = base_w * (1 + noise_w)
            w_tmp = np.clip(w_tmp, 1e-6, None)
            w_tmp = w_tmp / w_tmp.sum()

            noise_mw = rng.normal(
                loc=0.0, scale=method_weight_noise, size=base_mw.shape
            )
            mw_tmp = base_mw * (1 + noise_mw)
            mw_tmp = np.clip(mw_tmp, 1e-6, None)
            mw_tmp = mw_tmp / mw_tmp.sum()

            s1 = self.method_topsis(w=w_tmp)
            s2 = self.method_vikor(w=w_tmp)
            s3 = self.method_gra(w=w_tmp)

            df_tmp = pd.DataFrame(index=self.products)
            df_tmp["TOPSIS_Score"] = s1
            df_tmp["VIKOR_Score"] = s2
            df_tmp["GRA_Score"] = s3

            df_tmp["Rank_TOPSIS"] = df_tmp["TOPSIS_Score"].rank(
                ascending=False, method="min"
            )
            df_tmp["Rank_VIKOR"] = df_tmp["VIKOR_Score"].rank(
                ascending=False, method="min"
            )
            df_tmp["Rank_GRA"] = df_tmp["GRA_Score"].rank(
                ascending=False, method="min"
            )

            w_T, w_V, w_G = mw_tmp
            df_tmp["Weighted_Rank"] = (
                w_T * df_tmp["Rank_TOPSIS"]
                + w_V * df_tmp["Rank_VIKOR"]
                + w_G * df_tmp["Rank_GRA"]
            )

            ranks_runs[:, k] = df_tmp["Weighted_Rank"].values

        mean_rank = ranks_runs.mean(axis=1)
        std_rank = ranks_runs.std(axis=1)
        p_top3 = (ranks_runs <= 3.0).mean(axis=1)
        p_top5 = (ranks_runs <= 5.0).mean(axis=1)

        df_sens = pd.DataFrame(
            {
                "Mean_Rank_Sens": mean_rank,
                "Std_Rank_Sens": std_rank,
                "P_Rank<=3": p_top3,
                "P_Rank<=5": p_top5,
            },
            index=self.products,
        ).sort_values("Mean_Rank_Sens")

        out_csv = os.path.join(self.output_dir, "Sensitivity_Rank_Stats.csv")
        df_sens.to_csv(out_csv)
        print(f"    敏感性分析结果已保存: {out_csv}")
        return df_sens


def build_region_matrix(df_long: pd.DataFrame, region_abbr: str):
    """
    从长表抽取某个分区的“产品×指标”矩阵：
      - 行：Product
      - 列：所有数值型指标（剔除 Product/Region 元信息）
    若同一产品在该分区出现多行（理论上不应），取均值聚合。
    """
    df_r = df_long[df_long["Region_Abbr"] == region_abbr].copy()
    if df_r.empty:
        raise ValueError(f"分区 {region_abbr} 在输入表中不存在。")

    # 识别数值列
    non_metric_cols = {"Product", "Region_ID", "Region", "Region_Abbr"}
    metric_cols = [c for c in df_r.columns if c not in non_metric_cols]

    # 可选：去掉元数据列
    if DROP_META_COLS:
        metric_cols = [c for c in metric_cols if c not in META_COLS]

    # 仅保留数值列（保险：有些列可能是 object）
    numeric_cols = []
    for c in metric_cols:
        if pd.api.types.is_numeric_dtype(df_r[c]):
            numeric_cols.append(c)
        else:
            # 尝试转数值
            df_r[c] = pd.to_numeric(df_r[c], errors="coerce")
            if pd.api.types.is_numeric_dtype(df_r[c]):
                numeric_cols.append(c)

    df_m = df_r[["Product"] + numeric_cols].groupby("Product", as_index=True).mean()

    # 删除全 NaN 列（否则 preprocess 会提示并删）
    df_m = df_m.dropna(axis=1, how="all")

    return df_m


def run_mcdm_for_region(df_long: pd.DataFrame, region_id: int, region_name: str, region_abbr: str):
    """
    对单个分区执行完整 MCDM 流程，并输出到 OUTPUT_DIR/<Region_Abbr>_xxx
    返回：该分区最终排名表（带 Region 信息）
    """
    out_dir = os.path.join(OUTPUT_DIR, f"{region_id:02d}_{region_abbr}")
    os.makedirs(out_dir, exist_ok=True)

    df_ind = build_region_matrix(df_long, region_abbr=region_abbr)

    mcdm = EnsembleMCDMAdvanced(
        df_raw=df_ind,
        output_dir=out_dir,
        method_weights=METHOD_WEIGHTS,
        corr_prune_threshold=CORR_PRUNE_THRESHOLD,
        extreme_patterns=EXTREME_PATTERNS,
        extreme_min_score=EXTREME_MIN_SCORE,
        extreme_penalty=EXTREME_PENALTY,
    )

    mcdm.preprocess(
        abs_bias=True,
        fillna="median",
        drop_constant=True,
        prune_high_corr=True,
    )

    mcdm.calc_game_theory_weights()
    final_ranks = mcdm.run_ensemble()
    mcdm.plot_heatmap()

    sens = mcdm.run_sensitivity(
        n_runs=SENS_N_RUNS,
        weight_noise=SENS_WEIGHT_NOISE,
        method_weight_noise=SENS_METHOD_WEIGHT_NOISE,
    )

    # 带上分区信息，便于汇总
    out = final_ranks.copy()
    out.insert(0, "Region_ID", region_id)
    out.insert(1, "Region", region_name)
    out.insert(2, "Region_Abbr", region_abbr)

    # 也导出一份 TopN
    topn = out.head(SUMMARY_TOPN)
    topn.to_csv(os.path.join(out_dir, f"Top{SUMMARY_TOPN}_Products.csv"))

    print(f"\n[{region_abbr}] 推荐前 {SUMMARY_TOPN} 名: {out.index[:SUMMARY_TOPN].tolist()}")
    return out


# ====================== 主程序 ======================

if __name__ == "__main__":
    # 1) 读取“分区×产品”长表
    df_long = pd.read_csv(INPUT_CSV)

    # 基本字段检查
    required = {"Product", "Region_ID", "Region", "Region_Abbr"}
    miss = required - set(df_long.columns)
    if miss:
        raise KeyError(
            f"输入 CSV 缺少必要列 {sorted(list(miss))}。"
            f"请确认 INPUT_CSV 是脚本 B(v2-分区版) 生成的长表。"
        )

    # 2) 分区列表（按 Region_ID 排序）
    region_meta = (
        df_long[["Region_ID", "Region", "Region_Abbr"]]
        .drop_duplicates()
        .sort_values("Region_ID")
        .reset_index(drop=True)
    )

    print("\n>>> 将对以下分区分别执行 MCDM：")
    print(region_meta)

    all_results = []

    # 3) 逐分区执行 MCDM
    for _, r in region_meta.iterrows():
        rid = int(r["Region_ID"])
        rname = str(r["Region"])
        rabbr = str(r["Region_Abbr"])
        try:
            res = run_mcdm_for_region(df_long, rid, rname, rabbr)
            all_results.append(res)
        except Exception as e:
            print(f"[错误] 分区 {rabbr} 执行失败: {e}")
            continue

    # 4) 汇总输出：各分区 TopN
    if all_results:
        df_all = pd.concat(all_results, axis=0)
        df_all.to_csv(os.path.join(OUTPUT_DIR, "AllRegions_FinalRanking_Long.csv"))

        summary_rows = []
        for _, r in region_meta.iterrows():
            rid = int(r["Region_ID"])
            rabbr = str(r["Region_Abbr"])
            part = df_all[(df_all["Region_ID"] == rid) & (df_all["Region_Abbr"] == rabbr)]
            if part.empty:
                continue
            topn = part.sort_values("Final_Rank").head(SUMMARY_TOPN)
            for prod, row in topn.iterrows():
                summary_rows.append({
                    "Region_ID": rid,
                    "Region": row["Region"],
                    "Region_Abbr": row["Region_Abbr"],
                    "Product": prod,
                    "Final_Rank": float(row["Final_Rank"]),
                    "Weighted_Rank": float(row["Weighted_Rank"]),
                    "Mean_Rank": float(row["Mean_Rank"]),
                    "Flag_ExtremeFail": bool(row["Flag_ExtremeFail"]),
                })

        df_summary = pd.DataFrame(summary_rows)
        out_sum = os.path.join(OUTPUT_DIR, f"Summary_Top{SUMMARY_TOPN}_ByRegion.csv")
        df_summary.to_csv(out_sum, index=False)
        print(f"\n>>> 分区 Top{SUMMARY_TOPN} 汇总已保存: {out_sum}")
    else:
        print("\n[警告] 没有任何分区成功输出结果，请检查输入 CSV 与指标列。")
