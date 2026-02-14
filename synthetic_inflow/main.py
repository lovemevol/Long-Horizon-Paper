"""
合成入流生成主程序

流程：
1. 读取三峡1965-2025年日尺度实测入库流量
2. STL季节分解提取季节项与残差
3. 对残差拟合AR模型
4. 基于年径流量构建马尔可夫年型转换
5. 生成年型序列并模拟200年日尺度合成入流
6. 验证合成序列与历史序列统计特征的一致性
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, fields
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from diagnostics_plot import save_diagnostics
from stl_decomposition import STLDecomposer
from ar_model import ARModel
from markov_year_type import MarkovYearType
from statistical_validation import StatisticalValidator


DATA_FILE = pathlib.Path(__file__).resolve().parent / "实测径流.xlsx"
README_FILE = pathlib.Path(__file__).resolve().parent / "README.md"

PARAM_DESCRIPTIONS: Dict[str, str] = {
    "years_to_generate": "合成年份数量",
    "seasonal_period": "STL 季节周期（日）",
    "ar_max_lag": "AR 自动选阶的最大滞后数",
    "markov_types": "年型类别数（特丰/丰/平/枯/特枯）",
    "random_seed": "全流程随机种子",
    "output_dir": "结果输出目录",
    "residual_scale_strength": "月度残差方差匹配强度$\\kappa$",
    "residual_scale_min": "月度残差缩放下限$\\underline{\\alpha}$",
    "residual_scale_max": "月度残差缩放上限$\\overline{\\alpha}$",
    "low_flow_alignment_strength": "低流量分布映射强度$\\beta$",
    "dual_scale_strength": "分位+月份双重缩放强度$\\lambda$",
    "dual_scale_min": "双重缩放最小因子$\\alpha_{\\min}$",
    "dual_scale_max": "双重缩放最大因子$\\alpha_{\\max}$",
    "high_flow_enhance_strength": "高流量幂次增强强度$\\gamma$，0 表示关闭",
    "high_flow_enhance_quantile": "高流量增强参考分位阈值$\\tau$",
    "high_flow_enhance_exponent": "幂指数$\\eta$，决定放大量级随流量增长的速度",
    "high_flow_enhance_max_multiplier": "单点高流量缩放的上限$m_{\\max}$",
}


@dataclass
class GeneratorConfig:
    years_to_generate: int = 200  # 生成模拟年份数
    seasonal_period: int = 365  # 季节周期长度，对应日尺度一年
    ar_max_lag: int = 30  # AR模型最大滞后阶数
    markov_types: int = 5  # 年型分类数量
    random_seed: int = 10  # 全局随机种子
    output_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent / "output"  # 输出目录
    residual_scale_strength: float = 0.9  # 残差月度方差匹配强度
    residual_scale_min: float = 0.4  # 残差月度缩放下限
    residual_scale_max: float = 1.3  # 残差月度缩放上限
    low_flow_alignment_strength: float = 0.99  # 低流量分布同质化强度
    dual_scale_strength: float = 0.88  # 分位+月份双重缩放强度
    dual_scale_min: float = 0.05  # 双重缩放最小因子
    dual_scale_max: float = 1.4  # 双重缩放最大因子
    high_flow_enhance_strength: float = 0.2  # 极端高流量增强强度，0表示关闭
    high_flow_enhance_quantile: float = 0.9  # 参考分位阈值，用于识别高流量事件
    high_flow_enhance_exponent: float = 1.5  # 增强幂指数，决定放大量级随流量的增长速度
    high_flow_enhance_max_multiplier: float = 2.0  # 单点最大放大量级


class SyntheticInflowGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.observed_series: pd.Series | None = None
        self.seasonal_pattern: pd.Series | None = None
        self.trend_mean: float | None = None
        self.ar_model: ARModel | None = None
        self.year_type_model: MarkovYearType | None = None
        self.validator = StatisticalValidator()
        self.trend_series: pd.Series | None = None
        self.seasonal_series: pd.Series | None = None
        self.trend_plus_seasonal: pd.Series | None = None
        self.baseline_by_type: Dict[int, list] = {}
        self.residual_monthly_std: Dict[int, float] | None = None
        self.quantile_edges: np.ndarray | None = None
        self.quantile_rel_targets: np.ndarray | None = None
        self.month_rel_targets: np.ndarray | None = None
        self.global_rel_target: float | None = None
        self.high_flow_threshold: float | None = None
        self.annual_targets: Dict[int, float] = {}

    @staticmethod
    def load_observed_flow(file_path: pathlib.Path) -> pd.Series:
        df = pd.read_excel(file_path, header=None, names=["date", "flow"], parse_dates=[0])
        df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
        df = df.dropna()
        df = df.sort_values("date")
        df = df.set_index("date")

        # 剔除闰日以保证每年365天
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
        df = df.asfreq("D")
        df["flow"] = df["flow"].infer_objects(copy=False).interpolate(limit_direction="both")
        return df["flow"]

    def perform_stl(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        decomposer = STLDecomposer(seasonal_period=self.config.seasonal_period)
        trend, seasonal, residual = decomposer.decompose(self.observed_series)
        self.trend_series = trend
        self.seasonal_series = seasonal
        self.trend_plus_seasonal = trend + seasonal
        self.seasonal_pattern = decomposer.get_average_seasonal_pattern()
        self.trend_mean = trend.mean()
        self.residual_monthly_std = residual.groupby(residual.index.month).std(ddof=1).to_dict()
        return trend, seasonal, residual

    def _prepare_rel_change_targets(self) -> None:
        if self.observed_series is None or self.observed_series.empty:
            self.quantile_edges = None
            self.quantile_rel_targets = None
            self.month_rel_targets = None
            self.global_rel_target = None
            self.high_flow_threshold = None
            return

        flow = self.observed_series
        delta = flow.diff().abs()
        midpoint = (flow.shift(1) + flow) * 0.5
        rel = (delta / midpoint.replace(0, np.nan)).dropna()

        aligned_index = rel.index
        current_flow = flow.loc[aligned_index]
        months = aligned_index.month

        data = pd.DataFrame({
            "flow": current_flow.values,
            "rel": rel.values,
            "month": months,
        }, index=aligned_index)

        quantiles = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
        edges = np.quantile(data["flow"], quantiles)
        edges = np.unique(edges)
        if edges.size < 3:
            edges = np.linspace(data["flow"].min(), data["flow"].max(), num=3)

        edges[0] = min(edges[0], data["flow"].min())
        edges[-1] = max(edges[-1], data["flow"].max())

        buckets = pd.cut(
            data["flow"],
            bins=edges,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        )

        quantile_stats = data.groupby(buckets)["rel"].median()
        n_bins = len(edges) - 1
        fallback_rel = float(data["rel"].median()) if not data["rel"].empty else 0.03
        quantile_array = quantile_stats.reindex(range(n_bins), fill_value=np.nan).to_numpy()
        quantile_array = np.where(np.isfinite(quantile_array), quantile_array, fallback_rel)

        month_stats = data.groupby("month")["rel"].median()
        month_array = np.full(13, fallback_rel, dtype=float)
        for month, value in month_stats.items():
            if 1 <= month <= 12 and np.isfinite(value) and value > 0:
                month_array[month] = value

        self.quantile_edges = edges
        self.quantile_rel_targets = quantile_array
        self.month_rel_targets = month_array
        self.global_rel_target = fallback_rel

        q = np.clip(self.config.high_flow_enhance_quantile, 0.5, 0.995)
        threshold = float(self.observed_series.quantile(q))
        self.high_flow_threshold = threshold if np.isfinite(threshold) and threshold > 0 else None

    def fit_ar_model(self, residual: pd.Series) -> None:
        self.ar_model = ARModel(max_lag=self.config.ar_max_lag)
        self.ar_model.select_order(residual, plot=False)
        self.ar_model.fit(residual)

    def build_markov_model(self) -> None:
        self.year_type_model = MarkovYearType(n_types=self.config.markov_types)
        self.year_type_model.classify_years(self.observed_series)
        self.year_type_model.estimate_transition_matrix()
        self.prepare_baselines()

    def prepare_baselines(self) -> None:
        if self.trend_plus_seasonal is None:
            raise ValueError("请先执行perform_stl()以获取分解结果")
        if self.year_type_model is None:
            raise ValueError("请先构建年型模型")

        self.baseline_by_type = {idx: [] for idx in range(self.year_type_model.n_types)}
        combined = self.trend_plus_seasonal

        for year, type_idx in self.year_type_model.year_types.items():
            mask = combined.index.year == year
            yearly = combined[mask]
            if len(yearly) < self.config.seasonal_period:
                continue
            yearly = yearly.iloc[: self.config.seasonal_period]
            self.baseline_by_type[type_idx].append(yearly.to_numpy())

        # 若某类无样本，回退为平均季节模式
        fallback = (self.seasonal_pattern.values + self.trend_mean)
        for type_idx, baselines in self.baseline_by_type.items():
            if not baselines:
                self.baseline_by_type[type_idx].append(fallback)

    @staticmethod
    def _create_day_index(start_year: int, years: int) -> pd.DatetimeIndex:
        start = pd.Timestamp(f"{start_year}-01-01")
        end = pd.Timestamp(f"{start_year + years - 1}-12-31")
        date_range = pd.date_range(start=start, end=end, freq="D")
        mask = ~((date_range.month == 2) & (date_range.day == 29))
        return date_range[mask]

    def _generate_residual_series(self, days: int) -> pd.Series:
        residuals = self.ar_model.simulate(days)
        start_year = self.observed_series.index[-1].year + 1
        date_index = self._create_day_index(start_year, self.config.years_to_generate)
        series = pd.Series(residuals, index=date_index[:days])

        if self.residual_monthly_std:
            for month in range(1, 13):
                target_std = self.residual_monthly_std.get(month)
                if target_std is None or target_std <= 0:
                    continue
                mask = series.index.month == month
                if not mask.any():
                    continue
                sampled = series.loc[mask]
                sim_std = sampled.std(ddof=1)
                if sim_std and np.isfinite(sim_std) and sim_std > 0:
                    scale_raw = target_std / sim_std
                    strength = self.config.residual_scale_strength
                    scale = 1 + strength * (scale_raw - 1)
                    scale = max(self.config.residual_scale_min,
                                min(self.config.residual_scale_max, scale))
                    series.loc[mask] = sampled * scale

        return series

    def _sample_year_baseline(self, year_type: int) -> np.ndarray:
        baselines = self.baseline_by_type.get(year_type, [])
        if not baselines:
            return self.seasonal_pattern.values + self.trend_mean
        choice_idx = np.random.randint(0, len(baselines))
        return baselines[choice_idx]

    def _apply_dual_scaling(self, series: pd.Series) -> pd.Series:
        strength = self.config.dual_scale_strength
        if strength <= 0:
            return series
        if self.quantile_edges is None or self.quantile_rel_targets is None:
            return series
        if self.month_rel_targets is None or self.global_rel_target is None:
            return series

        values = series.to_numpy(copy=True)
        months = series.index.month
        edges = self.quantile_edges
        q_targets = self.quantile_rel_targets
        month_targets = self.month_rel_targets
        global_target = self.global_rel_target
        eps = 1e-6

        for i in range(1, len(values)):
            prev_val = values[i - 1]
            curr_val = values[i]
            diff = curr_val - prev_val
            base = (abs(prev_val) + abs(curr_val)) * 0.5
            if base <= eps:
                continue
            current_rel = abs(diff) / base
            if not np.isfinite(current_rel):
                continue

            flow_val = curr_val
            idx = np.searchsorted(edges, flow_val, side="right") - 1
            idx = int(np.clip(idx, 0, len(q_targets) - 1))

            target_candidates = [q_targets[idx], month_targets[months[i]], global_target]
            target_candidates = [v for v in target_candidates if np.isfinite(v) and v > 0]
            if not target_candidates:
                continue
            target_rel = float(np.mean(target_candidates))

            scale_raw = target_rel / max(current_rel, eps)
            scale = 1 + strength * (scale_raw - 1)
            scale = np.clip(scale, self.config.dual_scale_min, self.config.dual_scale_max)

            values[i] = values[i - 1] + diff * scale

        adjusted = pd.Series(values, index=series.index, name=series.name)
        return adjusted

    def _enhance_high_flows(self, series: pd.Series) -> pd.Series:
        strength = self.config.high_flow_enhance_strength
        threshold = self.high_flow_threshold
        if strength <= 0 or threshold is None or threshold <= 0:
            return series

        exponent = max(self.config.high_flow_enhance_exponent, 1.0)
        max_multiplier = max(self.config.high_flow_enhance_max_multiplier, 1.0)
        values = series.copy()
        year_list = sorted(set(values.index.year))

        for year in year_list:
            mask = values.index.year == year
            year_values = values.loc[mask].to_numpy()
            original_sum = year_values.sum()
            if not np.isfinite(original_sum) or original_sum <= 0:
                continue

            high_idx = year_values >= threshold
            if not np.any(high_idx):
                continue

            ratios = np.maximum(year_values[high_idx] / threshold, 1.0)
            multipliers = 1 + strength * (np.power(ratios, exponent) - 1)
            multipliers = np.clip(multipliers, 1.0, max_multiplier)
            year_values[high_idx] = year_values[high_idx] * multipliers

            enhanced_sum = year_values.sum()
            target = self.annual_targets.get(year, original_sum)
            if enhanced_sum > 0 and target > 0:
                year_values = year_values * (target / enhanced_sum)

            values.loc[mask] = year_values

        return values

    def _adjust_low_flows(self, series: pd.Series) -> pd.Series:
        blend = self.config.low_flow_alignment_strength
        if blend <= 0:
            return series

        low_q = self.validator.low_q
        max_q = max(low_q, min(0.3, low_q * 2))
        qs = np.linspace(0.0, max_q, num=8)

        hist_quantiles = self.observed_series.quantile(qs)
        synth_quantiles = series.quantile(qs)

        synth_vals = synth_quantiles.to_numpy()
        hist_vals = hist_quantiles.to_numpy()
        valid = np.isfinite(synth_vals) & np.isfinite(hist_vals)
        synth_vals = synth_vals[valid]
        hist_vals = hist_vals[valid]

        if synth_vals.size < 2:
            return series

        unique_synth, unique_idx = np.unique(synth_vals, return_index=True)
        unique_hist = hist_vals[unique_idx]

        if unique_synth.size < 2 or unique_synth[-1] <= unique_synth[0]:
            return series

        upper = unique_synth[-1]
        mask = series <= upper
        if not mask.any():
            return series

        mapped = np.interp(series.loc[mask], unique_synth, unique_hist)
        adjusted = (1 - blend) * series.loc[mask] + blend * mapped
        adjusted = np.maximum(adjusted, 0)
        series.loc[mask] = adjusted
        return series

    def _sample_annual_totals(self, year_types: np.ndarray) -> np.ndarray:
        annual_flow = self.year_type_model.annual_flow
        sample_totals = np.zeros_like(year_types, dtype=float)
        for type_idx in range(self.year_type_model.n_types):
            mask = self.year_type_model.year_types == type_idx
            pool = annual_flow[mask]
            if len(pool) == 0:
                pool = annual_flow
            sampled = np.random.choice(pool.values, size=(year_types == type_idx).sum(), replace=True)
            sample_totals[year_types == type_idx] = sampled
        return sample_totals

    def generate_synthetic_series(self) -> pd.Series:
        np.random.seed(self.config.random_seed)
        days = self.config.years_to_generate * self.config.seasonal_period
        residual_series = self._generate_residual_series(days)

        year_types = self.year_type_model.generate_year_type_sequence(
            n_years=self.config.years_to_generate,
            random_seed=self.config.random_seed,
        )

        sampled_totals = self._sample_annual_totals(year_types)

        synthetic = residual_series.copy()
        self.annual_targets = {}
        year_list = sorted(set(synthetic.index.year))
        for idx, year in enumerate(year_list):
            year_mask = synthetic.index.year == year
            baseline_values = self._sample_year_baseline(year_types[idx])
            baseline_values = baseline_values[: year_mask.sum()]
            year_values = baseline_values + synthetic[year_mask].values

            synthetic.loc[year_mask] = year_values

        for idx, year in enumerate(year_list):
            year_mask = synthetic.index.year == year
            year_sum = synthetic[year_mask].sum()
            target = sampled_totals[idx]
            scale = target / year_sum if year_sum > 0 else 1.0
            synthetic.loc[year_mask] *= scale
            self.annual_targets[year] = target

        synthetic = self._apply_dual_scaling(synthetic)
        synthetic = self._enhance_high_flows(synthetic)
        synthetic = self._adjust_low_flows(synthetic)
        synthetic = synthetic.clip(lower=0)
        synthetic.name = "synthetic_flow"
        return synthetic

    def run(self) -> Dict[str, Dict[str, float]]:
        self.observed_series = self.load_observed_flow(DATA_FILE)
        trend, seasonal, residual = self.perform_stl()
        self._prepare_rel_change_targets()
        self.fit_ar_model(residual)
        self.build_markov_model()
        synthetic_series = self.generate_synthetic_series()

        hist_summary = self.validator.summarize(self.observed_series)
        synth_summary = self.validator.summarize(synthetic_series)

        comparison = self.validator.compare_summaries(hist_summary, synth_summary)
        comparison.to_csv(self.output_dir / "statistical_comparison.csv", index=False)
        synthetic_series.to_csv(self.output_dir / "synthetic_inflow_series.csv", header=True)
        save_diagnostics(
            hist_summary=hist_summary,
            synth_summary=synth_summary,
            synthetic_series=synthetic_series,
            observed_series=self.observed_series,
            output_dir=self.output_dir,
        )

        summary = {
            "historical_basic": hist_summary.basic_stats,
            "synthetic_basic": synth_summary.basic_stats,
            "year_types_freq": dict(zip(self.year_type_model.type_names, np.round(self.year_type_model.stationary_dist, 4))),
        }

        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._update_readme(summary, hist_summary, synth_summary, synthetic_series)

        return summary

    @staticmethod
    def _format_number(value: float, digits: int = 2, use_comma: bool = True) -> str:
        if value is None or not np.isfinite(value):
            return "-"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}" if use_comma else str(int(value))
        fmt = f"{{:,.{digits}f}}" if use_comma else f"{{:.{digits}f}}"
        text = fmt.format(float(value))
        text = text.rstrip("0").rstrip(".")
        return text

    def _build_param_table(self) -> str:
        rows = ["| 参数 | 默认值 | 说明 |", "| --- | --- | --- |"]
        for field in fields(GeneratorConfig):
            name = field.name
            value = getattr(self.config, name)
            if isinstance(value, pathlib.Path):
                value_str = str(value)
            else:
                value_str = self._format_number(value, digits=2)
            desc = PARAM_DESCRIPTIONS.get(name, "")
            rows.append(f"| `{name}` | {value_str} | {desc} |")
        return "\n".join(rows)

    def _build_results_section(self,
                               hist_summary,
                               synth_summary,
                               synthetic_series: pd.Series) -> str:
        lines: List[str] = []

        def pct_delta(synth: float, hist: float) -> str:
            if hist == 0:
                return "0%"
            delta = (synth - hist) / hist * 100
            sign = "+" if delta >= 0 else ""
            return f"{sign}{self._format_number(delta, digits=2, use_comma=False)}%"

        lines.append("- 基本统计量与相对误差：\n")
        lines.append("  | 指标 | 历史 | 合成 | 相对误差 |")
        lines.append("  | --- | --- | --- | --- |")
        for key, label in [("mean", "均值 (m³/s)"), ("std", "标准差 (m³/s)"), ("skew", "偏度"), ("kurtosis", "峰度")]:
            hist_val = hist_summary.basic_stats.get(key)
            synth_val = synth_summary.basic_stats.get(key)
            lines.append(
                "  | {label} | {hist} | {synth} | {delta} |".format(
                    label=label,
                    hist=self._format_number(hist_val),
                    synth=self._format_number(synth_val),
                    delta=pct_delta(synth_val, hist_val),
                )
            )

        quantiles = [0.05, 0.1, 0.5, 0.9, 0.95]
        hist_quant = self.observed_series.quantile(quantiles)
        synth_quant = synthetic_series.quantile(quantiles)
        lines.append("\n- 分位点对比（m³/s）：\n")
        lines.append("  | 分位 | 历史 | 合成 | 合成−历史 |")
        lines.append("  | --- | --- | --- | --- |")
        for q in quantiles:
            hist_val = hist_quant.loc[q]
            synth_val = synth_quant.loc[q]
            diff = synth_val - hist_val
            lines.append(
                "  | {q:.0%} | {hist} | {synth} | {diff} |".format(
                    q=q,
                    hist=self._format_number(hist_val),
                    synth=self._format_number(synth_val),
                    diff=self._format_number(diff, digits=2, use_comma=True if abs(diff) >= 1000 else False),
                )
            )

        hist_month = hist_summary.monthly_mean
        synth_month = synth_summary.monthly_mean
        months = sorted(set(hist_month.index) | set(synth_month.index))
        lines.append("\n- 月均流量差异（m³/s）：\n")
        lines.append("  | 月份 | 历史 | 合成 | 合成−历史 |")
        lines.append("  | --- | --- | --- | --- |")
        for month in months:
            hist_val = hist_month.get(month, np.nan)
            synth_val = synth_month.get(month, np.nan)
            diff = synth_val - hist_val
            lines.append(
                "  | {month} | {hist} | {synth} | {diff} |".format(
                    month=month,
                    hist=self._format_number(hist_val),
                    synth=self._format_number(synth_val),
                    diff=self._format_number(diff, digits=2, use_comma=True if abs(diff) >= 1000 else False),
                )
            )

        persistence = [
            ("高", synth_summary.persistence_metrics.high_mean, hist_summary.persistence_metrics.high_mean, "平均"),
            ("高", synth_summary.persistence_metrics.high_std, hist_summary.persistence_metrics.high_std, "标准差"),
            ("高", synth_summary.persistence_metrics.high_max, hist_summary.persistence_metrics.high_max, "最大"),
            ("低", synth_summary.persistence_metrics.low_mean, hist_summary.persistence_metrics.low_mean, "平均"),
            ("低", synth_summary.persistence_metrics.low_std, hist_summary.persistence_metrics.low_std, "标准差"),
            ("低", synth_summary.persistence_metrics.low_max, hist_summary.persistence_metrics.low_max, "最大"),
        ]
        lines.append("\n- 丰枯持续期对比：\n")
        lines.append("  | 指标 | 历史 | 合成 | 合成−历史 |")
        lines.append("  | --- | --- | --- | --- |")
        labels_map = {
            ("高", "平均"): "丰水平均 (天)",
            ("高", "标准差"): "丰水标准差 (天)",
            ("高", "最大"): "丰水最大 (天)",
            ("低", "平均"): "枯水平均 (天)",
            ("低", "标准差"): "枯水标准差 (天)",
            ("低", "最大"): "枯水最大 (天)",
        }
        for mode, synth_val, hist_val, metric in persistence:
            label = labels_map[(mode, metric)]
            diff = synth_val - hist_val
            lines.append(
                "  | {label} | {hist} | {synth} | {diff} |".format(
                    label=label,
                    hist=self._format_number(hist_val),
                    synth=self._format_number(synth_val),
                    diff=self._format_number(diff, digits=2, use_comma=False),
                )
            )

        peak_hist = hist_summary.peak_metrics
        peak_synth = synth_summary.peak_metrics
        lines.append("\n- 高流量指标显著提升：")
        peak_entries = [
            ("最大日流量", peak_hist.max_flow, peak_synth.max_flow, "m³/s"),
            ("峰值系数", peak_hist.peak_factor, peak_synth.peak_factor, None),
            ("高流量段变异系数", peak_hist.peak_over_threshold_cv, peak_synth.peak_over_threshold_cv, None),
        ]
        for label, hist_val, synth_val, unit in peak_entries:
            delta = pct_delta(synth_val, hist_val)
            if unit:
                lines.append(
                    f"  - {label}：从 {self._format_number(hist_val)} {unit} 增至 {self._format_number(synth_val)} {unit}（{delta}）。"
                )
            else:
                lines.append(
                    f"  - {label}：从 {self._format_number(hist_val)} 提升至 {self._format_number(synth_val)}（{delta}）。"
                )

        return "\n".join(lines)

    @staticmethod
    def _replace_block(content: str, start_marker: str, end_marker: str, new_block: str) -> str:
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return content
        start_idx += len(start_marker)
        return content[:start_idx] + "\n" + new_block.strip() + "\n" + content[end_idx:]

    def _update_readme(self,
                       summary: Dict[str, Dict[str, float]],
                       hist_summary,
                       synth_summary,
                       synthetic_series: pd.Series) -> None:
        if not README_FILE.exists():
            return

        content = README_FILE.read_text(encoding="utf-8")
        param_table = self._build_param_table()
        result_section = self._build_results_section(hist_summary, synth_summary, synthetic_series)

        content = self._replace_block(content, "<!-- AUTO:PARAM_TABLE_START -->", "<!-- AUTO:PARAM_TABLE_END -->", param_table)
        content = self._replace_block(content, "<!-- AUTO:RESULT_SECTION_START -->", "<!-- AUTO:RESULT_SECTION_END -->", result_section)

        README_FILE.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于STL+AR+马尔可夫的合成入流生成")
    parser.add_argument("--years", type=int, default=None, help="生成年份数，默认取GeneratorConfig设置")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认取GeneratorConfig设置")
    parser.add_argument("--output", type=str, default=None, help="输出目录，默认取GeneratorConfig设置")
    parser.add_argument("--residual-scale-strength", type=float, default=None,
                        help="残差月度标准差调整强度，0表示不调整，1表示完全匹配历史")
    parser.add_argument("--residual-scale-min", type=float, default=None,
                        help="残差月度缩放下限")
    parser.add_argument("--residual-scale-max", type=float, default=None,
                        help="残差月度缩放上限")
    parser.add_argument("--low-flow-alignment-strength", type=float, default=None,
                        help="低流量分布同质化的融合系数，0表示不调整，1表示完全映射")
    parser.add_argument("--dual-scale-strength", type=float, default=None,
                        help="分位+月份双重缩放强度，0表示不启动该调整")
    parser.add_argument("--dual-scale-min", type=float, default=None,
                        help="双重缩放的最小缩放因子")
    parser.add_argument("--dual-scale-max", type=float, default=None,
                        help="双重缩放的最大缩放因子")
    parser.add_argument("--high-flow-enhance-strength", type=float, default=None,
                        help="极端高流量增强强度，0表示不启用")
    parser.add_argument("--high-flow-enhance-quantile", type=float, default=None,
                        help="高流量增强参考分位阈值")
    parser.add_argument("--high-flow-enhance-exponent", type=float, default=None,
                        help="高流量增强幂指数，控制放大量级随流量增长的速度")
    parser.add_argument("--high-flow-enhance-max-multiplier", type=float, default=None,
                        help="单点高流量的最大放大量级")
    return parser.parse_args()


def main():
    args = parse_args()
    config = GeneratorConfig()
    if args.years is not None:
        config.years_to_generate = args.years
    if args.seed is not None:
        config.random_seed = args.seed
    if args.output:
        config.output_dir = pathlib.Path(args.output)
        config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.residual_scale_strength is not None:
        config.residual_scale_strength = args.residual_scale_strength
    if args.residual_scale_min is not None:
        config.residual_scale_min = args.residual_scale_min
    if args.residual_scale_max is not None:
        config.residual_scale_max = args.residual_scale_max
    if args.low_flow_alignment_strength is not None:
        config.low_flow_alignment_strength = args.low_flow_alignment_strength
    if args.dual_scale_strength is not None:
        config.dual_scale_strength = args.dual_scale_strength
    if args.dual_scale_min is not None:
        config.dual_scale_min = args.dual_scale_min
    if args.dual_scale_max is not None:
        config.dual_scale_max = args.dual_scale_max
    if args.high_flow_enhance_strength is not None:
        config.high_flow_enhance_strength = args.high_flow_enhance_strength
    if args.high_flow_enhance_quantile is not None:
        config.high_flow_enhance_quantile = args.high_flow_enhance_quantile
    if args.high_flow_enhance_exponent is not None:
        config.high_flow_enhance_exponent = args.high_flow_enhance_exponent
    if args.high_flow_enhance_max_multiplier is not None:
        config.high_flow_enhance_max_multiplier = args.high_flow_enhance_max_multiplier

    generator = SyntheticInflowGenerator(config)
    summary = generator.run()
    print("合成入流生成完成，统计摘要：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()