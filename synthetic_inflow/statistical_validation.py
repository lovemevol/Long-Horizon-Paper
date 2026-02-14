"""
统计特征验证模块
用于比较历史序列与合成序列在多项统计特征上的一致性
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PersistenceMetrics:
    """丰枯持续期统计特征"""

    high_mean: float
    high_std: float
    high_max: int
    high_min: int
    low_mean: float
    low_std: float
    low_max: int
    low_min: int


@dataclass
class PeakMetrics:
    """偏峰特征统计指标"""

    max_flow: float
    peak_factor: float
    peak_over_threshold_freq: float
    peak_over_threshold_mean: float
    peak_over_threshold_cv: float


@dataclass
class StatisticalSummary:
    """综合统计指标集合"""

    basic_stats: Dict[str, float]
    peak_metrics: PeakMetrics
    persistence_metrics: PersistenceMetrics
    monthly_mean: pd.Series = field(default_factory=pd.Series)


class StatisticalValidator:
    """
    统计验证器，用于衡量历史序列与合成序列的同质性

    包括以下指标：
    1. 基本统计量：均值、标准差、偏度、峰度
    2. 偏峰特征：峰值、峰值系数（最大值/均值）、高分位超越特征
    3. 丰枯持续期：高流量与低流量事件的持续长度统计
    4. 月均流量：季节性对比
    """

    def __init__(self,
                 high_threshold_quantile: float = 0.9,
                 low_threshold_quantile: float = 0.1,
                 persistence_unit: str = "day"):
        """
        Args:
            high_threshold_quantile: 定义高流量事件的分位数阈值
            low_threshold_quantile: 定义低流量事件的分位数阈值
            persistence_unit: 持续期单位（目前支持day）
        """

        if not 0 < low_threshold_quantile < high_threshold_quantile < 1:
            raise ValueError("低阈值与高阈值量化必须满足 0 < low < high < 1")

        self.high_q = high_threshold_quantile
        self.low_q = low_threshold_quantile
        self.persistence_unit = persistence_unit

    @staticmethod
    def _basic_statistics(series: pd.Series) -> Dict[str, float]:
        series = series.dropna()
        return {
            "mean": series.mean(),
            "std": series.std(ddof=1),
            "var": series.var(ddof=1),
            "skew": stats.skew(series),
            "kurtosis": stats.kurtosis(series, fisher=False),
        }

    @staticmethod
    def _compute_monthly_mean(series: pd.Series) -> pd.Series:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("序列索引必须为DatetimeIndex以计算月均值")
        monthly = series.groupby([series.index.month, series.index.day])
        month_mean = monthly.mean().groupby(level=0).mean()
        month_mean.index = [f"{m:02d}" for m in month_mean.index]
        return month_mean

    @staticmethod
    def _compute_persistence_lengths(series: pd.Series,
                                     threshold: float,
                                     mode: str = "high") -> List[int]:
        if mode not in ("high", "low"):
            raise ValueError("mode必须是'high'或'low'")

        bool_series = series >= threshold if mode == "high" else series <= threshold

        lengths = []
        current = 0
        for flag in bool_series:
            if flag:
                current += 1
            else:
                if current > 0:
                    lengths.append(current)
                    current = 0
        if current > 0:
            lengths.append(current)

        if not lengths:
            return [0]
        return lengths

    def _persistence_statistics(self,
                                series: pd.Series,
                                high_threshold: float,
                                low_threshold: float) -> PersistenceMetrics:
        high_lengths = self._compute_persistence_lengths(series, high_threshold, mode="high")
        low_lengths = self._compute_persistence_lengths(series, low_threshold, mode="low")

        return PersistenceMetrics(
            high_mean=float(np.mean(high_lengths)),
            high_std=float(np.std(high_lengths, ddof=1) if len(high_lengths) > 1 else 0.0),
            high_max=int(np.max(high_lengths)),
            high_min=int(np.min(high_lengths)),
            low_mean=float(np.mean(low_lengths)),
            low_std=float(np.std(low_lengths, ddof=1) if len(low_lengths) > 1 else 0.0),
            low_max=int(np.max(low_lengths)),
            low_min=int(np.min(low_lengths)),
        )

    @staticmethod
    def _peak_characteristics(series: pd.Series,
                              threshold: float) -> PeakMetrics:
        series = series.dropna()
        max_flow = series.max()
        mean_flow = series.mean()
        peak_factor = max_flow / mean_flow if mean_flow > 0 else np.nan

        exceed = series[series >= threshold]
        freq = len(exceed) / len(series)
        exceed_mean = exceed.mean() if len(exceed) > 0 else 0.0
        exceed_cv = (exceed.std(ddof=1) / exceed_mean) if len(exceed) > 1 and exceed_mean > 0 else 0.0

        return PeakMetrics(
            max_flow=float(max_flow),
            peak_factor=float(peak_factor),
            peak_over_threshold_freq=float(freq),
            peak_over_threshold_mean=float(exceed_mean),
            peak_over_threshold_cv=float(exceed_cv),
        )

    def summarize(self, series: pd.Series) -> StatisticalSummary:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("序列索引必须为DatetimeIndex")

        basic_stats = self._basic_statistics(series)
        high_threshold = series.quantile(self.high_q)
        low_threshold = series.quantile(self.low_q)
        peak_metrics = self._peak_characteristics(series, high_threshold)
        persistence_metrics = self._persistence_statistics(series, high_threshold, low_threshold)
        monthly_mean = self._compute_monthly_mean(series)

        return StatisticalSummary(
            basic_stats=basic_stats,
            peak_metrics=peak_metrics,
            persistence_metrics=persistence_metrics,
            monthly_mean=monthly_mean,
        )

    @staticmethod
    def compare_summaries(hist_summary: StatisticalSummary,
                          synth_summary: StatisticalSummary) -> pd.DataFrame:
        rows = []

        for key, value in hist_summary.basic_stats.items():
            rows.append({
                "指标": key,
                "历史值": value,
                "合成值": synth_summary.basic_stats[key],
                "相对误差%": 100 * (synth_summary.basic_stats[key] - value) / value if value != 0 else np.nan,
            })

        peak_hist = hist_summary.peak_metrics
        peak_synth = synth_summary.peak_metrics
        peak_map = {
            "max_flow": "最大日流量",
            "peak_factor": "峰值系数",
            "peak_over_threshold_freq": "高流量频率",
            "peak_over_threshold_mean": "高流量平均值",
            "peak_over_threshold_cv": "高流量变异系数",
        }

        for attr, name in peak_map.items():
            hist_val = getattr(peak_hist, attr)
            synth_val = getattr(peak_synth, attr)
            rows.append({
                "指标": name,
                "历史值": hist_val,
                "合成值": synth_val,
                "相对误差%": 100 * (synth_val - hist_val) / hist_val if hist_val != 0 else np.nan,
            })

        pers_hist = hist_summary.persistence_metrics
        pers_synth = synth_summary.persistence_metrics
        persistence_map = {
            "high_mean": "丰水持续期均值",
            "high_std": "丰水持续期标准差",
            "high_max": "丰水持续期最大值",
            "high_min": "丰水持续期最小值",
            "low_mean": "枯水持续期均值",
            "low_std": "枯水持续期标准差",
            "low_max": "枯水持续期最大值",
            "low_min": "枯水持续期最小值",
        }

        for attr, name in persistence_map.items():
            hist_val = getattr(pers_hist, attr)
            synth_val = getattr(pers_synth, attr)
            rows.append({
                "指标": name,
                "历史值": hist_val,
                "合成值": synth_val,
                "相对误差%": 100 * (synth_val - hist_val) / hist_val if hist_val != 0 else np.nan,
            })

        df = pd.DataFrame(rows)
        df["相对误差%"] = df["相对误差%"].round(2)
        return df


if __name__ == "__main__":
    print("统计特征验证模块加载成功")