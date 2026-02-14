"""
多组实验结果对比可视化

生成多种对比图表，包括：
1. 基本统计量对比（均值、标准差、偏度、峰度）
2. 分位点对比
3. 月均流量对比
4. 持续期对比
5. 高流量指标对比
6. 时间序列对比
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from diagnostics_plot import save_multi_diagnostics
from statistical_validation import StatisticalValidator

fontsize = 20
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = fontsize
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


class ComparisonPlotter:
    """多组实验对比绘图器"""
    
    def __init__(self, results: List[Dict], output_dir: Path):
        """
        Parameters
        ----------
        results : List[Dict]
            实验结果列表，每个元素包含 name, output_dir, summary 等字段
        output_dir : Path
            对比图表输出目录
        """
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validator = StatisticalValidator()
        # 加载所有实验的数据
        self.load_all_data()
        
        # 定义颜色方案
        # 手动定义颜色方案
        self.colors = ['#1F77B4', '#FF7F0E', '#e377c2', "#2ca02c", '#1f77b4', '#a70eff', '#d62728', '#9467bd', 
                   '#8c564b', '#7f7f7f', '#bcbd22', '#17becf']
        # 确保颜色数量足够
        if len(results) > len(self.colors):
            # 如果实验数量超过预定义颜色数量，重复使用颜色
            self.colors = (self.colors * ((len(results) // len(self.colors)) + 1))[:len(results)]
        else:
            self.colors = self.colors[:len(results)]
    
    def load_all_data(self):
        """加载所有实验的数据"""
        self.experiment_data = []
        
        # 加载历史数据（从第一个实验的目录父级）
        first_exp_dir = Path(self.results[0]["output_dir"])
        data_file = first_exp_dir.parent.parent / "实测径流.xlsx"

        self.historical_series = None
        self.historical_summary = None

        if data_file.exists():
            df = pd.read_excel(data_file, header=None, names=["date", "flow"], parse_dates=[0])
            df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
            df = df.dropna()
            df = df.sort_values("date")
            df = df.set_index("date")
            df = df[~((df.index.month == 2) & (df.index.day == 29))]
            df = df.asfreq("D")
            df["flow"] = df["flow"].infer_objects(copy=False).interpolate(limit_direction="both")
            self.historical_series = df["flow"]
            try:
                self.historical_summary = self.validator.summarize(self.historical_series)
            except Exception as exc:
                print(f"⚠️  历史数据统计计算失败: {exc}")
        else:
            print(f"⚠️  警告: 未找到历史数据文件 {data_file}")
        
        for result in self.results:
            exp_dir = Path(result["output_dir"])
            
            # 加载合成序列
            synthetic_file = exp_dir / "synthetic_inflow_series.csv"
            synthetic_series = pd.read_csv(synthetic_file, index_col=0, parse_dates=True).squeeze()

            try:
                stats_summary = self.validator.summarize(synthetic_series)
            except Exception as exc:
                print(f"⚠️  统计摘要计算失败 ({result['name']}): {exc}")
                stats_summary = None

            # 加载统计对比
            comparison_file = exp_dir / "statistical_comparison.csv"
            comparison_df = pd.read_csv(comparison_file)

            self.experiment_data.append({
                "name": result["name"],
                "description": result.get("description", ""),
                "synthetic_series": synthetic_series,
                "comparison_df": comparison_df,
                "summary": result["summary"],
                "stats_summary": stats_summary,
            })
    
    def plot_basic_stats_comparison(self):
        """绘制基本统计量对比条形图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        stats = ["mean", "std", "skew", "kurtosis"]
        titles = ["Mean (m³/s)", "Std Dev (m³/s)", "Skewness", "Kurtosis"]
        
        for idx, (stat, title) in enumerate(zip(stats, titles)):
            ax = axes[idx]
            
            # 提取历史值和各实验合成值
            if self.historical_summary is not None:
                hist_val = self.historical_summary.basic_stats.get(stat, np.nan)
            else:
                hist_val = self.experiment_data[0]["summary"]["historical_basic"].get(stat, np.nan)

            synth_vals = []
            for exp in self.experiment_data:
                if exp["stats_summary"] is not None:
                    synth_vals.append(exp["stats_summary"].basic_stats.get(stat, np.nan))
                else:
                    synth_vals.append(exp["summary"]["synthetic_basic"].get(stat, np.nan))
            names = [exp["name"] for exp in self.experiment_data]
            
            # 绘制条形图
            x = np.arange(len(names))
            width = 0.6
            
            ax.axhline(y=hist_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Historical: {hist_val:.2f}', zorder=10)
            bars = ax.bar(x, synth_vals, width, color=self.colors, alpha=0.8)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, synth_vals)):
                height = bar.get_height()
                if np.isfinite(hist_val) and hist_val != 0:
                    error = abs(val - hist_val) / abs(hist_val) * 100
                else:
                    error = np.nan
                label = f"{val:.2f}"
                if np.isfinite(error):
                    label += f"\n({error:.1f}%)"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize - 2,
                )
            
            ax.set_ylabel(title)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.legend(frameon=False, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_basic_stats_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 基本统计量对比图已生成")
    
    def plot_quantile_comparison(self):
        """绘制分位点对比"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        
        # 绘制历史数据
        if self.historical_series is not None:
            hist_q_vals = [self.historical_series.quantile(q) for q in quantiles]
            ax.plot(quantiles, hist_q_vals, marker='D', label='Historical', 
                   color='red', linewidth=3, markersize=10, linestyle='--', zorder=100)
        
        # 绘制各实验合成数据
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            q_vals = [series.quantile(q) for q in quantiles]
            ax.plot(quantiles, q_vals, marker='o', label=exp["name"], 
                   color=self.colors[idx], linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Discharge (m³/s)")
        ax.set_title("Quantile Comparison Across Experiments")
        ax.legend(frameon=False, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_quantile_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 分位点对比图已生成")
    
    def plot_monthly_mean_comparison(self):
        """绘制月均流量对比"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        months = range(1, 13)
        
        # 绘制历史数据
        if self.historical_series is not None:
            hist_monthly = self.historical_series.groupby(self.historical_series.index.month).mean()
            ax.plot(months, hist_monthly.values, marker='D', label='Historical',
                   color='red', linewidth=3, markersize=10, linestyle='--', zorder=100)
        
        # 绘制各实验数据
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            monthly_mean = series.groupby(series.index.month).mean()
            ax.plot(months, monthly_mean.values, marker='s', label=exp["name"],
                   color=self.colors[idx], linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean Discharge (m³/s)")
        ax.set_title("Monthly Mean Discharge Comparison")
        ax.set_xticks(months)
        ax.legend(frameon=False, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_monthly_mean_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 月均流量对比图已生成")
    
    def plot_distribution_comparison(self):
        """绘制分布对比（直方图和密度曲线）"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 直方图
        ax1 = axes[0]
        bins = np.linspace(0, 40000, 50)
        
        # 绘制历史数据
        if self.historical_series is not None:
            ax1.hist(self.historical_series, bins=bins, alpha=0.7, label='Historical',
                    color='red', density=True, histtype='step', linewidth=3)
        
        # 绘制各实验数据
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            ax1.hist(series, bins=bins, alpha=0.5, label=exp["name"],
                    color=self.colors[idx], density=True)
        ax1.set_xlabel("Discharge (m³/s)")
        ax1.set_ylabel("Density")
        ax1.set_title("Distribution Comparison (Histogram)")
        ax1.legend(frameon=False, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 累积分布函数
        ax2 = axes[1]
        
        # 绘制历史数据
        if self.historical_series is not None:
            hist_sorted = np.sort(self.historical_series)
            hist_cumulative = np.arange(1, len(hist_sorted) + 1) / len(hist_sorted)
            ax2.plot(hist_sorted, hist_cumulative, label='Historical',
                    color='red', linewidth=3, linestyle='--', zorder=100)
        
        # 绘制各实验数据
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            sorted_data = np.sort(series)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax2.plot(sorted_data, cumulative, label=exp["name"],
                    color=self.colors[idx], linewidth=2, alpha=0.8)
        ax2.set_xlabel("Discharge (m³/s)")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title("Cumulative Distribution Function")
        ax2.legend(frameon=False, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 40000)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_distribution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 分布对比图已生成")
    
    def plot_extreme_values_comparison(self):
        """绘制极值统计对比"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 年最大值对比
        ax1 = axes[0]
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            annual_max = series.groupby(series.index.year).max()
            ax1.plot(annual_max.index, annual_max.values, marker='o',
                    label=exp["name"], color=self.colors[idx], alpha=0.7)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Annual Maximum (m³/s)")
        ax1.set_title("Annual Maximum Discharge")
        ax1.legend(frameon=False, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 年最小值对比
        ax2 = axes[1]
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            annual_min = series.groupby(series.index.year).min()
            ax2.plot(annual_min.index, annual_min.values, marker='o',
                    label=exp["name"], color=self.colors[idx], alpha=0.7)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Annual Minimum (m³/s)")
        ax2.set_title("Annual Minimum Discharge")
        ax2.legend(frameon=False, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "05_extreme_values_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 极值统计对比图已生成")
    
    def plot_variability_comparison(self):
        """绘制变异性对比"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 年内变异系数
        ax1 = axes[0]
        cv_values = []
        names = []
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            annual_cv = series.groupby(series.index.year).apply(lambda x: x.std() / x.mean())
            cv_values.append(annual_cv.mean())
            names.append(exp["name"])
        
        x_pos = np.arange(len(names))
        bars = ax1.bar(x_pos, cv_values, color=self.colors, alpha=0.8)
        for bar, val in zip(bars, cv_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=fontsize-2)
        ax1.set_ylabel("Mean Annual CV")
        ax1.set_title("Average Annual Coefficient of Variation")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 月度标准差
        ax2 = axes[1]
        months = range(1, 13)
        
        # 绘制历史数据
        if self.historical_series is not None:
            hist_monthly_std = self.historical_series.groupby(self.historical_series.index.month).std()
            ax2.plot(months, hist_monthly_std.values, marker='D', label='Historical',
                    color='red', linewidth=3, markersize=10, linestyle='--', zorder=100)
        
        # 绘制各实验数据
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            monthly_std = series.groupby(series.index.month).std()
            ax2.plot(months, monthly_std.values, marker='s', label=exp["name"],
                    color=self.colors[idx], linewidth=2, markersize=8, alpha=0.8)
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Standard Deviation (m³/s)")
        ax2.set_title("Monthly Standard Deviation")
        ax2.set_xticks(months)
        ax2.legend(frameon=False, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "06_variability_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 变异性对比图已生成")
    
    def plot_time_series_sample(self):
        """绘制时间序列样本对比（前3年）"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # 上图：历史数据（最近3年）
        if self.historical_series is not None:
            ax1 = axes[0]
            # 获取最近3年的历史数据
            end_year = self.historical_series.index[-1].year
            start_year = end_year - 3
            mask = self.historical_series.index.year >= start_year
            hist_sample = self.historical_series[mask]
            ax1.plot(hist_sample.index, hist_sample.values, 
                    color='red', linewidth=1.5, alpha=0.8, label='Historical')
            ax1.set_ylabel("Discharge (m³/s)")
            ax1.set_title(f"Historical Data (Last 4 Years: {start_year}-{end_year})")
            ax1.legend(frameon=False, loc='best')
            ax1.grid(True, alpha=0.3)
        
        # 下图：各实验合成数据（前3年）
        ax2 = axes[1]
        for idx, exp in enumerate(self.experiment_data):
            series = exp["synthetic_series"]
            # 只绘制前3年
            start_year = series.index[0].year
            mask = series.index.year < start_year + 4
            sample = series[mask]
            ax2.plot(sample.index, sample.values, label=exp["name"],
                   color=self.colors[idx], linewidth=1.5, alpha=0.8)
        
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Discharge (m³/s)")
        ax2.set_title("Synthetic Data (First 4 Years)")
        ax2.legend(frameon=False, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "07_time_series_sample.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 时间序列样本对比图已生成")
    
    def plot_error_metrics(self):
        """绘制相对于历史数据的误差指标"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 计算各统计量的相对误差
        metrics = ["mean", "std", "skew", "kurtosis"]
        metric_labels = ["Mean", "Std Dev", "Skewness", "Kurtosis"]
        
        x = np.arange(len(self.experiment_data))
        width = 0.2
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            if self.historical_summary is not None:
                hist_val = self.historical_summary.basic_stats.get(metric, np.nan)
            else:
                hist_val = self.experiment_data[0]["summary"]["historical_basic"].get(metric, np.nan)
            errors = []
            for exp in self.experiment_data:
                if exp["stats_summary"] is not None:
                    synth_val = exp["stats_summary"].basic_stats.get(metric, np.nan)
                else:
                    synth_val = exp["summary"]["synthetic_basic"].get(metric, np.nan)
                if np.isfinite(hist_val) and hist_val != 0:
                    error = abs(synth_val - hist_val) / abs(hist_val) * 100
                else:
                    error = np.nan
                errors.append(error)
            
            offset = (idx - len(metrics)/2) * width
            ax.bar(x + offset, errors, width, label=label, alpha=0.8)
        
        ax.set_ylabel("Relative Error (%)")
        ax.set_title("Relative Error in Basic Statistics (vs Historical)")
        ax.set_xticks(x)
        ax.set_xticklabels([exp["name"] for exp in self.experiment_data], 
                          rotation=45, ha='right')
        ax.legend(frameon=False, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "08_error_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 误差指标对比图已生成")
    
    def generate_summary_table(self):
        """生成汇总对比表格"""
        rows = []
        
        # 添加历史数据行
        if self.historical_series is not None:
            hist_row = {
                "实验名称": "Historical",
                "描述": "历史实测数据 (1965-2025)",
                "均值 (m³/s)": float(self.historical_series.mean()),
                "标准差 (m³/s)": float(self.historical_series.std()),
                "偏度": float(self.historical_series.skew()),
                "峰度": float(self.historical_series.kurtosis()),
                "最小值 (m³/s)": float(self.historical_series.min()),
                "最大值 (m³/s)": float(self.historical_series.max()),
            }
            rows.append(hist_row)
        
        # 添加各实验数据行
        for exp in self.experiment_data:
            series = exp["synthetic_series"]
            if exp["stats_summary"] is not None:
                basic_stats = exp["stats_summary"].basic_stats
            else:
                basic_stats = exp["summary"]["synthetic_basic"]
            
            row = {
                "实验名称": exp["name"],
                "描述": exp["description"],
                "均值 (m³/s)": basic_stats.get("mean", np.nan),
                "标准差 (m³/s)": basic_stats.get("std", np.nan),
                "偏度": basic_stats.get("skew", np.nan),
                "峰度": basic_stats.get("kurtosis", np.nan),
                "最小值 (m³/s)": float(series.min()),
                "最大值 (m³/s)": float(series.max()),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "summary_table.csv", index=False, encoding="utf-8-sig")
        
        # 同时保存为格式化的文本表格
        with open(self.output_dir / "summary_table.txt", "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
        
        print("✓ 汇总对比表格已生成")
    
    def plot_all(self):
        """生成所有对比图表"""
        print("\n" + "="*60)
        print("开始生成多组实验对比图表...")
        print("="*60 + "\n")
        
        if self.historical_series is not None and self.historical_summary is not None:
            synthetic_entries = []
            for exp in self.experiment_data:
                summary = exp["stats_summary"]
                if summary is None:
                    try:
                        summary = self.validator.summarize(exp["synthetic_series"])
                    except Exception:
                        continue
                synthetic_entries.append({
                    "name": exp["name"],
                    "series": exp["synthetic_series"],
                    "summary": summary,
                })
            if synthetic_entries:
                save_multi_diagnostics(
                    self.historical_summary,
                    self.historical_series,
                    synthetic_entries,
                    self.output_dir,
                )
                print("✓ 诊断样式多实验对比图已生成")

        self.plot_basic_stats_comparison()
        self.plot_quantile_comparison()
        self.plot_monthly_mean_comparison()
        self.plot_distribution_comparison()
        self.plot_extreme_values_comparison()
        self.plot_variability_comparison()
        self.plot_time_series_sample()
        self.plot_error_metrics()
        self.generate_summary_table()
        
        print("\n" + "="*60)
        print(f"所有对比图表已生成！共 {len(self.results)} 组实验")
        print(f"输出目录: {self.output_dir}")
        print("="*60 + "\n")
