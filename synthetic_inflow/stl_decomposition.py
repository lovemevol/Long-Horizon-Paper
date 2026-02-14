"""
STL季节分解模块
实现对时间序列的季节性-趋势分解（STL: Seasonal-Trend decomposition using Loess）
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from typing import Tuple, Dict

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class STLDecomposer:
    """
    STL季节分解器
    
    STL分解将时间序列分解为三个部分：
    Y(t) = T(t) + S(t) + R(t)
    
    其中：
    - T(t): 趋势项（Trend），反映长期变化趋势
    - S(t): 季节项（Seasonal），反映周期性变化模式
    - R(t): 残差项（Residual），去除趋势和季节性后的随机波动
    
    参数说明：
    - seasonal: 季节周期长度（对于日数据，年周期为365）
    - seasonal_deg: 季节性LOESS的多项式次数（0或1）
    - trend_deg: 趋势LOESS的多项式次数（0或1）
    - robust: 是否使用鲁棒拟合来处理异常值
    """
    
    def __init__(self, seasonal_period: int = 365, seasonal_deg: int = 1, 
                 trend_deg: int = 1, robust: bool = True):
        """
        初始化STL分解器
        
        Args:
            seasonal_period: 季节周期（日数据默认365天）
            seasonal_deg: 季节LOESS多项式次数
            trend_deg: 趋势LOESS多项式次数
            robust: 是否使用鲁棒拟合
        """
        self.seasonal_period = seasonal_period
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.robust = robust
        self.stl_result = None
        self.seasonal_patterns = {}  # 存储每年的季节模式
        
    def decompose(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        执行STL分解
        
        Args:
            data: 输入时间序列数据（带DatetimeIndex）
            
        Returns:
            trend: 趋势项
            seasonal: 季节项
            residual: 残差项
        """
        # 确保数据是Series并有DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是DatetimeIndex类型")
        
        # 执行STL分解
        stl = STL(data, 
                  seasonal=self.seasonal_period,
                  seasonal_deg=self.seasonal_deg,
                  trend_deg=self.trend_deg,
                  robust=self.robust)
        
        self.stl_result = stl.fit()
        
        trend = self.stl_result.trend
        seasonal = self.stl_result.seasonal
        residual = self.stl_result.resid
        
        # 提取每年的季节模式（用于后续合成）
        self._extract_seasonal_patterns(seasonal)
        
        return trend, seasonal, residual
    
    def _extract_seasonal_patterns(self, seasonal: pd.Series):
        """
        提取每年的季节模式
        
        Args:
            seasonal: 季节项序列
        """
        years = seasonal.index.year.unique()
        
        for year in years:
            year_data = seasonal[seasonal.index.year == year]
            # 标准化到365天（或366天闰年）
            day_of_year = year_data.index.dayofyear
            self.seasonal_patterns[year] = pd.Series(
                year_data.values, 
                index=day_of_year
            )
    
    def get_average_seasonal_pattern(self) -> pd.Series:
        """
        获取平均季节模式（跨年平均）
        
        Returns:
            平均季节模式（365个值）
        """
        if not self.seasonal_patterns:
            raise ValueError("请先执行decompose()方法")
        
        # 创建365天的平均季节模式
        avg_pattern = np.zeros(365)
        counts = np.zeros(365)
        
        for year, pattern in self.seasonal_patterns.items():
            for day in pattern.index:
                if day <= 365:  # 忽略闰年的第366天
                    avg_pattern[day-1] += pattern[day]
                    counts[day-1] += 1
        
        avg_pattern = avg_pattern / counts
        
        return pd.Series(avg_pattern, index=range(1, 366))
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取分解结果的统计信息
        
        Returns:
            包含各项统计指标的字典
        """
        if self.stl_result is None:
            raise ValueError("请先执行decompose()方法")
        
        trend = self.stl_result.trend
        seasonal = self.stl_result.seasonal
        residual = self.stl_result.resid
        
        stats = {
            'trend_mean': trend.mean(),
            'trend_std': trend.std(),
            'seasonal_amplitude': seasonal.max() - seasonal.min(),
            'seasonal_mean': seasonal.mean(),
            'seasonal_std': seasonal.std(),
            'residual_mean': residual.mean(),
            'residual_std': residual.std(),
            'residual_var': residual.var(),
            'trend_contribution': trend.var() / (trend.var() + seasonal.var() + residual.var()),
            'seasonal_contribution': seasonal.var() / (trend.var() + seasonal.var() + residual.var()),
            'residual_contribution': residual.var() / (trend.var() + seasonal.var() + residual.var()),
        }
        
        return stats
    
    def plot_decomposition(self, data: pd.Series, figsize: Tuple[int, int] = (15, 10),
                          save_path: str = None):
        """
        可视化STL分解结果
        
        Args:
            data: 原始时间序列
            figsize: 图形大小
            save_path: 保存路径（可选）
        """
        if self.stl_result is None:
            raise ValueError("请先执行decompose()方法")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # 原始数据
        axes[0].plot(data.index, data.values, 'b-', linewidth=0.5)
        axes[0].set_ylabel('原始数据 (m³/s)', fontsize=12)
        axes[0].set_title('STL季节分解结果', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 趋势项
        axes[1].plot(self.stl_result.trend.index, self.stl_result.trend.values, 
                     'r-', linewidth=1)
        axes[1].set_ylabel('趋势项 (m³/s)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 季节项
        axes[2].plot(self.stl_result.seasonal.index, self.stl_result.seasonal.values, 
                     'g-', linewidth=0.5)
        axes[2].set_ylabel('季节项 (m³/s)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # 残差项
        axes[3].plot(self.stl_result.resid.index, self.stl_result.resid.values, 
                     'k-', linewidth=0.5, alpha=0.7)
        axes[3].set_ylabel('残差项 (m³/s)', fontsize=12)
        axes[3].set_xlabel('日期', fontsize=12)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")
        
        plt.show()
    
    def plot_seasonal_pattern(self, figsize: Tuple[int, int] = (12, 6),
                            save_path: str = None):
        """
        可视化平均季节模式
        
        Args:
            figsize: 图形大小
            save_path: 保存路径（可选）
        """
        avg_pattern = self.get_average_seasonal_pattern()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(avg_pattern.index, avg_pattern.values, 'g-', linewidth=2)
        ax.fill_between(avg_pattern.index, avg_pattern.values, alpha=0.3, color='green')
        
        ax.set_xlabel('年内日序（天）', fontsize=12)
        ax.set_ylabel('季节项数值 (m³/s)', fontsize=12)
        ax.set_title('年内平均季节模式', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注关键月份
        month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                      '7月', '8月', '9月', '10月', '11月', '12月']
        
        ax.set_xticks(month_days)
        ax.set_xticklabels(month_names, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # 测试代码
    print("STL季节分解模块加载成功")
    print("主要功能：")
    print("1. 时间序列的STL分解（趋势+季节+残差）")
    print("2. 提取年内季节模式")
    print("3. 计算分解统计特征")
    print("4. 可视化分解结果")
