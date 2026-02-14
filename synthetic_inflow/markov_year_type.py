"""
马尔可夫年型转换模块
实现年型识别和基于马尔可夫链的年型序列生成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MarkovYearType:
    """
    马尔可夫年型转换模型
    
    年型分类：
    基于年径流量将历史年份分为不同类型（如丰水年、平水年、枯水年）
    分类方法：P = (Qᵢ - Q̄) / σ
    
    其中：
    - Qᵢ: 第i年的年径流量
    - Q̄: 多年平均年径流量
    - σ: 年径流量标准差
    - P: 频率指标
    
    分类标准（可调整）：
    - 特丰水年: P ≥ 0.8
    - 丰水年: 0.3 ≤ P < 0.8
    - 平水年: -0.3 ≤ P < 0.3
    - 枯水年: -0.8 ≤ P < -0.3
    - 特枯水年: P < -0.8
    
    马尔可夫链：
    P(Sₜ = j | Sₜ₋₁ = i) = pᵢⱼ
    
    状态转移矩阵P = [pᵢⱼ]，其中pᵢⱼ表示从状态i转移到状态j的概率
    """
    
    def __init__(self, n_types: int = 5, 
                 type_names: Optional[List[str]] = None,
                 thresholds: Optional[List[float]] = None):
        """
        初始化马尔可夫年型模型
        
        Args:
            n_types: 年型数量（3类或5类）
            type_names: 年型名称列表
            thresholds: 分类阈值（按P值划分）
        """
        self.n_types = n_types
        
        if type_names is None:
            if n_types == 5:
                self.type_names = ['特丰水年', '丰水年', '平水年', '枯水年', '特枯水年']
            elif n_types == 3:
                self.type_names = ['丰水年', '平水年', '枯水年']
            else:
                self.type_names = [f'类型{i+1}' for i in range(n_types)]
        else:
            self.type_names = type_names
        
        if thresholds is None:
            if n_types == 5:
                # 特丰/丰/平/枯/特枯
                self.thresholds = [0.8, 0.3, -0.3, -0.8]
            elif n_types == 3:
                # 丰/平/枯
                self.thresholds = [0.5, -0.5]
            else:
                # 等分位数
                self.thresholds = list(np.linspace(1, -1, n_types + 1)[1:-1])
        else:
            self.thresholds = thresholds
        
        self.year_types = None
        self.annual_flow = None
        self.transition_matrix = None
        self.stationary_dist = None
        
    def classify_years(self, data: pd.Series, return_stats: bool = True) -> pd.Series:
        """
        对历史年份进行分类
        
        Args:
            data: 日流量序列（带DatetimeIndex）
            return_stats: 是否返回统计信息
            
        Returns:
            年型序列（年份为索引，年型为值）
        """

        # 过滤不完整年份（天数不足365天）
        day_counts = data.groupby(data.index.year).count()
        valid_years = day_counts[day_counts >= 365].index
        filtered = data[data.index.year.isin(valid_years)]

        # 计算年径流量
        self.annual_flow = filtered.groupby(filtered.index.year).sum()

        # 计算频率指标P
        mean_flow = self.annual_flow.mean()
        std_flow = self.annual_flow.std()
        P = (self.annual_flow - mean_flow) / std_flow

        # 根据阈值分类
        year_types = pd.Series(index=P.index, dtype=float)

        for i, year in enumerate(P.index):
            p_value = P.iloc[i]

            # 从高到低判断
            type_idx = self.n_types - 1  # 默认最后一类
            for j, threshold in enumerate(self.thresholds):
                if p_value >= threshold:
                    type_idx = j
                    break

            year_types.iloc[i] = type_idx

        self.year_types = year_types.astype(int)

        # 打印统计信息
        if return_stats:
            print("\n年型分类统计:")
            print(f"  多年平均年径流量: {mean_flow:.2f} m³/s")
            print(f"  年径流量标准差: {std_flow:.2f} m³/s")
            print(f"  分类阈值: {self.thresholds}")
            print("\n各年型数量统计:")

            for type_idx in range(self.n_types):
                count = (year_types == type_idx).sum()
                freq = count / len(year_types) * 100
                print(f"  {self.type_names[type_idx]}: {count}年 ({freq:.1f}%)")

        return year_types
    def estimate_transition_matrix(self) -> np.ndarray:
        """
        估计状态转移概率矩阵
        
        转移概率计算：
        pᵢⱼ = nᵢⱼ / Σⱼnᵢⱼ
        
        其中nᵢⱼ是从状态i转移到状态j的次数
        
        Returns:
            转移概率矩阵 (n_types × n_types)
        """
        if self.year_types is None:
            raise ValueError("请先调用classify_years()方法")
        
        # 初始化转移计数矩阵
        transition_count = np.zeros((self.n_types, self.n_types))
        
        # 统计转移次数
        for i in range(len(self.year_types) - 1):
            current_type = self.year_types.iloc[i]
            next_type = self.year_types.iloc[i + 1]
            transition_count[current_type, next_type] += 1
        
        # 计算转移概率（行归一化）
        self.transition_matrix = np.zeros((self.n_types, self.n_types))
        
        for i in range(self.n_types):
            row_sum = transition_count[i, :].sum()
            if row_sum > 0:
                self.transition_matrix[i, :] = transition_count[i, :] / row_sum
            else:
                # 如果某个状态没有出现，使用均匀分布
                self.transition_matrix[i, :] = 1.0 / self.n_types
        
        print("\n状态转移概率矩阵:")
        print(self._format_matrix(self.transition_matrix))
        
        # 计算平稳分布
        self._compute_stationary_distribution()
        
        return self.transition_matrix
    
    def _compute_stationary_distribution(self):
        """
        计算马尔可夫链的平稳分布
        
        平稳分布π满足：π = πP，且Σπᵢ = 1
        
        通过求解特征值问题：πᵀ(P - I) = 0
        """
        # 求解转移矩阵的左特征向量（特征值为1）
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # 找到最接近1的特征值
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        
        # 归一化
        self.stationary_dist = stationary / stationary.sum()
        
        print("\n平稳分布（理论长期频率）:")
        for i in range(self.n_types):
            print(f"  {self.type_names[i]}: {self.stationary_dist[i]:.4f} ({self.stationary_dist[i]*100:.1f}%)")
    
    def generate_year_type_sequence(self, n_years: int, 
                                   initial_type: Optional[int] = None,
                                   random_seed: Optional[int] = None) -> np.ndarray:
        """
        生成年型序列
        
        Args:
            n_years: 生成年份数
            initial_type: 初始年型（如不指定则随机选择）
            random_seed: 随机种子
            
        Returns:
            年型序列数组
        """
        if self.transition_matrix is None:
            raise ValueError("请先调用estimate_transition_matrix()方法")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 初始化序列
        year_type_seq = np.zeros(n_years, dtype=int)
        
        # 确定初始状态
        if initial_type is None:
            # 根据平稳分布随机选择初始状态
            initial_type = np.random.choice(self.n_types, p=self.stationary_dist)
        
        year_type_seq[0] = initial_type
        
        # 根据转移矩阵生成序列
        for t in range(1, n_years):
            current_type = year_type_seq[t - 1]
            # 根据当前状态的转移概率选择下一状态
            year_type_seq[t] = np.random.choice(
                self.n_types, 
                p=self.transition_matrix[current_type, :]
            )
        
        return year_type_seq
    
    def get_year_flow_range(self, year_type: int) -> Tuple[float, float, float]:
        """
        获取指定年型的流量统计特征
        
        Args:
            year_type: 年型索引
            
        Returns:
            (均值, 标准差, 中位数)
        """
        if self.annual_flow is None or self.year_types is None:
            raise ValueError("请先调用classify_years()方法")
        
        # 筛选该年型的历史年份
        mask = self.year_types == year_type
        type_flows = self.annual_flow[mask]
        
        if len(type_flows) == 0:
            # 如果没有该类型的历史数据，返回总体统计
            return self.annual_flow.mean(), self.annual_flow.std(), self.annual_flow.median()
        
        return type_flows.mean(), type_flows.std(), type_flows.median()
    
    def _format_matrix(self, matrix: np.ndarray) -> str:
        """格式化矩阵输出"""
        lines = []
        
        # 表头
        header = "       " + "  ".join([f"{name[:4]:>6}" for name in self.type_names])
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        
        # 数据行
        for i in range(self.n_types):
            row = f"{self.type_names[i][:4]:>6} |"
            for j in range(self.n_types):
                row += f" {matrix[i, j]:>6.3f}"
            lines.append(row)
        
        return "\n  ".join(lines)
    
    def plot_transition_matrix(self, figsize: Tuple[int, int] = (10, 8),
                              save_path: str = None):
        """
        可视化转移概率矩阵
        
        Args:
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.transition_matrix is None:
            raise ValueError("请先调用estimate_transition_matrix()方法")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用热力图
        im = ax.imshow(self.transition_matrix, cmap='YlOrRd', aspect='auto', 
                      vmin=0, vmax=1)
        
        # 设置刻度
        ax.set_xticks(np.arange(self.n_types))
        ax.set_yticks(np.arange(self.n_types))
        ax.set_xticklabels(self.type_names, fontsize=10)
        ax.set_yticklabels(self.type_names, fontsize=10)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(self.n_types):
            for j in range(self.n_types):
                text = ax.text(j, i, f'{self.transition_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xlabel('转移到', fontsize=12, fontweight='bold')
        ax.set_ylabel('当前状态', fontsize=12, fontweight='bold')
        ax.set_title('年型马尔可夫转移概率矩阵', fontsize=14, fontweight='bold', pad=20)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('转移概率', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")
        
        plt.show()
    
    def plot_year_type_distribution(self, figsize: Tuple[int, int] = (12, 6),
                                   save_path: str = None):
        """
        可视化历史年型分布和序列
        
        Args:
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.year_types is None:
            raise ValueError("请先调用classify_years()方法")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：年型时序图
        colors = plt.cm.RdYlBu(np.linspace(0, 1, self.n_types))
        
        for year, year_type in self.year_types.items():
            axes[0].bar(year, 1, color=colors[year_type], 
                       edgecolor='black', linewidth=0.5)
        
        axes[0].set_xlabel('年份', fontsize=11)
        axes[0].set_ylabel('年型', fontsize=11)
        axes[0].set_title('历史年型时序分布', fontsize=12, fontweight='bold')
        axes[0].set_yticks(np.arange(self.n_types) / (self.n_types - 1) + 0.5 / self.n_types)
        axes[0].set_yticklabels(self.type_names, fontsize=9)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # 右图：年型频率柱状图
        type_counts = [sum(self.year_types == i) for i in range(self.n_types)]
        type_freqs = [count / len(self.year_types) * 100 for count in type_counts]
        
        bars = axes[1].bar(range(self.n_types), type_freqs, color=colors,
                          edgecolor='black', linewidth=1.5)
        
        # 添加数值标注
        for i, (bar, count, freq) in enumerate(zip(bars, type_counts, type_freqs)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}年\n{freq:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 添加平稳分布对比线（如果已计算）
        if self.stationary_dist is not None:
            axes[1].plot(range(self.n_types), self.stationary_dist * 100, 
                        'ro-', linewidth=2, markersize=8, 
                        label='理论平稳分布', zorder=10)
            axes[1].legend(fontsize=9)
        
        axes[1].set_xlabel('年型', fontsize=11)
        axes[1].set_ylabel('频率 (%)', fontsize=11)
        axes[1].set_title('年型频率分布', fontsize=12, fontweight='bold')
        axes[1].set_xticks(range(self.n_types))
        axes[1].set_xticklabels(self.type_names, rotation=45, ha='right', fontsize=9)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # 测试代码
    print("马尔可夫年型转换模块加载成功")
    print("主要功能：")
    print("1. 基于频率指标的年型分类")
    print("2. 估计状态转移概率矩阵")
    print("3. 计算平稳分布")
    print("4. 生成年型序列")
    print("5. 可视化转移矩阵和年型分布")
