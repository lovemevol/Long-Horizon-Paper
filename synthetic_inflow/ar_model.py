"""
AR模型模块
实现对残差序列的自回归建模和模拟
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ARModel:
    """
    自回归模型（AR Model）
    
    AR(p)模型数学形式：
    X(t) = c + φ₁X(t-1) + φ₂X(t-2) + ... + φₚX(t-p) + ε(t)
    
    其中：
    - X(t): t时刻的观测值
    - c: 常数项
    - φᵢ: 第i阶自回归系数
    - p: 自回归阶数
    - ε(t): 白噪声误差项，ε(t) ~ N(0, σ²)
    
    参数选择：
    - 通过AIC/BIC准则自动选择最优阶数
    - 通过PACF图判断截尾阶数
    """
    
    def __init__(self, max_lag: int = 30, ic: str = 'aic'):
        """
        初始化AR模型
        
        Args:
            max_lag: 最大滞后阶数
            ic: 信息准则 ('aic', 'bic', 'hqic')
        """
        self.max_lag = max_lag
        self.ic = ic
        self.model = None
        self.fitted_model = None
        self.optimal_lag = None
        self.residuals = None
        self.params = None
        
    def select_order(self, data: pd.Series, plot: bool = True) -> int:
        """
        通过信息准则选择最优AR阶数
        
        Args:
            data: 输入时间序列（残差序列）
            plot: 是否绘制ACF和PACF图
            
        Returns:
            最优滞后阶数
        """
        data_values = data.dropna()
        if np.isclose(data_values.std(ddof=1), 0.0):
            print("\n残差序列近似常数，跳过ADF检验，采用AR(1)模型")
            self.optimal_lag = 1
            return self.optimal_lag

        # 平稳性检验
        adf_result = adfuller(data_values)
        print(f"\nADF平稳性检验:")
        print(f"  ADF统计量: {adf_result[0]:.4f}")
        print(f"  p值: {adf_result[1]:.4f}")
        print(f"  临界值: {adf_result[4]}")

        if adf_result[1] > 0.05:
            print("  警告：序列可能非平稳，建议进行差分")
        else:
            print("  序列平稳")
        
        # 计算不同阶数的信息准则
        ic_values = []
        lags_range = range(1, min(self.max_lag + 1, len(data) // 2))
        
        for lag in lags_range:
            try:
                model = AutoReg(data.dropna(), lags=lag, trend='c')
                fitted = model.fit()
                
                if self.ic == 'aic':
                    ic_values.append(fitted.aic)
                elif self.ic == 'bic':
                    ic_values.append(fitted.bic)
                elif self.ic == 'hqic':
                    ic_values.append(fitted.hqic)
            except:
                ic_values.append(np.inf)
        
        # 选择最优阶数
        self.optimal_lag = lags_range[np.argmin(ic_values)]
        print(f"\n最优AR阶数（基于{self.ic.upper()}）: {self.optimal_lag}")
        
        # 绘制ACF和PACF
        if plot:
            self._plot_acf_pacf(data)
        
        return self.optimal_lag
    
    def fit(self, data: pd.Series, lag: Optional[int] = None) -> Dict:
        """
        拟合AR模型
        
        Args:
            data: 输入时间序列（残差序列）
            lag: 指定滞后阶数（如不指定则使用optimal_lag）
            
        Returns:
            模型参数字典
        """
        if lag is None:
            if self.optimal_lag is None:
                lag = self.select_order(data, plot=False)
            else:
                lag = self.optimal_lag
        
        # 拟合模型
        self.model = AutoReg(data.dropna(), lags=lag, trend='c')
        self.fitted_model = self.model.fit()
        self.residuals = self.fitted_model.resid
        
        # 提取参数
        self.params = {
            'const': self.fitted_model.params[0],
            'ar_coefs': self.fitted_model.params[1:].values,
            'sigma': np.std(self.residuals),
            'lag': lag,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
        }
        
        print(f"\nAR({lag})模型拟合完成:")
        print(f"  常数项 c: {self.params['const']:.6f}")
        print(f"  AR系数 φ: {self.params['ar_coefs']}")
        print(f"  残差标准差 σ: {self.params['sigma']:.6f}")
        print(f"  AIC: {self.params['aic']:.2f}")
        print(f"  BIC: {self.params['bic']:.2f}")
        
        return self.params
    
    def simulate(self, n_steps: int, burn_in: int = 100) -> np.ndarray:
        """
        模拟生成AR序列
        
        使用递推公式：
        X(t) = c + Σφᵢ·X(t-i) + ε(t)
        
        Args:
            n_steps: 生成序列长度
            burn_in: 预热期长度（丢弃前burn_in个点以消除初始值影响）
            
        Returns:
            模拟序列
        """
        if self.fitted_model is None:
            raise ValueError("请先调用fit()方法拟合模型")
        
        lag = self.params['lag']
        const = self.params['const']
        ar_coefs = self.params['ar_coefs']
        sigma = self.params['sigma']
        
        # 总长度包括预热期
        total_length = n_steps + burn_in
        
        # 初始化序列（从正态分布采样）
        series = np.random.normal(0, sigma, total_length)
        
        # 递推生成AR序列
        for t in range(lag, total_length):
            ar_term = sum(ar_coefs[i] * series[t-i-1] for i in range(lag))
            noise = np.random.normal(0, sigma)
            series[t] = const + ar_term + noise
        
        # 去掉预热期
        return series[burn_in:]
    
    def simulate_with_initial(self, n_steps: int, 
                             initial_values: np.ndarray) -> np.ndarray:
        """
        基于给定初始值模拟AR序列（用于连续生成）
        
        Args:
            n_steps: 生成序列长度
            initial_values: 初始值（长度应为lag）
            
        Returns:
            模拟序列
        """
        if self.fitted_model is None:
            raise ValueError("请先调用fit()方法拟合模型")
        
        lag = self.params['lag']
        const = self.params['const']
        ar_coefs = self.params['ar_coefs']
        sigma = self.params['sigma']
        
        if len(initial_values) < lag:
            raise ValueError(f"初始值长度必须至少为{lag}")
        
        # 初始化序列
        series = np.zeros(n_steps + lag)
        series[:lag] = initial_values[-lag:]
        
        # 递推生成
        for t in range(lag, n_steps + lag):
            ar_term = sum(ar_coefs[i] * series[t-i-1] for i in range(lag))
            noise = np.random.normal(0, sigma)
            series[t] = const + ar_term + noise
        
        return series[lag:]
    
    def _plot_acf_pacf(self, data: pd.Series, figsize: Tuple[int, int] = (12, 5)):
        """
        绘制自相关和偏自相关函数图
        
        Args:
            data: 输入序列
            figsize: 图形大小
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ACF图
        plot_acf(data.dropna(), lags=min(40, len(data)//2), ax=axes[0])
        axes[0].set_title('自相关函数 (ACF)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('滞后阶数', fontsize=10)
        axes[0].set_ylabel('相关系数', fontsize=10)
        
        # PACF图
        plot_pacf(data.dropna(), lags=min(40, len(data)//2), ax=axes[1])
        axes[1].set_title('偏自相关函数 (PACF)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('滞后阶数', fontsize=10)
        axes[1].set_ylabel('偏相关系数', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, figsize: Tuple[int, int] = (12, 8),
                      save_path: str = None):
        """
        绘制残差诊断图
        
        Args:
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.residuals is None:
            raise ValueError("请先调用fit()方法拟合模型")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 残差时序图
        axes[0, 0].plot(self.residuals.index, self.residuals.values, 'b-', linewidth=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[0, 0].set_title('残差时序图', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('时间', fontsize=9)
        axes[0, 0].set_ylabel('残差', fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 残差直方图
        axes[0, 1].hist(self.residuals.dropna(), bins=50, density=True, 
                       alpha=0.7, color='blue', edgecolor='black')
        
        # 叠加正态分布曲线
        mu, sigma = self.residuals.mean(), self.residuals.std()
        x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        axes[0, 1].plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * 
                       np.exp(-(x - mu)**2 / (2 * sigma**2)), 
                       'r-', linewidth=2, label='理论正态分布')
        axes[0, 1].set_title('残差分布直方图', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('残差值', fontsize=9)
        axes[0, 1].set_ylabel('概率密度', fontsize=9)
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 残差ACF
        plot_acf(self.residuals.dropna(), lags=min(40, len(self.residuals)//2), 
                ax=axes[1, 0])
        axes[1, 0].set_title('残差自相关图 (ACF)', fontsize=11, fontweight='bold')
        
        # Q-Q图
        from scipy import stats
        stats.probplot(self.residuals.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q图（正态性检验）', fontsize=11, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")
        
        plt.show()
    
    def get_diagnostic_stats(self) -> Dict:
        """
        获取模型诊断统计量
        
        Returns:
            诊断统计量字典
        """
        if self.residuals is None:
            raise ValueError("请先调用fit()方法拟合模型")
        
        from scipy import stats
        
        # Ljung-Box检验（残差独立性）
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(self.residuals.dropna(), lags=[10], return_df=True)
        
        # Jarque-Bera检验（正态性）
        jb_stat, jb_pvalue = stats.jarque_bera(self.residuals.dropna())
        
        stats_dict = {
            'mean': self.residuals.mean(),
            'std': self.residuals.std(),
            'skewness': stats.skew(self.residuals.dropna()),
            'kurtosis': stats.kurtosis(self.residuals.dropna()),
            'ljung_box_stat': lb_test['lb_stat'].values[0],
            'ljung_box_pvalue': lb_test['lb_pvalue'].values[0],
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
        }
        
        print("\n模型诊断统计量:")
        print(f"  残差均值: {stats_dict['mean']:.6f}")
        print(f"  残差标准差: {stats_dict['std']:.6f}")
        print(f"  偏度: {stats_dict['skewness']:.4f}")
        print(f"  峰度: {stats_dict['kurtosis']:.4f}")
        print(f"  Ljung-Box检验 p值: {stats_dict['ljung_box_pvalue']:.4f} " + 
              ("(残差独立)" if stats_dict['ljung_box_pvalue'] > 0.05 else "(残差相关)"))
        print(f"  Jarque-Bera检验 p值: {stats_dict['jarque_bera_pvalue']:.4f} " + 
              ("(正态分布)" if stats_dict['jarque_bera_pvalue'] > 0.05 else "(非正态)"))
        
        return stats_dict


if __name__ == "__main__":
    # 测试代码
    print("AR模型模块加载成功")
    print("主要功能：")
    print("1. 自动选择最优AR阶数（AIC/BIC准则）")
    print("2. 拟合AR模型并提取参数")
    print("3. 模拟生成AR序列")
    print("4. 残差诊断和模型检验")
