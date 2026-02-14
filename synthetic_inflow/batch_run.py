"""
批量运行多组配置的合成入流生成

使用方法：
1. 在 CONFIGURATIONS 中定义多组配置
2. 运行 python batch_run.py
3. 查看 output_comparison/ 目录下的对比图表
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List

import pandas as pd

from main import GeneratorConfig, SyntheticInflowGenerator


@dataclass
class ExperimentConfig:
    """单个实验配置"""
    name: str  # 实验名称
    config: GeneratorConfig  # 生成器配置
    description: str = ""  # 实验描述


class BatchRunner:
    """批量实验运行器"""
    
    def __init__(self, base_output_dir: pathlib.Path = None):
        if base_output_dir is None:
            base_output_dir = pathlib.Path(__file__).resolve().parent / "output_batch"
        elif isinstance(base_output_dir, str):
            base_output_dir = pathlib.Path(base_output_dir)
        self.base_output_dir = base_output_dir
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict] = []
        self.comparison_dir = self.base_output_dir / "comparison"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
    
    def add_experiment(self, experiment: ExperimentConfig):
        """添加实验到运行队列"""
        print(f"\n{'='*60}")
        print(f"运行实验: {experiment.name}")
        if experiment.description:
            print(f"描述: {experiment.description}")
        print(f"{'='*60}\n")
        
        # 为每个实验创建独立的输出目录
        exp_output_dir = self.base_output_dir / experiment.name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        experiment.config.output_dir = exp_output_dir
        
        # 运行实验
        generator = SyntheticInflowGenerator(experiment.config)
        summary = generator.run()
        
        # 保存实验配置
        config_dict = asdict(experiment.config)
        config_dict['output_dir'] = str(config_dict['output_dir'])
        with open(exp_output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        # 记录结果
        result = {
            "name": experiment.name,
            "description": experiment.description,
            "output_dir": str(exp_output_dir),
            "summary": summary,
            "config": config_dict,
        }
        self.results.append(result)
        
        print(f"\n实验 '{experiment.name}' 完成!")
        print(f"输出目录: {exp_output_dir}")
    
    def run_all(self, experiments: List[ExperimentConfig]):
        """运行所有实验"""
        for exp in experiments:
            self.add_experiment(exp)
        
        # 保存所有结果汇总
        with open(self.base_output_dir / "all_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"所有实验完成! 共 {len(self.results)} 个实验")
        print(f"结果保存在: {self.base_output_dir}")
        print(f"{'='*60}\n")
        
        # 生成对比图表
        self.generate_comparison_plots()
    
    def generate_comparison_plots(self):
        """生成多组实验对比图表"""
        from comparison_plot import ComparisonPlotter
        
        print("\n生成对比图表...")
        plotter = ComparisonPlotter(self.results, self.comparison_dir)
        plotter.plot_all()
        print(f"对比图表已保存至: {self.comparison_dir}")


# ============================================================================
# 配置示例：定义多组实验
# ============================================================================

def create_default_experiments() -> List[ExperimentConfig]:
    """创建默认的实验配置组"""
    
    experiments = []
    
    # 实验1: 基线配置
    config1 = GeneratorConfig(
        years_to_generate=200,  # 生成模拟年份数
        seasonal_period=365,  # 季节周期长度，对应日尺度一年
        ar_max_lag=30,  # AR模型最大滞后阶数
        markov_types=5,  # 年型分类数量
        random_seed=12,  # 全局随机种子
        residual_scale_strength=0.9,  # 残差月度方差匹配强度
        residual_scale_min=0.4,  # 残差月度缩放下限
        residual_scale_max=1.3,  # 残差月度缩放上限
        low_flow_alignment_strength=0.99,  # 低流量分布同质化强度
        dual_scale_strength=0.88,  # 分位+月份双重缩放强度
        dual_scale_min=0.05,  # 双重缩放最小因子
        dual_scale_max=1.4,  # 双重缩放最大因子
        high_flow_enhance_strength=0.2,  # 极端高流量增强强度，0表示关闭
        high_flow_enhance_quantile=0.9,  # 参考分位阈值，用于识别高流量事件
        high_flow_enhance_exponent=1.5,  # 增强幂指数，决定放大量级随流量的增长速度
        high_flow_enhance_max_multiplier=2.0  # 单点最大放大量级
    )
    experiments.append(ExperimentConfig(
        name="baseline",
        config=config1,
        description="基线配置：标准参数"
    ))
    
    # 实验2: 强化低流量对齐
    config2 = GeneratorConfig(
        years_to_generate=200,  # 生成模拟年份数
        seasonal_period=365,  # 季节周期长度，对应日尺度一年
        ar_max_lag=30,  # AR模型最大滞后阶数
        markov_types=5,  # 年型分类数量
        random_seed=12,  # 全局随机种子
        residual_scale_strength=0.8,  # 残差月度方差匹配强度
        residual_scale_min=0.3,  # 残差月度缩放下限
        residual_scale_max=1.3,  # 残差月度缩放上限
        low_flow_alignment_strength=1.2,  # 低流量分布同质化强度
        dual_scale_strength=0.9,  # 分位+月份双重缩放强度
        dual_scale_min=0.03,  # 双重缩放最小因子
        dual_scale_max=1.5,  # 双重缩放最大因子
        high_flow_enhance_strength=0.35,  # 极端高流量增强强度，0表示关闭
        high_flow_enhance_quantile=0.75,  # 参考分位阈值，用于识别高流量事件
        high_flow_enhance_exponent=1.5,  # 增强幂指数，决定放大量级随流量的增长速度
        high_flow_enhance_max_multiplier=2.2  # 单点最大放大量级
    )
    experiments.append(ExperimentConfig(
        name="enhanced_flow",
        config=config2,
        description="强化极端流量：low_flow_alignment_strength=1.2"
    ))
    
    # # 实验3: 强化高流量增强
    # config3 = GeneratorConfig(
    #     years_to_generate=200,
    #     random_seed=10,
    #     residual_scale_strength=0.9,
    #     low_flow_alignment_strength=0.99,
    #     dual_scale_strength=0.88,
    #     high_flow_enhance_strength=0.4,  # 更强的高流量增强
    #     high_flow_enhance_exponent=2.0,  # 更陡的增强曲线
    # )
    # experiments.append(ExperimentConfig(
    #     name="enhanced_high_flow",
    #     config=config3,
    #     description="强化高流量增强：strength=0.4, exponent=2.0"
    # ))
    
    # # 实验4: 弱化双重缩放
    # config4 = GeneratorConfig(
    #     years_to_generate=200,
    #     random_seed=10,
    #     residual_scale_strength=0.9,
    #     low_flow_alignment_strength=0.99,
    #     dual_scale_strength=0.5,  # 弱化双重缩放
    #     high_flow_enhance_strength=0.2,
    # )
    # experiments.append(ExperimentConfig(
    #     name="weak_dual_scale",
    #     config=config4,
    #     description="弱化双重缩放：dual_scale_strength=0.5"
    # ))
    
    # # 实验5: 无高流量增强
    # config5 = GeneratorConfig(
    #     years_to_generate=200,
    #     random_seed=10,
    #     residual_scale_strength=0.9,
    #     low_flow_alignment_strength=0.99,
    #     dual_scale_strength=0.88,
    #     high_flow_enhance_strength=0.0,  # 关闭高流量增强
    # )
    # experiments.append(ExperimentConfig(
    #     name="no_high_flow_enhance",
    #     config=config5,
    #     description="关闭高流量增强：high_flow_enhance_strength=0.0"
    # ))
    
    return experiments


def main():
    """主函数"""
    # 创建批量运行器
    runner = BatchRunner()
    
    # 获取实验配置
    experiments = create_default_experiments()
    
    # 运行所有实验
    runner.run_all(experiments)


if __name__ == "__main__":
    main()
