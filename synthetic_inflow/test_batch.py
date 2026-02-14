"""
测试批量运行功能（快速版本）

使用较少的生成年数来快速测试功能是否正常。
"""

from batch_run import BatchRunner, ExperimentConfig, GeneratorConfig


def test_batch_run():
    """测试批量运行（使用50年快速测试）"""
    print("🧪 测试模式：使用50年数据快速验证功能\n")
    
    experiments = []
    
    # 测试1: 基线
    experiments.append(ExperimentConfig(
        name="test_baseline",
        config=GeneratorConfig(
            years_to_generate=50,  # 仅50年用于快速测试
            random_seed=10,
            residual_scale_strength=0.9,
            low_flow_alignment_strength=0.99,
            dual_scale_strength=0.88,
            high_flow_enhance_strength=0.2,
        ),
        description="测试：基线配置"
    ))
    
    # 测试2: 强化高流量
    experiments.append(ExperimentConfig(
        name="test_high_flow",
        config=GeneratorConfig(
            years_to_generate=50,
            random_seed=10,
            residual_scale_strength=0.9,
            low_flow_alignment_strength=0.99,
            dual_scale_strength=0.88,
            high_flow_enhance_strength=0.4,
        ),
        description="测试：强化高流量"
    ))
    
    # 测试3: 无增强
    experiments.append(ExperimentConfig(
        name="test_no_enhance",
        config=GeneratorConfig(
            years_to_generate=50,
            random_seed=10,
            residual_scale_strength=0.9,
            low_flow_alignment_strength=0.99,
            dual_scale_strength=0.88,
            high_flow_enhance_strength=0.0,
        ),
        description="测试：无高流量增强"
    ))
    
    # 运行测试
    runner = BatchRunner(base_output_dir="output_test")
    runner.run_all(experiments)
    
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("📁 查看测试结果：output_test/comparison/")
    print("💡 如果一切正常，可以使用 quick_run.py 或 batch_run.py 进行完整实验")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_batch_run()
