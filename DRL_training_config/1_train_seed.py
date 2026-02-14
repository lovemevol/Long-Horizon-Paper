import argparse
import time
import subprocess
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_RUNS = 5 # 默认训练轮数
DEFAULT_INTERVAL = 5  # 秒
SEED_LOG_FILENAME = "seed_runs.log"


# 自动获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# tensorboard --logdir=E:\GitHub\Engine\agent\reservoir_single_agent\result_paperOne
# python agent/reservoir_single_agent/visible/run_dashboard.py

# 构建训练脚本的完整路径
scripts = [
    os.path.join(current_dir, "ppo_a_enhan_o5_y1.py"),
    os.path.join(current_dir, "ppo_a_enhan_o5_y2.py"),
    os.path.join(current_dir, "ppo_a_enhan_o5_y4.py"),
    os.path.join(current_dir, "ppo_a_enhan_o5_y6.py"),
    os.path.join(current_dir, "ppo_a_enhan_o5_y8.py"),
    os.path.join(current_dir, "ppo_b_enhan_o2.py"),
    os.path.join(current_dir, "ppo_b_enhan_o3.py"),
    os.path.join(current_dir, "ppo_b_enhan_o4.py"),
    os.path.join(current_dir, "ppo_b_enhan_o5.py"),
    os.path.join(current_dir, "ppo_b_enhan_o6.py"),
    os.path.join(current_dir, "ppo_b_enhan_o7.py"),
    os.path.join(current_dir, "ppo_b_enhan_o8.py"),
    os.path.join(current_dir, "ppo_c_base.py"),
    os.path.join(current_dir, "ppo_c_enhan.py"),
    os.path.join(current_dir, "ppo_c_his.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行多种训练脚本，并为每轮分配固定随机种子")
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"重复训练轮数，默认为 {DEFAULT_RUNS} 轮",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"每个脚本之间的等待秒数，默认为 {DEFAULT_INTERVAL}s",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="用于日志记录的基准值；可选",
    )
    return parser.parse_args()


def log_seed(run_idx: int, seed_value: int, base_seed: int | None) -> None:
    log_path = os.path.join(current_dir, SEED_LOG_FILENAME)
    with open(log_path, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        base_seed_repr = base_seed if base_seed is not None else "None"
        f.write(f"[{timestamp}] base_seed={base_seed_repr} run={run_idx} seed={seed_value}\n")


def run_script_with_seed(script_path: str, seed_value: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    cmd = [sys.executable, script_path, "--seed", str(seed_value)]
    print(f"运行脚本：{os.path.basename(script_path)} (seed={seed_value})")
    result = subprocess.run(cmd, env=env)
    print(f"{os.path.basename(script_path)} 执行完成，返回码：{result.returncode}")
    return result


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs 必须为正整数")

    base_seed = args.base_seed
    base_seed_repr = base_seed if base_seed is not None else "None"
    print(f"本次运行共 {args.runs} 轮，base_seed={base_seed_repr}，日志文件：{SEED_LOG_FILENAME}")

    for run_idx in range(1, args.runs + 1):
        seed_value = run_idx*10
        print("-" * 80)
        print(f"第 {run_idx} 轮训练，随机固定种子：{seed_value}")
        log_seed(run_idx, seed_value, base_seed)

        for idx, script in enumerate(scripts, start=1):
            result = run_script_with_seed(script, seed_value)
            if result.returncode != 0:
                print(f"警告：脚本 {script} 返回非零状态码 {result.returncode}")
            if idx < len(scripts) and args.interval > 0:
                print(f"等待 {args.interval}s 后继续...")
                time.sleep(args.interval)

    print("所有训练轮次执行完毕。")


if __name__ == "__main__":
    main()
