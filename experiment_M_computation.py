# -*- coding: utf-8 -*-
# experiment_M_computation.py
# 计算代价分析
# 两张表：(1) 完整仿真耗时（不同路径长度）；(2) 单步控制耗时

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

from base_model import make_straight_path
from coordinator import (SerialStrategy, FixedThresholdStrategy,
                          AdaptiveThresholdStrategy,
                          CoOptA, CoOptB, CoOptC,
                          simulate_transport)


# ============================================================
# 测法 1：完整仿真耗时
# ============================================================
def measure_full_simulation(strategy, path, params, n_trials=20):
    """测整个仿真的 wall-clock 耗时"""
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        simulate_transport(strategy, path, params)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)   # 转毫秒
    return np.array(times)


# ============================================================
# 测法 2：单步控制耗时（monkey patch）
# ============================================================
def measure_per_step(strategy, path, params):
    """通过临时包装策略方法，统计单步控制耗时"""
    orig_decide = strategy.decide_trigger
    orig_cmd = strategy.base_command

    total_time = [0.0]
    step_count = [0]

    def timed_decide(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_decide(*args, **kwargs)
        t1 = time.perf_counter()
        total_time[0] += (t1 - t0)
        step_count[0] += 1
        return result

    def timed_cmd(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_cmd(*args, **kwargs)
        t1 = time.perf_counter()
        total_time[0] += (t1 - t0)
        return result

    strategy.decide_trigger = timed_decide
    strategy.base_command = timed_cmd

    try:
        simulate_transport(strategy, path, params)
    finally:
        strategy.decide_trigger = orig_decide
        strategy.base_command = orig_cmd

    if step_count[0] == 0:
        return float("nan")
    return total_time[0] / step_count[0] * 1000.0   # ms/step


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}

    strategies = [
        ("serial",   SerialStrategy()),
        ("fixed",    FixedThresholdStrategy()),
        ("adaptive", AdaptiveThresholdStrategy()),
        ("cooptA",   CoOptA()),
        ("cooptB",   CoOptB()),
        ("cooptC",   CoOptC()),
    ]

    # ============================================================
    # 表 1：完整仿真耗时（不同路径长度）
    # ============================================================
    print("=" * 70)
    print("表 1：完整仿真耗时（单位：ms，20 次平均）")
    print("=" * 70)

    paths = [
        ("0.5 m",  make_straight_path((0.5, 0.5), (1.0, 0.5))),
        ("3.0 m",  make_straight_path((0.5, 0.5), (3.5, 0.5))),
        ("10.0 m", make_straight_path((0.5, 0.5), (10.5, 0.5))),
    ]

    print(f"{'strategy':<10}", end="")
    for label, _ in paths:
        print(f" {label:>12}", end="")
    print()
    print("-" * 52)

    results_full = {}
    for name, st in strategies:
        row = f"{name:<10}"
        row_vals = []
        for label, path in paths:
            times = measure_full_simulation(st, path, params)
            row += f" {times.mean():>12.3f}"
            row_vals.append(times.mean())
        results_full[name] = row_vals
        print(row)

    # ============================================================
    # 表 2：单步控制耗时
    # ============================================================
    print()
    print("=" * 70)
    print("表 2：单步控制耗时（单位：ms，10 次测量）")
    print("=" * 70)

    path_long = make_straight_path((0.5, 0.5), (10.5, 0.5))

    print(f"{'strategy':<10} {'mean(ms)':>12} {'max(ms)':>12}")
    print("-" * 36)
    results_step = {}
    for name, st in strategies:
        step_times = []
        for _ in range(10):
            t = measure_per_step(st, path_long, params)
            step_times.append(t)
        step_times = np.array(step_times)
        results_step[name] = step_times.mean()
        print(f"{name:<10} {step_times.mean():>12.5f} "
              f"{step_times.max():>12.5f}")

    # ============================================================
    # 绘图
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) 完整仿真耗时
    ax = axes[0]
    x = np.arange(len(strategies))
    width = 0.25
    colors = ["#888888", "#1f77b4", "#ff7f0e"]
    for i, (label, _) in enumerate(paths):
        vals = [results_full[name][i] for name, _ in strategies]
        ax.bar(x + (i - 1) * width, vals, width,
               label=label, color=colors[i], edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in strategies],
                        rotation=0, fontsize=11)
    ax.set_ylabel("完整仿真耗时 (ms)", fontsize=12)
    ax.set_title("(a) 完整仿真耗时 vs 策略", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # (b) 单步控制耗时
    ax = axes[1]
    names = [name for name, _ in strategies]
    vals = [results_step[name] for name in names]
    bars = ax.bar(names, vals, color="steelblue",
                   edgecolor="black", lw=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("单步控制耗时 (ms)", fontsize=12)
    ax.set_title("(b) 单步控制耗时", fontsize=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig14_computation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n已保存 fig14_computation.png")