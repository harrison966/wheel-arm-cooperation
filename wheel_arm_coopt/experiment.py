# -*- coding: utf-8 -*-
# experiment.py
# 六策略对比实验

import numpy as np
from env import build_default_map, resample_path
from robot import RobotParams
from strategies import (SerialStrategy, FixedThresholdStrategy,
                        AdaptiveThresholdStrategy,
                        CoOptA, CoOptB, CoOptC)
from simulate import simulate_clean


def run_once(strategy, start, goal, params):
    m = build_default_map()
    path = m.astar(start, goal)
    path, _ = resample_path(path, ds=0.05)
    return simulate_clean(strategy, path, params)


def run_once_with_map(strategy, start, goal, params):
    m = build_default_map()
    path = m.astar(start, goal)
    path, _ = resample_path(path, ds=0.05)
    result = simulate_clean(strategy, path, params)
    return result, m, path


def default_strategies():
    """返回 6 种策略"""
    return [
        SerialStrategy(),
        FixedThresholdStrategy(1.2),
        AdaptiveThresholdStrategy(1.15),
        CoOptA(k=0.5, trigger_margin=20),
        CoOptB(scale_min=0.6, trigger_margin=20),
        CoOptC(k=0.3, scale_min=0.6, trigger_margin=20),
    ]


# ============================================================
# 实验一：六策略基准对比
# ============================================================

def experiment_1(params=None):
    params = params or RobotParams()
    start = (0.5, 0.5)
    goal = (2.0, 1.5)

    strategies = default_strategies()

    print(f"{'strategy':<10} {'total':>8} {'arrive':>8} "
          f"{'wait':>8} {'overlap':>8}")
    print("-" * 50)
    results = {}
    for st in strategies:
        r = run_once(st, start, goal, params)
        results[st.name] = r
        print(f"{st.name:<10} {r['total_time']:8.3f} {r['t_arrive']:8.3f} "
              f"{r['wait_time']:8.3f} {r['overlap_time']:8.3f}")
    return results


# ============================================================
# 实验二：固定阈值敏感性
# ============================================================

def experiment_2(params=None):
    params = params or RobotParams()
    start = (0.5, 0.5)
    goal = (2.0, 1.5)

    thresholds = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
    print(f"{'d_trigger':>10} {'total':>8} {'wait':>8} {'overlap':>8}")
    print("-" * 40)
    for d in thresholds:
        st = FixedThresholdStrategy(d)
        r = run_once(st, start, goal, params)
        print(f"{d:>10.2f} {r['total_time']:8.3f} "
              f"{r['wait_time']:8.3f} {r['overlap_time']:8.3f}")


# ============================================================
# 实验三：路径长度鲁棒性（六策略）
# ============================================================

def experiment_3(params=None):
    params = params or RobotParams()
    start = (0.5, 0.5)
    goals = [(2.0, 1.5), (3.5, 2.0), (5.0, 5.0), (7.0, 4.0), (9.0, 5.0)]

    print(f"{'goal':>12} | "
          f"{'fixed':>16} | {'adap':>16} | "
          f"{'cooptA':>16} | {'cooptB':>16} | {'cooptC':>16}")
    print(f"{'':>12} | "
          f"{'T':>7} {'W':>7} | {'T':>7} {'W':>7} | "
          f"{'T':>7} {'W':>7} | {'T':>7} {'W':>7} | {'T':>7} {'W':>7}")
    print("-" * 110)

    for g in goals:
        r_f = run_once(FixedThresholdStrategy(1.2), start, g, params)
        r_a = run_once(AdaptiveThresholdStrategy(1.15), start, g, params)
        r_A = run_once(CoOptA(), start, g, params)
        r_B = run_once(CoOptB(), start, g, params)
        r_C = run_once(CoOptC(), start, g, params)
        print(f"{str(g):>12} | "
              f"{r_f['total_time']:7.3f} {r_f['wait_time']:7.3f} | "
              f"{r_a['total_time']:7.3f} {r_a['wait_time']:7.3f} | "
              f"{r_A['total_time']:7.3f} {r_A['wait_time']:7.3f} | "
              f"{r_B['total_time']:7.3f} {r_B['wait_time']:7.3f} | "
              f"{r_C['total_time']:7.3f} {r_C['wait_time']:7.3f}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("实验一：六策略基准对比")
    print("=" * 70)
    experiment_1()

    print("\n" + "=" * 70)
    print("实验二：固定阈值敏感性")
    print("=" * 70)
    experiment_2()

    print("\n" + "=" * 70)
    print("实验三：路径长度鲁棒性")
    print("=" * 70)
    experiment_3()