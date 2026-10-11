# -*- coding: utf-8 -*-
# plot_routes.py
# 作用：画六种策略的路线图，看触发位置和轨迹差异

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

from env import build_default_map, resample_path
from robot import RobotParams
from strategies import (SerialStrategy, FixedThresholdStrategy,
                        AdaptiveThresholdStrategy,
                        CoOptA, CoOptB, CoOptC)
from simulate import simulate_clean
from visualize import plot_trajectory


# ============================================================
# 参数：要改目标点，只改这一行
# ============================================================

START = (0.5, 0.5)
GOAL_SHORT = (0.8, 0.6)   # 极短路径（机械臂瓶颈）
GOAL_MID   = (1.5, 1.0)   # 短路径
GOAL_LONG  = (9.0, 5.0)   # 长路径（底盘瓶颈）

# 想画哪个场景，改这里
GOAL = GOAL_SHORT


# ============================================================
# 工具函数
# ============================================================

def run_with_map(strategy, start, goal, params):
    """构造地图 + 路径 + 仿真结果"""
    m = build_default_map()
    path = m.astar(start, goal)
    path, _ = resample_path(path, ds=0.05)
    result = simulate_clean(strategy, path, params)
    return result, m, path


def make_strategies():
    """六种策略（cooptB/C 用最优 scale_min=0.3）"""
    return [
        SerialStrategy(),
        FixedThresholdStrategy(1.2),
        AdaptiveThresholdStrategy(3.0),      # 短路径最优 alpha
        CoOptA(k=0.5, trigger_margin=20),
        CoOptB(scale_min=0.3, trigger_margin=20),   # ★ 改成 0.3
        CoOptC(k=0.3, scale_min=0.3, trigger_margin=20),
    ]


# ============================================================
# 主图：2x3 六策略路线对比
# ============================================================

def plot_six_routes(start=START, goal=GOAL, save=True):
    params = RobotParams()
    strategies = make_strategies()

    # 先跑完所有策略，拿到数据
    data = []
    for st in strategies:
        r, m, path = run_with_map(st, start, goal, params)
        data.append((st, r, m, path))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, (st, r, m, path) in zip(axes.flat, data):
        plot_trajectory(
            ax, r, m, path, start, goal,
            title=f"{st.name}  T={r['total_time']:.2f}s  "
                  f"W={r['wait_time']:.2f}s  O={r['overlap_time']:.2f}s"
        )

    fig.suptitle(f"Route comparison  (start={start}, goal={goal})",
                 fontsize=14, y=0.99)

    plt.tight_layout()
    if save:
        plt.savefig("routes_2x3.png", dpi=120, bbox_inches="tight")
        print("已保存：routes_2x3.png")
    plt.show()


# ============================================================
# 单策略路线图（调试 / 论文插图）
# ============================================================

def plot_one(strategy_name="cooptB", start=START, goal=GOAL):
    params = RobotParams()
    table = {st.name: st for st in make_strategies()}
    if strategy_name not in table:
        raise ValueError(f"未知策略: {strategy_name}，可选 {list(table.keys())}")

    st = table[strategy_name]
    r, m, path = run_with_map(st, start, goal, params)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_trajectory(
        ax, r, m, path, start, goal,
        title=f"{st.name}  T={r['total_time']:.2f}s  "
              f"W={r['wait_time']:.2f}s  O={r['overlap_time']:.2f}s"
    )
    plt.tight_layout()
    plt.savefig(f"route_{strategy_name}.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"已保存：route_{strategy_name}.png")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print(f"画六策略 2x3 对比图  (goal = {GOAL}) ...")
    plot_six_routes()
    print("完成。")