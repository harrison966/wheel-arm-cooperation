# utils.py
# 作用：画速度曲线、距离曲线、机械臂状态时间线，直观展示联合优化的对齐效果

import matplotlib.pyplot as plt
import numpy as np


def plot_run(result, title=None):
    """
    画一次仿真的三张图：
    1) 底盘速度 vs 时间
    2) 剩余距离 vs 时间
    3) 机械臂状态时间线
    """
    log = result["log"]
    t = np.array(log["t"])
    v = np.array(log["v"])
    d = np.array(log["dist"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # 图1：底盘速度
    axes[0].plot(t, v, label="base velocity")
    axes[0].set_ylabel("v (m/s)")
    axes[0].legend()
    axes[0].grid(True)

    # 图2：剩余距离（对数坐标更清楚）
    axes[1].plot(t, d, label="dist to goal")
    axes[1].set_ylabel("dist (m)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True)

    # 图3：机械臂状态
    states = log["arm_state"]
    state_map = {"idle": 0, "prepping": 1, "ready": 2,
                 "grasping": 3, "placing": 4, "done": 5}
    y = [state_map.get(s, 0) for s in states]
    axes[2].plot(t, y, drawstyle="steps-post")
    axes[2].set_yticks(list(state_map.values()))
    axes[2].set_yticklabels(list(state_map.keys()))
    axes[2].set_ylabel("arm state")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_runs(results_dict):
    """
    把多个策略的速度曲线画在一起，直观对比
    results_dict: {strategy_name: result}
    """
    plt.figure(figsize=(8, 4))
    for name, r in results_dict.items():
        log = r["log"]
        plt.plot(log["t"], log["v"], label=name)
    plt.xlabel("time (s)")
    plt.ylabel("base velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.title("Velocity comparison across strategies")
    plt.tight_layout()
    plt.show()