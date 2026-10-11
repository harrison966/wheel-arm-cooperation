# -*- coding: utf-8 -*-
# plot_all_v2.py
# 论文三张核心图（改进版）

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

from experiment import run_once
from strategies import (AdaptiveThresholdStrategy, CoOptA, CoOptB, CoOptC)
from robot import RobotParams
from env import build_default_map, resample_path
from simulate import simulate_clean


params = RobotParams()
start = (0.5, 0.5)
GOALS = [(0.8, 0.6), (1.0, 0.8), (1.5, 1.0), (2.0, 1.5),
         (3.5, 2.0), (5.0, 5.0), (9.0, 5.0)]


def get_path_length(goal):
    m = build_default_map()
    p = m.astar(start, goal)
    p, _ = resample_path(p, ds=0.05)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return seg.sum()


# ============================================================
# 图 A：路径长度 vs 总时间（主图 + 局部放大）
# ============================================================

def fig_A():
    lengths = [get_path_length(g) for g in GOALS]

    strategies = [
        ("adaptive (best)", AdaptiveThresholdStrategy(3.0), "o", "C0"),
        ("cooptA",          CoOptA(k=0.5, trigger_margin=20), "s", "C1"),
        ("cooptB",          CoOptB(scale_min=0.3, trigger_margin=20), "^", "C2"),
        ("cooptC",          CoOptC(k=0.3, scale_min=0.3, trigger_margin=20), "D", "C3"),
    ]

    results = {}
    for name, st, _, _ in strategies:
        results[name] = [run_once(st, start, g, params)["total_time"]
                         for g in GOALS]

    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    # -------- 主图 --------
    for name, _, mk, color in strategies:
        ax_main.plot(lengths, results[name], marker=mk, label=name,
                     linewidth=2.2, markersize=9, color=color)

    ax_main.set_xlabel("Path length (m)", fontsize=12)
    ax_main.set_ylabel("Total time (s)", fontsize=12)
    ax_main.set_title("Total time vs path length", fontsize=13)
    ax_main.legend(fontsize=11, loc="upper left")
    ax_main.grid(True, linestyle=":", alpha=0.5)

    # -------- 局部放大（前 3 个点）--------
    n_zoom = 3
    for name, _, mk, color in strategies:
        ax_zoom.plot(lengths[:n_zoom], results[name][:n_zoom],
                     marker=mk, label=name, linewidth=2.5,
                     markersize=12, color=color)

    # 标注最大差值
    diff_0 = results["adaptive (best)"][0] - results["cooptB"][0]
    ax_zoom.annotate(
        f"-{diff_0:.2f}s",
        xy=(lengths[0], results["cooptB"][0]),
        xytext=(lengths[0] + 0.3, results["cooptB"][0] - 0.5),
        fontsize=14, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=2)
    )

    ax_zoom.set_xlabel("Path length (m)", fontsize=12)
    ax_zoom.set_ylabel("Total time (s)", fontsize=12)
    ax_zoom.set_title("Zoom: short-path region", fontsize=12)
    ax_zoom.legend(fontsize=10, loc="upper right")
    ax_zoom.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig_A_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("已保存 fig_A_curve.png")


# ============================================================
# 图 B：参数敏感性（加参考线 + 饱和区）
# ============================================================

def fig_B():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -------- 左图：adaptive 的 alpha 扫描 --------
    ax = axes[0]
    goals = {"long (9.0, 5.0)": (9.0, 5.0), "short (1.5, 1.0)": (1.5, 1.0)}

    for label, g in goals.items():
        alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
        totals = [run_once(AdaptiveThresholdStrategy(a), start, g, params)["total_time"]
                  for a in alphas]
        ax.plot(alphas, totals, marker="o", label=f"adaptive, {label}",
                linewidth=2.2, markersize=8)

    # 参考线：cooptA 的最优值
    adap_long_best = run_once(AdaptiveThresholdStrategy(3.0),
                              start, (9.0, 5.0), params)["total_time"]
    coopt_long_best = run_once(CoOptA(0.5, 20),
                               start, (9.0, 5.0), params)["total_time"]
    ax.axhline(coopt_long_best, color="C1", linestyle="--", linewidth=2,
               label=f"cooptA best ({coopt_long_best:.2f}s)")
    # 阴影区：cooptA 优势区
    ax.axhspan(coopt_long_best - 0.5, coopt_long_best,
               alpha=0.15, color="C1")

    ax.set_xlabel("adaptive alpha", fontsize=12)
    ax.set_ylabel("Total time (s)", fontsize=12)
    ax.set_title("Sensitivity of adaptive", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # -------- 右图：cooptA 的 margin 扫描 --------
    ax = axes[1]
    for label, g in goals.items():
        margins = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]
        totals = [run_once(CoOptA(k=0.5, trigger_margin=m),
                           start, g, params)["total_time"]
                  for m in margins]
        ax.plot(margins, totals, marker="s", label=f"cooptA, {label}",
                linewidth=2.2, markersize=8)

    # 阴影区：饱和区（margin >= 18）
    ax.axvspan(18, 50, alpha=0.15, color="green",
               label="saturation zone (margin >= 18)")

    ax.set_xlabel("cooptA trigger_margin", fontsize=12)
    ax.set_ylabel("Total time (s)", fontsize=12)
    ax.set_title("Sensitivity of cooptA", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig_B_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("已保存 fig_B_sensitivity.png")


# ============================================================
# 图 C：速度曲线 + 剩余距离曲线
# ============================================================

def fig_C():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 只选短路径场景，差异明显
    goal = (1.0, 0.8)

    strategies = [
        ("adaptive", AdaptiveThresholdStrategy(3.0), "C0"),
        ("cooptA",   CoOptA(k=0.5, trigger_margin=20), "C1"),
        ("cooptB",   CoOptB(scale_min=0.3, trigger_margin=20), "C2"),
        ("cooptC",   CoOptC(k=0.3, scale_min=0.3, trigger_margin=20), "C3"),
    ]

    # 左图：速度曲线
    ax = axes[0]
    for name, st, color in strategies:
        m = build_default_map()
        p = m.astar(start, goal)
        p, _ = resample_path(p, ds=0.05)
        r = simulate_clean(st, p, params)
        log = r["log"]
        ax.plot(log["t"], log["v"], label=name, linewidth=2.2, color=color)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Base velocity (m/s)", fontsize=12)
    ax.set_title(f"Velocity profile (goal = {goal})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 0.55)

    # 右图：剩余距离（对数坐标）
    ax = axes[1]
    for name, st, color in strategies:
        m = build_default_map()
        p = m.astar(start, goal)
        p, _ = resample_path(p, ds=0.05)
        r = simulate_clean(st, p, params)
        log = r["log"]
        # 用 dist 数组，过滤掉 0 避免 log 报错
        t_arr = np.array(log["t"])
        d_arr = np.array(log["dist"])
        mask = d_arr > 1e-3
        ax.plot(t_arr[mask], d_arr[mask], label=name, linewidth=2.2, color=color)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Remaining distance (m)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"Remaining distance (goal = {goal})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, which="both")

    plt.tight_layout()
    plt.savefig("fig_C_velocity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("已保存 fig_C_velocity.png")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("生成图 A（路径长度 vs 总时间，含局部放大）...")
    fig_A()

    print("\n生成图 B（参数敏感性，含参考线和饱和区）...")
    fig_B()

    print("\n生成图 C（速度曲线 + 剩余距离曲线）...")
    fig_C()

    print("\n全部完成！三张图已保存。")