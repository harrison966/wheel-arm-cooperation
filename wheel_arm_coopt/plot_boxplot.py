# -*- coding: utf-8 -*-
# plot_boxplot.py
# 扰动鲁棒性箱线图：2 个场景 × 3 种扰动 × 4 个策略

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

from env import build_default_map, resample_path
from robot import RobotParams
from strategies import (AdaptiveThresholdStrategy, CoOptA, CoOptB, CoOptC)
from disturbance_exp import simulate_with_noise


def collect_data(n_trials=30):
    """收集所有配置下的 30 次仿真结果"""
    params = RobotParams()
    start = (0.5, 0.5)

    scenarios = [
        ("short", (1.0, 0.8)),
        ("long",  (9.0, 5.0)),
    ]

    strategies = [
        ("adaptive", AdaptiveThresholdStrategy(3.0)),
        ("cooptA",   CoOptA(k=0.5, trigger_margin=20)),
        ("cooptB",   CoOptB(scale_min=0.3, trigger_margin=20)),
        ("cooptC",   CoOptC(k=0.3, scale_min=0.3, trigger_margin=20)),
    ]

    disturbances = [
        ("none",       0.0, 0.0),
        ("v_noise",    0.1, 0.0),
        ("prep_noise", 0.0, 0.3),
    ]

    # data[scenario][disturb][strategy] = list of 30 totals
    data = {}
    for s_name, goal in scenarios:
        data[s_name] = {}
        # 预先生成路径
        m = build_default_map()
        p = m.astar(start, goal)
        p, _ = resample_path(p, ds=0.05)

        for d_name, v_noise, prep_noise in disturbances:
            data[s_name][d_name] = {}
            for st_name, st in strategies:
                totals = []
                for trial in range(n_trials):
                    r = simulate_with_noise(
                        st, p, params,
                        v_noise=v_noise,
                        prep_noise=prep_noise,
                        seed=trial,
                    )
                    totals.append(r["total_time"])
                data[s_name][d_name][st_name] = totals

    return data, [s[0] for s in scenarios], [d[0] for d in disturbances], [s[0] for s in strategies]


def plot_boxplots():
    data, scenarios, disturbances, strategies = collect_data(n_trials=30)

    # 2 行（short/long） × 3 列（none/v_noise/prep_noise）
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    colors = ["C0", "C1", "C2", "C3"]
    positions = [1, 2, 3, 4]

    for i, s_name in enumerate(scenarios):
        for j, d_name in enumerate(disturbances):
            ax = axes[i, j]

            box_data = [data[s_name][d_name][st_name] for st_name in strategies]
            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="red",
                               markeredgecolor="red", markersize=6),
            )

            # 给每个箱体上色
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(strategies, fontsize=9, rotation=15)
            ax.set_title(f"{s_name} / {d_name}", fontsize=11)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)

            # 只在最左列标 y 轴
            if j == 0:
                ax.set_ylabel("Total time (s)", fontsize=11)

    # 顶部加一个大标题
    fig.suptitle(
        "Disturbance robustness: distribution over 30 trials "
        "(red diamond = mean)",
        fontsize=13, y=1.00,
    )

    plt.tight_layout()
    plt.savefig("fig_D_boxplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("已保存：fig_D_boxplot.png")


if __name__ == "__main__":
    plot_boxplots()