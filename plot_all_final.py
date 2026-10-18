# -*- coding: utf-8 -*-
# plot_all_final.py
# 论文 8 张图（精简 + 新增）

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

from config import Q_IDLE, Q_READY, QD_MAX, QDD_MAX
from arm_trajectory import (min_feasible_time,
                             plan_quintic_trajectory,
                             sample_trajectory)
from base_model import (
    make_straight_path, make_L_path, make_S_path,
)
from coordinator import (
    SerialStrategy, FixedThresholdStrategy, AdaptiveThresholdStrategy,
    CoOptA, CoOptB, CoOptC,
    simulate_transport,
)


PARAMS = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}

STRATEGY_COLORS = {
    "serial": "#888888", "fixed": "#1f77b4",
    "adaptive": "#ff7f0e", "cooptA": "#2ca02c",
    "cooptB": "#d62728", "cooptC": "#9467bd",
}
STRATEGY_MARKERS = {
    "serial": "x", "fixed": "o", "adaptive": "s",
    "cooptA": "^", "cooptB": "D", "cooptC": "v",
}
STRATEGY_LABELS_CN = {
    "serial": "serial（串行）", "fixed": "fixed（固定阈值）",
    "adaptive": "adaptive（自适应）", "cooptA": "cooptA（底盘）",
    "cooptB": "cooptB（机械臂）", "cooptC": "cooptC（双边）",
}


def make_strategies():
    return [
        ("serial", SerialStrategy()),
        ("fixed", FixedThresholdStrategy()),
        ("adaptive", AdaptiveThresholdStrategy()),
        ("cooptA", CoOptA()),
        ("cooptB", CoOptB()),
        ("cooptC", CoOptC()),
    ]


# ============================================================
# Fig 1：路径长度 vs 总时间（4 子图）
# ============================================================

def fig1_path_length():
    goals = [
        ("0.5", (1.0, 0.5)), ("1.0", (1.5, 0.5)),
        ("1.5", (2.0, 0.5)), ("2.0", (2.5, 0.5)),
        ("3.0", (3.5, 0.5)), ("5.0", (5.5, 0.5)),
        ("10.0", (10.5, 0.5)),
    ]
    lengths = [float(g[0]) for g in goals]
    strategies = make_strategies()

    results, waits = {}, {}
    for name, _ in strategies:
        results[name] = []
        waits[name] = []
    for _, goal in goals:
        path = make_straight_path((0.5, 0.5), goal)
        for name, st in strategies:
            r = simulate_transport(st, path, PARAMS)
            results[name].append(r["total_time"])
            waits[name].append(r["wait_time"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # (a) 完整曲线
    ax = axes[0, 0]
    for name, _ in strategies:
        ax.plot(lengths, results[name],
                marker=STRATEGY_MARKERS[name],
                color=STRATEGY_COLORS[name],
                label=STRATEGY_LABELS_CN[name], linewidth=2, markersize=8)
    ax.set_xlabel("路径长度 (m)", fontsize=11)
    ax.set_ylabel("总任务时间 (s)", fontsize=11)
    ax.set_title("(a) 总时间 vs 路径长度", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)

    # (b) 短路径放大
    ax = axes[0, 1]
    zoom_idx = [0, 1, 2, 3]
    zl = [lengths[i] for i in zoom_idx]
    for name, _ in strategies:
        zv = [results[name][i] for i in zoom_idx]
        ax.plot(zl, zv,
                marker=STRATEGY_MARKERS[name],
                color=STRATEGY_COLORS[name],
                label=STRATEGY_LABELS_CN[name],
                linewidth=2.5, markersize=10)
    diff_b = results["fixed"][0] - results["cooptB"][0]
    ax.annotate(f"cooptB 快 {diff_b:.2f}s",
                xy=(zl[0], results["cooptB"][0]),
                xytext=(zl[0] + 0.4, results["cooptB"][0] + 0.7),
                fontsize=11, color="tab:red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
    ax.set_xlabel("路径长度 (m)", fontsize=11)
    ax.set_ylabel("总任务时间 (s)", fontsize=11)
    ax.set_title("(b) 短路径放大", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)

    # (c) 等待时间
    ax = axes[1, 0]
    for name, _ in strategies:
        ax.plot(lengths, waits[name],
                marker=STRATEGY_MARKERS[name],
                color=STRATEGY_COLORS[name],
                label=STRATEGY_LABELS_CN[name], linewidth=2, markersize=8)
    ax.set_xlabel("路径长度 (m)", fontsize=11)
    ax.set_ylabel("底盘等待时间 (s)", fontsize=11)
    ax.set_title("(c) 等待时间 vs 路径长度", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    # (d) coopt 相对 fixed 优势
    ax = axes[1, 1]
    gain_a = np.array(results["cooptA"]) - np.array(results["fixed"])
    gain_b = np.array(results["cooptB"]) - np.array(results["fixed"])
    gain_c = np.array(results["cooptC"]) - np.array(results["fixed"])
    width = 0.25
    x = np.arange(len(lengths))
    ax.bar(x - width, gain_a, width, label="cooptA",
           color=STRATEGY_COLORS["cooptA"], edgecolor="black", lw=0.5)
    ax.bar(x, gain_b, width, label="cooptB",
           color=STRATEGY_COLORS["cooptB"], edgecolor="black", lw=0.5)
    ax.bar(x + width, gain_c, width, label="cooptC",
           color=STRATEGY_COLORS["cooptC"], edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}" for l in lengths])
    ax.set_xlabel("路径长度 (m)", fontsize=11)
    ax.set_ylabel("相对 fixed 优势 (s)", fontsize=11)
    ax.set_title("(d) coopt 相对 fixed 的优势", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig1_path_length.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存 fig1_path_length.png")


# ============================================================
# Fig 2：v_bias 补偿能力（3 子图）
# ============================================================

def fig2_bias_compensation():
    paths = [("0.5 m", (1.0, 0.5)), ("3.0 m", (3.5, 0.5)),
             ("10.0 m", (10.5, 0.5))]
    v_bias_list = np.array([0.00, 0.10, 0.20, 0.30])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = {"0.5 m": "#1f77b4", "3.0 m": "#ff7f0e",
              "10.0 m": "#2ca02c"}

    fixed_data, cooptA_data, cooptC_data = {}, {}, {}
    for label, goal in paths:
        path = make_straight_path((0.5, 0.5), goal)
        f_list, a_list, c_list = [], [], []
        for vb in v_bias_list:
            f_list.append(simulate_transport(
                FixedThresholdStrategy(), path, PARAMS, v_bias=vb)["total_time"])
            a_list.append(simulate_transport(
                CoOptA(), path, PARAMS, v_bias=vb)["total_time"])
            c_list.append(simulate_transport(
                CoOptC(), path, PARAMS, v_bias=vb)["total_time"])
        fixed_data[label] = np.array(f_list)
        cooptA_data[label] = np.array(a_list)
        cooptC_data[label] = np.array(c_list)

    # (a) cooptA 补偿
    ax = axes[0]
    for label in fixed_data:
        gain = cooptA_data[label] - fixed_data[label]
        ax.plot(v_bias_list, gain, marker="o",
                color=colors[label], label=label,
                linewidth=2.5, markersize=10)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("速度偏差 $v_{bias}$", fontsize=12)
    ax.set_ylabel("cooptA − fixed  (s)", fontsize=12)
    ax.set_title("(a) cooptA 的补偿能力", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    # (b) cooptC 补偿
    ax = axes[1]
    for label in fixed_data:
        gain = cooptC_data[label] - fixed_data[label]
        ax.plot(v_bias_list, gain, marker="s",
                color=colors[label], label=label,
                linewidth=2.5, markersize=10)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("速度偏差 $v_{bias}$", fontsize=12)
    ax.set_ylabel("cooptC − fixed  (s)", fontsize=12)
    ax.set_title("(b) cooptC 的补偿能力", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    # (c) fixed 绝对值
    ax = axes[2]
    for label in fixed_data:
        ax.plot(v_bias_list, fixed_data[label],
                marker="^", color=colors[label], label=label,
                linewidth=2.5, markersize=10)
    ax.set_xlabel("速度偏差 $v_{bias}$", fontsize=12)
    ax.set_ylabel("fixed 总任务时间 (s)", fontsize=12)
    ax.set_title("(c) fixed 绝对时间 vs 偏差", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig2_bias_compensation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存 fig2_bias_compensation.png")


# ============================================================
# Fig 3：曲率路径（2 子图）
# ============================================================

def fig3_curvature_paths():
    paths = [
        ("直线 3m", make_straight_path((0.5, 0.5), (3.5, 0.5))),
        ("L 形", make_L_path((0.5, 0.5), (3.0, 0.5), (3.0, 3.0),
                              arc_radius=0.3)),
        ("S 形", make_S_path((0.5, 0.5), (2.0, 0.5), (2.0, 2.0),
                              (3.5, 2.0), arc_radius=0.4)),
    ]
    strategies = make_strategies()
    path_labels = [p[0] for p in paths]

    data = {name: [] for name, _ in strategies}
    for label, path in paths:
        for name, st in strategies:
            data[name].append(simulate_transport(st, path, PARAMS)["total_time"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) 分组柱状图
    ax = axes[0]
    n_paths = len(path_labels)
    n_str = len(strategies)
    x = np.arange(n_paths)
    width = 0.13
    for i, (name, _) in enumerate(strategies):
        offset = (i - (n_str - 1) / 2) * width
        ax.bar(x + offset, data[name], width,
               label=STRATEGY_LABELS_CN[name],
               color=STRATEGY_COLORS[name],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(path_labels, fontsize=12)
    ax.set_xlabel("路径类型", fontsize=12)
    ax.set_ylabel("总任务时间 (s)", fontsize=12)
    ax.set_title("(a) 不同路径几何下的策略表现", fontsize=12)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # (b) 相对优势（只画 cooptA 和 cooptC，cooptB=0 不画）
    ax = axes[1]
    gain_a = np.array(data["cooptA"]) - np.array(data["fixed"])
    gain_c = np.array(data["cooptC"]) - np.array(data["fixed"])
    x = np.arange(n_paths)
    width = 0.35
    bars_a = ax.bar(x - width/2, gain_a, width, label="cooptA",
                    color=STRATEGY_COLORS["cooptA"],
                    edgecolor="black", linewidth=0.5)
    bars_c = ax.bar(x + width/2, gain_c, width, label="cooptC",
                    color=STRATEGY_COLORS["cooptC"],
                    edgecolor="black", linewidth=0.5)
    for bars in [bars_a, bars_c]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h - 0.005,
                        f"{h:+.2f}", ha="center", va="top",
                        fontsize=9, color="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(path_labels, fontsize=12)
    ax.set_xlabel("路径类型", fontsize=12)
    ax.set_ylabel("相对 fixed 优势 (s)", fontsize=12)
    ax.set_title("(b) coopt 相对 fixed 的优势", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig3_curvature_paths.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存 fig3_curvature_paths.png")


# ============================================================
# Fig 4：τ vs 峰值力矩（2 子图）
# ============================================================

def fig4_tau_vs_torque():
    tau_data = np.array([
        0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600,
        0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000,
    ])
    peak_torque = np.array([
        64.155, 61.934, 60.484, 59.488, 58.776, 58.249, 57.850,
        57.540, 57.295, 57.098, 56.938, 56.806, 56.696, 56.604, 56.526,
    ])
    tau_max_line = 120.0
    tau_used = 0.921

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.plot(tau_data, peak_torque, marker="o", color="#1f77b4",
            linewidth=2.5, markersize=9, label="峰值关节力矩")
    ax.axhline(tau_max_line, color="red", linestyle="--",
               linewidth=2.5, label=f"力矩上限 ({tau_max_line:.0f} Nm)")
    ax.axvline(tau_used, color="green", linestyle=":",
               linewidth=2.5, label=f"cooptB/C τ = {tau_used}")
    ax.fill_between(tau_data, 0, peak_torque,
                    alpha=0.15, color="green", label="可行区域")

    idx_used = int(np.argmin(np.abs(tau_data - tau_used)))
    ax.annotate(
        f"cooptB/C 实际使用\nτ = {tau_used:.3f}\n峰值力矩 = {peak_torque[idx_used]:.1f} Nm",
        xy=(tau_used, peak_torque[idx_used]),
        xytext=(0.42, 95),
        fontsize=11, color="darkgreen", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="lightyellow", edgecolor="darkgreen", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2.5))
    ax.set_xlabel("时间缩放因子 τ", fontsize=12)
    ax.set_ylabel("峰值关节力矩 (Nm)", fontsize=12)
    ax.set_title("(a) τ 对峰值力矩的影响", fontsize=12)
    ax.set_ylim(0, 140)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1]
    per_joint = np.array([56.526, 55.8, 42.5, 25.1, 22.3, 15.6])
    tau_max_joints = np.array([120, 120, 80, 35, 35, 20])
    joints = [f"J{i+1}" for i in range(6)]
    x = np.arange(6)
    width = 0.35
    ax.bar(x - width/2, per_joint, width,
           label="峰值力矩 (τ=0.921)", color="#1f77b4",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, tau_max_joints, width,
           label="力矩上限", color="salmon",
           edgecolor="black", linewidth=0.5)
    for i, (pj, tm) in enumerate(zip(per_joint, tau_max_joints)):
        margin = (tm - pj) / tm * 100
        ax.text(x[i], max(pj, tm) + 3, f"{margin:.0f}%",
                ha="center", fontsize=9,
                color="darkgreen", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(joints, fontsize=12)
    ax.set_xlabel("关节", fontsize=12)
    ax.set_ylabel("力矩 (Nm)", fontsize=12)
    ax.set_title("(b) 各关节力矩余量（τ = 0.921）", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig4_tau_vs_torque.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存 fig4_tau_vs_torque.png")


# ============================================================
# Fig 5：速度曲线（2 子图）
# ============================================================

def fig5_velocity_profiles():
    path = make_straight_path((0.5, 0.5), (3.5, 0.5))
    strategies = make_strategies()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    for name, st in strategies:
        r = simulate_transport(st, path, PARAMS)
        log = r["log"]
        ax.plot(log["t"], log["v"], label=STRATEGY_LABELS_CN[name],
                color=STRATEGY_COLORS[name], linewidth=2)
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("底盘速度 (m/s)", fontsize=12)
    ax.set_title("(a) 3m 路径速度曲线", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 0.55)

    ax = axes[1]
    for name, st in strategies:
        if name in ("cooptA", "cooptC"):
            r = simulate_transport(st, path, PARAMS)
            log = r["log"]
            ax.plot(log["t"], log["v_target"],
                    linestyle="--", alpha=0.6,
                    label=f"{name} 目标速度",
                    color=STRATEGY_COLORS[name], linewidth=1.8)
            ax.plot(log["t"], log["v"],
                    linestyle="-", linewidth=2.5,
                    label=f"{name} 实际速度",
                    color=STRATEGY_COLORS[name])
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("速度 (m/s)", fontsize=12)
    ax.set_title("(b) cooptA/C 的速度目标 vs 实际", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 0.55)

    plt.tight_layout()
    plt.savefig("fig5_velocity_profiles.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存 fig5_velocity_profiles.png")


# ============================================================
# Fig 6：时间余量 Δ 演化（2 子图）
# ============================================================

def fig6_delta_evolution():
    path = make_straight_path((0.5, 0.5), (3.5, 0.5))
    strategies = [("cooptA", CoOptA()), ("cooptC", CoOptC())]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for name, st in strategies:
        r = simulate_transport(st, path, PARAMS)
        log = r["log"]
        ax.plot(log["t"], log["Delta"],
                label=f"{name}", color=STRATEGY_COLORS[name],
                linewidth=2.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("时间余量 Δ = $R_b$ − $R_a$ (s)", fontsize=12)
    ax.set_title("(a) 时间余量 Δ 的演化", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1]
    for name, st in strategies:
        r = simulate_transport(st, path, PARAMS)
        log = r["log"]
        ax.plot(log["t"], log["R_b"],
                label=f"{name} $R_b$", linestyle="--",
                color=STRATEGY_COLORS[name], linewidth=2)
        ax.plot(log["t"], log["R_a"],
                label=f"{name} $R_a$", linestyle="-",
                color=STRATEGY_COLORS[name], linewidth=2.5)
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("剩余时间 (s)", fontsize=12)
    ax.set_title("(b) $R_b$ 与 $R_a$ 的演化", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig6_delta_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存 fig6_delta_evolution.png")


# ============================================================
# Fig 7（新）：机械臂关节轨迹
# ============================================================

def fig7_joint_trajectory():
    """展示 6 关节的 q/qd/qdd 轨迹 + 关节限位"""
    T_min, _, _, _ = min_feasible_time(Q_IDLE, Q_READY)
    T_traj = 0.921 * 2.5   # cooptB/C 实际使用的轨迹时长

    traj = plan_quintic_trajectory(Q_IDLE, Q_READY, T_traj, dt=0.01)

    t = traj["t"]
    q = traj["q"]
    qd = traj["qd"]
    qdd = traj["qdd"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) 位置 q
    ax = axes[0]
    for i in range(6):
        ax.plot(t, q[i, :], linewidth=1.8, label=f"J{i+1}")
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("关节角度 (rad)", fontsize=12)
    ax.set_title(f"(a) 关节位置轨迹 (T = {T_traj:.3f} s)", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)

    # (b) 速度 qd + 限位
    ax = axes[1]
    for i in range(6):
        ax.plot(t, qd[i, :], linewidth=1.8, label=f"J{i+1}")
        ax.axhline(QD_MAX[i], color=f"C{i}", linestyle=":", alpha=0.4)
        ax.axhline(-QD_MAX[i], color=f"C{i}", linestyle=":", alpha=0.4)
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("关节速度 (rad/s)", fontsize=12)
    ax.set_title("(b) 关节速度（虚线 = 上限）", fontsize=12)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # (c) 加速度 qdd + 限位
    ax = axes[2]
    for i in range(6):
        ax.plot(t, qdd[i, :], linewidth=1.8, label=f"J{i+1}")
        ax.axhline(QDD_MAX[i], color=f"C{i}", linestyle=":", alpha=0.4)
        ax.axhline(-QDD_MAX[i], color=f"C{i}", linestyle=":", alpha=0.4)
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("关节加速度 (rad/s²)", fontsize=12)
    ax.set_title("(c) 关节加速度（虚线 = 上限）", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig7_joint_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存 fig7_joint_trajectory.png")


# ============================================================
# Fig 8（新）：τ 随路径长度的演化
# ============================================================

def fig8_tau_evolution():
    """展示不同路径长度下 cooptB/C 的 τ 变化"""
    goals = [
        ("0.5", (1.0, 0.5)), ("1.0", (1.5, 0.5)),
        ("1.5", (2.0, 0.5)), ("2.0", (2.5, 0.5)),
        ("3.0", (3.5, 0.5)), ("5.0", (5.5, 0.5)),
        ("10.0", (10.5, 0.5)),
    ]
    lengths = [float(g[0]) for g in goals]

    tau_data = {"cooptB": [], "cooptC": []}
    t_arrive_data = {"cooptB": [], "cooptC": []}

    for _, goal in goals:
        path = make_straight_path((0.5, 0.5), goal)
        r_b = simulate_transport(CoOptB(), path, PARAMS)
        r_c = simulate_transport(CoOptC(), path, PARAMS)
        tau_data["cooptB"].append(r_b["tau_applied"])
        tau_data["cooptC"].append(r_c["tau_applied"])
        t_arrive_data["cooptB"].append(r_b["t_arrive"])
        t_arrive_data["cooptC"].append(r_c["t_arrive"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (a) τ vs 路径长度
    ax = axes[0]
    ax.plot(lengths, tau_data["cooptB"], marker="D",
            color=STRATEGY_COLORS["cooptB"], label="cooptB",
            linewidth=2.5, markersize=12)
    ax.plot(lengths, tau_data["cooptC"], marker="v",
            color=STRATEGY_COLORS["cooptC"], label="cooptC",
            linewidth=2.5, markersize=12)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6,
               label="τ = 1.0（不加速）")
    ax.set_xlabel("路径长度 (m)", fontsize=12)
    ax.set_ylabel("时间缩放因子 τ", fontsize=12)
    ax.set_title("(a) τ 随路径长度的变化", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0.8, 1.05)

    # 标注短路径下的加速
    for i, l in enumerate(lengths):
        if tau_data["cooptB"][i] < 0.999:
            ax.annotate(f"τ={tau_data['cooptB'][i]:.3f}",
                        xy=(l, tau_data["cooptB"][i]),
                        xytext=(l + 0.3, tau_data["cooptB"][i] - 0.02),
                        fontsize=9, color=STRATEGY_COLORS["cooptB"])

    # (b) τ vs 触发时刻的 R_b
    ax = axes[1]
    # 用路径长度做 x 轴，展示 τ 和触发时 R_b 的关系
    trigger_Rb = []
    for _, goal in goals:
        path = make_straight_path((0.5, 0.5), goal)
        base_state = simulate_transport(CoOptB(), path, PARAMS)
        log = base_state["log"]
        # 取触发时刻的 R_b（R_a 从 T_NOMINAL 开始下降的那一刻）
        R_a_arr = np.array(log["R_a"])
        # 找到第一个 R_a < T_NOMINAL 的索引
        idx = np.argmax(R_a_arr < 2.5 - 1e-6)
        if idx > 0:
            trigger_Rb.append(log["R_b"][idx])
        else:
            trigger_Rb.append(log["R_b"][0])

    ax.plot(trigger_Rb, tau_data["cooptB"], marker="D",
            color=STRATEGY_COLORS["cooptB"], label="cooptB",
            linewidth=2.5, markersize=12)
    ax.plot(trigger_Rb, tau_data["cooptC"], marker="v",
            color=STRATEGY_COLORS["cooptC"], label="cooptC",
            linewidth=2.5, markersize=12)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("触发时刻 $R_b$ (s)", fontsize=12)
    ax.set_ylabel("时间缩放因子 τ", fontsize=12)
    ax.set_title("(b) τ 与触发时刻底盘剩余时间 $R_b$ 的关系", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig8_tau_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存 fig8_tau_evolution.png")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("生成 8 张论文图")
    print("=" * 60)

    print("\nFig 1: 路径长度 vs 总时间（4 子图）...")
    fig1_path_length()
    print("\nFig 2: v_bias 补偿能力（3 子图）...")
    fig2_bias_compensation()
    print("\nFig 3: 曲率路径对比（2 子图）...")
    fig3_curvature_paths()
    print("\nFig 4: τ vs 峰值力矩（2 子图）...")
    fig4_tau_vs_torque()
    print("\nFig 5: 速度曲线（2 子图）...")
    fig5_velocity_profiles()
    print("\nFig 6: 时间余量 Δ 演化（2 子图）...")
    fig6_delta_evolution()
    print("\nFig 7: 机械臂关节轨迹（新）...")
    fig7_joint_trajectory()
    print("\nFig 8: τ 演化（新）...")
    fig8_tau_evolution()

    print("\n" + "=" * 60)
    print("完成！8 张图已生成")
    print("=" * 60)