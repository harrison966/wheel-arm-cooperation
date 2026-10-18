# -*- coding: utf-8 -*-
# plot_multi_task.py
# 多任务累积优势可视化

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

# 从 experiment_L_multi_task.py 提取的数据
N_list = np.array([100, 500, 1000])

# cooptA 累积优势（s）
cooptA_gain = np.array([4.95, 26.25, 50.00])
# cooptC 累积优势（s）
cooptC_gain = np.array([3.60, 15.95, 29.75])
# cooptA 相对提升（%）
cooptA_pct = cooptA_gain / np.array([1911.80, 9627.35, 19313.70]) * 100

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# -------- (a) 累积优势 vs 任务数（带线性拟合）--------
ax = axes[0]
ax.plot(N_list, cooptA_gain, "o-", color="tab:green",
        linewidth=2.5, markersize=12, label="cooptA")
ax.plot(N_list, cooptC_gain, "s-", color="tab:purple",
        linewidth=2.5, markersize=12, label="cooptC")

# 线性拟合
z_a = np.polyfit(N_list, cooptA_gain, 1)
z_c = np.polyfit(N_list, cooptC_gain, 1)
N_fit = np.linspace(0, 1100, 100)
ax.plot(N_fit, np.polyval(z_a, N_fit), "--",
        color="tab:green", alpha=0.5,
        label=f"cooptA fit: {z_a[0]:.4f}·N")
ax.plot(N_fit, np.polyval(z_c, N_fit), "--",
        color="tab:purple", alpha=0.5,
        label=f"cooptC fit: {z_c[0]:.4f}·N")

# 数据标注
for n, g in zip(N_list, cooptA_gain):
    ax.annotate(f"{g:.2f}s", xy=(n, g),
                xytext=(n + 20, g - 4), fontsize=10,
                color="darkgreen", fontweight="bold")

ax.set_xlabel("任务数 $N$", fontsize=12)
ax.set_ylabel("累积优势 (s)", fontsize=12)
ax.set_title("(a) 累积优势 vs 任务数", fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, linestyle=":", alpha=0.5)

# -------- (b) 相对提升（%）--------
ax = axes[1]
x = np.arange(len(N_list))
width = 0.35
bars = ax.bar(x, cooptA_pct, width,
               color="tab:green", edgecolor="black", lw=0.5,
               label="cooptA")
# 数据标注
for b, v in zip(bars, cooptA_pct):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
            f"{v:.3f}%", ha="center", fontsize=11,
            color="darkgreen", fontweight="bold")
ax.axhline(np.mean(cooptA_pct), color="red", linestyle="--",
           linewidth=1.5, label=f"mean = {np.mean(cooptA_pct):.3f}%")
ax.set_xticks(x)
ax.set_xticklabels([f"{n}" for n in N_list])
ax.set_xlabel("任务数 $N$", fontsize=12)
ax.set_ylabel("相对提升 (%)", fontsize=12)
ax.set_title("(b) cooptA 相对提升", fontsize=13)
ax.set_ylim(0, 0.5)
ax.legend(fontsize=10)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

# -------- (c) 工业外推 --------
ax = axes[2]
tasks_per_shift = 1490          # 8h 班次完成的搬运次数
shifts = np.arange(1, 11)       # 1-10 个班次
daily_gain = cooptA_gain[2] / 1000 * tasks_per_shift * shifts

bars = ax.bar(shifts, daily_gain, color="steelblue",
               edgecolor="black", lw=0.5)
for b, v in zip(bars, daily_gain):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
            f"{v:.0f}", ha="center", fontsize=9)

ax.set_xlabel("累计班次数", fontsize=12)
ax.set_ylabel("cooptA 累积节省 (s)", fontsize=12)
ax.set_title(f"(c) 工业外推（{tasks_per_shift} 任务/班次）", fontsize=13)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("fig12_multi_task.png", dpi=150, bbox_inches="tight")
plt.show()
print("已保存 fig12_multi_task.png")