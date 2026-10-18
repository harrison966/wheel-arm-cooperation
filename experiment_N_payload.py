# -*- coding: utf-8 -*-
# experiment_N_payload.py
# 负载变化下的鲁棒性实验
# 改变实际预抓取时间 t_prep ∈ {1.5, 2.0, 2.5, 3.0, 4.0} s

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

from base_model import make_straight_path
from coordinator import (FixedThresholdStrategy, AdaptiveThresholdStrategy,
                          CoOptA, CoOptB, CoOptC,
                          simulate_transport)

params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}

paths = [
    ("0.5 m",  make_straight_path((0.5, 0.5), (1.0, 0.5))),
    ("3.0 m",  make_straight_path((0.5, 0.5), (3.5, 0.5))),
    ("10.0 m", make_straight_path((0.5, 0.5), (10.5, 0.5))),
]

strategies = [
    ("fixed",    FixedThresholdStrategy()),
    ("adaptive", AdaptiveThresholdStrategy()),
    ("cooptA",   CoOptA()),
    ("cooptB",   CoOptB()),
    ("cooptC",   CoOptC()),
]

payloads = [1.5, 2.0, 2.5, 3.0, 4.0]

# ============================================================
# 数据收集
# ============================================================
results = {}
for name, st in strategies:
    results[name] = {pl: [] for pl in payloads}

for name, st in strategies:
    for pl in payloads:
        for label, path in paths:
            r = simulate_transport(st, path, params,
                                    actual_prep_time=pl)
            results[name][pl].append(r["total_time"])

# ============================================================
# 打印数据表
# ============================================================
print(f"{'strategy':<10}", end="")
for pl in payloads:
    print(f" t_prep={pl:.1f}", end="")
print()
print("-" * 70)

for name, _ in strategies:
    row = f"{name:<10}"
    for pl in payloads:
        # 用 3.0 m 路径的数据
        row += f" {results[name][pl][1]:>11.3f}"
    print(row)

print()
print(f"{'strategy':<10}", end="")
for pl in payloads:
    print(f" t_prep={pl:.1f}", end="")
print()
print("-" * 70)
print("（10.0 m 路径）")
for name, _ in strategies:
    row = f"{name:<10}"
    for pl in payloads:
        row += f" {results[name][pl][2]:>11.3f}"
    print(row)
print()
print(f"{'strategy':<10}", end="")
for pl in payloads:
    print(f" t_prep={pl:.1f}", end="")
print()
print("-" * 70)
print("（0.5 m 路径）")
for name, _ in strategies:
    row = f"{name:<10}"
    for pl in payloads:
        row += f" {results[name][pl][0]:>11.3f}"   # ← 索引 0 = 0.5 m
    print(row)

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (path_label, path) in zip(axes, paths):
    for name, _ in strategies:
        # 用对应路径的索引
        idx = [p[0] for p in paths].index(path_label)
        totals = [results[name][pl][idx] for pl in payloads]
        ax.plot(payloads, totals, marker="o", label=name, linewidth=2)
    ax.set_xlabel("实际预抓取时间 $t_{prep}$ (s)", fontsize=12)
    ax.set_ylabel("总任务时间 (s)", fontsize=12)
    ax.set_title(f"路径: {path_label}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("fig15_payload.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n已保存 fig15_payload.png")