# -*- coding: utf-8 -*-
# experiment_G_param.py
# 控制器参数敏感性扫描

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

from base_model import make_straight_path
from coordinator import CoOptA, simulate_transport

params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}
paths = [("0.5 m", make_straight_path((0.5, 0.5), (1.0, 0.5))),
         ("3.0 m", make_straight_path((0.5, 0.5), (3.5, 0.5))),
         ("10.0 m", make_straight_path((0.5, 0.5), (10.5, 0.5)))]

# 扫描 k_v
k_v_list = [0.02, 0.05, 0.08, 0.12, 0.20, 0.30]
fig, ax = plt.subplots(figsize=(8, 5))
for label, path in paths:
    totals = [simulate_transport(CoOptA(k_v=kv), path, params)["total_time"]
              for kv in k_v_list]
    ax.plot(k_v_list, totals, marker="o", label=label, linewidth=2)
ax.set_xlabel("$k_v$"); ax.set_ylabel("total time (s)")
ax.set_title("cooptA: sensitivity to $k_v$")
ax.legend(); ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("fig9_kv_sensitivity.png", dpi=150)
plt.show()

# 扫描 m_trig
m_list = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
fig, ax = plt.subplots(figsize=(8, 5))
for label, path in paths:
    totals = [simulate_transport(CoOptA(trigger_margin=m), path, params)["total_time"]
              for m in m_list]
    ax.plot(m_list, totals, marker="s", label=label, linewidth=2)
ax.set_xlabel("$m_{trig}$ (s)"); ax.set_ylabel("total time (s)")
ax.set_title("cooptA: sensitivity to $m_{trig}$")
ax.legend(); ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("fig10_mtrig_sensitivity.png", dpi=150)
plt.show()

# 打印数据表
print(f"{'path':<8} {'metric':<8} " + " ".join(f"{v:>8}" for v in k_v_list))
for label, path in paths:
    totals = [simulate_transport(CoOptA(k_v=kv), path, params)["total_time"]
              for kv in k_v_list]
    print(f"{label:<8} {'k_v':<8} " + " ".join(f"{v:>8.3f}" for v in totals))

print()
print(f"{'path':<8} {'metric':<8} " + " ".join(f"{v:>8}" for v in m_list))
for label, path in paths:
    totals = [simulate_transport(CoOptA(trigger_margin=m), path, params)["total_time"]
              for m in m_list]
    print(f"{label:<8} {'m_trig':<8} " + " ".join(f"{v:>8.3f}" for v in totals))