# -*- coding: utf-8 -*-
# experiment_H_ablation.py
# 模块消融：拆解 cooptA/B/C 各自的贡献

from base_model import make_straight_path
from coordinator import (SerialStrategy, FixedThresholdStrategy,
                          CoOptA, CoOptB, CoOptC, simulate_transport)

params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}

goals = [("0.5 m", (1.0, 0.5)),
         ("3.0 m", (3.5, 0.5)),
         ("10.0 m", (10.5, 0.5))]

configs = [
    ("serial",           SerialStrategy()),
    ("fixed (baseline)", FixedThresholdStrategy()),
    ("cooptA only",      CoOptA()),
    ("cooptB only",      CoOptB()),
    ("cooptC (both)",    CoOptC()),
]

print(f"{'config':<22} {'0.5 m':>10} {'3.0 m':>10} {'10.0 m':>10}")
print("-" * 56)
for name, st in configs:
    row = f"{name:<22}"
    for _, goal in goals:
        path = make_straight_path((0.5, 0.5), goal)
        r = simulate_transport(st, path, params)
        row += f" {r['total_time']:>10.3f}"
    print(row)