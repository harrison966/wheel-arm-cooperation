# -*- coding: utf-8 -*-
# experiment_J_curvature_aware.py
# 曲率感知 cooptB/C

from base_model import make_L_path, make_S_path, make_straight_path
from coordinator import (FixedThresholdStrategy, CoOptB, CoOptC,
                          simulate_transport)

params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}

paths = [
    ("straight", make_straight_path((0.5, 0.5), (3.5, 0.5))),
    ("L-shape",  make_L_path((0.5, 0.5), (3.0, 0.5), (3.0, 3.0),
                              arc_radius=0.3)),
    ("S-shape",  make_S_path((0.5, 0.5), (2.0, 0.5), (2.0, 2.0),
                              (3.5, 2.0), arc_radius=0.4)),
]

configs = [
    ("fixed",           FixedThresholdStrategy()),
    ("cooptB (orig)",   CoOptB()),
    ("cooptC (orig)",   CoOptC()),
    ("cooptC (curv)",   CoOptC(curvature_aware=True)),
]

print(f"{'config':<18} " + " ".join(f"{p[0]:>12}" for p in paths))
print("-" * 60)
for name, st in configs:
    row = f"{name:<18}"
    for _, path in paths:
        r = simulate_transport(st, path, params)
        row += f" {r['total_time']:>12.3f}"
    print(row)