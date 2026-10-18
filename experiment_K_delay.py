# -*- coding: utf-8 -*-
# experiment_K_delay.py
# 控制时延鲁棒性

from base_model import make_straight_path
from coordinator import (FixedThresholdStrategy, CoOptA, CoOptC,
                          simulate_transport)

params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}
paths = [("0.5 m", make_straight_path((0.5, 0.5), (1.0, 0.5))),
         ("3.0 m", make_straight_path((0.5, 0.5), (3.5, 0.5))),
         ("10.0 m", make_straight_path((0.5, 0.5), (10.5, 0.5)))]

print(f"{'strategy':<10} {'path':<8} {'delay=0':>10} "
      f"{'delay=0.05':>12} {'delay=0.10':>12}")
print("-" * 60)
for name, st in [("fixed", FixedThresholdStrategy()),
                 ("cooptA", CoOptA()),
                 ("cooptC", CoOptC())]:
    for plabel, path in paths:
        row = f"{name:<10} {plabel:<8}"
        for d in [0.0, 0.05, 0.10]:
            r = simulate_transport(st, path, params, delay=d)
            row += f" {r['total_time']:>12.3f}"
        print(row)