# -*- coding: utf-8 -*-
# cooptB_scan.py
# 扫描 cooptB 的 scale_min 和 trigger_margin

from experiment import run_once
from strategies import CoOptB
from robot import RobotParams

params = RobotParams()
start = (0.5, 0.5)
goal = (0.8, 0.6)

print(f"cooptB scale_min 扫描  (goal = {goal}, margin = 20)")
print(f"{'scale_min':>10} {'total':>8} {'wait':>8} {'overlap':>8}")
print("-" * 40)

for s in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    st = CoOptB(scale_min=s, trigger_margin=20)
    r = run_once(st, start, goal, params)
    print(f"{s:>10.2f} {r['total_time']:>8.3f} "
          f"{r['wait_time']:>8.3f} {r['overlap_time']:>8.3f}")