# -*- coding: utf-8 -*-
# cooptC_scan.py
# 扫描 cooptC 的 scale_min 参数

from experiment import run_once
from strategies import CoOptC
from robot import RobotParams

params = RobotParams()
start = (0.5, 0.5)
goal = (0.8, 0.6)

print("cooptC scale_min 扫描  (goal = (0.8, 0.6))")
print("{:>8} {:>8} {:>8} {:>8}".format("s_min", "total", "wait", "overlap"))
print("-" * 40)

for s in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    st = CoOptC(k=0.3, scale_min=s, trigger_margin=20)
    r = run_once(st, start, goal, params)
    print("{:>8.2f} {:>8.3f} {:>8.3f} {:>8.3f}".format(
        s, r["total_time"], r["wait_time"], r["overlap_time"]))