# -*- coding: utf-8 -*-
# short_scan.py
# 短路径下的六策略对比

from experiment import run_once
from strategies import AdaptiveThresholdStrategy, CoOptA, CoOptB, CoOptC
from robot import RobotParams

params = RobotParams()
start = (0.5, 0.5)
goal = (1.5, 1.0)   # 短路径

strategies = [
    ("adaptive(a=1.0)",     AdaptiveThresholdStrategy(1.0)),
    ("adaptive(a=1.15)",    AdaptiveThresholdStrategy(1.15)),
    ("adaptive(a=2.0)",     AdaptiveThresholdStrategy(2.0)),
    ("cooptA(m=20)",        CoOptA(k=0.5, trigger_margin=20)),
    ("cooptB(s=0.6)",       CoOptB(scale_min=0.6, trigger_margin=20)),
    ("cooptC(k=0.3,s=0.6)", CoOptC(k=0.3, scale_min=0.6, trigger_margin=20)),
]

print(f"短路径  goal = {goal}")
print(f"{'strategy':<22} {'total':>8} {'wait':>8} {'overlap':>8}")
print("-" * 50)

for name, st in strategies:
    r = run_once(st, start, goal, params)
    print(f"{name:<22} {r['total_time']:8.3f} "
          f"{r['wait_time']:8.3f} {r['overlap_time']:8.3f}")