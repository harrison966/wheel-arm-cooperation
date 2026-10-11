# -*- coding: utf-8 -*-
# final_scan.py
# 补齐 1.0, 0.8 场景的数据

from experiment import run_once
from strategies import (AdaptiveThresholdStrategy, CoOptA, CoOptB, CoOptC)
from robot import RobotParams

params = RobotParams()
start = (0.5, 0.5)

print("{} {} {} {} {}".format("goal", "adap", "cooptA", "cooptB", "cooptC"))
print("-" * 55)

for goal in [(0.8, 0.6), (1.0, 0.8), (1.5, 1.0), (2.0, 1.5), (3.5, 2.0), (5.0, 5.0), (9.0, 5.0)]:
    r_a = run_once(AdaptiveThresholdStrategy(3.0), start, goal, params)
    r_A = run_once(CoOptA(0.5, 20), start, goal, params)
    r_B = run_once(CoOptB(0.3, 20), start, goal, params)
    r_C = run_once(CoOptC(0.3, 0.3, 20), start, goal, params)
    print("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        str(goal), r_a["total_time"], r_A["total_time"],
        r_B["total_time"], r_C["total_time"]))