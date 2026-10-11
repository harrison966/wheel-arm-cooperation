# short_scan3.py
from experiment import run_once
from strategies import (AdaptiveThresholdStrategy, CoOptA, CoOptB, CoOptC)
from robot import RobotParams

params = RobotParams()
start = (0.5, 0.5)
for goal in [(1.0, 0.8), (0.9, 0.7), (0.8, 0.6)]:
    print(f"\n=== goal = {goal} ===")
    for name, st in [
        ("adap(a=2.0)", AdaptiveThresholdStrategy(2.0)),
        ("cooptA(m=20)", CoOptA(0.5, 20)),
        ("cooptB(s=0.6)", CoOptB(0.6, 20)),
        ("cooptC(k=0.3,s=0.6)", CoOptC(0.3, 0.6, 20)),
    ]:
        r = run_once(st, start, goal, params)
        print(f"  {name:<22} total={r['total_time']:.3f} "
              f"wait={r['wait_time']:.3f} overlap={r['overlap_time']:.3f}")