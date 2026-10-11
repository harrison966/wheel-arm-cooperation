# short_scan2.py
from experiment import run_once
from strategies import AdaptiveThresholdStrategy
from robot import RobotParams

params = RobotParams()
start, goal = (0.5, 0.5), (1.5, 1.0)
print(f"{'alpha':>8} {'total':>8} {'wait':>8} {'overlap':>8}")
for a in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
    st = AdaptiveThresholdStrategy(alpha=a)
    r = run_once(st, start, goal, params)
    print(f"{a:>8.2f} {r['total_time']:>8.3f} "
          f"{r['wait_time']:>8.3f} {r['overlap_time']:>8.3f}")