# -*- coding: utf-8 -*-
# scan.py
# 参数扫描：alpha 扫描 + margin 扫描

from experiment import run_once
from strategies import AdaptiveThresholdStrategy, CoOptA
from robot import RobotParams


def scan_alpha(params=None):
    """扫描 adaptive 的 alpha 参数"""
    params = params or RobotParams()
    start, goal = (0.5, 0.5), (9.0, 5.0)

    print("=" * 60)
    print("adaptive 的 alpha 扫描  (goal = (9.0, 5.0))")
    print("=" * 60)
    print(f"{'alpha':>8} {'total':>10} {'wait':>10} {'overlap':>10}")
    print("-" * 45)

    best_total = 1e9
    best_alpha = None
    for a in [1.0, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0]:
        st = AdaptiveThresholdStrategy(alpha=a)
        r = run_once(st, start, goal, params)
        total = r["total_time"]
        print(f"{a:>8.2f} {total:>10.3f} "
              f"{r['wait_time']:>10.3f} {r['overlap_time']:>10.3f}")
        if total < best_total:
            best_total = total
            best_alpha = a

    print("-" * 45)
    print(f"adaptive 最优: alpha = {best_alpha}, total = {best_total:.3f}")
    return best_alpha, best_total


def scan_margin(params=None):
    """扫描 cooptA 的 trigger_margin 参数"""
    params = params or RobotParams()
    start, goal = (0.5, 0.5), (9.0, 5.0)

    print()
    print("=" * 60)
    print("cooptA 的 trigger_margin 扫描  (goal = (9.0, 5.0))")
    print("=" * 60)
    print(f"{'margin':>8} {'total':>10} {'arrive':>10} "
          f"{'wait':>10} {'overlap':>10}")
    print("-" * 55)

    best_total = 1e9
    best_margin = None
    for m in [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]:
        st = CoOptA(k=0.5, trigger_margin=m)
        r = run_once(st, start, goal, params)
        total = r["total_time"]
        print(f"{m:>8d} {total:>10.3f} {r['t_arrive']:>10.3f} "
              f"{r['wait_time']:>10.3f} {r['overlap_time']:>10.3f}")
        if total < best_total:
            best_total = total
            best_margin = m

    print("-" * 55)
    print(f"cooptA 最优: margin = {best_margin}, total = {best_total:.3f}")
    return best_margin, best_total


if __name__ == "__main__":
    _, adap_best = scan_alpha()
    _, coopt_best = scan_margin()

    print()
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"adaptive 最优 total = {adap_best:.3f}")
    print(f"cooptA  最优 total = {coopt_best:.3f}")
    print(f"cooptA 优势 = {adap_best - coopt_best:.3f} 秒")