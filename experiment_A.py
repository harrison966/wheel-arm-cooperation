# -*- coding: utf-8 -*-
# experiment_A.py
# 路径长度扫描：总时间、等待时间、重定时与关节峰值

from base_model import make_straight_path
from coordinator import (
    SerialStrategy,
    FixedThresholdStrategy,
    AdaptiveThresholdStrategy,
    CoOptA,
    CoOptB,
    CoOptC,
    simulate_transport,
)


if __name__ == "__main__":
    params = {
        "v_nominal": 0.35,
        "v_max": 0.50,
        "a_max": 0.40,
    }

    strategies = [
        SerialStrategy(),
        FixedThresholdStrategy(),
        AdaptiveThresholdStrategy(),
        CoOptA(),
        CoOptB(),
        CoOptC(),
    ]

    goals = [
        ("0.5 m", (1.0, 0.5)),
        ("1.0 m", (1.5, 0.5)),
        ("1.5 m", (2.0, 0.5)),
        ("2.0 m", (2.5, 0.5)),
        ("3.0 m", (3.5, 0.5)),
        ("5.0 m", (5.5, 0.5)),
        ("10.0 m", (10.5, 0.5)),
    ]

    print(
        f"{'path':>8} {'strategy':<10} {'total':>8} "
        f"{'arrive':>8} {'wait':>8} {'tau':>7} "
        f"{'qdot':>8} {'qddot':>8} {'status':>8}"
    )
    print("-" * 96)

    for label, goal in goals:
        path = make_straight_path((0.5, 0.5), goal)

        for strategy in strategies:
            result = simulate_transport(strategy, path, params)

            status = "timeout" if result["timed_out"] else "ok"

            print(
                f"{label:>8} {strategy.name:<10} "
                f"{result['total_time']:8.3f} "
                f"{result['t_arrive']:8.3f} "
                f"{result['wait_time']:8.3f} "
                f"{result['tau_applied']:7.3f} "
                f"{result['q_peak']:8.3f} "
                f"{result['qdd_peak']:8.3f} "
                f"{status:>8}"
            )

        print()