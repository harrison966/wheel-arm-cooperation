# -*- coding: utf-8 -*-
# experiment_C.py
# 曲率路径实验：直线、L 形和 S 形路径

from base_model import (
    make_straight_path,
    make_L_path,
    make_S_path,
)
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
        "a_lat_max": 0.50,
    }

    strategies = [
        SerialStrategy(),
        FixedThresholdStrategy(),
        AdaptiveThresholdStrategy(),
        CoOptA(),
        CoOptB(),
        CoOptC(),
    ]

    paths = [
        (
            "straight 3m",
            make_straight_path((0.5, 0.5), (3.5, 0.5)),
        ),
        (
            "L-shape",
            make_L_path(
                (0.5, 0.5),
                (3.0, 0.5),
                (3.0, 3.0),
                arc_radius=0.3,
            ),
        ),
        (
            "S-shape",
            make_S_path(
                (0.5, 0.5),
                (2.0, 0.5),
                (2.0, 2.0),
                (3.5, 2.0),
                arc_radius=0.4,
            ),
        ),
    ]

    for label, path in paths:
        max_curvature = abs(path.curvature).max()

        print(
            f"\n=== {label} | "
            f"length={path.total_length:.2f} m | "
            f"max curvature={max_curvature:.3f} 1/m ==="
        )
        print(
            f"{'strategy':<10} {'total':>8} {'arrive':>8} "
            f"{'wait':>8} {'tau':>7} {'qdot':>8} "
            f"{'qddot':>8} {'status':>8}"
        )
        print("-" * 84)

        for strategy in strategies:
            result = simulate_transport(strategy, path, params)

            status = "timeout" if result["timed_out"] else "ok"

            print(
                f"{strategy.name:<10} "
                f"{result['total_time']:8.3f} "
                f"{result['t_arrive']:8.3f} "
                f"{result['wait_time']:8.3f} "
                f"{result['tau_applied']:7.3f} "
                f"{result['q_peak']:8.3f} "
                f"{result['qdd_peak']:8.3f} "
                f"{status:>8}"
            )