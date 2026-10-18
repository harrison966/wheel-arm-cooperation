# -*- coding: utf-8 -*-
# experiment_F_noise.py
# Rb 观测噪声鲁棒性

import numpy as np
from base_model import make_straight_path
from coordinator import (FixedThresholdStrategy, CoOptA, CoOptC,
                          simulate_transport)


def run(strategy, path, params, sigma, n_trials=30):
    totals, rmses, oscs, fails = [], [], [], 0
    for seed in range(n_trials):
        r = simulate_transport(strategy, path, params,
                                seed=seed, rb_noise_std=sigma)
        if r["timed_out"]:
            fails += 1
            continue
        totals.append(r["total_time"])
        deltas = np.array(r["log"]["Delta"])
        rmses.append(np.sqrt(np.mean(deltas**2)))
        oscs.append(int(np.sum(np.diff(np.sign(deltas)) != 0)))
    if len(totals) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), fails
    return (np.mean(totals), np.std(totals),
            np.mean(rmses), np.mean(oscs), fails)


if __name__ == "__main__":
    params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}
    path = make_straight_path((0.5, 0.5), (10.5, 0.5))

    print(f"{'strategy':<10} {'sigma':>6} {'mean':>10} {'std':>8} "
          f"{'RMSE':>10} {'osc':>6} {'fail':>6}")
    print("-" * 62)
    for sigma in [0.0, 0.2, 0.4, 0.6]:
        for name, st in [("fixed", FixedThresholdStrategy()),
                         ("cooptA", CoOptA()),
                         ("cooptC", CoOptC())]:
            m, s, rmse, osc, fail = run(st, path, params, sigma)
            print(f"{name:<10} {sigma:>6.1f} {m:>10.3f} {s:>8.3f} "
                  f"{rmse:>10.4f} {osc:>6.1f} {fail:>6d}")