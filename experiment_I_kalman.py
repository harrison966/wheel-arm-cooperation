# -*- coding: utf-8 -*-
# experiment_I_kalman.py
# 卡尔曼滤波对比

import numpy as np
from base_model import make_straight_path
from coordinator import CoOptA, CoOptC, simulate_transport

params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}
path = make_straight_path((0.5, 0.5), (10.5, 0.5))

print(f"{'strategy':<10} {'sigma':>6} {'filter':>8} "
      f"{'mean':>10} {'std':>8} {'RMSE':>10}")
print("-" * 60)
for sigma in [0.2, 0.4, 0.6]:
    for name, st in [("cooptA", CoOptA()), ("cooptC", CoOptC())]:
        for use_filter in [False, True]:
            totals, rmses = [], []
            for seed in range(30):
                r = simulate_transport(st, path, params, seed=seed,
                                        rb_noise_std=sigma,
                                        rb_filter=use_filter)
                totals.append(r["total_time"])
                deltas = np.array(r["log"]["Delta"])
                rmses.append(np.sqrt(np.mean(deltas**2)))
            print(f"{name:<10} {sigma:>6.1f} {str(use_filter):>8} "
                  f"{np.mean(totals):>10.3f} {np.std(totals):>8.3f} "
                  f"{np.mean(rmses):>10.4f}")