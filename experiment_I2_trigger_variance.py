# -*- coding: utf-8 -*-
# experiment_I2_trigger_variance.py
# 卡尔曼滤波对"触发时刻方差"的影响

import numpy as np
from base_model import make_straight_path
from coordinator import (CoOptA, CoOptB, CoOptC,
                          simulate_transport, T_NOMINAL)

params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}
path = make_straight_path((0.5, 0.5), (10.5, 0.5))


def get_trigger_time(r):
    """从仿真 log 里提取触发时刻"""
    log = r["log"]
    t_arr = np.array(log["t"])
    R_a = np.array(log["R_a"])
    # 触发前 R_a = T_NOMINAL；触发后 R_a < T_NOMINAL
    below = R_a < T_NOMINAL - 1e-6
    if not np.any(below):
        return None
    idx = int(np.argmax(below))
    return float(t_arr[idx])


if __name__ == "__main__":
    print(f"T_NOMINAL = {T_NOMINAL:.3f} s")
    print(f"{'strategy':<10} {'sigma':>6} {'filter':>8} "
          f"{'trig_mean':>12} {'trig_std':>12} {'n_valid':>8}")
    print("-" * 62)

    for sigma in [0.0, 0.2, 0.4, 0.6]:
        for name, st in [("cooptA", CoOptA()),
                          ("cooptB", CoOptB()),
                          ("cooptC", CoOptC())]:
            for use_filter in [False, True]:
                triggers = []
                for seed in range(30):
                    r = simulate_transport(st, path, params, seed=seed,
                                            rb_noise_std=sigma,
                                            rb_filter=use_filter)
                    t_trig = get_trigger_time(r)
                    if t_trig is not None:
                        triggers.append(t_trig)

                if len(triggers) == 0:
                    print(f"{name:<10} {sigma:>6.1f} {str(use_filter):>8} "
                          f"{'nan':>12} {'nan':>12} {0:>8}")
                    continue

                triggers = np.array(triggers)
                print(f"{name:<10} {sigma:>6.1f} {str(use_filter):>8} "
                      f"{triggers.mean():>12.4f} {triggers.std():>12.4f} "
                      f"{len(triggers):>8d}")
        print()