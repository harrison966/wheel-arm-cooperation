# -*- coding: utf-8 -*-
# disturbance_exp.py
# 扰动鲁棒性实验：速度噪声 + 机械臂负载变化

import numpy as np
from env import build_default_map, resample_path
from robot import RobotParams
from strategies import (AdaptiveThresholdStrategy, CoOptA, CoOptB, CoOptC)
from simulate import simulate_clean


# ============================================================
# 带扰动的仿真：重写 BaseKinematics 和 ArmModel
# ============================================================

def simulate_with_noise(strategy, path, params, v_noise=0.0, prep_noise=0.0, seed=None):
    """
    带扰动的仿真：
    - v_noise: 底盘速度扰动标准差（m/s）
    - prep_noise: 机械臂预抓取时间扰动标准差（s）
    """
    if seed is not None:
        np.random.seed(seed)

    from robot import BaseKinematics, ArmModel

    base = BaseKinematics(path, params)
    arm = ArmModel(params)
    base.reset()
    arm.reset()

    dt = params.dt
    t = 0.0
    wait_time = 0.0
    overlap_time = 0.0
    prep_start_t = None

    # 对机械臂的预抓取时间加噪声
    original_t_prep = params.t_prep

    while t < 600.0:
        # 1) 触发判断
        if arm.state == "idle" and strategy.decide_prep(base, arm, params):
            if hasattr(strategy, "get_arm_time_scale"):
                scale = strategy.get_arm_time_scale(base, arm, params)
            else:
                scale = 1.0
            # 给 scale 加负载噪声
            if prep_noise > 0:
                scale_noisy = scale * (1.0 + np.random.normal(0, prep_noise / original_t_prep))
                scale_noisy = float(np.clip(scale_noisy, 0.1, 1.0))
            else:
                scale_noisy = scale
            arm.start_prep(time_scale=scale_noisy)
            prep_start_t = t

        # 2) 底盘加速度指令
        if hasattr(strategy, "base_accel_cmd"):
            a_cmd = strategy.base_accel_cmd(base, arm, params)
        else:
            d = base.dist_to_goal()
            if d > (base.v**2) / (2 * params.a_max) + 0.05:
                a_cmd = params.a_max
            else:
                a_cmd = -params.a_max

        # 3) 到达判断
        arrived = base.dist_to_goal() <= params.reach_thresh

        if arrived and not arm.is_ready():
            base.v = 0.0
            wait_time += dt
        elif arrived and arm.is_ready():
            base.v = 0.0
            break
        else:
            # 底盘运动 + 速度噪声
            a = np.clip(a_cmd, -params.a_max, params.a_max)
            base.v = np.clip(base.v + a * dt, 0, params.v_max)
            if v_noise > 0:
                base.v = np.clip(base.v + np.random.normal(0, v_noise * dt),
                                 0, params.v_max)
            base.s_cur = min(base.s_cur + base.v * dt, base.total)
            base.t += dt

        # 4) 机械臂状态推进
        arm.step()

        # 5) 重叠时间
        if arm.state == "prepping" and base.v > 1e-3:
            overlap_time += dt

        t += dt

    t_arrive = t
    t += params.t_grasp
    t += params.t_place

    return {
        "strategy": strategy.name,
        "total_time": t,
        "t_arrive": t_arrive,
        "wait_time": wait_time,
        "overlap_time": overlap_time,
    }


# ============================================================
# 实验：扰动下四策略的鲁棒性
# ============================================================

def run_disturbance(n_trials=30):
    params = RobotParams()
    start = (0.5, 0.5)

    scenarios = [
        ("short", (1.0, 0.8)),
        ("long",  (9.0, 5.0)),
    ]

    strategies = [
        ("adaptive", AdaptiveThresholdStrategy(3.0)),
        ("cooptA",   CoOptA(k=0.5, trigger_margin=20)),
        ("cooptB",   CoOptB(scale_min=0.3, trigger_margin=20)),
        ("cooptC",   CoOptC(k=0.3, scale_min=0.3, trigger_margin=20)),
    ]

    disturbance_levels = [
        ("none",   0.0,  0.0),
        ("v_noise", 0.1, 0.0),    # 底盘速度 ±0.1 m/s
        ("prep_noise", 0.0, 0.3), # 机械臂预抓取 ±0.3s
    ]

    print("=" * 80)
    print("Disturbance Robustness Experiment")
    print("=" * 80)

    for level_name, v_noise, prep_noise in disturbance_levels:
        print(f"\n--- Disturbance: {level_name} "
              f"(v_noise={v_noise}, prep_noise={prep_noise}) ---")
        print(f"{'scenario':<8} {'strategy':<10} {'mean':>8} {'std':>8} "
              f"{'min':>8} {'max':>8}")
        print("-" * 60)

        for scenario_name, goal in scenarios:
            m = build_default_map()
            p = m.astar(start, goal)
            p, _ = resample_path(p, ds=0.05)

            for st_name, st in strategies:
                totals = []
                for trial in range(n_trials):
                    r = simulate_with_noise(
                        st, p, params,
                        v_noise=v_noise,
                        prep_noise=prep_noise,
                        seed=trial,
                    )
                    totals.append(r["total_time"])

                totals = np.array(totals)
                print(f"{scenario_name:<8} {st_name:<10} "
                      f"{totals.mean():>8.3f} {totals.std():>8.3f} "
                      f"{totals.min():>8.3f} {totals.max():>8.3f}")


if __name__ == "__main__":
    run_disturbance(n_trials=30)