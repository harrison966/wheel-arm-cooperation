# -*- coding: utf-8 -*-
# simulate.py
# 仿真主循环

import numpy as np
from robot import RobotParams, BaseKinematics, ArmModel


def simulate_clean(strategy, path, params: RobotParams, max_time=120.0):
    """
    分两阶段：
    阶段1：底盘运动 + 机械臂预抓取
    阶段2：抓取 + 放置（固定耗时）
    """
    base = BaseKinematics(path, params)
    arm = ArmModel(params)
    base.reset()
    arm.reset()

    dt = params.dt
    t = 0.0
    wait_time = 0.0
    overlap_time = 0.0
    prep_start_t = None
    prep_end_t = None

    log = {"t": [], "s": [], "v": [], "arm_state": [], "dist": []}

    # ---------------- 阶段1 ----------------
    while t < max_time:
        # 1) 策略决定是否触发预抓取，并给出机械臂时间缩放
        if arm.state == "idle" and strategy.decide_prep(base, arm, params):
            # ★ 新增：取机械臂时间缩放
            if hasattr(strategy, "get_arm_time_scale"):
                scale = strategy.get_arm_time_scale(base, arm, params)
            else:
                scale = 1.0
            arm.start_prep(time_scale=scale)
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
            base.step(a_cmd)

        # 4) 机械臂状态推进
        prev = arm.state
        arm.step()
        if prev == "prepping" and arm.state == "ready":
            prep_end_t = t + dt

        # 5) 重叠时间统计
        if arm.state == "prepping" and base.v > 1e-3:
            overlap_time += dt

        # 6) 日志
        t += dt
        log["t"].append(t)
        log["s"].append(base.s_cur)
        log["v"].append(base.v)
        log["arm_state"].append(arm.state)
        log["dist"].append(base.dist_to_goal())

    t_arrive = t

    # ---------------- 阶段2 ----------------
    t += params.t_grasp
    t += params.t_place
    log["t"].append(t)
    log["s"].append(base.total)
    log["v"].append(0.0)
    log["arm_state"].append("done")
    log["dist"].append(0.0)

    return {
        "strategy": strategy.name,
        "total_time": t,
        "t_arrive": t_arrive,
        "wait_time": wait_time,
        "overlap_time": overlap_time,
        "prep_start_t": prep_start_t,
        "prep_end_t": prep_end_t,
        "log": log,
    }