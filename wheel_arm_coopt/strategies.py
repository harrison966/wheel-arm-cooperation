# -*- coding: utf-8 -*-
# strategies.py
# 六种调度策略：serial / fixed / adaptive / cooptA / cooptB / cooptC

import numpy as np
from robot import RobotParams


# ============================================================
# 基类：默认梯形速度曲线
# ============================================================

class BaseStrategy:
    name = "base"

    def decide_prep(self, base, arm, params):
        return False

    def get_arm_time_scale(self, base, arm, params):
        """默认不加速机械臂"""
        return 1.0

    def base_accel_cmd(self, base, arm, params):
        """默认梯形速度曲线：接近目标时减速"""
        d = base.dist_to_goal()
        if d > (base.v**2) / (2 * params.a_max) + 0.05:
            return params.a_max
        return -params.a_max


# ============================================================
# 1. 串行策略
# ============================================================

class SerialStrategy(BaseStrategy):
    name = "serial"

    def decide_prep(self, base, arm, params):
        return base.dist_to_goal() <= params.reach_thresh and base.v < 1e-3


# ============================================================
# 2. 固定阈值策略
# ============================================================

class FixedThresholdStrategy(BaseStrategy):
    name = "fixed"

    def __init__(self, d_trigger=1.2):
        self.d = d_trigger

    def decide_prep(self, base, arm, params):
        return base.dist_to_goal() <= self.d


# ============================================================
# 3. 速度自适应阈值策略
# ============================================================

class AdaptiveThresholdStrategy(BaseStrategy):
    name = "adaptive"

    def __init__(self, alpha=1.15):
        self.alpha = alpha

    def decide_prep(self, base, arm, params):
        v = max(base.v, 1e-3)
        d_trig = self.alpha * v * params.t_prep
        return base.dist_to_goal() <= d_trig


# ============================================================
# 4. cooptA：底盘减速型（只调底盘）
# ============================================================

class CoOptA(BaseStrategy):
    """delta>0 时加速，delta<0 时减速"""
    name = "cooptA"

    def __init__(self, k=0.5, trigger_margin=20):
        self.k = k
        self.trigger_margin = trigger_margin

    def decide_prep(self, base, arm, params):
        T_arr = base.time_to_goal()
        t_rem = arm.prep_remain()
        return T_arr <= t_rem + self.trigger_margin * params.dt

    def base_accel_cmd(self, base, arm, params):
        if arm.state == "idle":
            d = base.dist_to_goal()
            if d > (base.v**2) / (2 * params.a_max) + 0.05:
                return params.a_max
            return -params.a_max
        T_arr = base.time_to_goal()
        t_rem = arm.prep_remain()
        delta = T_arr - t_rem
        a_cmd = self.k * delta      # ★ 正号
        return float(np.clip(a_cmd, -params.a_max, params.a_max))


# ============================================================
# 5. cooptB：机械臂加速型（只调机械臂）
# ============================================================

class CoOptB(BaseStrategy):
    """
    底盘全速跑，机械臂按需要的速度加速预抓取。
    触发条件：T_arr <= t_prep_max（标准 2.5 秒）
    """
    name = "cooptB"

    def __init__(self, scale_min=0.6, trigger_margin=20):
        self.scale_min = scale_min      # 机械臂最快缩放（0.6 = 只用 60% 时间）
        self.trigger_margin = trigger_margin

    def decide_prep(self, base, arm, params):
        T_arr = base.time_to_goal()
        t_rem = arm.prep_remain()
        return T_arr <= t_rem + self.trigger_margin * params.dt

    def get_arm_time_scale(self, base, arm, params):
        """
        机械臂需要在 T_arr 秒内完成预抓取。
        如果 T_arr < t_prep，则加速（scale = T_arr / t_prep）
        """
        T_arr = base.dist_to_goal() / params.v_max
        scale = T_arr / params.t_prep
        scale = max(self.scale_min, min(scale, 1.0))
        return scale

    def base_accel_cmd(self, base, arm, params):
        # 底盘全速，不做对齐
        d = base.dist_to_goal()
        if d > (base.v**2) / (2 * params.a_max) + 0.05:
            return params.a_max
        return -params.a_max


# ============================================================
# 6. cooptC：双边调整（底盘 + 机械臂）
# ============================================================

class CoOptC(BaseStrategy):
    """
    底盘轻微调速 + 机械臂按需加速，两边一起配合。
    """
    name = "cooptC"

    def __init__(self, k=0.3, scale_min=0.6, trigger_margin=20):
        self.k = k
        self.scale_min = scale_min
        self.trigger_margin = trigger_margin

    def decide_prep(self, base, arm, params):
        T_arr = base.time_to_goal()
        t_rem = arm.prep_remain()
        return T_arr <= t_rem + self.trigger_margin * params.dt

    def get_arm_time_scale(self, base, arm, params):
    # 用剩余距离 / v_max 作为底盘到达时间的保守估计
        T_arr = base.dist_to_goal() / params.v_max
        scale = T_arr / params.t_prep
        scale = max(self.scale_min, min(scale, 1.0))
        return scale

    def base_accel_cmd(self, base, arm, params):
        if arm.state == "idle":
            d = base.dist_to_goal()
            if d > (base.v**2) / (2 * params.a_max) + 0.05:
                return params.a_max
            return -params.a_max
        T_arr = base.time_to_goal()
        t_rem = arm.prep_remain()
        delta = T_arr - t_rem
        a_cmd = self.k * delta      # ★ 正号，k 更小
        return float(np.clip(a_cmd, -params.a_max, params.a_max))