import numpy as np

from config import (
    Q_IDLE, Q_READY, DT,
    T_ALIGN, T_GRIPPER, T_PLACE, T_NOMINAL,
)
from arm_trajectory import (
    min_feasible_time,
    plan_quintic_trajectory,
    sample_trajectory,
)
from base_model import DiffDriveBase, BaseParams


T_MIN_TRAJ, _, _, _ = min_feasible_time(Q_IDLE, Q_READY)

if T_NOMINAL < T_MIN_TRAJ:
    raise ValueError("T_NOMINAL 不能小于机械臂轨迹最短可行时间。")

TAU_LOWER = T_MIN_TRAJ / T_NOMINAL


# ============================================================
# 一维卡尔曼滤波
# ============================================================
class KalmanFilter1D:
    def __init__(self, q=1e-3, r=0.04, known_rate=-1.0):
        self.x = 0.0
        self.P = 1.0
        self.Q = q
        self.R = r
        self.known_rate = known_rate

    def reset(self, x0):
        self.x = float(x0)
        self.P = 1.0

    def update(self, z, dt):
        self.x = self.x + self.known_rate * dt
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x


# ============================================================
# 底盘参数注入
# ============================================================
def make_base(path, params):
    base_params = BaseParams()
    keys = ["v_max", "a_max", "a_lat_max",
            "wheel_base", "wheel_radius", "v_wheel_max",
            "omega_max", "reach_thresh", "vel_thresh"]
    for key in keys:
        if key in params:
            setattr(base_params, key, float(params[key]))
    return DiffDriveBase(path, base_params)


def speed_tracking_accel(base, v_target, a_max):
    d_brake = base.v ** 2 / (2.0 * a_max) + 0.03
    if base.dist_to_goal() <= d_brake:
        return -a_max
    error = v_target - base.v
    if error > 0.01:
        return a_max
    if error < -0.01:
        return -a_max
    return 0.0


# ============================================================
# 策略基类
# ============================================================
class BaseStrategy:
    name = "base"

    def decide_trigger(self, base, arm_state, params):
        return False

    def arm_time_scale(self, base, arm_state, params):
        return 1.0

    def base_speed_target(self, base, arm_state, params):
        return params["v_nominal"]

    def base_command(self, base, arm_state, params):
        v_target = self.base_speed_target(base, arm_state, params)
        v_target = float(np.clip(v_target, 0.0, base.p.v_max))
        v_bias = params.get("v_bias", 0.0)
        v_target_actual = v_target * (1.0 - v_bias)
        arm_state["v_target"] = v_target
        arm_state["v_target_actual"] = v_target_actual
        a_cmd = speed_tracking_accel(base, v_target_actual, base.p.a_max)
        return a_cmd, v_target_actual


class SerialStrategy(BaseStrategy):
    name = "serial"

    def decide_trigger(self, base, arm_state, params):
        return (not arm_state["triggered"]
                and base.dist_to_goal() <= base.p.reach_thresh)


class FixedThresholdStrategy(BaseStrategy):
    name = "fixed"

    def __init__(self, d_trigger=1.2):
        self.d_trigger = float(d_trigger)

    def decide_trigger(self, base, arm_state, params):
        if arm_state["triggered"]:
            return False
        return base.dist_to_goal() <= max(self.d_trigger,
                                           base.p.reach_thresh)


class AdaptiveThresholdStrategy(BaseStrategy):
    name = "adaptive"

    def __init__(self, alpha=1.15):
        self.alpha = float(alpha)

    def decide_trigger(self, base, arm_state, params):
        if arm_state["triggered"]:
            return False
        t_prep_actual = arm_state.get("T_prep_actual", T_NOMINAL)
        d_trigger = self.alpha * max(base.v, 1e-3) * t_prep_actual
        return base.dist_to_goal() <= max(d_trigger,
                                           base.p.reach_thresh)


class CoOptA(BaseStrategy):
    name = "cooptA"

    def __init__(self, k_v=0.12, trigger_margin=0.30):
        self.k_v = float(k_v)
        self.k = k_v
        self.trigger_margin = float(trigger_margin)
        self.m_trig = trigger_margin

    def decide_trigger(self, base, arm_state, params):
        if arm_state["triggered"]:
            return False
        R_b = arm_state.get("R_b_obs",
                             base.time_to_goal(v_coord=params["v_nominal"]))
        t_prep_actual = arm_state.get("T_prep_actual", T_NOMINAL)
        return R_b <= t_prep_actual + self.trigger_margin

    def base_speed_target(self, base, arm_state, params):
        if not arm_state["triggered"]:
            return params["v_nominal"]
        r_b = arm_state.get("R_b_obs",
                             base.time_to_goal(v_coord=arm_state["v_target"]))
        delta = r_b - arm_state["R_a"]
        arm_state["delta"] = delta
        return params["v_nominal"] + self.k_v * delta


class CoOptB(BaseStrategy):
    name = "cooptB"

    def __init__(self, trigger_margin=0.30):
        self.trigger_margin = float(trigger_margin)
        self.m_trig = trigger_margin

    def decide_trigger(self, base, arm_state, params):
        if arm_state["triggered"]:
            return False
        R_b = arm_state.get("R_b_obs",
                             base.time_to_goal(v_coord=params["v_nominal"]))
        t_prep_actual = arm_state.get("T_prep_actual", T_NOMINAL)
        return R_b <= t_prep_actual + self.trigger_margin

    def arm_time_scale(self, base, arm_state, params):
        R_b = arm_state.get("R_b_obs",
                             base.time_to_goal(v_coord=params["v_nominal"]))
        t_prep_actual = arm_state.get("T_prep_actual", T_NOMINAL)
        tau = R_b / t_prep_actual
        return float(np.clip(tau, TAU_LOWER, 1.0))


class CoOptC(CoOptB):
    name = "cooptC"

    def __init__(self, k_v=0.06, trigger_margin=0.30, curvature_aware=False):
        super().__init__(trigger_margin=trigger_margin)
        self.k_v = float(k_v)
        self.k = k_v
        self.curvature_aware = curvature_aware

    def base_speed_target(self, base, arm_state, params):
        if not arm_state["triggered"]:
            return params["v_nominal"]
        r_b = arm_state.get("R_b_obs",
                             base.time_to_goal(v_coord=arm_state["v_target"]))
        delta = r_b - arm_state["R_a"]
        arm_state["delta"] = delta
        return params["v_nominal"] + self.k_v * delta

    def arm_time_scale(self, base, arm_state, params):
        R_b = arm_state.get("R_b_obs",
                             base.time_to_goal(v_coord=params["v_nominal"]))
        t_prep_actual = arm_state.get("T_prep_actual", T_NOMINAL)
        tau = R_b / t_prep_actual
        if self.curvature_aware:
            kappa = abs(base.current_curvature())
            gamma = 1.0 / (1.0 + 0.3 * max(0.0, kappa - 1.0))
            tau = tau * gamma
        return float(np.clip(tau, TAU_LOWER, 1.0))


# ============================================================
# 单次仿真
# ============================================================
def simulate_transport(
    strategy,
    path,
    params=None,
    max_time=60.0,
    verbose=False,
    v_noise=0.0,
    v_bias=0.0,
    rb_noise_std=0.0,
    rb_filter=False,
    delay=0.0,
    actual_prep_time=None,   # ★ 新增
    seed=None,
):
    default_params = {
        "dt": DT,
        "v_max": 0.50,
        "a_max": 0.40,
        "v_nominal": 0.35,
        "reach_thresh": 0.08,
        "vel_thresh": 0.01,
    }
    if params is None:
        params = {}
    params = {**default_params, **params}
    params["v_bias"] = float(v_bias)

    if actual_prep_time is None:
        actual_prep_time = T_NOMINAL
    actual_prep_time = float(actual_prep_time)

    base = make_base(path, params)
    rng = np.random.default_rng(seed)
    rb_filter_obj = KalmanFilter1D(q=1e-3, r=0.04, known_rate=-1.0) if rb_filter else None

    dt = params["dt"]
    delay_steps = int(round(delay / dt))
    a_cmd_buffer = [0.0] * delay_steps

    arm_state = {
        "triggered": False,
        "traj": None,
        "t_start": None,
        "R_a": actual_prep_time,           # ★ 用实际时间
        "R_b_obs": actual_prep_time,
        "T_prep_actual": actual_prep_time,  # ★ 新增
        "v_target": params["v_nominal"],
        "v_target_actual": params["v_nominal"] * (1.0 - v_bias),
        "delta": np.nan,
        "tau_requested": 1.0,
        "tau_applied": 1.0,
        "q_cur": Q_IDLE.copy(),
        "q_peak": 0.0,
        "qdd_peak": 0.0,
        "noise_injected": False,
        "filter_initialized": False,
    }

    log = {"t": [], "s": [], "v": [], "v_target": [],
           "R_a": [], "R_b": [], "Delta": []}

    t = 0.0
    wait_time = 0.0
    overlap_time = 0.0

    while t < max_time:
        # Rb 观测
        Rb_true = base.time_to_goal(v_coord=params["v_nominal"])
        if rb_noise_std > 0:
            Rb_obs = Rb_true + rng.normal(0.0, rb_noise_std)
            Rb_obs = max(Rb_obs, 0.0)
        else:
            Rb_obs = Rb_true

        if rb_filter_obj is not None:
            if not arm_state["filter_initialized"]:
                rb_filter_obj.reset(Rb_obs)
                arm_state["filter_initialized"] = True
            Rb_obs = rb_filter_obj.update(Rb_obs, dt)

        arm_state["R_b_obs"] = Rb_obs

        # 触发
        if (not arm_state["triggered"]
                and strategy.decide_trigger(base, arm_state, params)):
            tau_requested = strategy.arm_time_scale(base, arm_state, params)
            # ★ 实际轨迹时长 = tau × 实际预抓取时间
            t_prep = max(T_MIN_TRAJ, tau_requested * actual_prep_time)
            traj = plan_quintic_trajectory(
                Q_IDLE, Q_READY, t_prep,
                dt=dt, check_limits=True)
            if not traj["feasible"]:
                raise RuntimeError("机械臂轨迹不可行。")
            arm_state["triggered"] = True
            arm_state["traj"] = traj
            arm_state["t_start"] = t
            arm_state["R_a"] = t_prep
            arm_state["tau_requested"] = tau_requested
            arm_state["tau_applied"] = t_prep / actual_prep_time
            arm_state["q_peak"] = float(np.max(np.abs(traj["qd"])))
            arm_state["qdd_peak"] = float(np.max(np.abs(traj["qdd"])))

        # 推进机械臂
        if arm_state["triggered"]:
            elapsed = t - arm_state["t_start"]
            arm_state["R_a"] = max(0.0, arm_state["traj"]["T"] - elapsed)
            q_cur, _, _ = sample_trajectory(arm_state["traj"], elapsed)
            arm_state["q_cur"] = q_cur

        a_cmd, v_target = strategy.base_command(base, arm_state, params)

        arrived = base.is_arrived()
        arm_ready = (arm_state["triggered"] and arm_state["R_a"] <= 1e-9)

        if arrived and arm_ready:
            base.v = 0.0
            break
        if arrived:
            base.v = 0.0
            wait_time += dt
        else:
            if delay_steps > 0:
                a_cmd_buffer.append(a_cmd)
                a_cmd_used = a_cmd_buffer.pop(0)
            else:
                a_cmd_used = a_cmd
            base.step(a_cmd_used, dt, v_coord=v_target)
            if v_noise > 0.0 and not arm_state["noise_injected"]:
                bias = rng.normal(0.0, v_noise)
                base.v = float(np.clip(base.v + bias,
                                        0.0, base.p.v_max))
                arm_state["noise_injected"] = True

        if arm_state["triggered"] and base.v > 1e-3:
            overlap_time += dt

        log["t"].append(t)
        log["s"].append(base.s)
        log["v"].append(base.v)
        log["v_target"].append(v_target)
        log["R_a"].append(arm_state["R_a"])
        log["R_b"].append(Rb_obs)
        log["Delta"].append(Rb_obs - arm_state["R_a"])

        t += dt

    timed_out = t >= max_time
    return {
        "strategy": strategy.name,
        "total_time": t + T_ALIGN + T_GRIPPER + T_PLACE,
        "t_arrive": t,
        "wait_time": wait_time,
        "overlap_time": overlap_time,
        "tau_requested": arm_state["tau_requested"],
        "tau_applied": arm_state["tau_applied"],
        "q_peak": arm_state["q_peak"],
        "qdd_peak": arm_state["qdd_peak"],
        "timed_out": timed_out,
        "log": log,
    }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    from base_model import make_straight_path

    params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}

    strategies = [
        SerialStrategy(),
        FixedThresholdStrategy(),
        AdaptiveThresholdStrategy(),
        CoOptA(),
        CoOptB(),
        CoOptC(),
    ]

    goals = [("0.5 m", (1.0, 0.5)),
             ("3.0 m", (3.5, 0.5)),
             ("10.0 m", (10.5, 0.5))]

    print(f"T_NOMINAL = {T_NOMINAL:.3f} s")
    print(f"T_MIN_TRAJ = {T_MIN_TRAJ:.3f} s")
    print(f"TAU_LOWER = {TAU_LOWER:.3f}")

    for label, goal in goals:
        path = make_straight_path((0.5, 0.5), goal)
        print(f"\n=== 路径: {label} ===")
        for strategy in strategies:
            result = simulate_transport(strategy, path, params)
            print(
                f"{result['strategy']:<10} "
                f"total={result['total_time']:.3f}  "
                f"wait={result['wait_time']:.3f}  "
                f"tau={result['tau_applied']:.3f}"
            )