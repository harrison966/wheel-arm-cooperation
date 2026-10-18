import numpy as np


# ============================================================
# 底盘参数
# ============================================================

class BaseParams:
    """差速底盘参数"""
    def __init__(self):
        # 几何
        self.wheel_base = 0.4      # 轮距 L [m]
        self.wheel_radius = 0.05   # 轮半径 [m]

        # 速度 / 加速度限制
        self.v_max = 0.5           # 最大线速度 [m/s]
        self.a_max = 0.4           # 最大线加速度 [m/s^2]
        self.a_lat_max = 0.5       # 最大侧向加速度 [m/s^2]
        self.v_wheel_max = 10.0    # 单轮最大线速度 [m/s]

        # 角速度限制
        self.omega_max = 1.0       # 最大角速度 [rad/s]

        # 到达判据
        self.reach_thresh = 0.08   # 距离阈值 [m]
        self.vel_thresh = 0.01     # 速度阈值 [m/s]


# ============================================================
# 路径
# ============================================================

class Path:
    """按弧长参数化的路径"""

    def __init__(self, waypoints, ds=0.01):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self._resample(ds)
        self._compute_tangents_and_curvatures()

    def _resample(self, ds):
        p = self.waypoints
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        total = s[-1]
        n = max(int(total / ds), 2)
        s_new = np.linspace(0, total, n)
        x = np.interp(s_new, s, p[:, 0])
        y = np.interp(s_new, s, p[:, 1])
        self.xy = np.stack([x, y], axis=1)
        self.s = s_new
        self.total_length = total

    def _compute_tangents_and_curvatures(self):
        xy = self.xy
        dx = np.gradient(xy[:, 0], self.s)
        dy = np.gradient(xy[:, 1], self.s)
        ddx = np.gradient(dx, self.s)
        ddy = np.gradient(dy, self.s)

        self.tangent = np.stack([dx, dy], axis=1)
        norms = np.linalg.norm(self.tangent, axis=1, keepdims=True)
        self.tangent = self.tangent / np.maximum(norms, 1e-9)

        num = dx * ddy - dy * ddx
        den = (dx**2 + dy**2)**1.5
        self.curvature = num / np.maximum(den, 1e-9)

    def pose_at(self, s):
        s = np.clip(s, 0, self.total_length)
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        tx = np.interp(s, self.s, self.tangent[:, 0])
        ty = np.interp(s, self.s, self.tangent[:, 1])
        theta = np.arctan2(ty, tx)
        return np.array([x, y]), theta

    def curvature_at(self, s):
        s = np.clip(s, 0, self.total_length)
        return float(np.interp(s, self.s, self.curvature))


# ============================================================
# 差速底盘
# ============================================================

class DiffDriveBase:
    def __init__(self, path, params=None):
        self.path = path
        self.p = params if params is not None else BaseParams()
        self.reset()

    def reset(self):
        self.t = 0.0
        self.s = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.theta = 0.0

    def position(self):
        p, _ = self.path.pose_at(self.s)
        return p

    def dist_to_goal(self):
        return self.path.total_length - self.s

    def current_curvature(self):
        return self.path.curvature_at(self.s)

    def v_curve_limit(self):
        """曲率限速：v_curve = sqrt(a_lat_max / |κ|)"""
        kappa = abs(self.current_curvature())
        if kappa < 1e-6:
            return self.p.v_max
        return min(self.p.v_max, np.sqrt(self.p.a_lat_max / kappa))

    def v_ref(self, v_coord=None):
        """综合速度参考"""
        v = self.p.v_max
        v = min(v, self.v_curve_limit())
        if v_coord is not None:
            v = min(v, v_coord)
        return v

    def step(self, a_cmd, dt, v_coord=None):
        """按加速度指令步进 dt 秒"""
        a = np.clip(a_cmd, -self.p.a_max, self.p.a_max)
        v_lim = self.v_ref(v_coord)

        if self.v > v_lim:
            a = min(a, -0.5 * self.p.a_max)

        self.v = np.clip(self.v + a * dt, 0, v_lim)
        self.s = min(self.s + self.v * dt, self.path.total_length)

        _, theta = self.path.pose_at(self.s)
        self.theta = theta

        kappa = self.current_curvature()
        self.omega = self.v * kappa

        # 轮速检查
        v_L = self.v - self.omega * self.p.wheel_base / 2
        v_R = self.v + self.omega * self.p.wheel_base / 2
        if max(abs(v_L), abs(v_R)) > self.p.v_wheel_max:
            scale = self.p.v_wheel_max / max(abs(v_L), abs(v_R))
            self.v *= scale

        self.t += dt
        return self.position(), self.theta, self.v

    def can_stop_in(self, d_rem):
        d_brake = self.v**2 / (2 * self.p.a_max)
        return d_rem >= d_brake

    def time_to_goal(self, v_coord=None):
        """
        ★ 沿剩余路径采样，估计剩余到达时间。
        修复了之前"只看当前曲率，v_curve < vmax 永不成立"的问题。
        """
        d = self.dist_to_goal()
        if d <= self.p.reach_thresh:
            return 0.0

        # 沿剩余路径采样 20 个点
        n_samples = 20
        s_samples = np.linspace(self.s, self.path.total_length, n_samples)
        v_limits = []
        for s in s_samples:
            kappa = abs(self.path.curvature_at(s))
            if kappa < 1e-6:
                v_lim = self.p.v_max
            else:
                v_lim = min(self.p.v_max,
                            np.sqrt(self.p.a_lat_max / kappa))
            v_limits.append(v_lim)
        v_limits = np.array(v_limits)

        # 用平均限速作为目标速度（保守估计）
        v_target = float(np.mean(v_limits))
        if v_coord is not None:
            v_target = min(v_target, v_coord)

        # 梯形速度曲线
        v0 = self.v
        amax = self.p.a_max

        d_acc = max(0.0, (v_target**2 - v0**2) / (2 * amax))
        d_dec = v_target**2 / (2 * amax)

        if d <= d_acc + d_dec:
            # 三角速度曲线
            v_peak = np.sqrt(max(0.0, (2 * amax * d + v0**2) / 2.0))
            v_peak = min(v_peak, v_target)
            t = (v_peak - v0) / amax + v_peak / amax
        else:
            # 梯形速度曲线
            t_acc = (v_target - v0) / amax
            d_cruise = d - d_acc - d_dec
            t_cruise = d_cruise / v_target
            t_dec = v_target / amax
            t = t_acc + t_cruise + t_dec

        return max(0.0, t)

    def is_arrived(self):
        return (self.dist_to_goal() <= self.p.reach_thresh
                and self.v <= self.p.vel_thresh)


# ============================================================
# 路径生成工具
# ============================================================

def _smooth_corner_path(points, arc_radius=0.3, n_points=50):
    """
    通用多段折线转角平滑（二次贝塞尔）。
    关键：两段半径取小，避免转角重叠。
    """
    points = [np.asarray(p, dtype=float) for p in points]
    waypoints = [points[0]]

    for i in range(1, len(points) - 1):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]

        d1 = p_curr - p_prev
        d1 = d1 / np.linalg.norm(d1)
        d2 = p_next - p_curr
        d2 = d2 / np.linalg.norm(d2)

        L1 = np.linalg.norm(p_curr - p_prev)
        L2 = np.linalg.norm(p_next - p_curr)

        r1 = min(arc_radius, 0.5 * L1)
        r2 = min(arc_radius, 0.5 * L2)
        r = min(r1, r2)

        p_arc_start = p_curr - d1 * r
        p_arc_end = p_curr + d2 * r

        bez_t = np.linspace(0, 1, n_points)
        arc = ((1 - bez_t)[:, None]**2 * p_arc_start[None, :]
               + 2 * (1 - bez_t)[:, None] * bez_t[:, None] * p_curr[None, :]
               + bez_t[:, None]**2 * p_arc_end[None, :])

        waypoints.append(p_arc_start)
        waypoints.extend(arc[1:])

    waypoints.append(points[-1])
    return np.array(waypoints)


def make_straight_path(start, goal, n_points=100):
    """直线路径"""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    t = np.linspace(0, 1, n_points)
    waypoints = start[None, :] + (goal - start)[None, :] * t[:, None]
    return Path(waypoints)


def make_L_path(start, corner, goal, n_points=50, arc_radius=0.3):
    """L 形路径（转角平滑）"""
    start = np.asarray(start, dtype=float)
    corner = np.asarray(corner, dtype=float)
    goal = np.asarray(goal, dtype=float)
    waypoints = _smooth_corner_path([start, corner, goal],
                                     arc_radius=arc_radius,
                                     n_points=n_points)
    return Path(waypoints)


def make_S_path(start, mid1, mid2, goal, n_points=50, arc_radius=0.4):
    """S 形路径（两个转角）"""
    waypoints = _smooth_corner_path(
        [start, mid1, mid2, goal],
        arc_radius=arc_radius,
        n_points=n_points
    )
    return Path(waypoints)


def make_arc_path(center, radius, angle_start, angle_end, n_points=200):
    """圆弧路径（高曲率场景）"""
    angles = np.linspace(angle_start, angle_end, n_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    waypoints = np.stack([x, y], axis=1)
    return Path(waypoints)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    dt = 0.05

    print("=" * 60)
    print("测试 1：直线路径")
    print("=" * 60)
    path1 = make_straight_path((0.5, 0.5), (5.0, 0.5))
    print(f"路径总长: {path1.total_length:.3f} m")
    base1 = DiffDriveBase(path1)
    print(f"剩余时间（初态）: {base1.time_to_goal():.3f} s")
    for _ in range(20):
        base1.step(a_cmd=0.4, dt=dt)
    print(f"20 步后：v = {base1.v:.3f} m/s, s = {base1.s:.3f} m")
    print(f"剩余时间: {base1.time_to_goal():.3f} s")

    print("\n" + "=" * 60)
    print("测试 2：L 形路径")
    print("=" * 60)
    path2 = make_L_path((0.5, 0.5), (3.0, 0.5), (3.0, 3.0),
                        arc_radius=0.3)
    print(f"路径总长: {path2.total_length:.3f} m")
    curv2 = np.abs(path2.curvature)
    idx_max = int(np.argmax(curv2))
    print(f"最大曲率在 s = {path2.s[idx_max]:.3f} m，"
          f"曲率 = {path2.curvature[idx_max]:.3f}")
    v_lim2 = np.sqrt(0.5 / max(curv2[idx_max], 1e-6))
    print(f"最大曲率处限速 = {v_lim2:.3f} m/s")

    print("\n" + "=" * 60)
    print("测试 3：制动可行性")
    print("=" * 60)
    base3 = DiffDriveBase(make_straight_path((0, 0), (10, 0)))
    base3.v = 0.5
    print(f"v = {base3.v} m/s")
    print(f"制动距离 = {base3.v**2 / (2 * 0.4):.4f} m")
    print(f"能否在 1.0 m 内停下: {base3.can_stop_in(1.0)}")
    print(f"能否在 0.2 m 内停下: {base3.can_stop_in(0.2)}")

    print("\n" + "=" * 60)
    print("测试 4：圆弧路径")
    print("=" * 60)
    path4 = make_arc_path(center=(2.0, 2.0), radius=0.5,
                          angle_start=0, angle_end=np.pi)
    print(f"圆弧路径总长: {path4.total_length:.3f} m")
    mid_s = path4.total_length / 2
    print(f"实测曲率（中点）: {path4.curvature_at(mid_s):.3f}")
    print(f"理论曲率 = 1/0.5 = 2.000")

    print("\n" + "=" * 60)
    print("测试 5：S 形路径")
    print("=" * 60)
    path5 = make_S_path((0.5, 0.5), (2.0, 0.5), (2.0, 2.0), (3.5, 2.0),
                        arc_radius=0.4)
    print(f"S 形路径总长: {path5.total_length:.3f} m")
    curv5 = np.abs(path5.curvature)
    idx_max5 = int(np.argmax(curv5))
    print(f"最大曲率在 s = {path5.s[idx_max5]:.3f} m，"
          f"曲率 = {path5.curvature[idx_max5]:.3f}")
    v_lim5 = np.sqrt(0.5 / max(curv5[idx_max5], 1e-6))
    print(f"最大曲率处限速 = {v_lim5:.3f} m/s")