# robot.py
# 作用：定义机器人物理参数、底盘沿路径的一维运动、机械臂时序状态机

import numpy as np


class RobotParams:
    """机器人物理参数，全部实验保持固定（单一变量原则）"""
    v_max = 0.5          # 底盘最大线速度 m/s
    a_max = 0.4          # 底盘最大线加速度 m/s^2
    reach_thresh = 0.08  # 到达判定阈值 m
    t_prep = 2.5         # 机械臂待机 -> 预抓取姿态耗时 s
    t_grasp = 0.5        # 夹爪抓取耗时 s
    t_place = 2.5        # 机械臂抓取 -> 放置姿态耗时 s
    dt = 0.05            # 仿真时间步长 s


class BaseKinematics:
    """
    底盘运动学：把二维路径压缩成一维弧长 s，
    底盘沿 s 前进，位置通过插值得到。
    """

    def __init__(self, path, params: RobotParams):
        self.path = np.array(path)
        # 计算每个路径点的累积弧长
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.total = self.s[-1]  # 路径总长度
        self.p = params
        self.reset()

    def reset(self):
        """重置底盘状态到起点"""
        self.t = 0.0
        self.s_cur = 0.0  # 当前弧长位置
        self.v = 0.0      # 当前速度

    def pos(self, s):
        """给定弧长 s，返回对应的世界坐标 (x, y)"""
        s = np.clip(s, 0, self.total)
        x = np.interp(s, self.s, self.path[:, 0])
        y = np.interp(s, self.s, self.path[:, 1])
        return np.array([x, y])

    def dist_to_goal(self):
        """当前位置到路径终点的剩余弧长"""
        return self.total - self.s_cur

    def step(self, a_cmd):
        """
        按时间步长 dt 推进底盘状态。
        a_cmd: 期望加速度，会被 a_max 截断。
        返回：当前位置、当前速度。
        """
        a = np.clip(a_cmd, -self.p.a_max, self.p.a_max)  # 限制加速度
        # 速度积分，并限制在 [0, v_max]
        self.v = np.clip(self.v + a * self.p.dt, 0, self.p.v_max)
        # 弧长积分，不超过总长
        self.s_cur = min(self.s_cur + self.v * self.p.dt, self.total)
        self.t += self.p.dt
        return self.pos(self.s_cur), self.v

    def time_to_goal(self):
        """
        剩余时间预测：假设从当前速度出发，
        先加速到 v_max，再匀速，最后减速到 0。
        这是联合优化里 T_arrive 的核心估计。
        """
        d = self.dist_to_goal()
        v0 = self.v
        vmax = self.p.v_max
        amax = self.p.a_max

        # 从 v0 加速到 vmax 所需距离
        d_acc = max(0, (vmax**2 - v0**2) / (2 * amax))
        # 从 vmax 减速到 0 所需距离
        d_dec = vmax**2 / (2 * amax)

        if d <= d_acc + d_dec:
            # 剩余距离太短，达不到 vmax，走三角形速度曲线
            # 由 v_peak^2 = v0^2 + 2*amax*d_acc 且 v_peak^2 = 2*amax*d_dec
            # 两段距离之和 = d，解得 v_peak
            v_peak = np.sqrt(max(0, (2 * amax * d + v0**2) / 2))
            t = (v_peak - v0) / amax + v_peak / amax
            return max(t, 0.0)
        else:
            # 能达到 vmax，走梯形速度曲线
            t_acc = (vmax - v0) / amax          # 加速段
            d_cruise = d - d_acc - d_dec        # 匀速段距离
            t_cruise = d_cruise / vmax          # 匀速段
            t_dec = vmax / amax                 # 减速段
            return t_acc + t_cruise + t_dec


class ArmModel:
    """
    机械臂简化模型（支持时间缩放）
    状态机：idle -> prepping -> ready -> grasping -> placing -> done
    time_scale < 1 表示加速，= 1 表示标准速度
    """

    def __init__(self, params: RobotParams):
        self.p = params
        self.state = "idle"
        self.t_in_state = 0.0
        self.time_scale = 1.0

    def reset(self):
        self.state = "idle"
        self.t_in_state = 0.0
        self.time_scale = 1.0

    def start_prep(self, time_scale=1.0):
        """触发预抓取，time_scale 越小机械臂越快"""
        if self.state == "idle":
            self.state = "prepping"
            self.t_in_state = 0.0
            self.time_scale = float(np.clip(time_scale, 0.1, 1.0))

    def step(self):
        """按时间步长 dt 推进状态机"""
        self.t_in_state += self.p.dt
        eff_prep = self.p.t_prep * self.time_scale   # 有效预抓取时间
        if self.state == "prepping" and self.t_in_state >= eff_prep:
            self.state = "ready"
            self.t_in_state = 0.0
        elif self.state == "grasping" and self.t_in_state >= self.p.t_grasp:
            self.state = "placing"
            self.t_in_state = 0.0
        elif self.state == "placing" and self.t_in_state >= self.p.t_place:
            self.state = "done"
            self.t_in_state = 0.0

    def prep_remain(self):
        """剩余预抓取时间"""
        if self.state == "idle":
            return self.p.t_prep
        if self.state == "prepping":
            eff_prep = self.p.t_prep * self.time_scale
            return max(0.0, eff_prep - self.t_in_state)
        return 0.0

    def is_ready(self):
        return self.state in ("ready", "grasping", "placing", "done")