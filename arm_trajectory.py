import numpy as np
from config import QD_MAX, QDD_MAX, POLY_CV, POLY_CA, DT


# ============================================================
# 五次多项式时间律
# ============================================================

def quintic_s(u):
    """
    归一化五次多项式位置律。
    u ∈ [0, 1] → s ∈ [0, 1]
    满足 s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0
    """
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def quintic_sd(u):
    """ds/du = 30u² - 60u³ + 30u⁴"""
    return 30 * u**2 - 60 * u**3 + 30 * u**4


def quintic_sdd(u):
    """d²s/du² = 60u - 180u² + 120u³"""
    return 60 * u - 180 * u**2 + 120 * u**3


# ============================================================
# 最短可行时间
# ============================================================

def min_feasible_time(q_start, q_goal,
                      qd_max=None, qdd_max=None,
                      cv=None, ca=None):
    """
    计算从 q_start 到 q_goal 的最短可行时间。

    五次多项式时间律的峰值系数：
        max|ds/du|   = 1.875
        max|d²s/du²| = 5.77
    对每个关节 i：
        T_v_i = |Δq_i| * cv / qd_max_i
        T_a_i = sqrt(|Δq_i| * ca / qdd_max_i)
        T_i   = max(T_v_i, T_a_i)
    总最短时间 = max_i T_i
    """
    if qd_max is None:
        qd_max = QD_MAX
    if qdd_max is None:
        qdd_max = QDD_MAX
    if cv is None:
        cv = POLY_CV
    if ca is None:
        ca = POLY_CA

    dq = np.abs(q_goal - q_start)

    # 速度约束下的最短时间
    T_v = dq * cv / qd_max
    # 加速度约束下的最短时间
    T_a = np.sqrt(dq * ca / qdd_max)
    # 每个关节的最短时间
    T_per_joint = np.maximum(T_v, T_a)
    # 整条轨迹的最短时间（取最大）
    T_min = np.max(T_per_joint)

    return T_min, T_v, T_a, T_per_joint


# ============================================================
# 五次多项式轨迹生成
# ============================================================

def plan_quintic_trajectory(q_start, q_goal, T,
                            dt=None, qd_max=None, qdd_max=None,
                            check_limits=True, verbose=False):
    """
    生成五次多项式轨迹。
    """
    if dt is None:
        dt = DT
    if qd_max is None:
        qd_max = QD_MAX
    if qdd_max is None:
        qdd_max = QDD_MAX

    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    dq = q_goal - q_start

    # 时间序列
    n_steps = int(np.ceil(T / dt)) + 1
    t = np.linspace(0, T, n_steps)
    u = t / T

    # 位置、速度、加速度
    s = quintic_s(u)
    sd = quintic_sd(u)
    sdd = quintic_sdd(u)

    q = q_start[:, None] + dq[:, None] * s[None, :]
    qd = dq[:, None] * sd[None, :] / T
    qdd = dq[:, None] * sdd[None, :] / T**2

    # 检查约束
    peak_qd = np.max(np.abs(qd), axis=1)
    peak_qdd = np.max(np.abs(qdd), axis=1)

    ok_v = np.all(peak_qd <= qd_max + 1e-6)
    ok_a = np.all(peak_qdd <= qdd_max + 1e-6)
    feasible = ok_v and ok_a

    if check_limits and not feasible:
        if verbose:
            print("[Trajectory] 警告：轨迹超出约束")
            print("  峰值速度:", peak_qd)
            print("  速度限制:", qd_max)
            print("  峰值加速度:", peak_qdd)
            print("  加速度限制:", qdd_max)

    return {
        "t": t,
        "q": q,
        "qd": qd,
        "qdd": qdd,
        "T": T,
        "peak_qd": peak_qd,
        "peak_qdd": peak_qdd,
        "feasible": feasible,
    }


# ============================================================
# 重定时（时间缩放）
# ============================================================

def retime_trajectory(traj, tau):
    """
    把轨迹重定时到新时间 = tau * T_original。

    tau < 1：加速（需要在原约束上乘 1/tau）
    tau = 1：不变
    tau > 1：减速

    返回新的轨迹 dict，或者 None（如果不满足约束）。
    """
    T_orig = traj["T"]
    T_new = tau * T_orig

    q_start = traj["q"][:, 0]
    q_goal = traj["q"][:, -1]
    dt = traj["t"][1] - traj["t"][0]

    # 用新时间重新生成轨迹（并检查约束）
    new_traj = plan_quintic_trajectory(
        q_start, q_goal, T_new, dt=dt,
        check_limits=True, verbose=False
    )

    if not new_traj["feasible"]:
        return None
    return new_traj


# ============================================================
# 从轨迹中查询任意时刻的状态
# ============================================================

def sample_trajectory(traj, t_query):
    """
    从轨迹中采样给定时刻的 q, qd, qdd。
    使用线性插值。
    """
    t = traj["t"]
    q = traj["q"]
    qd = traj["qd"]
    qdd = traj["qdd"]

    t_query = np.clip(t_query, t[0], t[-1])

    q_s = np.array([np.interp(t_query, t, q[i, :]) for i in range(q.shape[0])])
    qd_s = np.array([np.interp(t_query, t, qd[i, :]) for i in range(qd.shape[0])])
    qdd_s = np.array([np.interp(t_query, t, qdd[i, :]) for i in range(qdd.shape[0])])
    return q_s, qd_s, qdd_s


def remaining_time(traj, t_current):
    """给定当前时刻，返回轨迹剩余时间"""
    return max(0.0, traj["T"] - t_current)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    from config import Q_IDLE, Q_READY

    print("=" * 60)
    print("最短可行时间测试")
    print("=" * 60)
    T_min, T_v, T_a, T_joint = min_feasible_time(Q_IDLE, Q_READY)
    print("各关节速度约束时间:", np.round(T_v, 4))
    print("各关节加速度约束时间:", np.round(T_a, 4))
    print("各关节最短时间:", np.round(T_joint, 4))
    print("整条轨迹最短时间 T_min:", round(T_min, 4), "s")

    print("\n" + "=" * 60)
    print("五次多项式轨迹测试（T = T_min）")
    print("=" * 60)
    traj = plan_quintic_trajectory(Q_IDLE, Q_READY, T_min,
                                  check_limits=True, verbose=True)
    print("时间步数:", len(traj["t"]))
    print("轨迹时长:", round(traj["T"], 4), "s")
    print("峰值速度:", np.round(traj["peak_qd"], 4))
    print("速度限制:", np.round(QD_MAX, 4))
    print("峰值加速度:", np.round(traj["peak_qdd"], 4))
    print("加速度限制:", np.round(QDD_MAX, 4))
    print("满足约束:", traj["feasible"])

    print("\n" + "=" * 60)
    print("重定时测试（加速到 0.7 倍）")
    print("=" * 60)
    new_traj = retime_trajectory(traj, tau=0.7)
    if new_traj is not None:
        print("成功！新时长:", round(new_traj["T"], 4), "s")
        print("峰值速度:", np.round(new_traj["peak_qd"], 4))
        print("峰值加速度:", np.round(new_traj["peak_qdd"], 4))
        print("满足约束:", new_traj["feasible"])
    else:
        print("失败：重定时后超约束")

    print("\n" + "=" * 60)
    print("轨迹采样测试")
    print("=" * 60)
    q_s, qd_s, qdd_s = sample_trajectory(traj, traj["T"] * 0.5)
    print("中点位置:", np.round(q_s, 4))
    print("中点速度:", np.round(qd_s, 4))
    print("中点加速度:", np.round(qdd_s, 4))
    print("剩余时间:", round(remaining_time(traj, traj["T"] * 0.5), 4), "s")