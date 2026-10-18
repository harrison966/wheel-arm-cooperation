import numpy as np

from config import ARM_DH, Q_IDLE, Q_READY, T_NOMINAL
from arm_trajectory import min_feasible_time, plan_quintic_trajectory


# ============================================================
# 近似机械臂物理参数
# ============================================================

# 各连杆质量，单位：kg
LINK_MASS = np.array([
    3.5, 8.0, 4.0, 2.5, 1.5, 0.8,
], dtype=float)

# 各关节连续力矩上限，单位：N m
TAU_MAX = np.array([
    120.0, 120.0, 80.0, 35.0, 35.0, 20.0,
], dtype=float)

# 关节粘性阻尼系数
VISCOUS_DAMPING = np.array([
    0.60, 0.80, 0.50, 0.25, 0.15, 0.08,
], dtype=float)

GRAVITY = np.array([0.0, 0.0, -9.81])


# ============================================================
# DH 正运动学
# ============================================================

def dh_transform(a, alpha, d, theta):
    """标准 DH 齐次变换矩阵。"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def forward_kinematics_chain(q, dh_params=ARM_DH):
    """
    返回：
    origins[i]：第 i 个关节原点在基坐标系的位置
    axes[i]：第 i 个关节 z 轴在基坐标系的方向
    frames[i]：第 i 个 DH 坐标系在基坐标系中的变换
    """
    transform = np.eye(4)

    origins = [transform[:3, 3].copy()]
    axes = [transform[:3, 2].copy()]
    frames = []

    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = q[i] + theta_offset

        transform = transform @ dh_transform(
            a,
            alpha,
            d,
            theta,
        )

        frames.append(transform.copy())
        origins.append(transform[:3, 3].copy())
        axes.append(transform[:3, 2].copy())

    return origins, axes, frames


# ============================================================
# 连杆惯量与雅可比
# ============================================================

def link_inertia(mass, length):
    """
    将每段连杆近似为细杆，返回其质心处惯量矩阵。
    """
    length = max(float(length), 0.05)

    i_perpendicular = mass * length ** 2 / 12.0
    i_axis = 0.02 * i_perpendicular

    return np.diag([
        i_axis,
        i_perpendicular,
        i_perpendicular,
    ])


def link_jacobians(link_index, origins, axes):
    """
    计算第 link_index 段连杆质心的线速度和角速度雅可比。
    """
    n_joint = len(axes) - 1

    # 将质心近似放在相邻关节原点的中点
    p_com = 0.5 * (
        origins[link_index] + origins[link_index + 1]
    )

    j_linear = np.zeros((3, n_joint))
    j_angular = np.zeros((3, n_joint))

    for joint_index in range(link_index + 1):
        axis = axes[joint_index]
        origin = origins[joint_index]

        j_linear[:, joint_index] = np.cross(
            axis,
            p_com - origin,
        )
        j_angular[:, joint_index] = axis

    return p_com, j_linear, j_angular


# ============================================================
# 动力学项
# ============================================================

def mass_matrix(q, dh_params=ARM_DH):
    """
    根据连杆雅可比构建关节空间质量矩阵 M(q)。
    """
    n_joint = len(q)

    origins, axes, frames = forward_kinematics_chain(
        q,
        dh_params,
    )

    matrix = np.zeros((n_joint, n_joint))

    for link_index in range(n_joint):
        a, _, d, _ = dh_params[link_index]
        mass = LINK_MASS[link_index]

        _, j_linear, j_angular = link_jacobians(
            link_index,
            origins,
            axes,
        )

        length = np.hypot(a, d)
        inertia_local = link_inertia(mass, length)

        rotation = frames[link_index][:3, :3]
        inertia_world = (
            rotation
            @ inertia_local
            @ rotation.T
        )

        matrix += (
            mass * (j_linear.T @ j_linear)
            + j_angular.T @ inertia_world @ j_angular
        )

    # 去除数值误差造成的非对称性
    return 0.5 * (matrix + matrix.T)


def potential_energy(q, dh_params=ARM_DH):
    """计算系统重力势能。"""
    origins, _, _ = forward_kinematics_chain(
        q,
        dh_params,
    )

    energy = 0.0

    for link_index, mass in enumerate(LINK_MASS):
        p_com = 0.5 * (
            origins[link_index]
            + origins[link_index + 1]
        )

        energy += -mass * GRAVITY.dot(p_com)

    return energy


def gravity_torque(q, eps=1e-6):
    """
    使用势能数值梯度计算重力广义力。
    """
    n_joint = len(q)
    torque = np.zeros(n_joint)

    for i in range(n_joint):
        dq = np.zeros(n_joint)
        dq[i] = eps

        torque[i] = (
            potential_energy(q + dq)
            - potential_energy(q - dq)
        ) / (2.0 * eps)

    return torque


def coriolis_torque(q, qd, eps=1e-5):
    """
    使用质量矩阵导数和 Christoffel 符号计算科氏和离心力项。
    """
    n_joint = len(q)

    d_mass = np.zeros((n_joint, n_joint, n_joint))

    for k in range(n_joint):
        dq = np.zeros(n_joint)
        dq[k] = eps

        d_mass[:, :, k] = (
            mass_matrix(q + dq)
            - mass_matrix(q - dq)
        ) / (2.0 * eps)

    torque = np.zeros(n_joint)

    for i in range(n_joint):
        for j in range(n_joint):
            for k in range(n_joint):
                gamma = 0.5 * (
                    d_mass[i, j, k]
                    + d_mass[i, k, j]
                    - d_mass[j, k, i]
                )

                torque[i] += gamma * qd[j] * qd[k]

    return torque


def inverse_dynamics(q, qd, qdd):
    """
    tau = M(q) qdd + C(q, qd) qd + G(q) + B qd
    """
    return (
        mass_matrix(q) @ qdd
        + coriolis_torque(q, qd)
        + gravity_torque(q)
        + VISCOUS_DAMPING * qd
    )


# ============================================================
# 轨迹力矩计算与时间缩放扫描
# ============================================================

def compute_torque_profile(traj):
    """
    输入 plan_quintic_trajectory 的输出。
    返回 shape=(6, N) 的关节力矩。
    """
    n_joint, n_step = traj["q"].shape
    torques = np.zeros((n_joint, n_step))

    for k in range(n_step):
        torques[:, k] = inverse_dynamics(
            traj["q"][:, k],
            traj["qd"][:, k],
            traj["qdd"][:, k],
        )

    return torques


def scan_tau_vs_torque(tau_values=None, dt=0.02):
    """
    扫描 tau，检查运动学和近似动力学力矩约束。

    返回：
    {
        tau,
        T_traj,
        peak_torque,
        feasible
    }
    """
    if tau_values is None:
        tau_values = np.linspace(0.30, 1.00, 15)

    t_min, _, _, _ = min_feasible_time(
        Q_IDLE,
        Q_READY,
    )

    results = []

    for tau in tau_values:
        t_traj = tau * T_NOMINAL

        if t_traj < t_min - 1e-9:
            continue

        traj = plan_quintic_trajectory(
            Q_IDLE,
            Q_READY,
            t_traj,
            dt=dt,
            check_limits=True,
        )

        if not traj["feasible"]:
            continue

        torques = compute_torque_profile(traj)

        peak_torque = np.max(
            np.abs(torques),
            axis=1,
        )

        feasible = bool(
            np.all(peak_torque <= TAU_MAX)
        )

        results.append({
            "tau": float(tau),
            "T_traj": float(t_traj),
            "peak_torque": peak_torque,
            "feasible": feasible,
        })

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("纯 NumPy 近似逆动力学时间缩放验证")
    print("=" * 72)

    t_min, _, _, _ = min_feasible_time(
        Q_IDLE,
        Q_READY,
    )

    print(f"名义预抓取时间: {T_NOMINAL:.3f} s")
    print(f"运动学最短可行时间: {t_min:.3f} s")
    print(f"关节力矩上限: {np.round(TAU_MAX, 1)} Nm")

    results = scan_tau_vs_torque()

    print(
        f"\n{'tau':>7} {'T_traj':>9} "
        f"{'peak torque':>14} {'feasible':>10}"
    )
    print("-" * 48)

    for item in results:
        print(
            f"{item['tau']:7.3f} "
            f"{item['T_traj']:9.3f} "
            f"{item['peak_torque'].max():14.3f} "
            f"{str(item['feasible']):>10}"
        )

    feasible_items = [
        item for item in results
        if item["feasible"]
    ]

    if feasible_items:
        best = min(
            feasible_items,
            key=lambda item: item["tau"],
        )

        print(
            "\n近似动力学下最小可行时间缩放："
            f"tau={best['tau']:.3f}, "
            f"T={best['T_traj']:.3f} s"
        )
    else:
        print("\n当前力矩上限下没有可行轨迹。")