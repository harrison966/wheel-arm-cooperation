import numpy as np
from config import ARM_DH, Q_MIN, Q_MAX, Q_IDLE


# ============================================================
# 基础工具
# ============================================================

def dh_transform(a, alpha, d, theta):
    """标准 DH 变换矩阵（单关节）"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1],
    ])


def forward_kinematics(q, dh_params=None):
    """正运动学：给定关节角 q（6,），返回末端位姿 T（4x4）"""
    if dh_params is None:
        dh_params = ARM_DH
    T = np.eye(4)
    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = q[i] + theta_offset
        T = T @ dh_transform(a, alpha, d, theta)
    return T


# ============================================================
# SE(3) 对数映射
# ============================================================

def so3_log(R):
    """SO(3) 对数映射，返回旋转向量 (3,)"""
    cos_theta = (np.trace(R) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-8:
        return np.zeros(3)

    if abs(theta - np.pi) < 1e-6:
        A = (R + np.eye(3)) / 2.0
        diag = np.clip(np.diag(A), 0, None)
        axis = np.sqrt(diag)
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]
        return theta * axis

    w_hat = (R - R.T) / (2 * np.sin(theta))
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    return theta * w


def se3_log(T):
    """SE(3) 对数映射，返回 6 维向量 [v; w]"""
    R = T[:3, :3]
    p = T[:3, 3]

    w = so3_log(R)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        v = p
    else:
        w_hat = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0],
        ])
        half_theta = theta / 2.0
        cot_half = np.cos(half_theta) / np.sin(half_theta)
        coef = (1.0 / theta**2) * (1.0 - half_theta * cot_half)
        V_inv = np.eye(3) - 0.5 * w_hat + coef * (w_hat @ w_hat)
        v = V_inv @ p

    return np.concatenate([v, w])


# ============================================================
# 数值雅可比（末端系）
# ============================================================

def jacobian(q, dh_params=None, eps=1e-6):
    """数值雅可比（6x6），末端系表示"""
    if dh_params is None:
        dh_params = ARM_DH

    T0 = forward_kinematics(q, dh_params)
    T0_inv = np.linalg.inv(T0)

    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        q_plus = q.copy()
        q_plus[i] += eps
        T_plus = forward_kinematics(q_plus, dh_params)
        dT = T0_inv @ T_plus
        J[:, i] = se3_log(dT) / eps
    return J


# ============================================================
# 单次阻尼最小二乘逆解
# ============================================================

def inverse_kinematics(T_target, q_init, dh_params=None,
                       q_min=None, q_max=None,
                       max_iter=800, tol=1e-4,
                       lam=0.01, adaptive_lam=True,
                       dq_max=0.2,
                       stall_patience=30,
                       verbose=False):
    """阻尼最小二乘逆解，返回 (q_solution, success)"""
    if dh_params is None:
        dh_params = ARM_DH
    if q_min is None:
        q_min = Q_MIN
    if q_max is None:
        q_max = Q_MAX

    q = q_init.copy().astype(float)
    eps_q = 1e-9

    lam_cur = lam
    prev_err = np.inf
    stall_count = 0

    for it in range(max_iter):
        T_cur = forward_kinematics(q, dh_params)
        err = se3_log(np.linalg.inv(T_cur) @ T_target)
        err_norm = np.linalg.norm(err)

        if verbose and (it % 10 == 0 or it < 5):
            print(f"[IK] iter={it:3d}  err={err_norm:.6e}  lam={lam_cur:.4e}  "
                  f"q={np.round(q, 3)}")

        # 收敛
        if err_norm < tol:
            if np.all(q >= q_min - eps_q) and np.all(q <= q_max + eps_q):
                q = np.clip(q, q_min, q_max)
                if verbose:
                    print(f"[IK] 收敛于 iter={it}，err={err_norm:.6e}")
                return q, True
            else:
                if verbose:
                    print("[IK] 收敛但超出限位")
                return q, False

        # 极限环检测
        if err_norm > prev_err * 0.999:
            stall_count += 1
        else:
            stall_count = 0
        prev_err = err_norm

        if stall_count >= stall_patience:
            q = q + 0.15 * np.random.randn(len(q))
            q = np.clip(q, q_min, q_max)
            lam_cur = 0.05
            stall_count = 0
            prev_err = np.inf
            if verbose:
                print(f"[IK] iter={it:3d}  检测到停滞，注入扰动")

        # 迭代一步
        J = jacobian(q, dh_params)

        if adaptive_lam:
            if err_norm > 0.1:
                lam_cur = 1e-3
            elif err_norm > 0.01:
                lam_cur = 1e-2
            else:
                lam_cur = 1e-1

        dq = J.T @ np.linalg.solve(J @ J.T + lam_cur**2 * np.eye(6), err)

        dq_norm = np.linalg.norm(dq)
        if dq_norm > dq_max:
            dq = dq * (dq_max / dq_norm)

        q = np.clip(q + dq, q_min, q_max)

    if verbose:
        T_cur = forward_kinematics(q, dh_params)
        err = se3_log(np.linalg.inv(T_cur) @ T_target)
        print(f"[IK] 达到最大迭代次数，最终 err={np.linalg.norm(err):.6e}")

    return q, False


# ============================================================
# 4 个关键点种子（覆盖典型构型）
# ============================================================

def _get_key_seeds():
    """
    生成 4 个典型初值：
        - 肘上 / 肘下
        - 腕翻转 / 腕不翻转
    对 6-DOF 串联臂，这 4 个点覆盖主要吸引域。
    """
    seeds = []
    for q2 in (-np.pi/4, -3*np.pi/4):
        for q4 in (-np.pi/4, -3*np.pi/4):
            seeds.append(np.array([
                0.0,
                q2,
                np.pi/2,
                q4,
                -np.pi/2,
                0.0,
            ]))
    return seeds


# ============================================================
# 鲁棒逆解：4 关键点 + 随机重启 + 就近选择
# ============================================================

def inverse_kinematics_robust(T_target, q_init=None,
                              n_restart=10,
                              dh_params=None,
                              q_min=None, q_max=None,
                              seed=None,
                              verbose=False,
                              **kwargs):
    """
    鲁棒逆解：
    1) 从 q_init 出发做一次 IK
    2) 从 4 个关键点种子各做一次 IK
    3) 再做 n_restart 次随机重启
    4) 从所有可行解中，选离 q_init 最近的一个

    返回: (q_solution, success)
    """
    if dh_params is None:
        dh_params = ARM_DH
    if q_min is None:
        q_min = Q_MIN
    if q_max is None:
        q_max = Q_MAX

    rng = np.random.default_rng(seed)
    candidates = []

    def try_solve(q0, tag=""):
        q_sol, ok = inverse_kinematics(T_target, q0,
                                       dh_params=dh_params,
                                       q_min=q_min, q_max=q_max,
                                       verbose=False, **kwargs)
        if ok:
            candidates.append(q_sol)
            if verbose:
                print(f"[IK-robust] {tag} 成功，"
                      f"距离(q_init) = "
                      f"{np.linalg.norm(q_sol - q_init) if q_init is not None else -1:.4f}")
        return ok

    # ---------- 第 1 轮：q_init ----------
    if q_init is not None:
        try_solve(q_init, "q_init")

    # ---------- 第 2 轮：4 个关键点 ----------
    for i, q0 in enumerate(_get_key_seeds()):
        try_solve(q0, f"key-seed-{i+1}")

    # ---------- 第 3 轮：随机重启 ----------
    for k in range(n_restart):
        q0 = rng.uniform(q_min, q_max)
        try_solve(q0, f"restart-{k+1}")

    if len(candidates) == 0:
        return None, False

    # ---------- 就近选择 ----------
    if q_init is not None:
        dists = [np.linalg.norm(c - q_init) for c in candidates]
        best = candidates[int(np.argmin(dists))]
        if verbose:
            print(f"[IK-robust] 共 {len(candidates)} 个解，"
                  f"最短距离 = {min(dists):.4f}")
    else:
        best = candidates[0]

    return best, True


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    print("=" * 60)
    print("正运动学测试")
    print("=" * 60)
    q_test = np.array([0.3, -0.5, 0.8, -0.6, -1.2, 0.4])
    T = forward_kinematics(q_test)
    print("末端位置:", np.round(T[:3, 3], 4))
    print("旋转矩阵:")
    print(np.round(T[:3, :3], 4))

    print("\n" + "=" * 60)
    print("逆解测试 1：真值 + 噪声初值")
    print("=" * 60)
    q_true = np.array([0.5, -0.8, 1.0, -0.7, -1.5, 0.3])
    T_target = forward_kinematics(q_true)
    q_init = q_true + 0.1 * np.random.randn(6)
    q_sol, ok = inverse_kinematics(T_target, q_init)
    print("成功:", ok)
    print("目标关节:", np.round(q_true, 4))
    print("求解关节:", np.round(q_sol, 4))
    if ok:
        print("关节误差:", np.linalg.norm(q_sol - q_true))
        T_sol = forward_kinematics(q_sol)
        print("位姿误差:", np.linalg.norm(
            se3_log(np.linalg.inv(T_sol) @ T_target)))

    print("\n" + "=" * 60)
    print("逆解测试 2：从 Q_IDLE 出发 + 4 关键点 + 就近选择")
    print("=" * 60)
    # 目标设成"真实构型" q_true，看能不能找到"离 Q_IDLE 最近"的等价解
    q_sol2, ok2 = inverse_kinematics_robust(
        T_target, q_init=Q_IDLE,
        n_restart=10, seed=0, verbose=True)
    print("成功:", ok2)
    if ok2:
        print("求解关节:", np.round(q_sol2, 4))
        print("离 Q_IDLE 距离:", np.linalg.norm(q_sol2 - Q_IDLE))
        print("离 q_true 距离:", np.linalg.norm(q_sol2 - q_true))
        T_sol2 = forward_kinematics(q_sol2)
        print("位姿误差:", np.linalg.norm(
            se3_log(np.linalg.inv(T_sol2) @ T_target)))

    print("\n" + "=" * 60)
    print("雅可比测试")
    print("=" * 60)
    J = jacobian(q_test)
    print("雅可比形状:", J.shape)
    print("雅可比条件数:", np.linalg.cond(J))