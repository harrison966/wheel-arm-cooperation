# visualize.py
# 作用：画单次仿真的路线图 + 时间着色 + 关键事件标记

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection


def draw_map(ax, grid_map, xlim=(0, 10), ylim=(0, 6)):
    """在地图上画障碍物和边界"""
    # 遍历所有栅格，把障碍画成灰色矩形
    # 为了效率，用 imshow 直接显示 grid 更方便
    ax.imshow(
        grid_map.grid.T,           # 转置，因为 imshow 的 (row, col) 与 (x, y) 相反
        origin="lower",
        extent=[grid_map.xlim[0], grid_map.xlim[1],
                grid_map.ylim[0], grid_map.ylim[1]],
        cmap="Greys",
        alpha=0.5,
        zorder=0,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle=":", alpha=0.3)


def plot_trajectory(ax, result, grid_map, path, start, goal,
                    title=None, show_events=True):
    """
    画一次仿真的完整路线图。
    参数：
        result: simulate_clean 返回的 dict
        grid_map: GridMap 实例
        path: 重采样后的路径 (N, 2)
        start, goal: 起点和目标点世界坐标
        show_events: 是否标注触发/等待事件
    """
    draw_map(ax, grid_map)

    # ---------- 1) 画 A* 路径（灰色虚线，作为参考） ----------
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "--", color="gray",
            linewidth=1.2, label="A* path", zorder=1)

    # ---------- 2) 从 log 里取机器人实际位置 ----------
    log = result["log"]
    s_log = np.array(log["s"])       # 每一步的弧长
    v_log = np.array(log["v"])       # 每一步的速度
    t_log = np.array(log["t"])       # 每一步的时间

    # 把弧长转成世界坐标
    # 用 BaseKinematics 的 pos 方法需要重新构造，这里直接插值
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_path = np.concatenate([[0], np.cumsum(seg)])
    x_traj = np.interp(s_log, s_path, path[:, 0])
    y_traj = np.interp(s_log, s_path, path[:, 1])

    # ---------- 3) 用速度给轨迹着色 ----------
    # 把轨迹分成小段，每段按速度上色
    points = np.array([x_traj, y_traj]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # 速度归一化到 [0, 1]，用于 colormap
    v_norm = v_log[:-1] / max(v_log.max(), 1e-6)

    lc = LineCollection(segments, cmap="jet", linewidth=3, zorder=3)
    lc.set_array(v_norm)
    lc.set_clim(0, 1)
    ax.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("velocity (normalized)")

    # ---------- 4) 标记起点、目标点 ----------
    ax.plot(start[0], start[1], "go", markersize=10,
            label="start", zorder=5)
    ax.plot(goal[0], goal[1], "r*", markersize=15,
            label="goal", zorder=5)

    # ---------- 5) 标记触发预抓取的位置 ----------
    if show_events:
        prep_t = result.get("prep_start_t")
        if prep_t is not None:
            # 找到 log 里最接近 prep_start_t 的索引
            idx = int(np.argmin(np.abs(t_log - prep_t)))
            ax.plot(x_traj[idx], y_traj[idx], "b^", markersize=12,
                    label="prep trigger", zorder=6)

        # 标记等待发生的位置
        # 等待 = 底盘速度接近 0 但还没到目标
        wait_mask = (v_log < 1e-3) & (s_log < s_path[-1] - 0.1)
        if np.any(wait_mask):
            ax.plot(x_traj[wait_mask], y_traj[wait_mask], "rx",
                    markersize=8, label="waiting", zorder=6)

    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)


def plot_run_with_map(result, grid_map, path, start, goal, title=None):
    """一行调用版：直接画一张图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_trajectory(ax, result, grid_map, path, start, goal, title=title)
    plt.tight_layout()
    plt.show()


def plot_four_strategies(results_dict, grid_map, path, start, goal):
    """
    把四种策略的路线图并排画在一张 2x2 图里。
    results_dict: {strategy_name: result}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    names = list(results_dict.keys())
    for ax, name in zip(axes.flat, names):
        r = results_dict[name]
        plot_trajectory(ax, r, grid_map, path, start, goal,
                        title=f"{name}  (T={r['total_time']:.2f}s, "
                              f"W={r['wait_time']:.2f}s)")
    plt.tight_layout()
    plt.show()