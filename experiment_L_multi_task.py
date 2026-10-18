# -*- coding: utf-8 -*-
# experiment_L_multi_task.py
# 多任务循环搬运（含 A* 避障）——用 Path 对象包装 A* 结果

import numpy as np
from env import build_default_map
from base_model import Path                # ★ 关键：用 Path 类
from coordinator import (SerialStrategy, FixedThresholdStrategy,
                          AdaptiveThresholdStrategy,
                          CoOptA, CoOptB, CoOptC,
                          simulate_transport)


def generate_tasks(n_tasks, grid_map, seed=42):
    """独立随机生成 n 个任务"""
    rng = np.random.default_rng(seed)
    tasks = []
    attempts = 0
    max_attempts = int(n_tasks * 200)
    while len(tasks) < n_tasks and attempts < max_attempts:
        attempts += 1
        start = (rng.uniform(0.8, 9.2), rng.uniform(0.8, 5.2))
        goal = (rng.uniform(0.8, 9.2), rng.uniform(0.8, 5.2))
        gi_s, gj_s = grid_map.world_to_grid(start)
        gi_g, gj_g = grid_map.world_to_grid(goal)
        if not grid_map.is_free(gi_s, gj_s):
            continue
        if not grid_map.is_free(gi_g, gj_g):
            continue
        if np.hypot(goal[0]-start[0], goal[1]-start[1]) < 2.0:
            continue
        try:
            _ = grid_map.astar(start, goal)
            tasks.append((start, goal))
        except Exception:
            continue
    return tasks


def make_path_from_astar(grid_map, start, goal):
    """把 A* 结果转成 Path 对象"""
    waypoints = grid_map.astar(start, goal)
    return Path(np.array(waypoints))


if __name__ == "__main__":
    params = {"v_nominal": 0.35, "v_max": 0.50, "a_max": 0.40}
    grid_map = build_default_map()

    strategies = [
        ("serial",    SerialStrategy()),
        ("fixed",     FixedThresholdStrategy()),
        ("adaptive",  AdaptiveThresholdStrategy()),
        ("cooptA",    CoOptA()),
        ("cooptB",    CoOptB()),
        ("cooptC",    CoOptC()),
    ]

    # ============================================================
    # 第一步：单任务测试
    # ============================================================
    print("=" * 60)
    print("第一步：单任务测试")
    print("=" * 60)

    tasks = generate_tasks(5, grid_map, seed=42)
    print(f"生成任务数: {len(tasks)}")

    start, goal = tasks[0]
    print(f"\n[测试] task 0: ({start[0]:.2f}, {start[1]:.2f}) "
          f"-> ({goal[0]:.2f}, {goal[1]:.2f})")

    path = make_path_from_astar(grid_map, start, goal)
    print(f"  路径总长: {path.total_length:.3f} m")

    st = FixedThresholdStrategy()
    r = simulate_transport(st, path, params)
    print(f"  总时间: {r['total_time']:.3f} s")

    # ============================================================
    # 第二步：多任务
    # ============================================================
    print("\n" + "=" * 60)
    print("第二步：多任务循环搬运")
    print("=" * 60)

    for n_tasks in [100, 500, 1000]:
        print(f"\n--- 连续 {n_tasks} 个任务 ---")
        tasks = generate_tasks(n_tasks, grid_map, seed=42)
        print(f"生成任务数: {len(tasks)}")

        # 预计算所有 Path 对象
        paths = []
        for s, g in tasks:
            try:
                p = make_path_from_astar(grid_map, s, g)
                paths.append(p)
            except Exception:
                continue
        print(f"成功规划路径数: {len(paths)}")

        print(f"{'strategy':<10} {'total(s)':>12} "
              f"{'mean(s)':>12} {'vs fixed':>12}")
        print("-" * 50)

        baseline = None
        for name, st in strategies:
            t_total = 0.0
            for p in paths:
                r = simulate_transport(st, p, params)
                t_total += r["total_time"]
            if name == "fixed":
                baseline = t_total
            row = f"{name:<10} {t_total:>12.2f} {t_total/len(paths):>12.3f}"
            if baseline is not None:
                delta = baseline - t_total
                row += f" {delta:>+12.2f}"
            print(row)