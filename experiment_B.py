# -*- coding: utf-8 -*-
# experiment_B.py
# 固定速度偏差扫描
# 展示 cooptA 的补偿能力随偏差的增大而增强

import numpy as np

from base_model import make_straight_path
from coordinator import (
    SerialStrategy,
    FixedThresholdStrategy,
    AdaptiveThresholdStrategy,
    CoOptA,
    CoOptB,
    CoOptC,
    simulate_transport,
)


def run_once(strategy, path, params, v_bias=0.0, v_noise=0.0, seed=0):
    return simulate_transport(
        strategy, path, params,
        v_noise=v_noise,
        v_bias=v_bias,
        seed=seed,
    )


if __name__ == "__main__":
    params = {
        "v_nominal": 0.35,
        "v_max": 0.50,
        "a_max": 0.40,
    }

    strategies = [
        ("serial",   SerialStrategy()),
        ("fixed",    FixedThresholdStrategy()),
        ("adaptive", AdaptiveThresholdStrategy()),
        ("cooptA",   CoOptA()),
        ("cooptB",   CoOptB()),
        ("cooptC",   CoOptC()),
    ]

    # 路径
    paths = [
        ("0.5 m", (1.0, 0.5)),
        ("3.0 m", (3.5, 0.5)),
        ("10.0 m", (10.5, 0.5)),
    ]

    # 扫描偏差
    v_bias_list = [0.00, 0.10, 0.20, 0.30]

    # ============================================================
    # 表格 1：每个 v_bias 下所有策略的 total（不同路径）
    # ============================================================

    for label, goal in paths:
        path = make_straight_path((0.5, 0.5), goal)

        print("\n" + "=" * 76)
        print(f"路径: {label}  (总长 {path.total_length:.2f} m)")
        print("=" * 76)

        # 表头
        header = f"{'strategy':<10}"
        for vb in v_bias_list:
            header += f"  bias={vb:.2f}"
        print(header)
        print("-" * 76)

        # 每行一个策略
        for name, st in strategies:
            row = f"{name:<10}"
            for vb in v_bias_list:
                r = run_once(st, path, params, v_bias=vb)
                row += f"  {r['total_time']:9.3f}"
            print(row)

        # 额外打印相对 fixed 的优势
        print("-" * 76)
        print("相对 fixed 的优势 (负值 = 更快)")
        print(f"{'strategy':<10}", end="")
        for vb in v_bias_list:
            print(f"  bias={vb:.2f}", end="")
        print()
        print("-" * 76)

        # 基准：fixed 的 total
        fixed_totals = []
        for vb in v_bias_list:
            r = run_once(FixedThresholdStrategy(), path, params, v_bias=vb)
            fixed_totals.append(r["total_time"])

        for name, st in strategies:
            if name == "fixed":
                continue
            row = f"{name:<10}"
            for i, vb in enumerate(v_bias_list):
                r = run_once(st, path, params, v_bias=vb)
                delta = r["total_time"] - fixed_totals[i]
                row += f"  {delta:+9.3f}"
            print(row)

    # ============================================================
    # 表格 2：cooptA 相对 fixed 的优势 vs v_bias
    # ============================================================

    print("\n\n" + "=" * 76)
    print("汇总：cooptA / cooptC 相对 fixed 的优势 vs v_bias")
    print("=" * 76)
    print(f"{'path':<8} {'v_bias':>8} "
          f"{'fixed':>10} {'cooptA':>10} {'cooptC':>10} "
          f"{'A-fixed':>10} {'C-fixed':>10}")
    print("-" * 76)

    for label, goal in paths:
        path = make_straight_path((0.5, 0.5), goal)
        for vb in v_bias_list:
            r_f = run_once(FixedThresholdStrategy(), path, params, v_bias=vb)
            r_a = run_once(CoOptA(), path, params, v_bias=vb)
            r_c = run_once(CoOptC(), path, params, v_bias=vb)

            print(
                f"{label:<8} {vb:>8.2f} "
                f"{r_f['total_time']:>10.3f} "
                f"{r_a['total_time']:>10.3f} "
                f"{r_c['total_time']:>10.3f} "
                f"{r_a['total_time'] - r_f['total_time']:>+10.3f} "
                f"{r_c['total_time'] - r_f['total_time']:>+10.3f}"
            )
        print()