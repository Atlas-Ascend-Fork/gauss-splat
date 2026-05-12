# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Gaussian Splatting Rasterization 性能基准测试脚本
支持功能:
  1. 功能验证测试（参数可配置）
  2. 端到端性能测试（warmup + 多轮计时）
  3. msprof profiling + op_summary 解析

用法:
  python perf_benchmark_rasterization.py                                     # 默认参数
  python perf_benchmark_rasterization.py --n_gs 55296 --width 244 --height 136  # 自定义参数
  python perf_benchmark_rasterization.py --perf_iters 20                      # 性能测试轮数
  python perf_benchmark_rasterization.py --skip_perf                          # 只跑功能测试
  python perf_benchmark_rasterization.py --skip_func                          # 只跑性能测试
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_npu

from gauss_splat import Rasterizer

torch.npu.set_device('npu:0')
torch.manual_seed(42)
np.set_printoptions(threshold=np.inf)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Gaussian Splatting Rasterization 性能基准测试')
    parser.add_argument('--n_gs', type=int, default=55296, help='高斯球数量')
    parser.add_argument('--width', type=int, default=244, help='图像宽度')
    parser.add_argument('--height', type=int, default=136, help='图像高度')
    parser.add_argument('--tile_size', type=int, default=64, help='Tile 大小')
    parser.add_argument('--sh_degree', type=int, default=3, help='球谐阶数 (0~3)')
    parser.add_argument('--warmup_iters', type=int, default=1, help='预热轮数')
    parser.add_argument('--perf_iters', type=int, default=10, help='性能测试轮数')
    parser.add_argument('--skip_func', action='store_true', help='跳过功能验证测试')
    parser.add_argument('--skip_perf', action='store_true', help='跳过性能测试')
    return parser.parse_args()


# ============================================================================
# 数据生成
# ============================================================================

def gen_splats(n_gs, sh_degree, device='npu'):
    means = (torch.rand(n_gs, 3, dtype=torch.float32) - 0.5) * 4.0
    quats = torch.randn(n_gs, 4, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.log(torch.ones(n_gs, 3, dtype=torch.float32) * 0.01)
    opacities = torch.logit(torch.full((n_gs,), 0.5, dtype=torch.float32))

    k = (sh_degree + 1) ** 2
    sh0 = torch.zeros(n_gs, 1, 3, dtype=torch.float32)
    shN = torch.randn(n_gs, k - 1, 3, dtype=torch.float32) * 0.01

    return {
        "means": means.to(device),
        "quats": quats.to(device),
        "scales": scales.to(device),
        "opacities": opacities.to(device),
        "sh0": sh0.to(device),
        "shN": shN.to(device),
    }


def gen_camera(width, height, device='npu'):
    camtoworlds = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    camtoworlds[0, 2, 3] = 4.0

    fx = fy = 300.0
    cx, cy = width / 2.0, height / 2.0
    Ks = torch.tensor([[[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]]], dtype=torch.float32, device=device)

    render_mode = "RGB"
    return (camtoworlds, Ks, render_mode)


# ============================================================================
# 功能验证测试
# ============================================================================

def run_func_tests(n_gs, width, height, tile_size, sh_degree):
    print("=" * 70)
    print("功能验证测试")
    print("=" * 70)
    print(f"  参数: n_gs={n_gs}, width={width}, height={height}, tile_size={tile_size}, sh_degree={sh_degree}")
    print()

    splats = gen_splats(n_gs, sh_degree)
    cam = gen_camera(width, height)
    rasterizer = Rasterizer()

    render_colors, render_depths, meta = rasterizer.rasterization(
        cam=cam,
        size=(width, height),
        tile_size=tile_size,
        active_sh_degree=sh_degree,
        splats=splats,
        camera_model="pinhole",
    )

    passed = 0
    failed = 0
    C = 1

    # Test 1: 输出形状
    try:
        assert tuple(render_colors.shape) == (C, height, width, 3), \
            f"colors shape: expected ({C}, {height}, {width}, 3), got {render_colors.shape}"
        assert tuple(render_depths.shape) == (C, height, width, 1), \
            f"depths shape: expected ({C}, {height}, {width}, 1), got {render_depths.shape}"
        print(f"  [PASS] test_output_shape: colors={render_colors.shape}, depths={render_depths.shape}")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_output_shape: {e}")
        failed += 1

    # Test 2: NaN/Inf 检查
    try:
        assert not torch.isnan(render_colors).any(), "render_colors contains NaN"
        assert not torch.isinf(render_colors).any(), "render_colors contains Inf"
        assert not torch.isnan(render_depths).any(), "render_depths contains NaN"
        assert not torch.isinf(render_depths).any(), "render_depths contains Inf"
        print(f"  [PASS] test_output_valid: no NaN/Inf")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_output_valid: {e}")
        failed += 1

    # Test 3: Padding 维度
    try:
        expected_pw = math.ceil(width / tile_size) * tile_size
        expected_ph = math.ceil(height / tile_size) * tile_size
        assert rasterizer.padded_width == expected_pw, \
            f"padded_width: expected {expected_pw}, got {rasterizer.padded_width}"
        assert rasterizer.padded_height == expected_ph, \
            f"padded_height: expected {expected_ph}, got {rasterizer.padded_height}"
        print(f"  [PASS] test_padded_dims: padded=({rasterizer.padded_width}, {rasterizer.padded_height})")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_padded_dims: {e}")
        failed += 1

    # Test 4: 颜色值范围
    try:
        c_min = render_colors.min().item()
        c_max = render_colors.max().item()
        assert c_min >= 0.0, f"colors min={c_min} < 0"
        assert c_max <= 1.5, f"colors max={c_max} > 1.5"
        print(f"  [PASS] test_colors_range: [{c_min:.4f}, {c_max:.4f}]")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] test_colors_range: {e}")
        failed += 1

    # Test 5: meta 信息
    try:
        assert meta["width"] == width
        assert meta["height"] == height
        assert meta["n_cameras"] == C
        assert "means2d" in meta
        assert "radii" in meta
        print(f"  [PASS] test_meta_info: width={meta['width']}, height={meta['height']}, n_cameras={meta['n_cameras']}")
        passed += 1
    except (AssertionError, KeyError) as e:
        print(f"  [FAIL] test_meta_info: {e}")
        failed += 1

    print()
    print(f"  结果: {passed} passed, {failed} failed, {passed + failed} total")
    return failed == 0


# ============================================================================
# 性能基准测试
# ============================================================================

def run_perf_benchmark(n_gs, width, height, tile_size, sh_degree, warmup_iters, perf_iters):
    print("=" * 70)
    print("性能基准测试")
    print("=" * 70)
    print(f"  参数: n_gs={n_gs}, width={width}, height={height}, tile_size={tile_size}, sh_degree={sh_degree}")
    print(f"  预热轮数: {warmup_iters}, 测试轮数: {perf_iters}")
    print()

    splats = gen_splats(n_gs, sh_degree)
    cam = gen_camera(width, height)

    # ---------- warmup ----------
    print(f"  Warmup ({warmup_iters} iters)...", end=" ", flush=True)
    rasterizer = Rasterizer()
    for _ in range(warmup_iters):
        rasterizer.tile_grid = None
        rasterizer.pix_coord = None
        render_colors, render_depths, meta = rasterizer.rasterization(
            cam=cam,
            size=(width, height),
            tile_size=tile_size,
            active_sh_degree=sh_degree,
            splats=splats,
            camera_model="pinhole",
        )
        torch.npu.synchronize()
    print("done")

    # ---------- benchmark ----------
    durations = []
    print(f"  Benchmark ({perf_iters} iters)...", flush=True)
    for i in range(perf_iters):
        rasterizer.tile_grid = None
        rasterizer.pix_coord = None

        t0 = time.time()
        render_colors, render_depths, meta = rasterizer.rasterization(
            cam=cam,
            size=(width, height),
            tile_size=tile_size,
            active_sh_degree=sh_degree,
            splats=splats,
            camera_model="pinhole",
        )
        torch.npu.synchronize()
        t1 = time.time()
        dur_ms = (t1 - t0) * 1000.0
        durations.append(dur_ms)
        print(f"    iter {i:3d}: {dur_ms:.3f} ms")

    # ---------- 统计 ----------
    dur_arr = np.array(durations)
    print()
    print(f"  端到端性能统计 (ms):")
    print(f"    Mean:   {dur_arr.mean():.3f}")
    print(f"    Median: {np.median(dur_arr):.3f}")
    print(f"    Min:    {dur_arr.min():.3f}")
    print(f"    Max:    {dur_arr.max():.3f}")
    print(f"    Std:    {dur_arr.std():.3f}")
    print(f"    P50:    {np.percentile(dur_arr, 50):.3f}")
    print(f"    P90:    {np.percentile(dur_arr, 90):.3f}")
    print(f"    P99:    {np.percentile(dur_arr, 99):.3f}")
    print(f"    FPS:    {1000.0 / dur_arr.mean():.1f}")

    perf_result = {
        "n_gs": n_gs,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "sh_degree": sh_degree,
        "warmup_iters": warmup_iters,
        "perf_iters": perf_iters,
        "mean_ms": float(dur_arr.mean()),
        "median_ms": float(np.median(dur_arr)),
        "min_ms": float(dur_arr.min()),
        "max_ms": float(dur_arr.max()),
        "std_ms": float(dur_arr.std()),
        "p50_ms": float(np.percentile(dur_arr, 50)),
        "p90_ms": float(np.percentile(dur_arr, 90)),
        "p99_ms": float(np.percentile(dur_arr, 99)),
        "fps": float(1000.0 / dur_arr.mean()),
        "durations": durations,
    }
    return perf_result


# ============================================================================
# msprof op_summary 解析
# ============================================================================

RASTERIZATION_OPS = [
    "ProjectionThreeDimsGaussianForward",
    "GaussianFilter",
    "FlashGaussianBuildMask",
    "GaussianSort",
    "GetRenderSchedule",
    "CalcRenderFwdDoubleClipGsids",
]

RASTERIZATION_OPS_LOWER = [op.lower() for op in RASTERIZATION_OPS]


def find_op_summary_csv(prof_dir: str) -> Optional[str]:
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            if f.startswith("op_summary") and f.endswith(".csv"):
                return os.path.join(root, f)
    return None


def parse_op_summary(csv_path: str) -> List[Dict]:
    ops = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_name = row.get("Op Name", "").strip()
            op_type = row.get("OP Type", "").strip()
            task_dur_str = row.get("Task Duration(us)", "0").strip().replace('\t', '')
            try:
                task_dur = float(task_dur_str)
            except ValueError:
                task_dur = 0.0
            ops.append({
                "op_name": op_name,
                "op_type": op_type,
                "task_duration_us": task_dur,
                "row": row,
            })
    return ops


def analyze_rasterization_ops(ops: List[Dict]) -> Tuple[Dict, float]:
    raster_ops = defaultdict(list)
    total_duration = 0.0

    for op in ops:
        name = op["op_name"]
        name_lower = name.lower()
        is_rast = False
        for i, pattern in enumerate(RASTERIZATION_OPS_LOWER):
            if pattern in name_lower:
                raster_ops[RASTERIZATION_OPS[i]].append(op)
                total_duration += op["task_duration_us"]
                is_rast = True
                break
        if not is_rast:
            for keyword in ["spherical", "sh_", "quat", "covar"]:
                if keyword in name_lower:
                    raster_ops["OtherProjection"].append(op)
                    total_duration += op["task_duration_us"]
                    break

    return dict(raster_ops), total_duration


def print_op_summary_table(raster_ops: Dict, total_us: float):
    print("=" * 70)
    print("msprof op_summary 算子耗时统计")
    print("=" * 70)
    print()

    op_stats = []
    for op_name, op_list in raster_ops.items():
        durations = [op["task_duration_us"] for op in op_list]
        count = len(durations)
        total = sum(durations)
        mean = total / count if count > 0 else 0
        op_stats.append({
            "op_name": op_name,
            "count": count,
            "total_us": total,
            "mean_us": mean,
            "min_us": min(durations) if durations else 0,
            "max_us": max(durations) if durations else 0,
        })

    op_stats.sort(key=lambda x: x["total_us"], reverse=True)

    header = f"  {'Op Name':<45} {'Count':>5} {'Total(us)':>12} {'Mean(us)':>10} {'Min(us)':>10} {'Max(us)':>10} {'Ratio':>8}"
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    for s in op_stats:
        ratio = s["total_us"] / total_us * 100 if total_us > 0 else 0
        print(f"  {s['op_name']:<45} {s['count']:>5} {s['total_us']:>12.1f} {s['mean_us']:>10.1f} {s['min_us']:>10.1f} {s['max_us']:>10.1f} {ratio:>7.1f}%")

    print(sep)
    print(f"  {'TOTAL':<45} {'':>5} {total_us:>12.1f} {'':>10} {'':>10} {'':>10} {'100.0%':>8}")
    print(f"  {'':45} {'':5} {'=' + str(round(total_us / 1000, 3)) + ' ms':>12}")
    print()

    return op_stats, total_us


def run_msprof_analysis(prof_dir: str):
    csv_path = find_op_summary_csv(prof_dir)
    if csv_path is None:
        print(f"  [WARN] op_summary CSV not found under {prof_dir}")
        return None, None
    print(f"  解析: {csv_path}")
    ops = parse_op_summary(csv_path)
    raster_ops, total_us = analyze_rasterization_ops(ops)
    if not raster_ops:
        print("  [WARN] 未找到 rasterization 相关算子")
        return None, None
    op_stats, total = print_op_summary_table(raster_ops, total_us)
    return op_stats, total


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(func_ok, perf_result, op_stats, total_us, args, report_path="perf_report.txt"):
    print("=" * 70)
    print("性能测试报告")
    print("=" * 70)
    print()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Gaussian Splatting Rasterization 性能测试报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"测试参数:\n")
        f.write(f"  高斯球数量 (n_gs):  {args.n_gs}\n")
        f.write(f"  图像宽度 (width):   {args.width}\n")
        f.write(f"  图像高度 (height):  {args.height}\n")
        f.write(f"  Tile 大小:    {args.tile_size}\n")
        f.write(f"  球谐阶数 (sh_degree): {args.sh_degree}\n")
        f.write(f"  预热轮数:       {args.warmup_iters}\n")
        f.write(f"  测试轮数:      {args.perf_iters}\n\n")

        f.write(f"功能验证: {'ALL PASSED' if func_ok else 'FAILED'}\n\n")

        if perf_result:
            f.write(f"端到端性能:\n")
            f.write(f"  Mean:   {perf_result['mean_ms']:.3f} ms\n")
            f.write(f"  Median: {perf_result['median_ms']:.3f} ms\n")
            f.write(f"  Min:    {perf_result['min_ms']:.3f} ms\n")
            f.write(f"  Max:    {perf_result['max_ms']:.3f} ms\n")
            f.write(f"  Std:    {perf_result['std_ms']:.3f} ms\n")
            f.write(f"  FPS:    {perf_result['fps']:.1f}\n\n")

            f.write(f"各轮耗时 (ms):\n")
            for i, d in enumerate(perf_result['durations']):
                f.write(f"  iter {i:3d}: {d:.3f}\n")
            f.write("\n")

        if op_stats:
            f.write(f"msprof 算子耗时统计 (Task Duration):\n")
            total_ms = total_us / 1000.0 if total_us else 0
            f.write(f"  {'Op Name':<45} {'Count':>5} {'Total(us)':>12} {'Mean(us)':>10} {'Min(us)':>10} {'Max(us)':>10} {'Ratio':>8}\n")
            f.write(f"  {'-' * 100}\n")
            for s in op_stats:
                ratio = s["total_us"] / total_us * 100 if total_us else 0
                f.write(f"  {s['op_name']:<45} {s['count']:>5} {s['total_us']:>12.1f} {s['mean_us']:>10.1f} {s['min_us']:>10.1f} {s['max_us']:>10.1f} {ratio:>7.1f}%\n")
            f.write(f"  {'-' * 100}\n")
            f.write(f"  {'TOTAL Task Duration':<45} {'':5} {total_us:>12.1f} {'':>10} {'':>10} {'':>10} {'100.0%':>8}\n")
            f.write(f"  {'Total (ms)':<45} {'':5} {total_ms:>12.3f}\n")
            f.write("\n")

    print(f"  报告已保存到: {report_path}")
    print()


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    print()
    print("*" * 70)
    print("  Gaussian Splatting Rasterization 性能基准测试")
    print("*" * 70)
    print(f"  n_gs={args.n_gs}, size=({args.width}, {args.height}), tile_size={args.tile_size}, sh_degree={args.sh_degree}")
    print(f"  warmup={args.warmup_iters}, iters={args.perf_iters}")
    print(f"  skip_func={args.skip_func}, skip_perf={args.skip_perf}")
    print()

    func_ok = True
    perf_result = None
    op_stats = None
    total_us = None

    # 1. 功能验证
    if not args.skip_func:
        func_ok = run_func_tests(args.n_gs, args.width, args.height, args.tile_size, args.sh_degree)
        if not func_ok:
            print("[ERROR] 功能验证失败，跳过性能测试")
            sys.exit(1)
    else:
        print("[INFO] 跳过功能验证测试")
        print()

    # 2. 性能基准
    if not args.skip_perf:
        perf_result = run_perf_benchmark(
            args.n_gs, args.width, args.height, args.tile_size, args.sh_degree,
            args.warmup_iters, args.perf_iters,
        )
    else:
        print("[INFO] 跳过性能测试")
        print()

    # 3. msprof op_summary 解析
    prof_dir_env = os.environ.get("MSPROF_OUTPUT_DIR", "")
    if prof_dir_env and os.path.isdir(prof_dir_env):
        print(f"[INFO] 检测到 MSPROF_OUTPUT_DIR={prof_dir_env}, 解析 op_summary")
        op_stats, total_us = run_msprof_analysis(prof_dir_env)
    else:
        prof_subdirs = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for d in [script_dir, os.getcwd(), "/tmp"]:
            prof_path = os.path.join(d, "prof")
            if os.path.isdir(prof_path):
                for sub in os.listdir(prof_path):
                    sub_full = os.path.join(prof_path, sub)
                    if os.path.isdir(sub_full) and sub.startswith("PROF_"):
                        prof_subdirs.append(sub_full)
        if prof_subdirs:
            latest_prof = max(prof_subdirs, key=lambda p: os.path.getmtime(p))
            print(f"[INFO] 发现 profiling 目录: {latest_prof}")
            op_stats, total_us = run_msprof_analysis(latest_prof)

    if op_stats is None and perf_result is not None:
        print()
        print("[HINT] 若需 msprof 算子级分析，请使用以下命令运行本脚本:")
        print(f"  msprof --output=./prof --application=\"python {os.path.abspath(__file__)} --n_gs {args.n_gs} --width {args.width} --height {args.height} --tile_size {args.tile_size}\" ")
        print()

    # 4. 生成报告
    generate_report(func_ok, perf_result, op_stats, total_us, args)

    print("Done.")


if __name__ == "__main__":
    main()