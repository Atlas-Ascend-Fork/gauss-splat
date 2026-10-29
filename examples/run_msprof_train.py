#!/usr/bin/env python3

import sys
import os
import csv
import json
import glob
import shutil
import subprocess

MODE = sys.argv[1]  # forward | backward
SH_DEGREE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
PERF_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 and MODE == 'backward' else 10

HERE = os.path.dirname(os.path.abspath(__file__))  # examples/arch35
CKPT_DIR = 'checkpoints'

CKPT_LEVELS = [
    (200_000, 'ckpt_31_rank0_20w.pt'),
    (400_000, 'ckpt_31_rank0_40w.pt'),
    (600_000, 'ckpt_31_rank0_60w.pt'),
    (800_000, 'ckpt_31_rank0_80w.pt'),
    (1_000_000, 'ckpt_31_rank0_100w.pt'),
    (1_400_000, 'ckpt_31_rank0_140w.pt'),
    (1_700_000, 'ckpt_31_rank0_170w.pt'),
    (2_000_000, 'ckpt_31_rank0_200w.pt'),
    (3_000_000, 'ckpt_31_rank0_300w.pt'),
    (4_000_000, 'ckpt_31_rank0_400w.pt'),
]

# 3DGS forward operators (SIMD + SIMT variants)
FWD_OPS = [
    'SphericalHarmonicsForward', 'QuatScalesToCovars',
    'ProjectionThreeDimsGaussianForward', 'GaussianFilter',
    'FlashGaussianBuildMask', 'GaussianSort',
    'CalcRenderSimtFwd', 'CalcRenderFwdDoubleClipGsids',
]
# 3DGS backward operators (SIMD + SIMT variants)
BWD_OPS = [
    'SphericalHarmonicsBwd', 'FullyFusedProjectionBwd',
    'CalcRenderSimtBwd', 'CalcRenderBwdVarClipGsids',
]
OP_NAMES = FWD_OPS if MODE == 'forward' else BWD_OPS + FWD_OPS

import subprocess as _sp
_BRANCH = _sp.check_output(['git', 'branch', '--show-current'], cwd=HERE, text=True).strip()
BASE_OUT = os.path.join(HERE, f'out_train_{_BRANCH}_{MODE}_sh{SH_DEGREE}')


def run_msprof_for_ckpt(n_label, ckpt_name):
    out_dir = os.path.join(BASE_OUT, ckpt_name.replace('.pt', ''))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    result_dir = os.path.join(HERE, f'results/perf_train_{MODE}_{ckpt_name.replace(".pt","")}')
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    train_py = os.path.join(HERE, 'train.py')
    if MODE == 'forward':
        train_cmd = (f'{train_py} --data_dir data/360_v2/garden '
                     f'--result_dir {result_dir} --ckpt {ckpt_path} '
                     f'--sh_degree {SH_DEGREE} --data_factor 4 --depth_loss '
                     f'--tile_size 32 --steps_scaler 0.1 --render_backend 3dgs')
    else:
        train_cmd = (f'{train_py} --data_dir data/360_v2/garden '
                     f'--result_dir {result_dir} --ckpt {ckpt_path} '
                     f'--sh_degree {SH_DEGREE} --data_factor 4 --depth_loss '
                     f'--tile_size 32 --perf_steps {PERF_STEPS} --steps_scaler 0.1 '
                     f'--render_backend 3dgs --ssim_lambda 0.0')
    msprof_cmd = ['bash', '-c',
                  f'export ASCEND_RT_VISIBLE_DEVICES=0 && '
                  f'export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256" && '
                  f'cd {HERE} && '
                  f'msprof --output={out_dir} --aic-mode=task-based '
                  f'--aic-metrics=PipeUtilization --task-time=on '
                  f'--ascendcl=on --ai-core=on '
                  f'--application="$(which python) {train_cmd}"']
    print(f'\n=== {ckpt_name} (N~{n_label:,}, mode={MODE}, sh_degree={SH_DEGREE}, steps={PERF_STEPS}) ===',
          flush=True)
    p = subprocess.run(msprof_cmd, capture_output=True, text=True, timeout=7200)
    if p.returncode != 0:
        print('  STDERR tail:', p.stderr[-300:], flush=True)
    stdout_tail = p.stdout[-500:]
    has_oom = ('out of memory' in stdout_tail.lower()
               or 'out of memory' in p.stderr.lower())
    csvs = glob.glob(os.path.join(out_dir, 'PROF_*/mindstudio_profiler_output/op_summary_*.csv'))
    if not csvs:
        print(f'  ERROR: no op_summary.csv (oom={has_oom})', flush=True)
        print('  stdout tail:', stdout_tail[-300:], flush=True)
        return {'error': 'no op_summary', 'oom': has_oom}
    ops = parse_op_summary(csvs[0])
    import torch
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    N = ck['splats']['means'].shape[0]
    means_nan = torch.isnan(ck['splats']['means']).any().item()
    return {'N': N, 'means_nan': means_nan, 'oom': has_oom, 'ops': ops}


def parse_op_summary(csv_file):
    op_durations = {name: [] for name in OP_NAMES}
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def start_key(r):
        try:
            return float(r.get('Task Start Time(us)', '0').strip().rstrip('\t'))
        except ValueError:
            return 0.0
    rows.sort(key=start_key)

    for row in rows:
        op_name = row.get('Op Name', '')
        try:
            dur = float(row['Task Duration(us)'])
        except (ValueError, KeyError):
            continue
        for target in OP_NAMES:
            if target in op_name:
                op_durations[target].append(dur)

    result = {}
    for op, durs in op_durations.items():
        if not durs:
            result[op] = None
            continue
        warmup = min(max(len(durs) // 10, 2), 10)
        kept = durs[warmup:] if len(durs) > warmup else durs
        result[op] = round(sum(kept) / len(kept), 3)
    return result


def main():
    all_results = {}
    for n_label, ckpt_name in CKPT_LEVELS:
        res = run_msprof_for_ckpt(n_label, ckpt_name)
        all_results[ckpt_name] = res
        print(f'  {ckpt_name}: {res}', flush=True)

    out_json = os.path.join(BASE_OUT, f'{MODE}_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved {out_json}')


if __name__ == '__main__':
    main()
