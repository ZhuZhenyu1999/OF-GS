#!/usr/bin/env python3
"""Re-run the 2026-11-09 ablations without overwriting earlier results.

Run on the Linux dataset server, in the existing test2-slam environment:
  python scripts/rerun_ablation_review.py --plan missing --gpu 2
  python scripts/rerun_ablation_review.py --plan missing --gpu 2 --execute
  python scripts/rerun_ablation_review.py --plan gate-check --gpu 2 --execute

Default is a command preview, with no training or output-directory creation.
missing: KITTI 02/05/06/10 x direct=off x gate=off/on, seed=0 (8 runs).
gate-check: KITTI 02/06/10 x direct=on x gate=off/on, seeds=0/1/2 (18 runs).
full: both datasets x all four configurations x seeds=0/1/2 (204 runs).
Use full only after fixing failures and choosing the final code/configuration.

Commands reproduce the explicit settings in the supplied ablation summaries.
Unspecified settings come from the server's args.py and are saved for each run;
this script cannot reconstruct an unavailable historical code version.
Each batch is sequential on one GPU and stops at its first unsuccessful run
unless --keep-going is supplied. GPU usage is sampled every two seconds, not
an exact PyTorch peak-memory measurement. SLAM's own Time is kept separately
from the runner's full process wall time. No model or algorithm code is changed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
import zipfile

WAYMO = ['100613', '13476', '106762', '132384', '152706', '153495',
         '158686', '163453', '405841']
KITTI = ['00', '02', '03', '05', '06', '07', '08', '10']
COMMON = shlex.split('''--test_hold 8 --num_kpts 8000 --mono_depth_backend da2
--local_ba_window_size 10 --local_ba_recent_keyframes 5 --local_ba_fixed_keyframes 2
--local_ba_max_points 1500 --local_ba_min_shared_points 30 --local_ba_min_tracks 50
--local_ba_iterations 15 --local_ba_every 5 --local_ba_huber_delta 3.0
--local_ba_max_reprojection_error 8.0 --local_ba_min_parallax_deg 1.0
--local_ba_max_pose_change_ratio 0.2 --local_ba_max_rotation_deg 5.0
--pose_gaussian_min_update_count 5 --pose_gaussian_min_observed_keyframes 3
--densification_interval 50 --densify_grad_threshold 0.0002 --percent_dense 0.01
--depth_loss_weight_init 0.01 --depth_loss_weight_decay 0.9 --viewer_mode none''')
CAPTURE_ARGS = '''import json,runpy,sys
root=sys.argv.pop(1)
parser=runpy.run_path(root+'/args.py')['get_args']
print(json.dumps(vars(parser()),ensure_ascii=False,default=str))
'''


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_jobs(plan, seeds=None):
    if seeds is None:
        seeds = [0] if plan == 'missing' else [0, 1, 2]
    if len(seeds) != len(set(seeds)) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('Seeds must be unique integers in [0, 2**32).')
    sets = [('kitti', ['02', '05', '06', '10'])] if plan == 'missing' else (
        [('kitti', ['02', '06', '10'])] if plan == 'gate-check' else
        [('waymo', WAYMO), ('kitti', KITTI)])
    directs = ['off'] if plan == 'missing' else (['on'] if plan == 'gate-check' else ['off', 'on'])
    return [dict(dataset=ds, sequence=s, direct=d, gate=g, seed=seed)
            for seed in seeds for ds, seqs in sets for s in seqs
            for d in directs for g in ['off', 'on']]


def job_command(job, args, batch):
    ds = job['dataset']
    source = (args.kitti_root if ds == 'kitti' else args.waymo_root) / job['sequence']
    output = (batch / f"seed_{job['seed']}" /
              f"{ds}_direct_{job['direct']}_gate_{job['gate']}" / ds / job['sequence'])
    cli = ['-s', str(source), '--dataset_type', ds]
    cli += ['--kitti_camera', '2'] if ds == 'kitti' else [
        '--camera_name', 'FRONT', '--waymo_image_model', 'opencv']
    cli += COMMON + ['--seed', str(job['seed']),
                     '--direct_gaussian_sampling', job['direct'],
                     '--pose_gaussian_gate', job['gate'], '-m', str(output)]
    return [sys.executable, '-u', str(args.repo / 'slam.py'), *cli], output, source


def probe(command, cwd):
    try:
        r = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                           errors='replace', timeout=30)
        return {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'error': str(e)}


def source_snapshot(repo, batch):
    excluded = {'__pycache__', 'node_modules', 'models', 'data', 'datasets',
                'build', 'dist', 'artifacts', 'tmp', 'tests', '论文', '图表'}
    extensions = {'.py', '.cpp', '.cu', '.cuh', '.h', '.hpp'}
    manifest = {}
    with zipfile.ZipFile(batch / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for base, dirs, names in os.walk(repo):
            dirs[:] = sorted(d for d in dirs if d not in excluded
                             and not d.startswith(('.', 'results'))
                             and not Path(base, d).is_symlink())
            for name in sorted(names):
                p = Path(base, name)
                if p.suffix not in extensions or p.is_symlink():
                    continue
                relative = p.relative_to(repo).as_posix()
                if p.is_relative_to(batch):
                    continue
                data = p.read_bytes()
                manifest[relative] = hashlib.sha256(data).hexdigest()
                z.writestr(relative, data)
    write_json(batch / 'source_sha256.json', manifest)


def run_one(job, command, output, repo, env, gpu):
    output.mkdir(parents=True, exist_ok=False)
    record = dict(job, dataset_type=job['dataset'], output_dir=str(output),
                  command='CUDA_VISIBLE_DEVICES=' + shlex.quote(gpu) + ' ' + shlex.join(command),
                  argv=command, started_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'run_record.json', record)
    parsed = subprocess.run([sys.executable, '-S', '-c', CAPTURE_ARGS,
                             str(repo), *command[3:]], cwd=repo, env=env,
                            capture_output=True, text=True, errors='replace')
    if parsed.returncode:
        (output / 'run.log').write_text(parsed.stdout + parsed.stderr, encoding='utf-8')
        record.update(returncode=parsed.returncode, completed=False, error='Argument validation failed')
        write_json(output / 'run_record.json', record)
        return record
    write_json(output / 'resolved_args.json', json.loads(parsed.stdout))
    write_json(output / 'gpu_before.json', probe(['nvidia-smi', '-i', gpu], repo))
    monitor = None
    start = time.perf_counter()
    try:
        with (output / 'gpu_samples.csv').open('w', encoding='utf-8') as samples, \
             (output / 'gpu_monitor.log').open('w', encoding='utf-8') as monitor_log, \
             (output / 'run.log').open('w', encoding='utf-8') as log:
            try:
                monitor = subprocess.Popen([
                    'nvidia-smi', '-i', gpu,
                    '--query-gpu=timestamp,index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu',
                    '--format=csv', '-l', '2'], cwd=repo, stdout=samples, stderr=monitor_log)
            except OSError as e:
                monitor_log.write(str(e))
            log.write(record['command'] + '\n'); log.flush()
            print('  Log: ' + str(output / 'run.log'), flush=True)
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['returncode'] = result.returncode
    except KeyboardInterrupt:
        record.update(returncode=130, interrupted=True)
    except OSError as e:
        record.update(returncode=-1, error=str(e))
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill(); monitor.wait()
        record['process_wall_seconds'] = time.perf_counter() - start
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
    metrics_file = output / 'slam_metrics.json'
    metrics = None
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
        except (ValueError, OSError) as e:
            record['metrics_error'] = str(e)
    required = ['PSNR', 'SSIM', 'LPIPS', 'RMSE ATE [m]', 'time', 'num_gaussians']
    record['completed'] = record['returncode'] == 0 and isinstance(metrics, dict) and all(
        isinstance(metrics.get(k), (int, float)) and not isinstance(metrics[k], bool)
        and math.isfinite(metrics[k]) for k in required)
    if record['completed']:
        record['completed'] = metrics['time'] > 0 and isinstance(metrics['num_gaussians'], int) and metrics['num_gaussians'] >= 0
    if isinstance(metrics, dict):
        record['metrics'] = metrics
        record.update({k: metrics[k] for k in ['PSNR', 'SSIM', 'LPIPS', 'num_gaussians', 'FPS'] if k in metrics})
        record.update(ATE=metrics.get('RMSE ATE [m]'), TIME=metrics.get('time'))
    elif metrics is not None:
        record['metrics_error'] = 'slam_metrics.json must contain an object'
    if not record['completed'] and not record.get('error'):
        record['error'] = 'Nonzero process exit or missing required final metrics; inspect run.log'
    write_json(output / 'gpu_after.json', probe(['nvidia-smi', '-i', gpu], repo))
    write_json(output / 'run_record.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', choices=['missing', 'gate-check', 'full'], required=True)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--gpu', default='2', help='One physical GPU index or UUID; default 2')
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--kitti-root', type=Path, default=Path('/mnt/data_23/zzy/dataset/KITTI'))
    p.add_argument('--waymo-root', type=Path, default=Path('/mnt/data_23/zzy/dataset/waymo'))
    p.add_argument('--output-root', type=Path, default=Path('results/ablation_review'))
    p.add_argument('--execute', action='store_true', help='Actually run; otherwise only print the plan')
    p.add_argument('--keep-going', action='store_true', help='Continue after unsuccessful jobs')
    args = p.parse_args()
    args.repo = args.repo.expanduser().resolve()
    args.kitti_root = args.kitti_root.expanduser().resolve()
    args.waymo_root = args.waymo_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser()
    if not args.gpu or ',' in args.gpu:
        p.error('--gpu must identify a single GPU')
    if not args.output_root.is_absolute():
        args.output_root = args.repo / args.output_root
    args.output_root = args.output_root.resolve()
    try:
        jobs = make_jobs(args.plan, args.seeds)
    except ValueError as e:
        p.error(str(e))
    batch = args.output_root / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.plan)
    commands = [job_command(job, args, batch) for job in jobs]
    print(f'Plan: {args.plan}; planned runs: {len(jobs)}; GPU: {args.gpu}', flush=True)
    print(f'Batch: {batch}', flush=True)
    if not args.execute:
        for i, (cmd, _, _) in enumerate(commands, 1):
            print(f'[{i}/{len(jobs)}] CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {shlex.join(cmd)}')
        print('Preview only. Add --execute to run on the dataset server.')
        return 0
    for path in [args.repo / 'slam.py', args.repo / 'args.py']:
        if not path.is_file():
            p.error(f'Required source file not found: {path}')
    for _, _, source in commands:
        if not source.is_dir():
            p.error(f'Dataset sequence not found: {source}')
    gpu_info = probe(['nvidia-smi', '-i', args.gpu], args.repo)
    if gpu_info.get('returncode') != 0:
        p.error('Cannot query requested GPU: ' + str(gpu_info))
    uuid_info = probe(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid', '--format=csv,noheader'], args.repo)
    gpu_uuid = uuid_info.get('stdout', '').strip()
    if uuid_info.get('returncode') != 0 or not gpu_uuid.startswith('GPU-') or '\n' in gpu_uuid:
        p.error('Cannot resolve requested GPU to one UUID: ' + str(uuid_info))
    print(f'Physical GPU {args.gpu} resolves to {gpu_uuid}', flush=True)
    batch.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu_uuid, PYTHONUNBUFFERED='1')
    source_snapshot(args.repo, batch)
    write_json(batch / 'environment.json', {
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'requested_gpu': args.gpu, 'resolved_gpu_uuid': gpu_uuid,
        'gpu': gpu_info, 'git_head': probe(['git', 'rev-parse', 'HEAD'], args.repo),
        'git_status': probe(['git', 'status', '--short'], args.repo),
        'packages': sorted((d.metadata.get('Name', ''), d.version) for d in importlib.metadata.distributions()),
        'environment': {k: env.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                          'PYTHONHASHSEED', 'PYTORCH_CUDA_ALLOC_CONF', 'CUBLAS_WORKSPACE_CONFIG']}})
    write_json(batch / 'plan.json', [dict(job, argv=cmd, output_dir=str(out))
                                    for job, (cmd, out, _) in zip(jobs, commands)])
    records = []
    for i, (job, (command, output, _)) in enumerate(zip(jobs, commands), 1):
        print(f'[{i}/{len(jobs)}] {job}', flush=True)
        record = run_one(job, command, output, args.repo, env, gpu_uuid)
        records.append(record)
        write_json(batch / 'rerun_summary.json', records)
        print(f"  returncode={record['returncode']}; completed={record['completed']}", flush=True)
        if record.get('interrupted') or (not record['completed'] and not args.keep_going):
            print('Stopped. See the last run.log before rerunning; earlier results remain intact.')
            break
    print('Summary: ' + str(batch / 'rerun_summary.json'))
    return 0 if len(records) == len(jobs) and all(r['completed'] for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
