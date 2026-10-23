"""Private CPU prep and GPU parity/native-window probe; no downloads or fallback."""
from __future__ import annotations
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

PROJECT = Path('/opt/project')
DATA = Path('/data')
RESULTS = DATA / 'results'
EXPECTED = {'torch': '2.7.1+cu126', 'triton': '3.3.1', 'fla-core': '0.5.2',
            'einops': '0.8.1', 'numpy': '2.2.6', 'safetensors': '0.5.3'}
FLA_CHUNK_SHA = 'a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, indent=2) + '\n')


def claim(job, stage):
    import modal
    volume = modal.Volume.from_name(job['volume_name'], create_if_missing=False)
    volume.reload()
    if time.time() >= job['deadline_unix']:
        raise TimeoutError('Absolute session deadline expired before worker start')
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Durable guard also rejects provider restarts after this marker is committed.
    with (RESULTS / ('attempt_' + stage + '.json')).open('x') as f:
        json.dump({'stage': stage, 'started_at_unix': time.time(), 'automatic_retry': False}, f)
    volume.commit()
    return volume


def verify_sources(job):
    result = []
    for row in job['source_files']:
        path = PROJECT / row['path']
        if not path.resolve().is_relative_to(PROJECT) or path.is_symlink():
            raise ValueError('Source path escaped allowlist')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Remote source hash mismatch: ' + row['path'])
        result.append(row)
    return result


def verify_assets(job):
    result = []
    for row in job['assets']:
        path = DATA / row['remote_path']
        if not path.resolve().is_relative_to(DATA) or path.is_symlink():
            raise ValueError('Asset path escaped allowlist')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Remote asset hash mismatch: ' + row['role'])
        result.append(row)
    return result


def runtime(gpu=False):
    import torch
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED or torch.__version__ != EXPECTED['torch'] or torch.version.cuda != '12.6':
        raise RuntimeError('Runtime exact-version/CUDA gate failed: ' + json.dumps(versions))
    if not platform.python_version().startswith('3.12.'):
        raise RuntimeError('Expected Python 3.12 runtime')
    chunk = importlib.import_module('fla.ops.kda.chunk')
    chunk_hash = sha(chunk.__file__)
    if chunk_hash != FLA_CHUNK_SHA:
        raise RuntimeError('Installed official FLA KDA source SHA mismatch')
    result = {'python': platform.python_version(), 'versions': versions, 'cuda': torch.version.cuda,
              'fla_kda_chunk_source_sha256': chunk_hash,
              'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, timeout=15).splitlines(),
              'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True, timeout=5).splitlines()[0]}
    if gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Requires exactly one BF16-capable CUDA GPU')
        name = torch.cuda.get_device_name()
        if 'L40S' not in name:
            raise RuntimeError('Unexpected GPU: ' + name)
        result.update({'gpu': name, 'gpu_count': 1, 'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                       'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True, timeout=5).strip()})
    return result


def prepare(job):
    volume = claim(job, 'prepare')
    started = time.time()
    try:
        source_files = verify_sources(job)
        assets = verify_assets(job)
        # CPU prep must not import KDA GPU kernels: runtime() is GPU-only below.
        versions = {name: importlib.metadata.version(name) for name in EXPECTED}
        if versions != EXPECTED:
            raise RuntimeError('CPU preparation exact-version gate failed')
        import torch
        if torch.version.cuda != '12.6' or not platform.python_version().startswith('3.12.'):
            raise RuntimeError('CPU preparation Python/CUDA runtime mismatch')
        from safetensors import safe_open
        ckpt = next(DATA / x['remote_path'] for x in assets if x['role'] == 'checkpoint')
        with safe_open(ckpt, framework='pt', device='cpu') as f:
            keys = list(f.keys())
        if len(keys) != 294:
            raise RuntimeError('Expected 294 Pupu teacher tensors')
        result = {'status': 'passed', 'scope': 'CPU hash/runtime/header preparation only, no GPU used',
                  'versions': versions, 'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
                  'source_files': source_files, 'assets': assets, 'checkpoint_tensors': len(keys),
                  'billing_scope': job['billing_scope'], 'provenance': job['provenance'],
                  'duration_seconds': time.time() - started}
        write('prepare.json', result)
        return result
    except Exception as exc:
        result = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'automatic_retry': False}
        write('prepare.json', result)
        return result
    finally:
        volume.commit()


def run_process(argv, deadline, log):
    remaining = deadline - time.time()
    if remaining < 10:
        raise TimeoutError('Insufficient bounded worker time; no new process started')
    with log.open('a') as stream:
        stream.write('\nCOMMAND ' + json.dumps(argv) + '\n'); stream.flush()
        child = subprocess.Popen(argv, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            returncode = child.wait(timeout=max(1, remaining - 5))
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise
    if returncode:
        raise RuntimeError(f'Subprocess exited {returncode}; stop, no reference/smaller-model retry')


def probe(job):
    volume = claim(job, 'probe')
    started = time.time()
    deadline = min(job['deadline_unix'] - 10, started + 580)
    result = {'status': 'failed', 'scope': 'S0a synthetic-head numerical path check, not SFT or timing-quality validation',
              'automatic_retry': False, 'run_id': job['run_id'], 'started_at_unix': started,
              'deadline_unix': deadline, 'expected_total_parameters': job['expected_total_parameters'],
              'billing_scope': job['billing_scope'], 'provenance': job['provenance']}
    try:
        verify_sources(job)
        assets = verify_assets(job)
        info = runtime(gpu=True)
        info['source_files'] = job['source_files']; info['assets'] = assets
        write('runtime.json', info)
        parity = RESULTS / 'fla_parity.json'
        run_process([sys.executable, 'scripts/check_fla.py', '--output', str(parity)],
                    min(deadline, time.time() + 240), RESULTS / 'probe.log')
        check = json.loads(parity.read_text())
        if not check.get('passed') or check.get('tested_lengths') != [16, 129]:
            raise RuntimeError('Parity gate failed; full model never started')
        checkpoint = next(DATA / x['remote_path'] for x in assets if x['role'] == 'checkpoint')
        audio = next(DATA / x['remote_path'] for x in assets if x['role'] == 'audio')
        output = RESULTS / 'cuda_fla_native_probe.json'
        run_process([sys.executable, 'scripts/probe.py', '--checkpoint', str(checkpoint),
                     '--device', 'cuda', '--backend', 'fla', '--parity-report', str(parity),
                     '--mel-frames', '1024', '--groups', '2', '--steps', '1', '--audio', str(audio),
                     '--audio-offset-seconds', '15', '--output', str(output)], deadline, RESULTS / 'probe.log')
        report = json.loads(output.read_text())
        if (report.get('status') != 'passed' or report.get('inventory', {}).get('total') != job['expected_total_parameters']
                or report.get('steps') != 1 or report.get('mel_frames') != 1024
                or report.get('backend') != 'fla' or not report.get('native_window_test')
                or report.get('audio_offset_seconds') != 15):
            raise RuntimeError('Final full-size native-window scope/report gate failed')
        result.update({'status': 'passed', 'parity_passed': True, 'native_probe_passed': True,
                       'total_parameters': report['inventory']['total'], 'gpu': info['gpu']})
    except Exception as exc:
        result.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
    finally:
        result['ended_at_unix'] = time.time()
        result['worker_seconds'] = result['ended_at_unix'] - started
        write('launcher_status.json', result)
        volume.commit()
    return result
