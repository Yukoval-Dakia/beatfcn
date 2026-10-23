#!/usr/bin/env python3
"""Explicit, one-attempt Modal S0a actions. Import/help/plan are entirely local."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SDK_VERSION = '1.6.2'
TORCH = '2.7.1+cu126'
TRITON = '3.3.1'
FLA = '0.5.2'
FLA_HASH = '5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761'
WEIGHT_HASH = '6a97605541ba9d3680f6758e2ae19019736f03e5db14e8bbfbd71746b19304f4'
WEIGHT_BYTES = 1612597160
HIGHER_HASH = 'ffc5a731ccd5797e81e95fb75a8f575dc92c9cca6ab8116a17a8b1c65582cd98'
EXPECTED_PARAMS = 476739917
CPU = 2
MEM_MIB = 16384
GPU_SECONDS = 600
SESSION_SECONDS = 900
AUTHORIZED_GROSS_CEILING_USD = 2.0
TARGET_GROSS_USD = 1.0
SCALEDOWN_SECONDS = 2
GPU_RATE = .000542
CPU_RATE = .0000131
MEM_RATE = .00000222
SOURCE_NAMES = (
    'beatfcn_s0/__init__.py', 'beatfcn_s0/kda.py', 'beatfcn_s0/model.py',
    'beatfcn_s0/preprocess.py', 'beatfcn_s0/pupu.py',
    'scripts/check_fla.py', 'scripts/probe.py', 'scripts/modal_worker_s0a.py',
    'scripts/modal_launch_s0a.py', 'requirements-fla.txt', 'requirements-modal.txt',
)
ARTIFACT_NAMES = ('prepare.json', 'runtime.json', 'fla_parity.json',
                  'cuda_fla_native_probe.json', 'launcher_status.json', 'probe.log')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_file(path):
    p = Path(path).absolute()
    if any(part.startswith('.') for part in p.parts[1:]):
        raise ValueError('Hidden paths are outside the file allowlist')
    if any(parent.is_symlink() for parent in (p, *p.parents)):
        raise ValueError('Symlink paths are forbidden')
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def sources():
    result = []
    for name in SOURCE_NAMES:
        path = safe_file(ROOT / name)
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Source escapes project root')
        if path.stat().st_size > 256 * 1024:
            raise ValueError('Unexpectedly large source; review before upload')
        result.append({'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size})
    return result


def read_json(path):
    return json.loads(safe_file(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def validate_run_id(value):
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,47}', value):
        raise ValueError('run-id must be 1..48 lowercase letters/digits/hyphens')
    return value


def create_manifest(args):
    """Local-only selection, no credential lookup or upload."""
    candidate = read_json(args.candidate)
    if candidate.get('chart_key') != 'osu4k:4067533' or candidate.get('audio_sha256') != HIGHER_HASH:
        raise ValueError('This bounded launch is pinned to the curated Higher candidate')
    rows = []
    for role, path, remote, expected in (
        ('checkpoint', args.checkpoint, 'assets/teacher_encoder.safetensors', WEIGHT_HASH),
        ('audio', args.audio or candidate['audio_path'], 'assets/Higher.mp3', HIGHER_HASH),
    ):
        p = safe_file(path)
        if sha(p) != expected:
            raise ValueError(f'{role} checksum mismatch')
        rows.append({'role': role, 'local_path': str(p), 'remote_path': remote,
                     'sha256': expected, 'bytes': p.stat().st_size})
    if rows[0]['bytes'] != WEIGHT_BYTES:
        raise ValueError('Unexpected checkpoint length')
    provenance_keys = ('chart_key', 'title', 'artist', 'creator', 'version', 'source_url',
                       'dataset', 'dataset_revision', 'chart_sha256', 'audio_sha256',
                       'human_authorship_evidence', 'independent_audio_alignment_audit',
                       'gold', 'rights_status', 'purpose')
    manifest = {'schema': 1, 'assets': rows, 'source_files': sources(),
                'provenance': {k: candidate[k] for k in provenance_keys if k in candidate},
                'probe': {'audio_offset_seconds': 15.0, 'mel_frames': 1024, 'window_seconds': 10.24,
                          'groups': 2, 'steps': 1, 'head_supervision': 'synthetic, not chart labels'}}
    write_json(args.asset_manifest, manifest)
    return manifest


def checked_manifest(path, check_local_assets=True):
    m = read_json(path)
    if m.get('schema') != 1 or m.get('source_files') != sources():
        raise ValueError('Missing/stale manifest: source changed after review')
    rows = m.get('assets', [])
    if len(rows) != 2 or {r.get('role') for r in rows} != {'checkpoint', 'audio'}:
        raise ValueError('Exactly the pinned checkpoint and Higher audio are allowed')
    expected = {'checkpoint': ('assets/teacher_encoder.safetensors', WEIGHT_HASH),
                'audio': ('assets/Higher.mp3', HIGHER_HASH)}
    for row in rows:
        if (row.get('remote_path'), row.get('sha256')) != expected[row['role']]:
            raise ValueError('Asset identity mismatch')
        if not isinstance(row.get('bytes'), int) or row['bytes'] <= 0:
            raise ValueError('Invalid asset size')
        if row['role'] == 'checkpoint' and row['bytes'] != WEIGHT_BYTES:
            raise ValueError('Checkpoint size mismatch')
        if check_local_assets:
            p = safe_file(row['local_path'])
            if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
                raise ValueError('Asset changed since manifest creation')
    if sum(r['bytes'] for r in rows) > 2 * 1024**3:
        raise ValueError('Asset allowlist exceeds 2 GiB ceiling')
    if m.get('probe') != {'audio_offset_seconds': 15.0, 'mel_frames': 1024, 'window_seconds': 10.24,
                          'groups': 2, 'steps': 1, 'head_supervision': 'synthetic, not chart labels'}:
        raise ValueError('Probe scope differs from authorized S0a')
    return m


def check_billing(record, now=None):
    """Operator evidence gate, NOT a billing API or a self-created spend cap."""
    now = time.time() if now is None else now
    if record.get('schema') != 1 or record.get('operator_launch_clearance') is not True:
        raise ValueError('Current operator billing clearance is required')
    timestamp = record.get('account_checked_at_unix', 0)
    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or not 0 <= now - timestamp <= 900:
        raise ValueError('Read-only account check must be current (within 15 minutes)')
    if record.get('account_read_check_completed') is not True:
        raise ValueError('Recheck the account read-only after authentication')
    for key in ('remaining_credits_usd', 'gross_usage_so_far_usd', 'workspace_usage_cap_usd', 'net_spend_limit_usd'):
        v = record.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f'Billing field must be a known nonnegative number: {key}')
    if record['remaining_credits_usd'] < 1 or record['workspace_usage_cap_usd'] > AUTHORIZED_GROSS_CEILING_USD or record['net_spend_limit_usd'] != 0:
        raise ValueError('Requires >=$1 applicable credits, authorized workspace gross cap <=$2, net spend limit $0')
    if record['workspace_usage_cap_usd'] - record['gross_usage_so_far_usd'] < .50:
        raise ValueError('Less than $0.50 gross headroom; reassess before any remote action')
    if record.get('cap_scope') != 'workspace_all_usage_before_credits':
        raise ValueError('Environment compute budget alone does not include build/storage/egress scope')
    for key in ('credits_evidence', 'limit_evidence'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError('Record non-secret billing evidence, not credentials')
    if record.get('limit_evidence_status') not in ('provider_verified', 'user_reported'):
        raise ValueError('Distinguish provider-verified limits from user-reported settings')
    if record.get('other_workloads_accounted_for') is not True:
        raise ValueError('Account for other workspace usage before this bounded run')
    return record


def require_sdk():
    # A denied alternate endpoint must never be used as an automatic fallback.
    endpoint = os.environ.get('MODAL_SERVER_URL', 'https://api.modal.com')
    if endpoint != 'https://api.modal.com':
        raise RuntimeError('Only https://api.modal.com is permitted; no alternate hosts or fallback')
    os.environ['MODAL_SERVER_URL'] = endpoint
    import modal
    if modal.__version__ != SDK_VERSION:
        raise RuntimeError(f'This launcher is API-checked only with modal=={SDK_VERSION}')
    return modal


@contextmanager
def watchdog(seconds):
    """POSIX client ceiling; app context exits/cancels on alarm. Provider cap remains essential."""
    if seconds <= 0:
        raise TimeoutError('Shared 900-second session deadline expired')
    def expired(_signum, _frame):
        raise TimeoutError('Local action watchdog expired; no automatic retry')
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def runtime_image(modal):
    # No GPU for image construction. Official CUDA wheel index, no third-party wheels.
    return (modal.Image.debian_slim(python_version='3.12')
        .apt_install('ffmpeg', 'gcc', 'libc6-dev')
        .pip_install(f'torch=={TORCH}', index_url='https://download.pytorch.org/whl/cu126')
        .pip_install(f'triton=={TRITON}', 'einops==0.8.1', 'numpy==2.2.6', 'safetensors==0.5.3')
        .pip_install_from_requirements(str(ROOT / 'requirements-fla.txt'),
                                       extra_options='--no-deps --require-hashes')
        .run_commands('python -m pip check')
        .env({'PYTHONPATH': '/opt/project', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
              'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'}))


def mount_sources(image):
    for row in sources():
        image = image.add_local_file(ROOT / row['path'], '/opt/project/' + row['path'], copy=False)
    return image


def cpu_entry(job):
    from scripts.modal_worker_s0a import prepare
    return prepare(job)


def gpu_entry(job):
    from scripts.modal_worker_s0a import probe
    return probe(job)


def function_options(seconds):
    return dict(cpu=(CPU, CPU), memory=(MEM_MIB, MEM_MIB), max_containers=1,
                min_containers=0, buffer_containers=0, scaledown_window=SCALEDOWN_SECONDS,
                retries=0, timeout=seconds, startup_timeout=60, single_use_containers=True,
                include_source=False, serialized=True, block_network=True)


def plan():
    rate = GPU_RATE + CPU * CPU_RATE + (MEM_MIB / 1024) * MEM_RATE
    return {'scope': 'S0a only; shared Pupu24 + four full KDA; 476739917 parameters',
            'steps': 'BF16 parity at T=16,129; then one 1024-mel/10.24s forward/backward/manual SGD',
            'gpu': 'one L40S 48 GiB', 'gpu_worker_seconds': GPU_SECONDS,
            'session_absolute_seconds': SESSION_SECONDS, 'automatic_application_retries': 0,
            'max_containers': 1, 'cpu_cores': CPU, 'memory_MiB': MEM_MIB,
            'single_use_containers': True, 'min_containers': 0, 'scaledown_seconds': SCALEDOWN_SECONDS,
            'execution_baseline_usd': round(GPU_SECONDS * rate, 6),
            'gpu_startup_60s_plus_scaledown_2s_reserve_usd': round(62 * rate, 6),
            'cpu_prepare_60s_baseline_usd': round(60 * (CPU * CPU_RATE + 16 * MEM_RATE), 6),
            'build_storage_egress': 'Additional; included in workspace gross $2 ceiling, $1 target; not omitted or asserted free',
            'max_total_authorized_usd': AUTHORIZED_GROSS_CEILING_USD, 'target_gross_usd': TARGET_GROSS_USD, 'max_out_of_pocket_usd': 0,
            'billing_warning': 'Estimate is not a cap. Operator must verify account/credits and record evidence. No automatic billing changes.',
            'runtime': {'python': '3.12', 'torch': TORCH, 'triton': TRITON, 'fla-core': FLA,
                        'fla_wheel_sha256': FLA_HASH, 'einops': '0.8.1', 'numpy': '2.2.6',
                        'safetensors': '0.5.3', 'modal_sdk': SDK_VERSION},
            'source_files': sources(), 'remote_action_performed': False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', default='plan', choices=(
        'plan', 'create-manifest', 'upload-assets', 'prepare', 'probe', 'retrieve', 'cleanup'))
    p.add_argument('--run-id', default='s0a-higher-001')
    p.add_argument('--state', default=str(ROOT / 'reports/modal_session.json'))
    p.add_argument('--asset-manifest', default=str(ROOT / 'reports/modal_assets.json'))
    p.add_argument('--billing-record')
    p.add_argument('--candidate', default=str(ROOT / 'contracts/modal_probe_candidate.json'))
    p.add_argument('--audio', help='Local path override for the same SHA-pinned Higher audio; no substitute recording')
    p.add_argument('--checkpoint', default='teacher_encoder.safetensors')
    p.add_argument('--output-dir', default=str(ROOT / 'reports/modal_result'))
    p.add_argument('--approved-upload', action='store_true')
    p.add_argument('--approved-compute', action='store_true')
    p.add_argument('--approved-cleanup', action='store_true', help='Explicit approval to delete this run\'s uploaded files after retrieval')
    a = p.parse_args(argv)
    validate_run_id(a.run_id)
    if a.action == 'plan':
        print(json.dumps(plan(), indent=2)); return 0
    if a.action == 'create-manifest':
        print(json.dumps(create_manifest(a), indent=2)); return 0
    state_path = Path(a.state)
    if a.action in ('retrieve', 'cleanup'):
        state = read_json(state_path)
        if (state.get('schema') != 1 or state['run_id'] != a.run_id
                or state.get('volume_name') != 'beatfcn-shared-s0a-' + a.run_id):
            raise ValueError('Run-id/state/volume mismatch')
        if a.action == 'cleanup' and not a.approved_cleanup:
            p.error('cleanup requires explicit --approved-cleanup')
        if a.action == 'cleanup' and not state.get('retrieved'):
            raise ValueError('Retrieve available evidence before deleting run files')
        modal = require_sdk()
        volume = modal.Volume.from_name(state['volume_name'], create_if_missing=False)
        if a.action == 'retrieve':
            dest = Path(a.output_dir); dest.mkdir(parents=True, exist_ok=True)
            retrieved = []; missing = []
            for name in ARTIFACT_NAMES:
                data = bytearray()
                try:
                    for block in volume.read_file('/results/' + name):
                        data.extend(block)
                        if len(data) > 32 * 1024**2:
                            raise ValueError('Result exceeds retrieval size ceiling')
                except FileNotFoundError:
                    missing.append(name); continue
                (dest / name).write_bytes(data); retrieved.append(name)
            state['retrieved'] = {'at_unix': time.time(), 'files': retrieved, 'missing': missing}
            write_json(state_path, state)
            print(json.dumps(state['retrieved'], indent=2)); return 0
        if state.get('cleaned') or state.get('cleanup_attempt'):
            raise ValueError('Cleanup already attempted; inspect uncertain outcomes before any repeat')
        # Explicit deletion only of the per-run assets/results, never another run's volume.
        state['cleanup_attempt'] = {'at_unix': time.time(), 'status': 'pending_or_uncertain'}
        write_json(state_path, state)
        volume.remove_file('/', recursive=True)
        state['cleanup_attempt']['status'] = 'complete'
        state['cleaned'] = {'at_unix': time.time(), 'empty_volume_metadata_may_remain': True}
        write_json(state_path, state); print(json.dumps(state['cleaned'])); return 0
    if not a.approved_upload:
        p.error('Remote stages require explicit --approved-upload for allowlisted files')
    if a.action in ('prepare', 'probe') and not a.approved_compute:
        p.error('Compute requires explicit --approved-compute')
    if not a.billing_record:
        p.error('Remote stages require --billing-record with current operator evidence')
    billing = check_billing(read_json(a.billing_record))
    manifest = checked_manifest(a.asset_manifest, check_local_assets=a.action == 'upload-assets')
    if a.action == 'upload-assets':
        if state_path.exists():
            raise ValueError('Session already exists; do not repeat an uncertain upload')
        now = time.time()
        state = {'schema': 1, 'run_id': a.run_id, 'volume_name': 'beatfcn-shared-s0a-' + a.run_id,
                 'created_at_unix': now, 'deadline_unix': now + SESSION_SECONDS,
                 'manifest_sha256': canonical_hash(manifest), 'source_files': manifest['source_files'],
                 'billing': billing, 'attempts': {}, 'no_remote_credentials_mounted': True}
    else:
        state = read_json(state_path)
        if (state.get('schema') != 1 or state['run_id'] != a.run_id
                or state.get('volume_name') != 'beatfcn-shared-s0a-' + a.run_id
                or state['manifest_sha256'] != canonical_hash(manifest)):
            raise ValueError('Session/manifest mismatch')
        prerequisite = 'upload-assets' if a.action == 'prepare' else 'prepare'
        if state['attempts'].get(prerequisite, {}).get('status') != 'complete':
            raise ValueError('Required earlier stage did not complete; stop rather than retry')
    if a.action in state['attempts']:
        raise ValueError('Stage already attempted; uncertain/failed outcomes must be inspected, never retried automatically')
    remaining = state['deadline_unix'] - time.time()
    if remaining < 30:
        raise TimeoutError('Insufficient time remains in the absolute 900-second session')
    state['attempts'][a.action] = {'status': 'pending_or_uncertain', 'started_at_unix': time.time(), 'billing': billing}
    write_json(state_path, state)  # Reserve one attempt before any external call.
    try:
        modal = require_sdk()
        volume = modal.Volume.from_name(state['volume_name'], create_if_missing=a.action == 'upload-assets')
        if a.action == 'upload-assets':
            with watchdog(min(180, remaining)), volume.batch_upload(force=False) as batch:
                for row in manifest['assets']:
                    batch.put_file(row['local_path'], '/' + row['remote_path'])
            result = {'status': 'complete', 'uploaded_assets': len(manifest['assets']), 'private_per_run_volume': state['volume_name']}
        else:
            app = modal.App('beatfcn-shared-kda-s0a-' + a.run_id)
            job = {'run_id': a.run_id, 'volume_name': state['volume_name'],
                   'deadline_unix': state['deadline_unix'], 'source_files': manifest['source_files'],
                   'assets': [{k: v for k, v in r.items() if k != 'local_path'} for r in manifest['assets']],
                   'provenance': manifest['provenance'], 'probe': manifest['probe'], 'billing_scope': billing,
                   'expected_total_parameters': EXPECTED_PARAMS}
            if a.action == 'prepare':
                image = runtime_image(modal)
                fn = app.function(image=mount_sources(image), volumes={'/data': volume},
                                  **function_options(60))(cpu_entry)
                stage_limit = min(240, remaining)
            else:
                # No package install/image rebuild is permitted on the GPU path.
                image = modal.Image.from_id(state['image_id'])
                fn = app.function(image=mount_sources(image), gpu='L40S', volumes={'/data': volume},
                                  **function_options(GPU_SECONDS))(gpu_entry)
                stage_limit = min(GPU_SECONDS + 65, remaining)
            call = None
            with watchdog(stage_limit), modal.enable_output(), app.run():
                try:
                    if a.action == 'prepare':
                        state['image_id'] = image.object_id
                        write_json(state_path, state)
                    call = fn.spawn(job)
                    state['attempts'][a.action]['function_call_id'] = call.object_id
                    write_json(state_path, state)
                    result = call.get(timeout=max(1, min(stage_limit, state['deadline_unix'] - time.time()) - 5))
                except BaseException:
                    if call is not None:
                        call.cancel(terminate_containers=True)
                    raise
            if result.get('status') != 'passed':
                raise RuntimeError('Remote stage failed closed: ' + json.dumps(result))
        state['attempts'][a.action].update({'status': 'complete', 'ended_at_unix': time.time(), 'result': result})
        write_json(state_path, state)
        print(json.dumps(result, indent=2)); return 0
    except BaseException as exc:
        state['attempts'][a.action].update({'status': 'failed_or_uncertain', 'ended_at_unix': time.time(),
                                          'error_type': type(exc).__name__, 'error': str(exc),
                                          'automatic_retry': False})
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
