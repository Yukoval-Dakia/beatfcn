"""Offline-only launcher contract tests. No authentication, SDK requests, or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

# macOS /tmp and /var are symlinks; the launcher rejects symlinked paths by design.
REAL_TMP = Path(tempfile.gettempdir()).resolve()

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

launcher = load('modal_launch_s0a')
worker = load('modal_worker_s0a')

class LauncherContract(unittest.TestCase):
    def billing(self):
        return dict(schema=1, operator_launch_clearance=True, account_read_check_completed=True,
                    account_checked_at_unix=1000, remaining_credits_usd=30, gross_usage_so_far_usd=0,
                    workspace_usage_cap_usd=1, net_spend_limit_usd=0,
                    cap_scope='workspace_all_usage_before_credits', credits_evidence='account read',
                    limit_evidence='user confirmed settings', limit_evidence_status='user_reported',
                    other_workloads_accounted_for=True)

    def test_help_plan_and_import_are_local(self):
        for args in (['--help'], ['plan']):
            code = ('import socket,runpy,sys; '
                    'socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError("network")); '
                    f'sys.argv={[str(ROOT / "scripts/modal_launch_s0a.py"), *args]!r}; '
                    f'runpy.run_path({str(ROOT / "scripts/modal_launch_s0a.py")!r},run_name="__main__")')
            p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
            self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn('modal', sys.modules)

    def test_exact_compute_limits(self):
        opts = launcher.function_options(600)
        self.assertEqual(opts['timeout'], 600)
        self.assertEqual(opts['max_containers'], 1)
        self.assertEqual(opts['min_containers'], 0)
        self.assertEqual(opts['buffer_containers'], 0)
        self.assertEqual(opts['retries'], 0)
        self.assertEqual(opts['cpu'], (2, 2))
        self.assertEqual(opts['memory'], (16384, 16384))
        self.assertTrue(opts['single_use_containers'])
        self.assertFalse(opts['include_source'])
        self.assertTrue(opts['block_network'])
        self.assertEqual(launcher.SESSION_SECONDS, 900)
        self.assertEqual(launcher.plan()['execution_baseline_usd'], .362232)

    def test_billing_requires_fresh_full_scope_evidence(self):
        valid = self.billing()
        self.assertEqual(launcher.check_billing(valid, now=1001), valid)
        for change in ({'net_spend_limit_usd': 1}, {'workspace_usage_cap_usd': 3},
                       {'cap_scope': 'environment_compute_only'}, {'remaining_credits_usd': .5},
                       {'operator_launch_clearance': False}, {'account_read_check_completed': False},
                       {'account_checked_at_unix': 0}, {'gross_usage_so_far_usd': .51},
                       {'remaining_credits_usd': float('nan')}, {'limit_evidence_status': 'assumed'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                launcher.check_billing({**valid, **change}, now=1001)
        verified = {**valid, 'limit_evidence_status': 'provider_verified'}
        launcher.check_billing(verified, now=1001)
        launcher.check_billing({**verified, 'workspace_usage_cap_usd': 2}, now=1001)

    def test_explicit_permissions_precede_sdk(self):
        with patch.object(launcher, 'require_sdk', side_effect=AssertionError('must not reach SDK')):
            for action in ('upload-assets', 'prepare', 'probe', 'cleanup'):
                with self.subTest(action=action), self.assertRaises((SystemExit, FileNotFoundError)):
                    launcher.main([action, '--state', str(REAL_TMP / 'does-not-exist-s0a-test.json')])

    def test_reject_alternate_modal_endpoint(self):
        with patch.dict(os.environ, {'MODAL_SERVER_URL': 'https://api.modal2.com'}):
            with self.assertRaisesRegex(RuntimeError, 'Only https://api.modal.com'):
                launcher.require_sdk()

    def test_allowlist_no_workspace_auth_or_weights(self):
        rows = launcher.sources()
        self.assertEqual(len(rows), len(launcher.SOURCE_NAMES))
        self.assertTrue(all(r['bytes'] < 256 * 1024 for r in rows))
        self.assertTrue(all(not any(x in r['path'] for x in ('.env', '.modal', '.safetensors', 'auth', 'token')) for r in rows))
        for name in ('../escape', 'BadUppercase', '', 'x/y'):
            with self.assertRaises(ValueError):
                launcher.validate_run_id(name)

    def test_source_symlink_denied(self):
        with tempfile.TemporaryDirectory(dir=REAL_TMP) as d:
            a = Path(d) / 'a.py'; a.write_text('pass\n')
            launcher.safe_file(a)  # the real file passes; only the link must fail
            b = Path(d) / 'b.py'; b.symlink_to(a)
            with self.assertRaises(ValueError):
                launcher.safe_file(b)

    def test_manifest_scope_rejects_before_reading_assets(self):
        m = {'schema': 1, 'source_files': launcher.sources(), 'assets': []}
        with tempfile.TemporaryDirectory(dir=REAL_TMP) as d:
            p = Path(d) / 'manifest.json'; p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'Exactly'):
                launcher.checked_manifest(p, check_local_assets=False)

    def test_no_reference_backend_or_train_action(self):
        self.assertNotIn("'train'", (ROOT / 'scripts/modal_launch_s0a.py').read_text())
        text = (ROOT / 'scripts/modal_worker_s0a.py').read_text()
        self.assertIn("'--backend', 'fla'", text)
        self.assertNotIn("'--backend', 'reference'", text)
        self.assertIn("'--mel-frames', '1024'", text)
        self.assertIn("'--steps', '1'", text)
        self.assertIn("'--audio-offset-seconds', '15'", text)

    def test_expired_process_does_not_start(self):
        with patch.object(worker.subprocess, 'Popen', side_effect=AssertionError('must not start')):
            with self.assertRaises(TimeoutError):
                worker.run_process(['unused'], time.time() - 1, Path('/tmp/unused-log'))

    def test_failed_parity_never_starts_model(self):
        class Volume:
            def commit(self):
                pass
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            calls = []
            def run(argv, deadline, log):
                calls.append(argv)
                (dest / 'fla_parity.json').write_text(json.dumps({'passed': False, 'tested_lengths': [16, 129]}))
            job = {'run_id': 'test', 'deadline_unix': time.time() + 600,
                   'expected_total_parameters': 476739917, 'billing_scope': {}, 'provenance': {},
                   'source_files': []}
            with patch.object(worker, 'RESULTS', dest), patch.object(worker, 'claim', return_value=Volume()), \
                 patch.object(worker, 'verify_sources'), patch.object(worker, 'verify_assets', return_value=[]), \
                 patch.object(worker, 'runtime', return_value={'gpu': 'L40S'}), patch.object(worker, 'run_process', side_effect=run):
                result = worker.probe(job)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], 'scripts/check_fla.py')
            self.assertTrue((dest / 'launcher_status.json').is_file())

if __name__ == '__main__':
    unittest.main()
