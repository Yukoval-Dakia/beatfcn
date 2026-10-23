import unittest
import numpy as np
from beatfcn.events.osu_mania import parse_osu
from beatfcn.events.phira import parse_phira
from beatfcn.ingest import offset as OFF
from beatfcn.ingest.timing import is_simple_ratio, segment_metrics, snap_deviation_sec, timing_reliability
from beatfcn.ingest.pipeline import assign_splits, finalize_rows
from events_helpers import FIX


def impulse_env(times_sec, dur=60.0):
    env = np.zeros(int(dur * 100))
    for t in times_sec:
        env[int(round(t * 100))] = 5.0
    return env


class OffsetEstimator(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.audio = np.sort(rng.choice(np.arange(100, 5800), 200, replace=False)) / 100.0

    def test_recovers_shift_sign(self):
        for shift in (-80, 0, 40, 120):
            heads = self.audio - shift / 1000.0   # notes early by `shift` ms -> delta = +shift
            d = OFF.estimate_offset(heads, env=impulse_env(self.audio))
            self.assertEqual(d['status'], 'ok')
            self.assertLessEqual(abs(d['delta_ms'] - shift), 5, d['delta_ms'])
            self.assertFalse(d['edge'])
            self.assertGreater(d['agree_delta'], 0.95)
            self.assertAlmostEqual(OFF.agree_at(d, d['delta_ms']), d['agree_delta'], places=3)

    def test_too_few_and_edge(self):
        self.assertEqual(OFF.estimate_offset(self.audio[:5], env=impulse_env(self.audio))['status'], 'too_few_notes')
        wide = np.convolve(impulse_env(self.audio[::4]), np.bartlett(81), 'same')  # +-400 ms bumps
        d = OFF.estimate_offset(self.audio[::4] - 0.3, env=wide)
        self.assertTrue(d['edge'])
        self.assertEqual(d['delta_ms'], 200)

    def test_flux_envelope_on_mel(self):
        mel = np.zeros((3000, 128), np.float16)
        for t in (500, 1000, 1500, 2000):
            mel[t:t + 20] = 3.0
        env = OFF.flux_envelope(mel)
        self.assertEqual(env.shape, (3000,))
        self.assertTrue(all(env[t - 1:t + 2].max() > 3 for t in (500, 1000, 1500, 2000)))


def diag(delta, conf=4.0, edge=False, a0=0.2, ar=None, n=100, at=None):
    counts = [0] * 401
    counts[200] = int(a0 * n)
    counts[at if at is not None else delta + 200] = int((ar if ar is not None else 0.8) * n)
    return {'status': 'ok', 'n': n, 'delta_ms': delta, 'conf': conf, 'edge': edge, 'agree_counts': counts}


class OffsetDecision(unittest.TestCase):
    def test_keep_within_25_regardless_of_conf(self):
        d = OFF.decide(diag(30, conf=1.0), c_ms=10)
        self.assertEqual((d['action'], d['shift_ms'], d['r_ms']), ('keep', 0, 20))

    def test_shift(self):
        d = OFF.decide(diag(60, a0=0.2, ar=0.8, at=50 + 200), c_ms=10)
        self.assertEqual((d['action'], d['shift_ms']), ('shift', 50))
        self.assertEqual(d['agree_after'], 0.8)

    def test_quarantine_reasons(self):
        self.assertEqual(OFF.decide(diag(150), 0)['reasons'], ['offset_residual_too_large'])
        self.assertEqual(OFF.decide(diag(60, conf=2.5), 0)['reasons'], ['offset_low_conf'])
        self.assertEqual(OFF.decide(diag(60, a0=0.7, ar=0.8), 0)['reasons'], ['offset_insufficient_gain'])
        self.assertEqual(OFF.decide(diag(60, a0=0.0, ar=0.25), 0)['reasons'], ['offset_low_agree_after'])
        self.assertIn('offset_search_edge', OFF.decide(diag(200, edge=True), 0)['reasons'])
        self.assertEqual(OFF.decide({'status': 'too_few_notes', 'n': 3}, 0)['reasons'], ['offset_too_few_notes'])

    def test_disagreement(self):
        f = OFF.disagreement_flags([('a', 'x', 0), ('b', 'x', 30), ('c', 'y', 0), ('d', 'y', 20), ('e', 'x', None)])
        self.assertEqual(f, {'a', 'b'})


class Timing(unittest.TestCase):
    def test_simple_ratios(self):
        for r in (0.5, 2 / 3, 1.0, 1.5, 2.0, 170 / 255):
            self.assertTrue(is_simple_ratio(r), r)
        for r in (1.1, 0.8, 3.0):
            self.assertFalse(is_simple_ratio(r), r)

    def test_segment_metrics_drops_zero_length(self):
        m = segment_metrics([(0, 120), (8, 150), (8, 180), (8.5, 120), (20, 97)])
        self.assertEqual(m['n_segments'], 4)
        self.assertEqual(m['min_segment_beats'], 0.5)
        self.assertEqual(m['n_nonsimple_ratios'], 1)   # 120->180 = 3/2, 180->120 = 2/3, 120->97 non-simple

    def test_snap_relative_to_segment(self):
        beat = np.array([10.3 + 0.25, 10.3 + 1 / 3, 10.3 + 0.1])
        dev = snap_deviation_sec(beat, np.full(3, 10.3), np.full(3, 120.0))
        self.assertTrue(dev[0] < 1e-9 and dev[1] < 1e-9 and dev[2] > 0.003)

    def test_fixture_charts(self):
        osu = timing_reliability(parse_osu((FIX / 'mania4k_synthetic.osu').read_bytes()))
        self.assertEqual(osu['snap_rate'], 1.0)
        self.assertEqual(osu['n_segments'], 2)
        rpe = timing_reliability(parse_phira((FIX / 'rpe_multibpm_synthetic.json').read_bytes()))
        self.assertIsNotNone(rpe['snap_rate'])
        pgr = timing_reliability(parse_phira((FIX / 'pgr_v3_synthetic.json').read_bytes()))
        self.assertEqual(pgr['n_segments'], 1)
        for t in (osu, rpe, pgr):
            self.assertEqual(t['unreliable'], bool(t['triggers']))


def row(cid, src, title, sha, delta, codec='mp3', title_alt=''):
    return {'chart_id': cid, 'source': src, 'title': title, 'title_alt': title_alt, 'audio_sha256': sha,
            'codec': codec, 'status': 'parsed', 'p0_quarantine': [], 'validation_quarantine': [],
            'offset': diag(delta) if delta is not None else {'status': 'too_few_notes', 'n': 1},
            'timing': {'unreliable': False, 'triggers': []}}


class GroupAndFinalize(unittest.TestCase):
    def test_cross_source_title_and_audio_union(self):
        rows = [row('osu:1', 'osu', '雲鶴サクリファイス', 'a', 0, title_alt='Unkaku Sacrifice'),
                row('osu:2', 'osu', 'Other', 'a', 0),
                row('phira:9', 'phira', '雲鶴サクリファイス (TV size)', 'b', 0),
                row('phira:10', 'phira', 'unkaku sacrifice', 'c', 0),
                row('phira:11', 'phira', 'Different', 'd', 0)]
        _, c2g, _ = assign_splits(rows)
        self.assertEqual(len({c2g[c] for c in ('osu:1', 'osu:2', 'phira:9', 'phira:10')}), 1)
        self.assertNotEqual(c2g['phira:11'], c2g['osu:1'])

    def test_finalize_c_from_train_only(self):
        rows = [row(f'osu:{i}', 'osu', f'song {i}', f's{i}', 10) for i in range(200)]
        rows.append(row('osu:x', 'osu', 'few', 'sx', None))
        out, ctab = finalize_rows(rows)
        self.assertEqual(ctab['osu|mp3']['c_ms'], 10)
        self.assertEqual(ctab['osu|mp3']['n_train'], sum(r['split'] == 'train' for r in out) - (out[-1]['split'] == 'train'))
        self.assertTrue(all(r['kept'] for r in out[:-1]))
        self.assertEqual(out[-1]['quarantine_reasons'], ['offset_too_few_notes'])
        self.assertEqual({r['split'] for r in out}, {'train', 'val', 'test'})


if __name__ == '__main__':
    unittest.main()
