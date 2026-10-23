"""Whole-song quality metrics of design/FIRST_CANDIDATE_v2.md §1 on deterministic synthetic charts."""
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from beatfcn.events.canonical import Event, LaneLoc
from beatfcn.events.requests import build_request
from beatfcn.model.evaluation import (QUALITY_MEANS, density_curve, editable_union, evaluate_songs, ioi_histogram_kl,
                                      onset_error_distribution, quality_metrics, repetition_consistency, style_variance)
from beatfcn.model.p1 import P1Model
from events_helpers import DURATION, _chart, synthetic_mania, synthetic_phira
from test_p1_model import CFG, song

SEC = 1_000_000
LONG = 120 * SEC
PERIOD = 8 * SEC
PERIODS = LONG // PERIOD  # 15 segments of 8 s


def periodic_mania():
    """Two 8 s motifs A/B alternating (chart period 16 s) on a 250 ms grid; collision-free.
    A and B never share (slot, lane): their lane-aware onset F1 is exactly 0."""
    evs = []
    for p in range(PERIODS):
        m = p % 2
        for i in range(32):
            if (i + m) % 3 == 0:
                continue
            t = p * PERIOD + i * 250_000
            lane = (3 * i + 3 * m) % 4
            if (i + m) % 5 == 0:
                evs.append(Event(f'h:{len(evs)}', t, t + 200_000, 'hold', LaneLoc(lane)))
            else:
                evs.append(Event(f't:{len(evs)}', t, t, 'tap', LaneLoc(lane)))
            if m == 0 and i % 8 == 4:
                evs.append(Event(f'c:{len(evs)}', t, t, 'tap', LaneLoc((lane + 2) % 4)))
    return _chart('mania_4k', evs)


def periodic_summary(seed=0):
    """S20-like rows [S, 8, D]: four distinct row vectors per motif, so windows with the same offset and the
    same motif phase (|dt| a multiple of 16 s) are the only pairs with cosine 1."""
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((2, 4, 8, 32)).astype(np.float32)
    return np.stack([rows[(r // 4) % 2, r % 4] for r in range(LONG // (2 * SEC))])


def moved(events, dt_us=0, lanes=0):
    return [Event(e.event_id, e.onset_us + dt_us, e.end_us + dt_us, e.type,
                  LaneLoc((e.location.lane + lanes) % 4)) for e in events]


class QualityMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chart = periodic_mania()
        sample = build_request(chart, kind='from_scratch', seed=0, audio_id='a', audio_duration_us=LONG,
                               empty_overlay=None, control_dropout=False)
        assert sample.ok
        cls.truth = [e for g in sample.model_view.target_groups for e in g.events]
        cls.regions = sample.request['regions']
        cls.runs = editable_union(cls.regions)
        cls.summary = periodic_summary()
        assert len(cls.truth) == len(chart.events)

    def test_identical_chart_scores_perfectly(self):
        q = quality_metrics(self.truth, self.truth, self.regions, 'mania_4k', self.summary)
        oe = q['onset_error_50ms']
        self.assertEqual((oe['match_rate'], oe['median_ms'], oe['p90_ms']), (1.0, 0.0, 0.0))
        self.assertEqual(oe['matched'], len(self.truth))
        dc = q['density_curve']
        self.assertEqual(dc['blocks'], LONG // (2 * SEC))
        self.assertAlmostEqual(dc['pearson_r'], 1.0, places=9)
        self.assertEqual(dc['count_ratio'], 1.0)
        self.assertEqual(q['ioi_kl']['kl_nats'], 0.0)
        self.assertGreater(q['ioi_kl']['true'], 0)
        rep = q['repetition']
        self.assertEqual(rep['source'], 'summary_s20')
        self.assertEqual(rep['kept_pairs'], math.ceil(0.05 * rep['candidate_pairs']))
        self.assertGreaterEqual(rep['human_reuse_pairs'], 3)
        self.assertEqual(rep['p_model_given_human'], 1.0)
        # Only the 16 s-periodic pairs tie at cosine 1 in this audio, so the top 5 % are all human reuses.
        self.assertEqual(rep['human_no_reuse_pairs'], 0)
        self.assertIsNone(rep['p_model_given_not_human'])
        sv = q['style_variance']
        self.assertEqual((sv['segments_generated'], sv['segments_true']), (PERIODS, PERIODS))
        self.assertEqual(sv['hold_frac_var_generated'], sv['hold_frac_var_true'])
        self.assertEqual(sv['chord_frac_var_generated'], sv['chord_frac_var_true'])
        self.assertGreater(sv['hold_frac_var_true'], 0)
        self.assertGreater(sv['chord_frac_var_true'], 0)
        json.dumps(q, allow_nan=False)

    def test_repetition_pairs_are_audio_defined_and_lane_aware(self):
        # Every candidate pair kept: 57 windows at a 2 s stride, |dt| >= 16 s -> 1225 pairs; the chart repeats
        # every 16 s (8 rows), so exactly the pairs with dt in {16, 32, ..., 112} s are human reuses.
        rep = repetition_consistency(self.truth, self.truth, self.runs, 'mania_4k', self.summary, top_fraction=1.0)
        self.assertEqual(rep['windows'], 57)
        self.assertEqual((rep['candidate_pairs'], rep['kept_pairs']), (1225, 1225))
        self.assertEqual(rep['human_reuse_pairs'], sum(57 - d for d in range(8, 57, 8)))
        self.assertEqual((rep['p_model_given_human'], rep['p_model_given_not_human']), (1.0, 0.0))
        # Lanes rotated in every second 16 s: mania sees the rotation (pairs 32 s apart still match, 16 s apart
        # do not), Phira ignores lanes.
        rotated = [e for p in range(PERIODS) for e in moved([x for x in self.truth if x.onset_us // PERIOD == p],
                                                              lanes=(p // 2) % 2)]
        rep = repetition_consistency(rotated, self.truth, self.runs, 'mania_4k', self.summary, top_fraction=1.0)
        self.assertEqual(rep['human_reuse_pairs'], 175)
        self.assertAlmostEqual(rep['p_model_given_human'], 75 / 175)
        # Lane-agnostic, the rotated chart has exactly the truth's rhythm: it reuses wherever the human does
        # (this motif also repeats lane-agnostically under 2 s shifts, so more pairs count as human reuses).
        rep = repetition_consistency(rotated, self.truth, self.runs, 'phira', self.summary, top_fraction=1.0)
        self.assertGreater(rep['human_reuse_pairs'], 175)
        self.assertEqual((rep['p_model_given_human'], rep['p_model_given_not_human']), (1.0, 0.0))
        # A random chart does not reuse where the human does.
        random_chart = list(synthetic_mania(seed=3, duration=LONG).events)
        rep = repetition_consistency(random_chart, self.truth, self.runs, 'mania_4k', self.summary)
        self.assertGreaterEqual(rep['human_reuse_pairs'], 3)
        self.assertLess(rep['p_model_given_human'], 0.5)
        # Too few pairs (30 s of audio) and no audio at all: probabilities None, counts kept.
        short = repetition_consistency(self.truth, self.truth, [(0, 30 * SEC)], 'mania_4k', self.summary[:15])
        self.assertEqual((short['windows'], short['candidate_pairs'], short['kept_pairs']), (12, 10, 0))
        self.assertIsNone(short['p_model_given_human'])
        none = repetition_consistency(self.truth, self.truth, self.runs, 'mania_4k', None)
        self.assertIsNone(none['source'])
        self.assertIsNone(none['p_model_given_human'])

    def test_shifted_and_thinned_charts_degrade_sensibly(self):
        shifted = moved(self.truth, 30_000)
        oe = onset_error_distribution(shifted, self.truth, self.runs)
        self.assertEqual((oe['match_rate'], oe['median_ms'], oe['p90_ms']), (1.0, 30.0, 30.0))
        far = onset_error_distribution(moved(self.truth, 60_000), self.truth, self.runs)
        self.assertEqual((far['matched'], far['match_rate'], far['median_ms']), (0, 0.0, None))
        # Nearest matching: a close duplicate takes the pair, the far one stays unmatched.
        truth = [Event('t', SEC, SEC, 'tap', LaneLoc(0))]
        pred = [Event('a', SEC + 40_000, SEC + 40_000, 'tap', LaneLoc(0)), Event('b', SEC + 5_000, SEC + 5_000, 'tap', LaneLoc(0))]
        oe = onset_error_distribution(pred, truth, [(0, 2 * SEC)])
        self.assertEqual((oe['matched'], oe['pred'], oe['median_ms']), (1, 2, 5.0))
        thinned = self.truth[::2]
        oe = onset_error_distribution(thinned, self.truth, self.runs)
        self.assertAlmostEqual(oe['match_rate'], len(thinned) / len(self.truth))
        dc = density_curve(thinned, self.truth, self.runs)
        self.assertAlmostEqual(dc['count_ratio'], len(thinned) / len(self.truth))
        self.assertLess(dc['pearson_r'], 1.0)
        self.assertGreater(dc['pearson_r'], 0.0)
        half = [e for e in self.truth if e.onset_us < LONG // 2]
        dh = density_curve(half, self.truth, self.runs)
        self.assertAlmostEqual(dh['count_ratio'], len(half) / len(self.truth))
        self.assertLess(dh['pearson_r'], dc['pearson_r'])
        self.assertIsNone(density_curve(self.truth, self.truth, [(0, 10 * SEC)])['pearson_r'])  # 5 blocks < 10
        kl = ioi_histogram_kl(thinned, self.truth, self.runs)
        self.assertGreater(kl['kl_nats'], 0.05)
        self.assertLess(kl['generated'], kl['true'])
        self.assertIsNone(ioi_histogram_kl(self.truth, [], self.runs)['kl_nats'])
        no_holds = [Event(e.event_id, e.onset_us, e.onset_us, 'tap', e.location) for e in self.truth]
        sv = style_variance(no_holds, self.truth, self.runs)
        self.assertEqual(sv['hold_frac_var_generated'], 0.0)
        self.assertGreater(sv['hold_frac_var_true'], 0.0)
        self.assertIsNone(style_variance(self.truth[:3], self.truth, [(0, 8 * SEC)])['hold_frac_var_generated'])

    def test_whole_song_evaluation_reports_quality_per_song_and_per_mode(self):
        charts = {'mania': synthetic_mania(), 'phira': synthetic_phira()}
        rows = [{'chart_id': key, 'audio_sha256': key, 'mode': chart.mode,
                 'split': 'val', 'audio_sec': DURATION / 1e6} for key, chart in charts.items()]
        cache = song(0)
        corpus = SimpleNamespace(train_rows=lambda rows: rows, chart=charts.__getitem__,
                                 song=lambda key, in_memory: cache)
        with patch('beatfcn.model.sample._pick', return_value=0):
            result = evaluate_songs(P1Model(CFG).eval(), corpus, rows, n_songs=2)
        self.assertTrue(result['valid_for_comparison'])
        for record in result['songs']:
            q = record['quality']
            self.assertEqual(set(q), set(QUALITY_MEANS))
            self.assertEqual(q['onset_error_50ms']['match_rate'], 0.0)  # END chosen at once: nothing generated
            self.assertEqual(q['density_curve']['count_ratio'], 0.0)
            self.assertEqual(q['repetition']['source'], 'summary_s20')
            self.assertGreater(q['ioi_kl']['kl_nats'], 0.0)
        for mode, s in result['summary'].items():
            for section, fields in QUALITY_MEANS.items():
                for name in fields:
                    self.assertEqual(set(s[section][name]), {'mean', 'songs'})
            self.assertEqual(s['onset_error_50ms']['match_rate'], {'mean': 0.0, 'songs': 1})
            self.assertEqual(s['onset_error_50ms']['median_ms'], {'mean': None, 'songs': 0})
            self.assertEqual(s['repetition']['kept_pairs_total'], 0)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
