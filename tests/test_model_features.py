import dataclasses
import unittest
import numpy as np
from beatfcn.events.requests import build_request, KINDS
from beatfcn.model import features as F
from events_helpers import synthetic_mania, synthetic_phira, DURATION


def cores(chart, seeds=range(12)):
    for kind in KINDS:
        for seed in seeds:
            try:
                s = build_request(chart, kind=kind, seed=seed, audio_id='a', audio_duration_us=DURATION)
            except (ValueError, AssertionError):
                continue
            if not s.ok:
                continue
            rng = np.random.default_rng(seed)
            try:
                c0 = F.choose_core(s, rng, DURATION)
                yield s, c0, F.build_core(s, chart.events, c0, DURATION)
            except F.SkipExample:
                continue


class FeatureTests(unittest.TestCase):
    def test_targets_are_legal_and_ordered(self):
        n = 0
        for chart in (synthetic_mania(1), synthetic_phira(2)):
            for s, c0, ex in cores(chart):
                prev = -1
                for g in ex.groups:
                    self.assertTrue(0 <= g['t_rel_us'] < F.CORE_US)
                    self.assertGreater(g['t_rel_us'], prev)
                    prev = g['t_rel_us']
                    self.assertTrue(ex.bin_ok[g['bin'], g['residual']], (s.recipe['kind'], g))
                    if g['copy'] >= 0:
                        self.assertEqual(ex.copy_times[g['copy']], g['t_rel_us'])
                    for nt in g['notes']:
                        if 'end_rel_us' in nt:
                            lo, hi = nt['end_ok_us']
                            self.assertTrue(lo <= nt['end_rel_us'] <= hi, nt)
                    n += 1
        self.assertGreater(n, 50)

    def test_hidden_targets_never_in_prefix(self):
        checked = 0
        for chart in (synthetic_mania(3), synthetic_phira(4)):
            for s, c0, ex in cores(chart):
                if not ex.groups:
                    continue
                later = [h for h in s.hidden if h.onset_us >= c0]
                # Deleting every hidden target at/after c0 must not change any prefix array. Controls are
                # excluded on purpose: P0 derives them (as bands) from the full chart by design (spec §8).
                kept = tuple(h for h in s.hidden if h.onset_us < c0)
                ex2 = F.build_core(dataclasses.replace(s, hidden=kept), chart.events, c0, DURATION)
                for name in ('known', 'refs', 'cell_state', 'bin_ok', 'copy_times', 'summary_stats'):
                    np.testing.assert_array_equal(getattr(ex, name), getattr(ex2, name), err_msg=name)
                gen = ex.known[ex.known[:, 7] == 1]
                self.assertTrue((gen[:, 0] < 0).all(), 'generated context must precede the core')
                self.assertTrue(later)
                checked += 1
        self.assertGreater(checked, 10)

    def test_generated_context_flags(self):
        for s, c0, ex in cores(synthetic_mania(5)):
            obs = ex.known[ex.known[:, 7] == 0]
            np.testing.assert_array_equal(obs[:, 8], (obs[:, 0] >= 0).astype(np.float32))


if __name__ == '__main__':
    unittest.main()


class PoolTests(unittest.TestCase):
    def test_resident_pool_reads_one_song_before_first_draw_and_grows(self):
        from beatfcn.train.loop import CoreSampler
        from events_helpers import synthetic_mania, DURATION
        chart = synthetic_mania(0)
        rows = [{'chart_id': f'c{i}', 'song': f's{i}', 'mode': 'mania_4k', 'duration_us': DURATION} for i in range(20)]
        reads = []
        smp = CoreSampler(rows, lambda cid: chart, lambda sha: reads.append(sha) or object(), 0, pool=6, residence=16)
        smp.draw(4)
        self.assertEqual(len(reads), 1)
        for _ in range(10):
            smp.draw(4)
        self.assertEqual(len(smp.resident['mania_4k']), 6)


class SamplerRejectsOversizedGroups(unittest.TestCase):
    def test_core_with_more_than_max_notes_is_skipped_not_crashed(self):
        from unittest import mock
        from beatfcn.train.loop import CoreSampler
        from beatfcn.model.batch import MAX_NOTES
        from events_helpers import synthetic_mania, DURATION
        chart = synthetic_mania(0)
        rows = [{'chart_id': 'c0', 'song': 's0', 'mode': 'mania_4k', 'duration_us': DURATION}]
        smp = CoreSampler(rows, lambda cid: chart, lambda sha: object(), 0)
        real = F.build_core

        def oversized(*a, **k):
            ex = real(*a, **k)
            if not ex.groups:  # only cores with an oversized group reach the sampler's check
                raise F.SkipExample('empty core in fixture')
            ex.groups[0]['notes'] = ex.groups[0]['notes'][:1] * (MAX_NOTES + 1)
            return ex
        with mock.patch.object(F, 'build_core', oversized):
            with self.assertRaises(RuntimeError):  # every draw is skipped, never a crash from collate
                smp.draw(1)
        self.assertTrue(any('MAX_NOTES' in k for k in smp.skipped), smp.skipped)


class HoldContextTests(unittest.TestCase):
    """Hold limits and busy lanes use known context only, never the hidden future or the event's own copy."""

    def _core(self, evs, c0, kind='from_scratch'):
        from beatfcn.events.canonical import Event, LaneLoc
        from beatfcn.events.requests import build_request
        from events_helpers import _chart
        chart = _chart('mania_4k', [Event(f'h{i}', a, b, 'hold' if b > a else 'tap', LaneLoc(l), f'x#{i}') for i, (a, b, l) in enumerate(evs)])
        s = build_request(chart, kind=kind, seed=0, audio_id='a', audio_duration_us=DURATION, control_dropout=False)
        return F.build_core(s, chart.events, c0, DURATION)

    def test_hold_limit_ignores_its_own_unquantized_copy(self):
        ex = self._core([(1_000_400, 2_000_400, 0), (6_000_000, 6_000_000, 1)], 0)
        hold = next(n for g in ex.groups for n in g['notes'] if 'end_rel_us' in n)
        self.assertGreater(hold['end_ok_us'][1], 1_000_000)
        self.assertEqual(F.check_targets(ex), [])

    def test_hold_limit_does_not_depend_on_hidden_future(self):
        a = self._core([(1_000_000, 2_000_000, 0), (3_000_000, 3_000_000, 0)], 0)
        b = self._core([(1_000_000, 2_000_000, 0), (5_000_000, 5_000_000, 0)], 0)
        ha = next(n for g in a.groups for n in g['notes'] if 'end_rel_us' in n)
        hb = next(n for g in b.groups for n in g['notes'] if 'end_rel_us' in n)
        self.assertEqual(ha['end_ok_us'], hb['end_ok_us'])

    def test_active_hold_from_before_the_halo_is_known_and_busy(self):
        ex = self._core([(1_000_000, 12_000_000, 0), (9_000_000, 9_000_000, 1)], 8_000_000)
        self.assertEqual(len(ex.known), 1)  # the long hold is generated context although it started at 1 s
        g = next(g for g in ex.groups if g['t_rel_us'] == 1_000_000)
        self.assertTrue(bool(g['busy_lanes'][0]))
