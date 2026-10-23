import dataclasses
import unittest
import numpy as np
import torch
from beatfcn.events.requests import build_request
from beatfcn.model import features as F
from beatfcn.model.p1 import P1Model
from beatfcn.model.sample import generate_core, core_context, match_f1
from events_helpers import synthetic_mania, synthetic_phira, DURATION
from test_p1_model import CFG, song


class SampleTests(unittest.TestCase):
    def test_generation_is_legal(self):
        torch.manual_seed(0)
        m = P1Model(CFG).eval()
        n_groups = 0
        for ci, chart in enumerate((synthetic_mania(7), synthetic_phira(8))):
            li = F.line_index_of(chart.events)
            for seed in range(3):
                s = build_request(chart, kind='middle_holes', seed=seed, audio_id='a', audio_duration_us=DURATION)
                c0 = F.choose_core(s, np.random.default_rng(seed), DURATION)
                ex = F.build_core(s, chart.events, c0, DURATION)
                holds, runs, heads = core_context(s, ex, li)
                groups = generate_core(m, ex, song(ci), holds, runs, heads, seed=seed, temperature=1.0)
                prev = -1
                for g in groups:
                    t = g['t_rel_us']
                    self.assertGreater(t, prev)
                    prev = t
                    self.assertTrue(ex.bin_ok[t // F.BIN_US, t % F.BIN_US // F.MS])
                    lanes = [n['lane'] for n in g['notes'] if n['lane'] >= 0]
                    self.assertEqual(len(lanes), len(set(lanes)))
                    self.assertFalse(any(g['busy_lanes'][l] for l in lanes))
                    for n in g['notes']:
                        if 'end_rel_us' in n:
                            end = c0 + t + n['end_rel_us']
                            self.assertGreater(end, c0 + t)
                            self.assertTrue(any(a <= c0 + t and end <= b for a, b in runs))
                    n_groups += 1
        self.assertGreater(n_groups, 5)

    def test_match_f1(self):
        self.assertEqual(match_f1([0, 100, 205], [0, 110, 400], 10)[:3], (2, 3, 3))
        self.assertEqual(match_f1([], [], 10)[3], 1.0)


if __name__ == '__main__':
    unittest.main()


class IncrementalDecodingTests(unittest.TestCase):
    """encode_step (K/V cache) must reproduce the full re-encode row by row, and greedy generation must match."""

    def _example(self):
        import numpy as np
        from beatfcn.events.requests import build_request
        from beatfcn.model import features as F
        from test_p1_model import song as mk_song
        from events_helpers import synthetic_mania, DURATION
        chart = synthetic_mania(2)
        for k in range(40):
            try:
                s = build_request(chart, kind='from_scratch', seed=300 + k, audio_id='a', audio_duration_us=DURATION)
                ex = F.build_core(s, chart.events, F.choose_core(s, np.random.default_rng(k), DURATION), DURATION)
                if len(ex.groups) >= 4:
                    return ex, mk_song(1), s
            except (ValueError, AssertionError):
                continue
        self.fail('no example with >= 4 groups')

    def test_encode_step_matches_full_encode_rows(self):
        import copy
        import numpy as np
        from beatfcn.model.batch import collate, dequant
        from beatfcn.model.p1 import P1Model
        from test_p1_model import CFG
        ex, song, _ = self._example()
        torch.manual_seed(0)
        m = P1Model(dataclasses.replace(CFG, dropout=0.0, selection=True)).eval()
        full_bt = dequant(collate([ex], [song], CFG.taps, CFG.top_tap))
        with torch.no_grad():
            h_full, pa_full, _ = m.encode(full_bt)
            intent_full = m._intent_h.clone()
            cache = {}
            for k in range(4):
                work = copy.copy(ex)
                work.groups = ex.groups[:k]
                work.right = ex.right[:k + 1]
                bt = dequant(collate([work], [song], CFG.taps, CFG.top_tap))
                if k == 0:
                    h, pa, _ = m.encode(bt, cache)
                    row = h[:, 0:1]
                else:
                    row, pa, _ = m.encode_step(bt, cache)
                self.assertTrue(torch.allclose(row, h_full[:, k:k + 1], atol=2e-5), f'row {k}')
            self.assertTrue(torch.allclose(pa, pa_full, atol=1e-6))
            self.assertTrue(torch.allclose(m._intent_h, intent_full, atol=1e-6))

    def test_greedy_generation_identical_with_cache(self):
        from beatfcn.model.p1 import P1Model
        from beatfcn.model.sample import generate_core, core_context
        from test_p1_model import CFG
        ex, song, s = self._example()
        torch.manual_seed(0)
        m = P1Model(dataclasses.replace(CFG, dropout=0.0)).eval()
        holds, runs, heads = core_context(s, ex, {})
        a = generate_core(m, ex, song, holds, runs, heads, greedy=True, seed=1, incremental=False)
        b = generate_core(m, ex, song, holds, runs, heads, greedy=True, seed=1, incremental=True)
        self.assertEqual([(g['t_rel_us'], [(n['type'], n['lane']) for n in g['notes']]) for g in a],
                         [(g['t_rel_us'], [(n['type'], n['lane']) for n in g['notes']]) for g in b])
        self.assertGreater(len(a), 0)
