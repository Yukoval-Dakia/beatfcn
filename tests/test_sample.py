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
