import copy
import dataclasses
import unittest
import numpy as np
import torch
from beatfcn.model import cf
from beatfcn.model.p1 import P1Model
from test_p1_model import CFG, batch, song
from events_helpers import DURATION

SEL = dataclasses.replace(CFG, selection=True)


class SelectionTests(unittest.TestCase):
    def test_views_differ_only_in_selected_bit(self):
        rng = np.random.default_rng(0)
        a = np.arange(0, DURATION, 250_000)
        b = np.arange(125_000, DURATION, 250_000)
        p = cf.build_pair(a, b, DURATION, rng)
        self.assertIsNotNone(p)
        va, vb = cf.views(p)
        diff = np.argwhere(va != vb)
        self.assertTrue(set(diff[:, 1]) <= {1, 7})
        self.assertTrue(len(p['cells_a']) >= 4 and not set(p['cells_a']) & set(p['cells_b']))
        # support lies outside the target core and its halo
        rel = p['support'] - p['c0']
        self.assertTrue(((rel < -4_000_000) | (rel >= 12_000_000)).all())

    def test_field_reads_only_references_and_audio(self):
        torch.manual_seed(0)
        m = P1Model(SEL).eval()
        bt = batch(8, seed=4)
        self.assertTrue(bt['refs_valid'].any(), 'fixture must contain reference evidence')
        with torch.no_grad():
            m(bt)
            s1 = m._sel_core.clone()
            bt2 = copy.deepcopy(bt)
            bt2['known'][..., 0] += 0.5
            bt2['known'][..., 5] *= -1
            bt2['note_type'] = bt2['note_type'].flip(-1)
            bt2['grp_t'] = bt2['grp_t'] * 0.5
            bt2['controls'][:] = 0
            m(bt2)
            torch.testing.assert_close(m._sel_core, s1, atol=1e-5, rtol=1e-5)
            bt3 = copy.deepcopy(bt)
            bt3['refs'][..., 7] = 1 - bt3['refs'][..., 7]
            m(bt3)
            self.assertGreater((m._sel_core - s1).abs().max().item(), 1e-6)

    def test_selection_model_trains_and_cf_learns(self):
        torch.manual_seed(1)
        m = P1Model(SEL)
        bt = batch(4, seed=6)
        loss, met = m(bt)
        self.assertIn('select', met)
        loss.backward()
        rng = np.random.default_rng(3)
        a = np.arange(0, DURATION, 200_000)
        b = np.arange(100_000, DURATION, 230_000)
        sc = song(0)
        for l in SEL.taps:  # A heads sound different from B heads (band 0 vs band 5) so the task is learnable
            t = sc.taps[l].astype(np.float32)
            ia, ib = (a // 40_000)[a // 40_000 < len(t)], (b // 40_000)[b // 40_000 < len(t)]
            t[ia, 0, :16] += 4  # a contrast pattern survives the taps' LayerNorm (a constant offset would not)
            t[ia, 0, 16:] -= 4
            t[ib, 5, :16] -= 4
            t[ib, 5, 16:] += 4
            sc.taps[l] = t
        pairs = [p for p in (cf.build_pair(a, b, DURATION, rng) for _ in range(6)) if p]
        cbt = cf.collate_pairs(pairs, [sc] * len(pairs), SEL.taps, None)
        torch.manual_seed(1)
        m = P1Model(SEL)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(100):
            l, acc = cf.cf_loss(m, cbt)
            opt.zero_grad()
            l.backward()
            opt.step()
        self.assertLess(l.item(), 0.3)
        self.assertEqual(acc.item(), 1.0)


if __name__ == '__main__':
    unittest.main()
