import copy
import unittest
import numpy as np
import torch
from beatfcn.events.requests import build_request
from beatfcn.model import features as F
from beatfcn.model.batch import SongCache, collate
from beatfcn.model.p1 import P1Config, P1Model
from events_helpers import synthetic_mania, synthetic_phira, DURATION

CFG = P1Config(d=64, layers=2, heads=4, audio_kv_heads=2, ffn=96, dropout=0.0, tap_dim=32, summary_dim=32, ptr=32, gru=48)


def song(seed):
    rng = np.random.default_rng(seed)
    cells = DURATION // F.CELL_US
    return SongCache({l: rng.standard_normal((cells, 8, 32)).astype(np.float16) for l in CFG.taps},
                     rng.standard_normal((DURATION // F.BLOCK_US, 8, 32)).astype(np.float32),
                     rng.standard_normal((cells * 4, 128)).astype(np.float16), DURATION)


def batch(n=4, seed=0):
    exs, songs, k = [], [], 0
    charts = [synthetic_mania(seed), synthetic_phira(seed + 1)]
    while len(exs) < n:
        chart = charts[k % 2]
        try:
            s = build_request(chart, kind=('middle_holes', 'from_scratch', 'sparse_anchors', 'reference')[k % 4],
                              seed=seed * 100 + k, audio_id='a', audio_duration_us=DURATION)
            c0 = F.choose_core(s, np.random.default_rng(k), DURATION)
            ex = F.build_core(s, chart.events, c0, DURATION)
            if ex.groups:
                exs.append(ex)
                songs.append(song(k % 2))
        except (ValueError, AssertionError):
            pass
        k += 1
    return collate(exs, songs, CFG.taps, CFG.top_tap)


class P1Tests(unittest.TestCase):
    def test_forward_backward_all_fields(self):
        import dataclasses
        bt = batch(6)
        placement = {'line_q', 'line_tab', 'side_h', 'x_h'}
        for loc in (False, True):
            torch.manual_seed(0)
            m = P1Model(dataclasses.replace(CFG, phira_location=loc))
            loss, met = m(bt)
            self.assertTrue(torch.isfinite(loss))
            for k in ('time', 'residual', 'cont', 'type', 'lane', 'end_block', 'end_coarse', 'end_fine'):
                self.assertIn(k, met)
            for k in ('line', 'side', 'x'):
                self.assertEqual(k in met, loc)
            loss.backward()
            none = {n.split('.')[0] for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()}
            self.assertEqual(none, set() if loc else placement)

    def test_future_groups_do_not_change_earlier_time_logits(self):
        torch.manual_seed(1)
        m = P1Model(CFG).eval()
        bt = batch(2, seed=3)
        G = bt['grp_valid'].shape[1]
        i = max(1, int(bt['grp_valid'][0].sum()) // 2)
        bt2 = copy.deepcopy(bt)
        for key in ('note_type', 'note_lane', 'note_line', 'note_side'):
            bt2[key][:, i:] = bt2[key][:, i:].flip(-1)
        bt2['note_x'][:, i:] += 0.3
        bt2['note_dur'][:, i:] += 1.0
        with torch.no_grad():
            h, pa, fi, _ = m.encode(bt)
            h2, _, _, _ = m.encode(bt2)
        torch.testing.assert_close(h[:, :i + 1], h2[:, :i + 1], atol=1e-5, rtol=1e-5)
        self.assertGreater((h[:, i + 1:] - h2[:, i + 1:]).abs().max().item(), 1e-4)

    def test_overfit_small_batch(self):
        torch.manual_seed(2)
        m = P1Model(CFG)
        bt = batch(4, seed=5)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        first = None
        for step in range(150):
            loss, met = m(bt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        self.assertLess(loss.item(), 0.3 * first, (first, loss.item(), met))


if __name__ == '__main__':
    unittest.main()


class LegalityTests(unittest.TestCase):
    def test_illegal_target_is_counted_on_device_and_raised_at_summary(self):
        from beatfcn.model.p1 import masked_ce
        from beatfcn.train.loop import merge, summarize
        logits = torch.randn(3, 5)
        mask = torch.ones(3, 5, dtype=torch.bool)
        mask[1, 2] = False
        valid = torch.tensor([True, True, False])
        loss, n, c, bad = masked_ce(logits, torch.tensor([0, 2, -1]), valid, mask)
        self.assertEqual((int(n), int(bad)), (2, 1))
        acc = {}
        merge(acc, {'lane': (loss.detach(), n, c), '_illegal': bad})
        with self.assertRaises(AssertionError):
            summarize(acc)
        loss, n, c, bad = masked_ce(logits, torch.tensor([0, 3, -1]), valid, mask)  # legal; padded row ignored
        ref = torch.nn.functional.cross_entropy(logits[:2].masked_fill(~mask[:2], -1e9), torch.tensor([0, 3]))
        self.assertEqual(int(bad), 0)
        self.assertTrue(torch.allclose(loss, ref))


class PackedTests(unittest.TestCase):
    def test_packed_flex_path_matches_padded_sdpa_path(self):
        """Packing valid rows + FlexAttention block mask is the same computation as padded SDPA + dense mask."""
        import dataclasses
        bt = batch(6)
        res = {}
        for packed in (False, True):
            torch.manual_seed(0)
            m = P1Model(dataclasses.replace(CFG, packed=packed, dropout=0.0, selection=True)).eval()
            b = dict(bt)
            b['w_select'] = 0.2
            loss, met = m(b)
            loss.backward()
            res[packed] = (float(loss), {k: tuple(float(x) for x in v) for k, v in met.items() if k != '_illegal'},
                           torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
        (l0, m0, g0), (l1, m1, g1) = res[False], res[True]
        self.assertAlmostEqual(l0, l1, places=4)
        for k in m0:
            self.assertEqual(m0[k][1:], m1[k][1:], k)
        self.assertLess(float((g0 - g1).norm() / g0.norm()), 1e-5)


class BlockMaskTests(unittest.TestCase):
    def test_block_level_mask_is_superset_of_dense_reference(self):
        """The block list built from per-block core/time ranges must include every block the element-exact
        mask_mod (evaluated densely by create_block_mask) marks non-empty."""
        from torch.nn.attention.flex_attention import create_block_mask
        from beatfcn.model import features as FE
        from beatfcn.model.batch import dequant
        bt = dequant(batch(5, seed=2))
        m = P1Model(CFG).eval()
        with torch.no_grad():
            rows, lens, times, timed, valid = m._rows(bt)
            Y, ty, vy = m._y_rows(bt, m.group_embedding(bt))
            rv = torch.cat(valid + [vy], 1)
            B, Np = rv.shape
            kinds = torch.cat([torch.full((n,), k) for k, n in enumerate(lens + [Np - sum(lens)])])
            idx = rv.reshape(-1).nonzero()[:, 0]
            N = idx.shape[0]
            pad = -(-N // m.PACK_BUCKET) * m.PACK_BUCKET - N
            core = torch.cat([idx // Np, torch.full((pad,), B)])
            pos = idx % Np
            kind = torch.cat([kinds[pos], torch.zeros(pad, dtype=kinds.dtype)])
            ypos = torch.cat([pos - sum(lens), torch.zeros(pad, dtype=pos.dtype)])
            t_q = torch.cat([torch.cat(times + [ty], 1).reshape(-1)[idx], torch.zeros(pad)])
            bm = m._block_mask(bt, core, kind, ypos, t_q, 8)
            BNa = B * FE.LOCAL_CELLS * 8
            ref = create_block_mask(bm.mask_mod, None, None, N + pad, BNa + N + pad, device='cpu', BLOCK_SIZE=128)
            ours, theirs = bm.to_dense()[0, 0].bool(), ref.to_dense()[0, 0].bool()
        self.assertEqual(int((theirs & ~ours).sum()), 0)
        self.assertLess(float(ours.float().mean()), 0.25)  # and it is sparse, not a dense superset
