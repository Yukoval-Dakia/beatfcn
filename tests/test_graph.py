"""GraphedStep (eager mode, CPU) == plain training step; static collate shapes."""
import copy
import dataclasses
import unittest
import numpy as np
import torch
from beatfcn.model import features as F
from beatfcn.model.batch import STATIC, collate, collate_meta, SongInfo, attach_audio, _CpuSong
from beatfcn.model.p1 import P1Model
from beatfcn.train.graph import GraphedStep
from test_p1_model import CFG, batch, song
from events_helpers import synthetic_mania, DURATION


class _Run:
    def __init__(self, model, opt):
        self.model, self.opt = model, opt


def _static_batch(n, seed):
    from beatfcn.events.requests import build_request
    exs, k = [], 0
    chart = synthetic_mania(seed)
    while len(exs) < n:
        try:
            s = build_request(chart, kind=('middle_holes', 'reference')[k % 2], seed=seed * 50 + k, audio_id='a', audio_duration_us=DURATION)
            ex = F.build_core(s, chart.events, F.choose_core(s, np.random.default_rng(k), DURATION), DURATION)
            if ex.groups:
                exs.append(ex)
        except (ValueError, AssertionError):
            pass
        k += 1
    sg = song(seed)
    infos = [SongInfo(0, sg.taps[CFG.taps[0]].shape[0], sg.summary.shape[0])] * n
    bt = collate_meta(exs, infos, CFG.taps, CFG.top_tap, 32, 32, False, static=STATIC)
    cpu = _CpuSong(sg, CFG.taps)
    return attach_audio(bt, lambda b: cpu, 'cpu', None)


class GraphedStepTests(unittest.TestCase):
    def test_accumulation_keeps_dynamic_shapes_and_matches_plain_updates(self):
        torch.manual_seed(42)
        reference = P1Model(CFG)
        candidate = copy.deepcopy(reference)
        a = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        b = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
        batches = [batch(2, seed=1), batch(3, seed=5)]
        graph = GraphedStep([_Run(candidate, b)], 1., 'cpu', capture=False)
        for _ in range(2):
            for bt in batches:
                loss, _ = reference(dict(bt))
                (loss / len(batches)).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            a.step()
            a.zero_grad(set_to_none=True)
            graph.fill_many(batches, 0.)
            graph.step()
        for (name, p), q in zip(reference.named_parameters(), candidate.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5, msg=name)
        with self.assertRaisesRegex(ValueError, 'accumulation count'):
            graph.fill_many(batches[:1], 0.)

    def test_static_batches_share_one_shape(self):
        a, b = _static_batch(3, 1), _static_batch(5, 2)
        for k in ('grp_t', 'known', 'refs', 'copy_t', 'song_cells', 'hold_index', 'hold_detail'):
            self.assertEqual(a[k].shape[1:], b[k].shape[1:], k)
        self.assertEqual(a['grp_t'].shape[1], STATIC['G'])
        self.assertEqual(a['song_cells'].shape[2], STATIC['C'] // F.STREAM_STRIDE)
        self.assertEqual(a['hold_index'].shape[0], STATIC['H'])

    def test_static_overflow_is_a_skip(self):
        from beatfcn.model.batch import _size
        with self.assertRaises(F.SkipExample):
            _size(STATIC['K'] + 1, 'K', STATIC)
        self.assertEqual(_size(5, 'K', STATIC), STATIC['K'])

    def test_eager_graphed_step_equals_plain_loop(self):
        cfg = dataclasses.replace(CFG, dropout=0.0, selection=True)
        batches = [_static_batch(4, 7), _static_batch(4, 8)]
        torch.manual_seed(0)
        m_ref = P1Model(cfg)
        m_gs = copy.deepcopy(m_ref)
        o_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-3)
        o_gs = torch.optim.AdamW(m_gs.parameters(), lr=1e-3)
        gs = GraphedStep([_Run(m_gs, o_gs)], 1.0, 'cpu', capture=False)
        for bt in batches:
            b = dict(bt)
            b['w_select'] = 0.2
            loss, _ = m_ref(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m_ref.parameters(), 1.0)
            o_ref.step()
            o_ref.zero_grad(set_to_none=True)
            gs.fill(bt, 0.2)
            mets, norms = gs.step()
        for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_gs.named_parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6), n1)
        self.assertIn('select', mets[0])


class TieredStepTests(unittest.TestCase):
    def _dyn(self, n, seed):
        return batch(n, seed=seed)

    def test_tier_selection_and_padding_preserve_losses(self):
        from beatfcn.model.batch import TIERS, batch_dims, tier_of, tier_shape, AXIS
        from beatfcn.train.graph import TieredStep
        bt = self._dyn(3, 4)
        dims = batch_dims(bt)
        self.assertEqual(tier_of(dims), 0)
        self.assertIsNone(tier_of({**dims, 'K': 10 ** 6}))
        torch.manual_seed(0)
        m = P1Model(dataclasses.replace(CFG, dropout=0.0, selection=True)).eval()
        ts = TieredStep([_Run(m, torch.optim.AdamW(m.parameters(), lr=1e-3))], 1.0, 'cpu', capture=False)
        tier = ts.tiers[1]  # pad up to tier 1 explicitly and compare with the unpadded batch
        ts._alloc(tier, [bt])
        ts._fill(tier, [bt], 0.2)
        padded = dict(tier.slots[0])
        for k in bt:
            if torch.is_tensor(bt[k]) and k in AXIS:
                self.assertEqual(tuple(padded[k].shape), tier_shape(k, bt[k].shape, tier.spec, bt['mode'].shape[0]), k)
        with torch.no_grad():
            la, ma = m(dict(bt, w_select=0.2))
            lb, mb = m(padded)
        self.assertAlmostEqual(float(la), float(lb), places=4)
        for k in ma:
            if k != '_illegal':
                self.assertEqual(int(ma[k][1]), int(mb[k][1]), k)
                self.assertEqual(int(ma[k][2]), int(mb[k][2]), k)

    def test_tier_padding_preserves_losses_with_compact_inner(self):
        from beatfcn.model.batch import prepare_inner_indices
        from beatfcn.train.graph import TieredStep
        bt = prepare_inner_indices(self._dyn(3, 4))
        torch.manual_seed(0)
        m = P1Model(dataclasses.replace(CFG, dropout=0.0, selection=True, compact_inner=True)).eval()
        ts = TieredStep([_Run(m, torch.optim.AdamW(m.parameters(), lr=1e-3))], 1.0, 'cpu', capture=False)
        tier = ts.tiers[1]
        ts._alloc(tier, [bt])
        ts._fill(tier, [bt], 0.2)
        with torch.no_grad():
            la, ma = m(dict(bt, w_select=0.2))
            lb, mb = m(dict(tier.slots[0]))
        self.assertAlmostEqual(float(la), float(lb), places=4)
        for k in ma:
            if k != '_illegal':
                self.assertEqual(int(ma[k][1]), int(mb[k][1]), k)
                self.assertEqual(int(ma[k][2]), int(mb[k][2]), k)

    def test_metadata_batch_attached_into_tier_slots(self):
        """A worker-side metadata batch (collate_meta + _audio recipe) filled into a tier slot with the attach
        callback gives the same losses as the plain attached batch."""
        from beatfcn.model.batch import collate_meta, attach_audio, SongInfo, _CpuSong
        from beatfcn.train.graph import TieredStep
        from test_p1_model import batch_parts
        exs, songs = batch_parts(3, 4)
        infos = [SongInfo.of(s, b, CFG.taps[0]) for b, s in enumerate(songs)]
        meta = collate_meta(exs, infos, CFG.taps, CFG.top_tap, CFG.tap_dim, CFG.summary_dim, False)
        cpu = [_CpuSong(s, CFG.taps) for s in songs]
        plain = batch(3, 4)
        torch.manual_seed(0)
        m = P1Model(dataclasses.replace(CFG, dropout=0.0, selection=True, compact_inner=True)).eval()
        calls = []
        ts = TieredStep([_Run(m, torch.optim.AdamW(m.parameters(), lr=1e-3))], 1.0, 'cpu', capture=False,
                        attach=lambda bt, out=None: calls.append(out is not None) or attach_audio(bt, lambda b: cpu[b], 'cpu', None, out=out))
        tier = ts.tiers[1]
        ts._alloc(tier, [meta])
        ts._fill(tier, [meta], 0.2)
        self.assertEqual(calls, [True])
        slot = tier.slots[0]
        for k in ('audio', 'song_cells', 'ref_audio', 'hold_blocks', 'hold_detail'):
            self.assertIn(k, slot)
        with torch.no_grad():
            la, ma = m(dict(plain, w_select=0.2))
            lb, mb = m(dict(slot))
        self.assertAlmostEqual(float(la), float(lb), places=4)
        for k in ma:
            if k != '_illegal':
                self.assertEqual(int(ma[k][1]), int(mb[k][1]), k)
                self.assertEqual(int(ma[k][2]), int(mb[k][2]), k)
        eager = ts._to_device([collate_meta(exs, infos, CFG.taps, CFG.top_tap, CFG.tap_dim, CFG.summary_dim, False)], 0.2)
        self.assertEqual(calls, [True, False])
        with torch.no_grad():
            lc, _ = m(eager[0])
        self.assertAlmostEqual(float(la), float(lc), places=4)

    def test_eager_tiered_step_equals_plain_accumulation(self):
        from beatfcn.train.graph import TieredStep
        batches = [self._dyn(2, 11), self._dyn(3, 12)]
        torch.manual_seed(3)
        ref = P1Model(dataclasses.replace(CFG, dropout=0.0))
        cand = copy.deepcopy(ref)
        oa, ob = torch.optim.AdamW(ref.parameters(), lr=1e-3), torch.optim.AdamW(cand.parameters(), lr=1e-3)
        ts = TieredStep([_Run(cand, ob)], 1.0, 'cpu', capture=False)
        for _ in range(2):
            for bt in batches:
                loss, _ = ref(dict(bt, w_select=0.0))
                (loss / len(batches)).backward()
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            oa.step()
            oa.zero_grad(set_to_none=True)
            ts.step(batches, 0.0)
        for (n, p), q in zip(ref.named_parameters(), cand.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5, msg=n)
        self.assertEqual(ts.stats['eager_steps'], 2)

    def test_compact_inner_tier_uses_an_invalid_sentinel_slot(self):
        from beatfcn.model.batch import batch_dims, tier_of, prepare_inner_indices
        bt = prepare_inner_indices(self._dyn(3, 6))
        dims = batch_dims(bt)
        ti = tier_of(dims)
        self.assertIsNotNone(ti)
        from beatfcn.model.batch import TIERS
        self.assertLess(dims['G'], TIERS[ti]['G'])  # the last slot of every core is padding -> invalid
