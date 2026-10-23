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
    def test_static_batches_share_one_shape(self):
        a, b = _static_batch(3, 1), _static_batch(5, 2)
        for k in ('grp_t', 'known', 'refs', 'copy_t', 'summary', 'hold_index', 'hold_detail'):
            self.assertEqual(a[k].shape[1:], b[k].shape[1:], k)
        self.assertEqual(a['grp_t'].shape[1], STATIC['G'])
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
