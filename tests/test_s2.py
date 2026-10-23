import unittest
import numpy as np
import torch
from beatfcn.audio import memory as M
from beatfcn.audio.pupu import FrozenLargeTeacher
from beatfcn.train import s2 as S2
from test_p1_model import CFG, batch
from events_helpers import DURATION
from beatfcn.model import features as F


class S2Tests(unittest.TestCase):
    def test_attach_matches_direct_encoding_and_has_grad(self):
        torch.manual_seed(0)
        t = FrozenLargeTeacher(dim=32, depth=24, heads=4).eval()
        lora = S2.add_lora(t, rank=2, alpha=4, dropout=0.0)
        bt = batch(3, seed=2)
        exs = bt.pop('_examples', None)
        # rebuild the examples the fixture used (cells per song from duration)
        from test_p1_model import song
        import test_p1_model as T
        rng = np.random.default_rng(0)
        mel = rng.standard_normal((DURATION // 10_000, 128)).astype(np.float32)
        ex = F.CoreExample(0, 8_000_000, (8_000_000 - F.HALO_US) // F.CELL_US, np.zeros((0, 9), np.float32),
                           np.array([[20.0, 0, 0, -1, -1, 0, 0, 1, 0]], np.float32), np.full(len(F.CONTROL_FIELDS), -1),
                           np.ones(4, bool), np.zeros(F.LOCAL_CELLS, np.int64), np.ones((F.CORE_BINS, 10), bool),
                           np.zeros(0, np.int64), np.zeros((15, 6), np.float32), [], 0, np.zeros((1, 4), np.float32))
        bt = {'refs': torch.from_numpy(ex.refs)[None], 'hold_index': torch.zeros(0, 3, dtype=torch.long),
              'hold_onset_us': torch.zeros(0, dtype=torch.long), 'hold_block': torch.zeros(0, dtype=torch.long),
              'audio': torch.zeros(1, 3, F.LOCAL_CELLS, 8, 32), 'ref_audio': torch.zeros(1, 1, 3, 8, 32),
              'hold_detail': torch.zeros(0, 50, 8, 32)}
        cells = -(-mel.shape[0] // 4)
        plan, il, ir, ih = S2.plan_windows([ex], bt, [cells])
        win = S2.mel_windows(plan, [mel], M.silence_row())
        eye, zero = torch.eye(32), torch.zeros(32)
        out = S2.attach_top(dict(bt), t, win, il, ir, ih, zero, eye, 2)
        # direct: encode the whole song's top tap and slice the same cells
        with torch.no_grad():
            full = torch.cat([S2.encode_top(t, M.window_batch(mel, [s], M.silence_row())) for s in M.plan(mel.shape[0]).starts]).reshape(-1, 8, 32)
        torch.testing.assert_close(out['audio'][0, 2].detach(), full[ex.song_cell0:ex.song_cell0 + F.LOCAL_CELLS], atol=1e-5, rtol=1e-5)
        ref_cell = int(round((ex.c0_us + 20.0 * F.SEC) / F.CELL_US))
        torch.testing.assert_close(out['ref_audio'][0, 0, 2].detach(), full[ref_cell], atol=1e-5, rtol=1e-5)
        out['audio'].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora))


if __name__ == '__main__':
    unittest.main()
