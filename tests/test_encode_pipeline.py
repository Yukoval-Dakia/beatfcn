"""Plumbing of the grouped / padded / async cache encoder against a per-window reference."""
import tempfile
import unittest
import numpy as np
import torch
from beatfcn.audio import memory as M
from beatfcn.audio import encode_pipeline as P

D = 16


def fake_fwd(want):
    """Deterministic stand-in for the teacher: per-window centre cells, 8 bands of 16 mel bins, x layer."""
    def fwd(w):
        x = w[:, 0].float()  # [B, 1024, 128]
        c = x[:, M.EDGE:M.EDGE + M.HOP].reshape(x.shape[0], P.CENTRE_CELLS, M.FRAMES_PER_CELL, 128).mean(2)
        c = c.reshape(x.shape[0], P.CENTRE_CELLS, M.BANDS, D)
        return [(c * l).to(torch.bfloat16) for l in want]
    return fwd


class PipelineTests(unittest.TestCase):
    def test_grouped_padded_pipeline_matches_per_window_reference(self):
        rng = np.random.default_rng(0)
        mels = {f's{i}': rng.standard_normal((int(n), 128)).astype(np.float32)
                for i, n in enumerate([3000, 700, 5200, 1800, 2600])}  # songs of 3..11 windows
        layers, want = (9, 16, 24), [9, 16, 20, 24]
        mean = {l: torch.full((D,), 0.1 * l) for l in layers}
        scale = {l: torch.full((D,), 0.05 * l) for l in layers}
        quant = P.quantiser(layers, mean, scale)
        runner = P.TapRunner(fake_fwd(want), want, layers, quant, batch=4, window_shape=(1, M.WINDOW, 128),
                             device='cpu', mode='eager')
        saved = {}
        stats = P.run(list(mels), lambda s: mels[s], lambda s, arrs: saved.__setitem__(s, {k: v.copy() for k, v in arrs}), runner, layers,
                      group=2, writers=3, device='cpu')
        self.assertEqual(stats['songs'], len(mels))
        fwd = fake_fwd(want)
        for sha, mel in mels.items():
            pl = M.plan(mel.shape[0])
            ref = {l: [] for l in want}
            for s in pl.starts:  # one window at a time: no batching, no padding
                w = M.window_batch(mel, [s], M.silence_row()).to(torch.bfloat16)
                for l, t in zip(want, fwd(w)):
                    ref[l].append(t[0])
            for l in layers:
                exp = quant(l, torch.cat(ref[l])[:pl.cells]).numpy()
                np.testing.assert_array_equal(saved[sha][f'L{l}'], exp, err_msg=f'{sha} L{l}')
            s20 = P.summary_half(torch.cat(ref[20])[:pl.cells]).numpy()
            np.testing.assert_array_equal(saved[sha]['S20'], s20)
            self.assertEqual(list(saved[sha]), ['L9', 'L16', 'L24', 'S20'])  # S20 written last

    def test_rings_and_fallbacks_match_reference(self):
        """Tiny rings: songs longer than max_cells and oversize groups take the fallback buffers; slots recycle."""
        rng = np.random.default_rng(1)
        mels = {f't{i}': rng.standard_normal((int(n), 128)).astype(np.float16)
                for i, n in enumerate([900, 5200, 1500, 400, 3100, 2200, 700])}
        layers, want = (9, 16, 24), [9, 16, 20, 24]
        quant = P.quantiser(layers)
        runner = P.TapRunner(fake_fwd(want), want, layers, quant, batch=3, window_shape=(1, M.WINDOW, 128),
                             device='cpu', mode='eager')
        saved = {}
        with torch.inference_mode():  # as in encode_app: ring slots become inference tensors, readers run in other threads
            stats = P.run(list(mels), lambda s: mels[s], lambda s, arrs: saved.__setitem__(s, {k: v.copy() for k, v in arrs}),
                          runner, layers, group=2, writers=2, readers=2, device='cpu', out_slots=2, in_slots=2, max_cells=500)
        self.assertGreater(stats['fallback_buffers'], 0)
        fwd = fake_fwd(want)
        for sha, mel in mels.items():
            pl = M.plan(mel.shape[0])
            w = M.window_batch(mel.astype(np.float32), pl.starts, M.silence_row()).to(torch.bfloat16)
            ref = [t.reshape(-1, *t.shape[2:])[:pl.cells] for t in fwd(w)]
            for l, t in zip(want, ref):
                if l in layers:
                    np.testing.assert_array_equal(saved[sha][f'L{l}'], quant(l, t).numpy(), err_msg=f'{sha} L{l}')
            np.testing.assert_array_equal(saved[sha]['S20'], P.summary_half(ref[2]).numpy())

    def test_summary_half_matches_memory_summary_means(self):
        t = torch.randn(237, 8, D)
        np.testing.assert_allclose(P.summary_half(t).float().numpy(), M.summary_means(t).half().float().numpy())


if __name__ == '__main__':
    unittest.main()
