import unittest
import numpy as np
import torch
from beatfcn.audio import memory as M
from beatfcn.audio.pupu import FrozenLargeTeacher, make_rope


def tiny():
    torch.manual_seed(0)
    return FrozenLargeTeacher(dim=32, depth=4, heads=4).eval()


class AudioMemoryTests(unittest.TestCase):
    def test_plan_covers_every_cell_once(self):
        for n in (1, 3, 4, 511, 512, 1024, 1025, 3001, 15900):
            p = M.plan(n)
            cover = np.zeros(p.cells + 400, int)
            for s in p.starts:
                a = (s + M.EDGE) // 4
                cover[a:a + M.HOP // 4] += 1
            self.assertTrue((cover[:p.cells] == 1).all(), n)
            self.assertEqual(p.cells, -(-n // 4))

    def test_cells_come_from_matching_window_centre(self):
        enc = M.TapEncoder(tiny())
        rng = np.random.default_rng(1)
        mel = rng.standard_normal((2100, 128)).astype(np.float32)
        pad = M.silence_row()
        taps = M.encode_song(enc, mel, [2, 4], batch=3, pad_row=pad, out_dtype=torch.float32)
        self.assertEqual(taps[4].shape, (M.plan(2100).cells, 8, 32))
        # Recompute window 2 directly through the reference module and compare its centre.
        s = M.plan(2100).starts[2]
        w = M.window_batch(mel, [s], pad)
        t = enc.teacher
        x = t.patch_embed(w)
        rope = make_rope((256, 8), t.head_dim)
        for blk in t.teacher.blocks:
            x = blk(x, rope)
        ref = x.reshape(1, 256, 8, 32)[0, 64:192]
        a = (s + M.EDGE) // 4
        torch.testing.assert_close(taps[4][a:a + 128], ref, atol=0, rtol=0)

    def test_final_tap_matches_pretrained_forward_before_norm(self):
        t = tiny()
        mel = torch.randn(1, 1, 1024, 128)
        with torch.no_grad():
            full = t(mel, return_tokens=True)  # final LayerNorm applied
            mine = M.TapEncoder(t)(mel, [4])[4]
        torch.testing.assert_close(t.teacher.norm(mine), full[:, 64:192], atol=1e-5, rtol=1e-5)

    def test_summary_means_and_key(self):
        tap = torch.arange(120 * 8 * 2, dtype=torch.float32).reshape(120, 8, 2)
        s = M.summary_means(tap)
        self.assertEqual(s.shape, (3, 8, 2))
        torch.testing.assert_close(s[2], tap[100:].mean(0))
        k1 = M.cache_key('a' * 64, 'b' * 64, [8, 16, 24], 'float16')
        self.assertNotEqual(k1, M.cache_key('a' * 64, 'b' * 64, [8, 16, 24], 'float16', lora='r8'))
        self.assertEqual(k1, M.cache_key('a' * 64, 'b' * 64, [24, 16, 8], 'float16'))


if __name__ == '__main__':
    unittest.main()
