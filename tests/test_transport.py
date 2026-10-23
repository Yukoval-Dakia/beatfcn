"""Compact batch transport (int8 taps, float16 mel/summary) must equal the float path after dequant."""
import unittest
import numpy as np
import torch
from beatfcn.events.requests import build_request
from beatfcn.model import cf
from beatfcn.model import features as F
from beatfcn.model.batch import SongCache, collate, dequant
from beatfcn.train.data import QuantArray
from test_p1_model import CFG
from events_helpers import synthetic_mania, DURATION


def songs_pair(seed):
    """The same song as int8 QuantArray taps and as their exact float32 dequantised copy."""
    rng = np.random.default_rng(seed)
    cells = DURATION // F.CELL_US
    q = {l: rng.integers(-127, 128, (cells, 8, 32)).astype(np.int8) for l in CFG.taps}
    sc = {l: rng.uniform(0.01, 0.1, 32).astype(np.float32) for l in CFG.taps}
    summary = rng.standard_normal((DURATION // F.BLOCK_US, 8, 32)).astype(np.float16)
    mel = rng.standard_normal((cells * 4, 128)).astype(np.float16)
    quant = SongCache({l: QuantArray(q[l], sc[l]) for l in CFG.taps}, summary, mel, DURATION)
    flt = SongCache({l: q[l].astype(np.float32) * sc[l] for l in CFG.taps}, summary, mel, DURATION)
    return quant, flt


def examples(n=6):
    chart, exs, k = synthetic_mania(3), [], 0
    while len(exs) < n:
        try:
            s = build_request(chart, kind=('middle_holes', 'reference')[k % 2], seed=k, audio_id='a', audio_duration_us=DURATION)
            ex = F.build_core(s, chart.events, F.choose_core(s, np.random.default_rng(k), DURATION), DURATION)
            if ex.groups:
                exs.append(ex)
        except (ValueError, AssertionError):
            pass
        k += 1
    return exs


class TransportTests(unittest.TestCase):
    def test_int8_batch_matches_float_batch(self):
        quant, flt = songs_pair(0)
        exs = examples()
        a = collate(exs, [quant] * len(exs), CFG.taps, CFG.top_tap)
        b = collate(exs, [flt] * len(exs), CFG.taps, CFG.top_tap)
        self.assertEqual(a['audio'].dtype, torch.int8)
        self.assertGreater(len(a['hold_detail']), 0, 'fixture must contain holds')
        da, db = dequant(dict(a)), dequant(dict(b))
        self.assertEqual(set(da), set(db))
        for k in db:
            if torch.is_tensor(db[k]):
                self.assertEqual(da[k].dtype, db[k].dtype, k)
                self.assertTrue(torch.allclose(da[k], db[k], atol=1e-6), k)

    def test_int8_cf_pairs_match_float(self):
        quant, flt = songs_pair(1)
        rng = np.random.default_rng(0)
        p = cf.build_pair(np.arange(0, DURATION, 250_000), np.arange(125_000, DURATION, 250_000), DURATION, rng)
        a = dequant(cf.collate_pairs([p], [quant], CFG.taps, None))
        b = dequant(cf.collate_pairs([p], [flt], CFG.taps, None))
        for k in ('audio', 'ref_audio'):
            self.assertTrue(torch.allclose(a[k], b[k], atol=1e-6), k)


if __name__ == '__main__':
    unittest.main()


class PoolTests(unittest.TestCase):
    def test_meta_plus_pool_attach_matches_cpu_collate(self):
        """collate_meta in a worker + attach_audio from a SongPool (CPU device here) == collate."""
        from beatfcn.model.batch import collate_meta, attach_audio, SongInfo
        from beatfcn.train.pool import SongPool
        quant, flt = songs_pair(3)
        exs = examples(5)
        ref = collate(exs, [quant] * len(exs), CFG.taps, CFG.top_tap)
        infos = [SongInfo('s0', quant.taps[CFG.taps[0]].shape[0], quant.summary.shape[0])] * len(exs)
        bt = collate_meta(exs, infos, CFG.taps, CFG.top_tap, 32, 32, True)
        self.assertLess(sum(v.numel() * v.element_size() for v in bt.values() if torch.is_tensor(v)), 2_000_000)
        loader = lambda sha: ({l: quant.taps[l].raw for l in CFG.taps}, quant.summary, quant.mel10)
        pool = SongPool(loader, CFG.taps, 'cpu', capacity=2)
        pool.request(['s0'])
        attach_audio(bt, pool.get, 'cpu', {l: quant.taps[l].scale for l in CFG.taps})
        pool.release(['s0'])
        self.assertEqual(set(bt), set(ref))
        for k in ref:
            if torch.is_tensor(ref[k]):
                self.assertEqual(ref[k].dtype, bt[k].dtype, k)
                self.assertTrue(torch.equal(ref[k], bt[k]), k)
        self.assertEqual(pool.stats['loads'], 1)


class PoolRefcountTests(unittest.TestCase):
    def test_song_referenced_by_two_batches_survives_one_release(self):
        from beatfcn.train.pool import SongPool
        quant, _ = songs_pair(4)
        loader = lambda sha: ({l: quant.taps[l].raw for l in CFG.taps}, quant.summary, quant.mel10)
        pool = SongPool(loader, CFG.taps, 'cpu', capacity=1)
        pool.request(['a'])
        pool.request(['a'])
        pool.get('a')
        pool.release(['a'])
        pool.request(['b'])
        pool.get('b')  # capacity 1, but 'a' is still referenced by the second batch
        self.assertIn('a', pool.resident)
        pool.release(['a'])
        pool.request(['c'])
        pool.get('c')
        self.assertNotIn('a', pool.resident)
