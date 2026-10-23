import random
import unittest
from beatfcn.events.group_split import (group_rows, normalize_title, old_split_leakage, split_manifest, split_of,
                                          straddling_groups, work_key)


def rows(n=400):
    out = []
    for i in range(n):
        song = i // 3      # three charts (difficulties) per song
        out.append({'id': i, 'name': f'Song {song}' + (' (TV Size)' if i % 3 == 2 else ''), 'composer': 'c'})
    out.append({'id': 10_000, 'name': 'ＳＯＮＧ　1', 'composer': 'x'})                 # full-width variant of Song 1
    out.append({'id': 10_001, 'name': 'Totally different', 'audio_sha256': 'abc'})
    out.append({'id': 10_002, 'name': 'Song 2 [Speed Up]', 'audio_sha256': 'abc'})   # audio bridges two titles
    out.append({'id': 10_003, 'name': ''})
    return out


class Split(unittest.TestCase):
    def test_normalisation(self):
        self.assertEqual(normalize_title('ＳＯＮＧ　1'), normalize_title('song 1'))
        self.assertEqual(work_key('Song 1 (TV Size)'), work_key('song1'))
        self.assertEqual(work_key('雲鶴サクリファイス【Remaster】'), normalize_title('雲鶴サクリファイス'))

    def test_grouping_evidence(self):
        groups, c2g = group_rows(rows())
        self.assertEqual(c2g['3'], c2g['5'])                 # 'Song 1' and 'Song 1 (TV Size)'
        self.assertEqual(c2g['3'], c2g['10000'])             # full-width title
        self.assertEqual(c2g['10001'], c2g['6'])             # audio hash bridges to Song 2
        self.assertTrue(any(k.startswith('audio:') for k in groups[c2g['10001']]['evidence']))
        self.assertEqual(groups[c2g['10003']]['members'], ['10003'])

    def test_deterministic_and_no_group_straddles(self):
        r = rows()
        m1 = split_manifest(r, seed=20261022)
        shuffled = r[:]
        random.Random(0).shuffle(shuffled)
        m2 = split_manifest(shuffled, seed=20261022)
        self.assertEqual(m1['chart_split'], m2['chart_split'])
        self.assertEqual(m1['groups'], m2['groups'])
        _, c2g = group_rows(r)
        self.assertEqual(straddling_groups(m1['chart_split'], c2g), {})
        self.assertNotEqual(m1['chart_split'], split_manifest(r, seed=1)['chart_split'])
        n = m1['n_groups']
        self.assertAlmostEqual(m1['counts']['train']['groups'] / n, 0.8, delta=0.1)
        self.assertEqual(sum(v['charts'] for v in m1['counts'].values()), len(r))

    def test_adding_rows_does_not_move_existing_groups(self):
        r = rows()
        a = split_manifest(r, seed=5)
        b = split_manifest(r + [{'id': 99_999, 'name': 'Brand new song'}], seed=5)
        self.assertEqual({c: s for c, s in b['chart_split'].items() if c in a['chart_split']}, a['chart_split'])
        self.assertEqual(split_of('gabc', 5), split_of('gabc', 5))

    def test_old_split_leak_report(self):
        r = rows(9)   # songs 0, 1, 2 with 3 charts each (+ extras)
        old = {'0': 'train', '1': 'val', '2': 'train', '3': 'test', '4': 'test', '5': 'test', '6': 'train', '7': 'train',
               '8': 'train', '777': 'train'}
        rep = old_split_leakage(old, r)
        self.assertEqual(rep['straddling_groups'], 1)
        self.assertEqual(rep['charts_in_straddling_groups'], 3)
        self.assertEqual(rep['val_or_test_charts_sharing_song_with_other_split'], 1)
        self.assertEqual(rep['old_charts_without_metadata'], 1)
        self.assertEqual(rep['combos'], {'train+val': 1})


if __name__ == '__main__':
    unittest.main()
