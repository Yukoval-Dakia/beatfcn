import unittest
from collections import Counter
from fractions import Fraction
from beatfcn.events.canonical import Event, LaneLoc, UnsupportedFormat
from beatfcn.events.osu_mania import ExportError, export_osu, parse_osu
from events_helpers import FIX

RAW = (FIX / 'mania4k_synthetic.osu').read_bytes()


def codes(chart):
    return {i.code for i in chart.issues}


def sigs(chart):
    return Counter((e.onset_us, e.end_us, e.type, e.location.lane) for e in chart.events)


class OsuParse(unittest.TestCase):
    def test_round_trip_byte_identical_variants(self):
        lf = RAW.replace(b'\r\n', b'\n')
        for raw in (RAW, lf, lf.rstrip(b'\n'), b'\xef\xbb\xbf' + RAW, RAW.replace(b'\r\n', b'\n', 3)):
            c = parse_osu(raw)
            self.assertFalse(c.quarantined, c.issues)
            self.assertEqual(export_osu(c, []), raw)
            self.assertEqual(export_osu(c), raw)

    def test_negative_beatlength_is_sv_not_bpm(self):
        c = parse_osu(RAW)
        self.assertEqual([s.bpm for s in c.timing.segments], [120, 160])
        self.assertEqual([r['sv'] for r in c.sidecar['sv_rows']], [2, Fraction(1, 2)])
        # 6500 ms: 8 beats at 120 BPM from 1000 ms, then 1500 ms at 160 BPM = 4 beats
        self.assertEqual(c.native_beats['osu:L38'][0], 12)
        only_sv = RAW.replace(b'1000,500,4,2,0,60,1,0\r\n', b'').replace(b'5000,375,4,2,0,60,1,0\r\n', b'')
        self.assertIn('no_uninherited_timing_point', codes(parse_osu(only_sv)))

    def test_holds_and_lanes(self):
        c = parse_osu(RAW)
        h = c.by_id()['osu:L34']
        self.assertEqual((h.type, h.onset_us, h.end_us, h.location.lane), ('hold', 1_500_000, 2_500_000, 2))
        self.assertEqual(sorted({e.location.lane for e in c.events}), [0, 1, 2, 3])
        self.assertEqual(c.by_id()['osu:L35'].type, 'tap')   # type 5 = circle + new combo

    def test_rejects_other_modes_keys_versions(self):
        for bad in (RAW.replace(b'CircleSize:4', b'CircleSize:7'), RAW.replace(b'Mode: 3', b'Mode: 0'),
                    RAW.replace(b'v14', b'v12'), b'not an osu file'):
            with self.assertRaises(UnsupportedFormat):
                parse_osu(bad)

    def test_bad_objects_quarantined_not_fixed(self):
        cases = {b'448,192,6500,1,': b'512,192,6500,1,',            # x out of lane range
                 b'64,192,3000,1,': b'64,192,3000,2,',              # slider in mania
                 b'192,192,5000,128,0,6125': b'192,192,5000,128,0,5000',  # non-positive hold
                 b'448,192,2000,5,': b'320,192,2000,5,'}            # head inside lane-2 hold
        want = ['lane_out_of_range', 'invalid_mania_object_type', 'non_positive_hold', 'lane_collision']
        for (a, b), code in zip(cases.items(), want):
            c = parse_osu(RAW.replace(a, b))
            self.assertTrue(c.quarantined)
            self.assertIn(code, codes(c))
            self.assertEqual(export_osu(c, []), RAW.replace(a, b))


class OsuExport(unittest.TestCase):
    def test_merge_preserves_every_original_line(self):
        c = parse_osu(RAW)
        adds = [Event('new-a', 500_000, 500_000, 'tap', LaneLoc(2)),
                Event('new-b', 3_000_000, 4_000_000, 'hold', LaneLoc(3)),
                Event('new-c', 9_000_000, 9_000_000, 'tap', LaneLoc(0))]
        out = export_osu(c, adds)
        orig, new = RAW.splitlines(keepends=True), out.splitlines(keepends=True)
        it = iter(new)
        self.assertTrue(all(any(x == ln for x in it) for ln in orig), 'original lines must be an in-order subsequence')
        self.assertEqual(len(new), len(orig) + 3)
        self.assertEqual(new[-1], b'64,192,9000,1,0,0:0:0:0:\r\n')
        self.assertEqual(new[orig.index(b'64,192,1000,1,0,0:0:0:0:\r\n')], b'320,192,500,1,0,0:0:0:0:\r\n')
        re = parse_osu(out)
        self.assertEqual(sigs(re), sigs(c) + Counter((a.onset_us, a.end_us, a.type, a.location.lane) for a in adds))

    def test_merge_without_trailing_newline(self):
        raw = RAW.rstrip(b'\r\n')
        out = export_osu(parse_osu(raw), [Event('n', 9_000_000, 9_000_000, 'tap', LaneLoc(0))])
        self.assertTrue(out.startswith(raw))
        self.assertEqual(out[len(raw):], b'\r\n64,192,9000,1,0,0:0:0:0:')

    def test_export_fails_closed(self):
        c = parse_osu(RAW)
        for bad in ([Event('x', 500_500, 500_500, 'tap', LaneLoc(0))],          # not integer ms
                    [Event('x', 2_000_000, 2_000_000, 'tap', LaneLoc(2))],      # inside lane-2 hold
                    [Event('osu:L32', 9_000_000, 9_000_000, 'tap', LaneLoc(0))]):  # id clash
            with self.assertRaises(ExportError):
                export_osu(c, bad)
        q = parse_osu(RAW.replace(b'448,192,6500,1,', b'512,192,6500,1,'))
        with self.assertRaises(ExportError):
            export_osu(q, [Event('x', 9_000_000, 9_000_000, 'tap', LaneLoc(0))])


if __name__ == '__main__':
    unittest.main()
