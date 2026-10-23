import json
import unittest
from fractions import Fraction
from beatfcn.events.canonical import UnsupportedFormat, build_model_view
from beatfcn.events.phira import parse_phira
from events_helpers import FIX

RPE = json.loads((FIX / 'rpe_multibpm_synthetic.json').read_text())
PGR = json.loads((FIX / 'pgr_v3_synthetic.json').read_text())


def rpe(mut=None, **kw):
    d = json.loads(json.dumps(RPE))
    if mut:
        mut(d)
    return parse_phira(json.dumps(d).encode(), 'rpe.json', **kw)


def codes(c):
    return {i.code for i in c.issues}


class RpeParse(unittest.TestCase):
    def test_multi_bpm_beat_to_us_exact(self):
        c = rpe()
        self.assertFalse(c.quarantined, c.issues)
        # segments: 120 BPM from beat 0; 180 from beat 4 (2 s); 150 from beat 21/2 (25/6 s); META.offset +250 ms
        want = {'rpe:0:0': (750_000, 750_000),                 # beat 1 -> 0.5 s
                'rpe:0:1': (1_416_667, 2_805_556),             # 7/3 -> 7/6 s ; 17/3 -> 23/9 s
                'rpe:1:0': (2_250_000, 2_250_000),             # beat 4 -> 2 s
                'rpe:0:2': (3_027_778, 3_027_778),             # 19/3 -> 25/9 s
                'rpe:0:3': (4_616_667, 4_616_667)}             # 11 -> 25/6 + 0.2 s
        self.assertEqual({e.event_id: (e.onset_us, e.end_us) for e in c.events}, want)
        self.assertEqual(c.native_beats['rpe:0:1'], (Fraction(7, 3), Fraction(17, 3)))
        self.assertEqual(c.timing.audio_sec(Fraction(17, 3)), Fraction(23, 9) + Fraction(1, 4))
        self.assertEqual([s.start_sec for s in c.timing.segments], [0, 2, Fraction(25, 6)])

    def test_types_sides_lines_scene(self):
        c = rpe()
        e = c.by_id()
        self.assertEqual([e[k].type for k in ('rpe:0:0', 'rpe:0:1', 'rpe:0:2', 'rpe:0:3')], ['tap', 'hold', 'flick', 'drag'])
        self.assertEqual(e['rpe:0:1'].location.side, 'below')   # above == 2
        self.assertEqual(e['rpe:0:3'].location.side, 'below')   # above == 0
        self.assertEqual(e['rpe:0:0'].location.side, 'above')
        self.assertEqual(e['rpe:0:1'].location.native_x, -200.5)
        self.assertEqual([s['father'] for s in c.sidecar['scene']], [-1, 0])

    def test_fake_notes_excluded_from_playable(self):
        c = rpe()
        self.assertNotIn('rpe:0:4', c.by_id())
        self.assertEqual([f.event_id for f in c.fakes], ['rpe:0:4'])
        self.assertEqual(c.fakes[0].onset_us, 1_750_000)
        self.assertEqual(len(c.events), 5)

    def test_info_offset_and_sign(self):
        c = rpe(info_offset_sec=0.1)
        self.assertEqual(c.by_id()['rpe:0:0'].onset_us, 850_000)
        def neg(d): d['META']['offset'] = -250
        self.assertEqual(rpe(neg).by_id()['rpe:0:0'].onset_us, 250_000)

    def test_bpmfactor_documented_semantics_but_quarantined(self):
        def bf(d): d['judgeLineList'][1]['bpmfactor'] = 2.0
        c = rpe(bf)
        self.assertIn('bpmfactor_semantics_unverified', codes(c))
        self.assertTrue(c.quarantined)
        self.assertEqual(c.by_id()['rpe:1:0'].onset_us, 4_250_000)  # BPM / 2 -> 60 BPM: 4 beats = 4 s

    def test_fail_closed_cases(self):
        muts = {
            'ambiguous_is_fake': lambda d: d['judgeLineList'][0]['notes'][0].update(isFake=2),
            'unknown_note_type': lambda d: d['judgeLineList'][0]['notes'][0].update(type=5),
            'bad_note_time': lambda d: d['judgeLineList'][0]['notes'][0].update(startTime=[1, 0.5, 2]),
            'bpm_first_not_zero': lambda d: d['BPMList'][0].update(startTime=[1, 0, 1]),
            'meta_offset_not_int': lambda d: d['META'].update(offset=12.5),
            'unknown_note_fields': lambda d: d['judgeLineList'][0]['notes'][0].update(mystery=1),
            'non_positive_hold': lambda d: d['judgeLineList'][0]['notes'][1].update(endTime=[2, 1, 3]),
        }
        for code, m in muts.items():
            c = rpe(m)
            self.assertIn(code, codes(c), code)
            self.assertTrue(c.quarantined, code)

    def test_model_view_phira_ordering(self):
        c = rpe()
        v = build_model_view('phira', [], c.events)
        self.assertTrue(v.ok)
        self.assertEqual([g.time_us for g in v.target_groups], [750_000, 1_417_000, 2_250_000, 3_028_000, 4_617_000])


class OfficialAndUnsupported(unittest.TestCase):
    def test_official_v3_times(self):
        c = parse_phira((FIX / 'pgr_v3_synthetic.json').read_bytes(), 'pgr.json')
        self.assertFalse(c.quarantined, c.issues)
        got = {e.event_id: (e.onset_us, e.end_us, e.type, e.location.side) for e in c.events}
        self.assertEqual(got, {'pgr:0:above:0': (600_000, 600_000, 'tap', 'above'),
                               'pgr:0:above:1': (1_100_000, 1_850_000, 'hold', 'above'),
                               'pgr:0:below:0': (1_600_000, 1_600_000, 'drag', 'below'),
                               'pgr:0:below:1': (1_662_500, 1_662_500, 'flick', 'below'),
                               'pgr:1:above:0': (250_000, 250_000, 'tap', 'above')})
        self.assertIsNone(c.timing)  # two different line BPMs: no single chart-wide map
        self.assertEqual(c.native_beats['pgr:0:above:1'], (Fraction(2), Fraction(7, 2)))

    def test_unsupported_formats(self):
        with self.assertRaises(UnsupportedFormat):
            parse_phira((FIX / 'pec_synthetic.pec').read_bytes())
        d = dict(PGR, formatVersion=1)
        with self.assertRaises(UnsupportedFormat):
            parse_phira(json.dumps(d).encode())
        with self.assertRaises(UnsupportedFormat):
            parse_phira(json.dumps(RPE).encode() + json.dumps(RPE).encode())  # concatenated JSON
        with self.assertRaises(UnsupportedFormat):
            parse_phira(b'{"something": 1}')


if __name__ == '__main__':
    unittest.main()
