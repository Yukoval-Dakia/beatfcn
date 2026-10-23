import unittest
from fractions import Fraction
from beatfcn.events.canonical import (Event, Group, LaneLoc, LineLoc, TimingMap, build_model_view, check_groups,
                                         quantize_us, round_half_up)
from beatfcn.events.validate import model_view_issues


def tap(i, t, lane):
    return Event(f'e{i}', t, t, 'tap', LaneLoc(lane), f'x#{i}')


class Quantization(unittest.TestCase):
    def test_round_half_up(self):
        self.assertEqual([quantize_us(x) for x in (999_499, 999_500, 1_000_499, 1_000_500)],
                         [999_000, 1_000_000, 1_000_000, 1_001_000])
        self.assertEqual(round_half_up(Fraction(5, 2)), 3)
        self.assertEqual(round_half_up(Fraction(-5, 2)), -2)

    def test_collision_after_quantization_is_quarantined(self):
        v = build_model_view('mania_4k', [], [tap(1, 1_000_400, 0), tap(2, 999_700, 0)])
        self.assertFalse(v.ok)
        self.assertEqual([i.code for i in v.quarantine], ['lane_collision_after_quantization'])
        self.assertTrue(model_view_issues(v))

    def test_collision_with_observed_after_quantization(self):
        obs = [Event('h', 0, 2_000_300, 'hold', LaneLoc(1), 'x#h')]     # observed end is not on the ms grid
        v = build_model_view('mania_4k', obs, [tap(1, 2_000_600, 1)])   # -> 2001 ms: after the hold, fine
        self.assertTrue(v.ok, v.quarantine)
        v = build_model_view('mania_4k', obs, [tap(1, 2_000_400, 1)])   # -> 2000 ms: now inside the hold
        self.assertEqual([i.code for i in v.quarantine], ['lane_collision_after_quantization'])
        v = build_model_view('mania_4k', obs, [tap(1, 2_000_200, 1)])   # inside in source; quantizing would hide it
        self.assertEqual([i.code for i in v.quarantine], ['lane_collision_in_source'])

    def test_non_positive_hold_after_quantization(self):
        v = build_model_view('mania_4k', [], [Event('h', 2_000_200, 2_000_400, 'hold', LaneLoc(0))])
        self.assertEqual([i.code for i in v.quarantine], ['non_positive_hold_after_quantization'])
        self.assertEqual(v.target_groups, ())

    def test_observed_never_quantized_and_copy_time(self):
        obs = [tap(0, 1_234_567, 0)]
        v = build_model_view('mania_4k', obs, [tap(1, 1_234_567, 2), tap(2, 1_234_890, 3)])
        self.assertTrue(v.ok)
        self.assertEqual(v.observed_groups[0].events[0], obs[0])
        self.assertEqual([(g.time_us, g.copy_time) for g in v.target_groups], [(1_234_567, True), (1_235_000, False)])
        self.assertEqual(dict(v.quantized_from), {'e2': (1_234_890, 1_234_890)})
        self.assertEqual(model_view_issues(v, obs), [])


class Grouping(unittest.TestCase):
    def test_mania_group_order_and_strict_increase(self):
        v = build_model_view('mania_4k', [], [tap(1, 5_000_000, 3), tap(2, 5_000_000, 0), tap(3, 5_000_000, 2),
                                              tap(4, 4_000_000, 1)])
        self.assertEqual([g.time_us for g in v.target_groups], [4_000_000, 5_000_000])
        self.assertEqual([e.location.lane for e in v.target_groups[1].events], [0, 2, 3])
        self.assertEqual(check_groups(v.target_groups), [])

    def test_phira_group_order(self):
        def ev(i, line, side, x, ty):
            return Event(f'p{i}', 1_000_000, 1_000_000 if ty != 'hold' else 1_500_000, ty, LineLoc(line, side, x))
        evs = [ev(1, '10', 'above', 0.0, 'tap'), ev(2, '2', 'below', -5.0, 'tap'), ev(3, '2', 'above', 3.0, 'drag'),
               ev(4, '2', 'above', 3.0, 'tap'), ev(5, '2', 'above', -1.0, 'flick')]
        v = build_model_view('phira', [], evs)
        self.assertEqual([e.event_id for e in v.target_groups[0].events], ['p5', 'p4', 'p3', 'p2', 'p1'])

    def test_check_groups_detects_violations(self):
        a, b = tap(1, 1000, 0), tap(2, 1000, 2)
        bad = [Group(1000, (b, a)), Group(1000, (a,)), Group(2000, ()), Group(3000, (tap(3, 2999, 0),))]
        msgs = ' | '.join(check_groups(bad))
        for frag in ('not in canonical order', 'not >', 'empty', 'member onset'):
            self.assertIn(frag, msgs)

    def test_phira_duplicate_after_quantization(self):
        e1 = Event('a', 1_000_100, 1_000_100, 'tap', LineLoc('0', 'above', 1.0))
        e2 = Event('b', 1_000_200, 1_000_200, 'tap', LineLoc('0', 'above', 1.0))
        self.assertEqual([i.code for i in build_model_view('phira', [], [e1, e2]).quarantine],
                         ['duplicate_object_after_quantization'])


class Objects(unittest.TestCase):
    def test_event_and_location_validation(self):
        with self.assertRaises(TypeError):
            Event('x', 1.5, 1.5, 'tap', LaneLoc(0))
        with self.assertRaises(ValueError):
            Event('x', 1, 1, 'slide', LaneLoc(0))
        with self.assertRaises(ValueError):
            LaneLoc(4)
        with self.assertRaises(ValueError):
            LineLoc('0', 'left', 0.0)
        e = Event('x', 1, 1, 'tap', LaneLoc(0))
        with self.assertRaises(Exception):
            e.onset_us = 2  # frozen

    def test_timing_map_exact(self):
        tm = TimingMap.from_bpm_points([(0, 120), (Fraction(4), 180), (Fraction(21, 2), 150)], Fraction(1, 4))
        self.assertEqual(tm.audio_sec(Fraction(17, 3)), Fraction(23, 9) + Fraction(1, 4))
        self.assertEqual(tm.audio_us(Fraction(17, 3)), 2_805_556)
        self.assertEqual(tm.chart_sec(-1), Fraction(-1, 2))  # first BPM extends backwards


if __name__ == '__main__':
    unittest.main()


class SourceStackTests(unittest.TestCase):
    """DATA_RULES v4: identical stacked notes (<= 4) are intentional and must reach training."""

    def _ev(self, i, t, typ='drag', x=0.5):
        from beatfcn.events.canonical import Event, LineLoc
        return Event(f'e{i}', t, t, typ, LineLoc('L0', 'above', x), f'src#{i}')

    def test_allowed_stack_is_not_quarantined(self):
        from beatfcn.events.canonical import build_model_view
        v = build_model_view('phira', (), tuple(self._ev(i, 1_000_000) for i in range(4)))
        self.assertTrue(v.ok, [i.code for i in v.quarantine])

    def test_oversized_stack_is_quarantined(self):
        from beatfcn.events.canonical import build_model_view
        v = build_model_view('phira', (), tuple(self._ev(i, 1_000_000) for i in range(5)))
        self.assertIn('duplicate_object_in_source', [i.code for i in v.quarantine])

    def test_quantization_merge_is_still_quarantined(self):
        from beatfcn.events.canonical import build_model_view
        v = build_model_view('phira', (), (self._ev(0, 1_000_200), self._ev(1, 1_000_400)))  # distinct in source, same at 1 ms
        self.assertIn('duplicate_object_after_quantization', [i.code for i in v.quarantine])
