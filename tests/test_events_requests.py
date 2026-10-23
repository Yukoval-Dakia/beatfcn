import itertools
import unittest
from beatfcn.events import controls as C
from beatfcn.events.canonical import Event, LaneLoc
from beatfcn.events.requests import (KINDS, LeakError, assert_no_leak, build_reference, build_request, resolve,
                                        sample_request)
from beatfcn.events.validate import (controls_recompute_issues, errors, leakage_issues, reference_evidence_issues,
                                        validate_request)
import numpy as np
from events_helpers import DURATION, synthetic_mania, synthetic_phira

MANIA, PHIRA = synthetic_mania(0), synthetic_phira(1)


def build(chart=MANIA, **kw):
    kw.setdefault('kind', 'from_scratch')
    kw.setdefault('seed', 0)
    return build_request(chart, audio_id='syn-audio', audio_duration_us=DURATION, **kw)


def empty_span(req):
    return [(r['start_us'], r['end_us']) for r in req['regions'] if r['state'] == 'empty']


class RequestKinds(unittest.TestCase):
    def test_every_kind_and_overlay_is_valid_and_leak_free(self):
        for chart, kind, empty, bh, seed in itertools.product((MANIA, PHIRA), KINDS, (None, 'natural', 'simulated'),
                                                              (False, True), (0, 1)):
            with self.subTest(mode=chart.mode, kind=kind, empty=empty, bh=bh, seed=seed):
                s = build(chart, kind=kind, seed=seed, empty_overlay=empty, boundary_hold=bh)
                self.assertEqual(errors(validate_request(s.request)), [])
                self.assertEqual(leakage_issues(s.request, s.hidden, s.evidence), [])
                self.assertEqual(controls_recompute_issues(s.request, s.hidden), [])
                self.assertTrue(s.ok, s.model_view.quarantine)
                obs = {d['event_id'] for d in s.request['observed_events']}
                hid = {e.event_id for e in s.hidden}
                rem = {r['event']['event_id'] for r in s.removed}
                self.assertEqual(obs | hid | rem, {e.event_id for e in chart.events})
                self.assertFalse(obs & hid or obs & rem or hid & rem)
                if kind == 'from_scratch' and not bh:
                    self.assertEqual(obs, set())
                if kind == 'reference':
                    self.assertTrue(s.request['references'])

    def test_determinism(self):
        a, b = sample_request(MANIA, 7, 'a', DURATION), sample_request(MANIA, 7, 'a', DURATION)
        self.assertEqual(a.request, b.request)
        self.assertEqual(a.hidden, b.hidden)
        self.assertNotEqual(sample_request(MANIA, 8, 'a', DURATION).request, a.request)

    def test_sampled_mix_runs(self):
        kinds = {sample_request(MANIA, s, 'a', DURATION).recipe['kind'] for s in range(40)}
        self.assertEqual(kinds, set(KINDS))


class Boundaries(unittest.TestCase):
    def test_hold_crossing_boundary_becomes_observed_whole(self):
        h1 = Event('h1', 4_000_000, 6_000_000, 'hold', LaneLoc(0), 'x#h1')  # head in complete -> O
        h2 = Event('h2', 6_500_000, 9_000_000, 'hold', LaneLoc(1), 'x#h2')  # head editable, tail in complete -> O whole
        h3 = Event('h3', 5_500_000, 7_000_000, 'hold', LaneLoc(2), 'x#h3')  # fully editable -> hidden
        regions = [(0, 5_000_000, 'complete'), (5_000_000, 8_000_000, 'editable'), (8_000_000, 10_000_000, 'complete')]
        obs, hid, prov = resolve([h1, h2, h3], regions)
        self.assertEqual([e.event_id for e in obs], ['h1', 'h2'])
        self.assertEqual(obs[1], h2)  # never cropped
        self.assertEqual([e.event_id for e in hid], ['h3'])
        self.assertEqual(prov, [{'event_id': 'h2', 'reason': 'hold_crosses_permission_boundary_observed_whole'}])

    def test_boundary_hold_overlay(self):
        hit = 0
        for seed in range(10):
            s = build(kind='from_scratch', seed=seed, boundary_hold=True)
            ov = [o for o in s.recipe['overlays'] if o['type'] == 'boundary_hold']
            if not ov:
                continue
            hit += 1
            h = MANIA.by_id()[ov[0]['hold']]
            obs = {d['event_id']: d for d in s.request['observed_events']}
            self.assertIn(h.event_id, obs)
            self.assertEqual((obs[h.event_id]['onset_us'], obs[h.event_id]['end_us']), (h.onset_us, h.end_us))
            self.assertIn(h.event_id, [p['event_id'] for p in s.recipe['permission_provenance']])
        self.assertGreater(hit, 5)

    def test_simulated_empty_removes_hidden_whole(self):
        removed_any = False
        for seed in range(12):
            s = build(kind='from_scratch', seed=seed, empty_overlay='simulated')
            (a, b), = empty_span(s.request)
            for r in s.removed:
                e = Event.from_json(r['event'])
                self.assertTrue(C.overlaps([(a, b)], *e.occupancy()))
                removed_any = True
            for e in s.hidden:
                self.assertFalse(C.overlaps([(a, b)], *e.occupancy()))
            for d in s.request['observed_events']:
                self.assertFalse(C.overlaps([(a, b)], *Event.from_json(d).occupancy()))
        self.assertTrue(removed_any)

    def test_natural_empty_is_really_blank(self):
        s = build(kind='middle_holes', seed=3, empty_overlay='natural')
        spans = empty_span(s.request)
        self.assertTrue(spans or 'natural_empty_unavailable' in s.recipe['notes'])
        for e in MANIA.events:
            self.assertFalse(C.overlaps(spans, *e.occupancy()))
        self.assertEqual(s.removed, ())

    def test_controls_recomputed_after_augmentation(self):
        s = build(kind='from_scratch', seed=2, empty_overlay='simulated', control_dropout=False)
        E = [(r['start_us'], r['end_us']) for r in s.request['regions'] if r['state'] == 'editable']
        heads = [e for e in s.hidden if C.contains(E, e.onset_us)]
        self.assertAlmostEqual(s.raw_controls['density_notes_per_second'], len(heads) * 1e6 / C.length(E))
        self.assertTrue(s.removed)
        self.assertEqual(s.request['controls']['density_notes_per_second'],
                         C.band_of('density_notes_per_second', s.raw_controls['density_notes_per_second']))


class References(unittest.TestCase):
    def test_reference_evidence_rules(self):
        cands = list(range(0, DURATION, 37_000))
        for seed in range(6):
            s = build(kind='reference', seed=seed, salient_candidates_us=cands)
            self.assertEqual(reference_evidence_issues(s.evidence, MANIA.events), [])
            for ev in s.evidence:
                self.assertEqual(ev['coverage'], 'complete')
                self.assertLessEqual(len(ev['selected']), 128)
                n_sal = ev['unselected_source'].count('salient')
                self.assertLessEqual(n_sal, len(ev['unselected_us']) // 2)
                if len(ev['unselected_us']) >= 2:
                    self.assertGreater(n_sal, 0)
                ids = {d['event_id'] for d in ev['selected']}
                self.assertFalse(ids & {e.event_id for e in s.hidden})

    def test_sparse_reference_has_no_unselected(self):
        ev = build_reference(MANIA.events, (0, 8_000_000), 'sparse', np.random.default_rng(0), salient_candidates_us=[10_000])
        self.assertEqual(ev['unselected_us'], [])
        self.assertTrue(ev['selected'])

    def test_unselected_exclusions_and_shortening(self):
        evs = [Event(f't{i}', 100_000 + i * 20_000, 100_000 + i * 20_000, 'tap', LaneLoc(i % 4), f'x#{i}') for i in range(200)]
        evs.append(Event('h', 5_000_000, 6_000_000, 'hold', LaneLoc(0), 'x#h'))
        ev = build_reference(evs, (0, 8_000_000), 'complete', np.random.default_rng(1))
        self.assertEqual(len(ev['selected']), 128)
        self.assertEqual(ev['span'], (0, evs[128].onset_us))
        self.assertEqual(ev['shortened_from'], (0, 8_000_000))
        self.assertEqual(reference_evidence_issues([ev], evs), [])
        ev = build_reference(evs, (4_000_000, 8_000_000), 'complete', np.random.default_rng(2), n_unselected=100)
        self.assertTrue(all(not 5_000_000 <= t < 6_000_000 for t in ev['unselected_us']))
        self.assertEqual(len(ev['unselected_us']), 100)


class Leakage(unittest.TestCase):
    def setUp(self):
        self.s = build(kind='middle_holes', seed=4)
        self.assertTrue(self.s.hidden)

    def test_clean_sample_passes(self):
        assert_no_leak(self.s.request, self.s.hidden, self.s.evidence)

    def test_injected_hidden_event_fires(self):
        req = dict(self.s.request, observed_events=self.s.request['observed_events'] + [self.s.hidden[0].to_json(locked=True)])
        with self.assertRaisesRegex(LeakError, 'hidden_id_in_condition'):
            assert_no_leak(req, self.s.hidden)

    def test_injected_hidden_content_with_new_id_fires(self):
        d = dict(self.s.hidden[0].to_json(locked=True), event_id='renamed')
        req = dict(self.s.request, observed_events=self.s.request['observed_events'] + [d])
        with self.assertRaisesRegex(LeakError, 'hidden_content_in_condition'):
            assert_no_leak(req, self.s.hidden)

    def test_reference_over_target_fires(self):
        h = self.s.hidden[0]
        ref = {'reference_id': 'bad', 'audio_id': 'syn-audio', 'mode': 'mania_4k', 'coverage': 'sparse',
               'span': {'start_us': h.onset_us, 'end_us': h.onset_us + 1_000_000}, 'events': [], 'native_chart_ref': 'x'}
        with self.assertRaisesRegex(LeakError, 'reference_span_overlaps_hidden_target'):
            assert_no_leak(dict(self.s.request, references=[ref]), self.s.hidden)

    def test_exact_count_control_fires(self):
        req = dict(self.s.request, controls={'density_notes_per_second': [3.37, 3.37]})
        with self.assertRaisesRegex(LeakError, 'control_not_binned'):
            assert_no_leak(req, self.s.hidden)


if __name__ == '__main__':
    unittest.main()
