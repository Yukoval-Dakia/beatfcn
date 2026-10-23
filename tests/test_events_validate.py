import copy
import json
import unittest
from beatfcn.events.osu_mania import parse_osu
from beatfcn.events.requests import build_request
from beatfcn.events.validate import cf_pair_issues, errors, validate_request, validate_response
from events_helpers import CONTRACTS, FIX

REQ = json.loads((CONTRACTS / 'request.example.json').read_text())
RESP = json.loads((CONTRACTS / 'response.example.json').read_text())


def codes(issues):
    return {i['code'] for i in errors(issues)}


class PackageExample(unittest.TestCase):
    def test_example_request_passes_schema_and_semantics(self):
        out = validate_request(REQ)
        self.assertEqual(errors(out), [], out)

    def test_example_response_passes(self):
        self.assertEqual(validate_response(REQ, RESP), [])


class Mutations(unittest.TestCase):
    def mutate(self, f):
        r = copy.deepcopy(REQ)
        f(r)
        return codes(validate_request(r))

    def test_region_and_event_errors(self):
        self.assertIn('region_overlap', self.mutate(lambda r: r['regions'][1].update(start_us=7_000_000)))
        self.assertIn('region_gap', self.mutate(lambda r: r['regions'][1].update(start_us=9_000_000)))
        self.assertIn('regions_do_not_cover_audio', self.mutate(lambda r: r['regions'][3].update(end_us=29_000_000)))
        self.assertIn('duplicate_region_id', self.mutate(lambda r: r['regions'][1].update(region_id='done-left')))
        self.assertIn('non_positive_hold', self.mutate(lambda r: r['observed_events'][2].update(end_us=7_000_000)))
        self.assertIn('non_hold_end_ne_onset', self.mutate(lambda r: r['observed_events'][0].update(end_us=1_000_001)))
        self.assertIn('event_outside_audio', self.mutate(lambda r: r['observed_events'][3].update(onset_us=31_000_000, end_us=31_000_000)))
        self.assertIn('schema', self.mutate(lambda r: r['observed_events'][0].update(type='flick')))

    def test_reference_and_control_errors(self):
        self.assertIn('reference_event_outside_span', self.mutate(lambda r: r['references'][0]['span'].update(end_us=2_000_000)))
        self.assertIn('complete_reference_without_native_chart', self.mutate(lambda r: r['references'][0].update(native_chart_ref=None)))
        self.assertIn('band_low_gt_high', self.mutate(lambda r: r['controls'].update(density_notes_per_second=[4, 2])))
        self.assertIn('unsupported_type_for_mode', self.mutate(lambda r: r['controls'].update(allowed_types=['tap', 'flick'])))
        self.assertIn('unsupported_control_for_mode', self.mutate(lambda r: r['controls'].update(flick_ratio=[0.1, 0.2])))
        self.assertIn('unsatisfiable_hold_control', self.mutate(lambda r: r['controls'].update(allowed_types=['tap'])))
        self.assertIn('two_finger_without_profile', self.mutate(lambda r: r.update(hard_profile_ref=None) or r['controls'].update(max_active_contacts=2)))
        self.assertIn('locked_lane_conflict', self.mutate(lambda r: r['observed_events'][0].update(
            onset_us=8_000_000, end_us=8_000_000, location={'kind': 'lane', 'lane': 1})))

    def test_response_errors(self):
        def resp(*adds, **kw):
            return {i['code'] for i in validate_response(REQ, dict(RESP, additions=list(adds), **kw))}
        def ev(i, on, end, ty, lane):
            return {'event_id': f'g{i}', 'onset_us': on, 'end_us': end, 'type': ty, 'location': {'kind': 'lane', 'lane': lane}}
        self.assertIn('addition_outside_editable', resp(ev(1, 5_000_000, 5_000_000, 'tap', 0)))          # complete
        self.assertIn('addition_outside_editable', resp(ev(1, 15_000_000, 17_000_000, 'hold', 0)))       # into empty
        self.assertIn('addition_lane_conflict', resp(ev(1, 8_500_000, 8_500_000, 'tap', 1)))             # active O hold
        self.assertIn('addition_lane_conflict', resp(ev(1, 9_000_000, 9_000_000, 'tap', 2), ev(2, 9_000_000, 9_000_000, 'tap', 2)))
        self.assertIn('addition_id_reuses_observed', resp(dict(ev(1, 10_000_000, 10_000_000, 'tap', 0), event_id='human-1')))
        self.assertIn('observed_ids_not_preserved', resp(unchanged_observed_event_ids=['human-1']))
        self.assertEqual(resp(ev(1, 9_200_000, 9_200_000, 'tap', 1)), set())   # hold end is half-open

    def test_phira_scene_line_ids(self):
        r = copy.deepcopy(REQ)
        r.update(mode='phira', scene_ref='scene', references=[], controls={})
        for d in r['observed_events']:
            d['location'] = {'kind': 'line', 'line_id': '0', 'side': 'above', 'native_x': 0.0}
        self.assertEqual(errors(validate_request(r, scene_line_ids={'0'})), [])
        r['observed_events'][0]['location']['line_id'] = '7'
        self.assertIn('unknown_line_id', codes(validate_request(r, scene_line_ids={'0'})))


class ChartBacked(unittest.TestCase):
    def test_observed_must_equal_import(self):
        chart = parse_osu((FIX / 'mania4k_synthetic.osu').read_bytes(), 'm.osu')
        s = build_request(chart, kind='continuation', seed=3, audio_id='a', audio_duration_us=8_000_000)
        self.assertEqual(errors(validate_request(s.request, chart=chart)), [])
        req = copy.deepcopy(s.request)
        if req['observed_events']:
            req['observed_events'][0]['onset_us'] += 1
            req['observed_events'][0]['end_us'] += 1
            self.assertIn('observed_differs_from_import', codes(validate_request(req, chart=chart)))

    def test_complete_reference_missing_event(self):
        chart = parse_osu((FIX / 'mania4k_synthetic.osu').read_bytes(), 'm.osu')
        r = copy.deepcopy(REQ)
        r['references'][0].update(native_chart_ref='m.osu', span={'start_us': 0, 'end_us': 2_000_000}, events=[])
        self.assertIn('complete_reference_missing_event', codes(validate_request(r, ref_charts={'m.osu': chart})))


class Counterfactual(unittest.TestCase):
    def test_cf_pair_checks(self):
        a = [{'time_us': 100_000, 'attrs': 'MASK', 'selected': True}, {'time_us': 300_000, 'attrs': 'MASK', 'selected': False}]
        b = [dict(x, selected=not x['selected']) for x in a]
        self.assertEqual(cf_pair_issues(a, b), [])
        self.assertIn('non_selected_field_differs', {i['code'] for i in cf_pair_issues(a, [dict(b[0], time_us=1), b[1]])})
        self.assertIn('attributes_not_masked', {i['code'] for i in cf_pair_issues([dict(a[0], attrs={'lane': 1}), a[1]],
                                                                                [dict(b[0], attrs={'lane': 1}), b[1]])})
        shared = [dict(a[0]), dict(a[1], time_us=110_000)]
        self.assertIn('shared_selection_cell', {i['code'] for i in cf_pair_issues(shared, [dict(x, selected=not x['selected']) for x in shared])})


if __name__ == '__main__':
    unittest.main()
