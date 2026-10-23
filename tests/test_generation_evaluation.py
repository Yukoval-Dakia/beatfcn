from dataclasses import replace
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from beatfcn.events.canonical import Event, LaneLoc, ModelView
from beatfcn.events.requests import build_request
from beatfcn.model import features as F
from beatfcn.model.evaluation import generate_song, legality_issues, hold_matches, score_events, evaluate_songs
from beatfcn.model.sample import _hold_end, generate_core, core_context
from beatfcn.model.p1 import P1Model
from events_helpers import synthetic_mania, synthetic_phira, DURATION
from test_p1_model import CFG, song


def ev(id,on,end=None,lane=0):
    return Event(id,on,on if end is None else end,'tap' if end is None else 'hold',LaneLoc(lane))


class EvaluationTests(unittest.TestCase):
    def test_onset_match_does_not_hide_wrong_lane_or_note_type(self):
        truth = [ev('t', 1_000_000, lane=0)]
        for pred in ([ev('wrong-lane', 1_000_000, lane=1)],
                     [ev('wrong-type', 1_000_000, 2_000_000, lane=0)]):
            score = score_events(pred, truth, 'mania_4k')
            self.assertEqual(score['onset_20ms']['f1'], 1)
            self.assertEqual(score['note_20ms']['f1'], 0)
        score = score_events(truth * 2, truth, 'mania_4k')
        self.assertEqual(score['note_20ms']['tp'], 1)
        self.assertEqual(score['note_20ms']['pred'], 2)

    def test_empty_evaluation_cannot_be_valid_comparison(self):
        for count in (0, -1):
            with self.assertRaises(ValueError):
                evaluate_songs(None, None, [], n_songs=count)
        with self.assertRaises(ValueError):
            evaluate_songs(None, None, [], n_songs=4)

    def test_whole_evaluation_runs_both_modes_without_credit_for_empty_output(self):
        charts = {'mania': synthetic_mania(), 'phira': synthetic_phira()}
        rows = [{'chart_id': key, 'audio_sha256': key, 'mode': chart.mode,
                 'split': 'val', 'audio_sec': DURATION / 1e6} for key, chart in charts.items()]
        cache = song(0)
        corpus = SimpleNamespace(train_rows=lambda rows: rows, chart=charts.__getitem__,
                                 song=lambda key, in_memory: cache)
        # Actual tiny P1 encoder/decoder, with END chosen deterministically in each core.
        with patch('beatfcn.model.sample._pick', return_value=0):
            result = evaluate_songs(P1Model(CFG).eval(), corpus, rows, n_songs=2)
        self.assertTrue(result['valid_for_comparison'])
        self.assertEqual(set(result['summary']), {'mania_4k', 'phira'})
        for record in result['songs']:
            self.assertTrue(record['complete'])
            self.assertEqual(len(record['cores']), 4)
            self.assertEqual(record['scores']['note_20ms']['f1'], 0)
            self.assertGreater(record['scores']['note_20ms']['true'], 0)
        json.dumps(result, allow_nan=False)

    def test_wrong_tail_cannot_pass_joint_metric(self):
        pred=[ev('p',1_000_000,4_000_000)]
        truth=[ev('t',1_000_000,2_000_000)]
        score=score_events(pred,truth,'mania_4k')
        self.assertEqual(score['hold_head_20ms']['f1'],1)
        self.assertEqual(score['hold_joint_20ms']['f1'],0)
        self.assertEqual(score['tail_error_given_head_20ms']['mae_ms'],2000)
        self.assertEqual(hold_matches(pred*2,truth,20_000)['tp'],1)

    def test_joint_matching_finds_augmenting_path(self):
        p=[ev('p0',100,300),ev('p1',100,200)]
        t=[ev('t0',100,200),ev('t1',100,400)]
        self.assertEqual(hold_matches(p,t,0,100)['tp'],2)

    def test_independent_legality_checks_cross_core_and_permissions(self):
        s=build_request(synthetic_mania(),kind='from_scratch',seed=0,audio_id='a',audio_duration_us=DURATION)
        req={**s.request,'observed_events':[ev('locked',12_000_000).to_json()],
             'regions':[{'start_us':0,'end_us':20_000_000,'state':'editable'},
                        {'start_us':20_000_000,'end_us':DURATION,'state':'empty'}]}
        issues=legality_issues(req,[ev('long',7_000_000,13_000_000),ev('next',9_000_000),ev('bad',21_000_000)])
        codes={i['code'] for i in issues}
        self.assertIn('head_inside_hold',codes)
        self.assertIn('outside_editable',codes)

    def test_rollout_uses_predictions_and_never_hidden_truth(self):
        s=build_request(synthetic_mania(),kind='from_scratch',seed=0,audio_id='a',audio_duration_us=DURATION)
        # Poison both hidden representations: any use as context would be observable at the second core.
        poisoned=replace(s,hidden=(ev('poison',1_000_000,25_000_000,3),),model_view=ModelView('mania_4k',(),(),()))
        observations=[]
        def fake(model,ex,song,holds,runs,heads,**kwargs):
            observations.append((ex.c0_us,ex.known.copy(),list(holds)))
            self.assertEqual(ex.groups,[])
            kwargs['diagnostics'].update(terminated_by_end=True,group_limit_reached=False)
            if ex.c0_us==0:
                return [{'t_rel_us':7_000_000,'notes':[{'type':1,'lane':1,'end_rel_us':5_000_000}]}]
            return []
        model=SimpleNamespace(c=SimpleNamespace(phira_location=False))
        out=generate_song(model,poisoned,None,core_generator=fake)
        self.assertTrue(out['complete'])
        self.assertEqual(len(observations),4)
        self.assertEqual(observations[1][2],[(7_000_000,12_000_000,1)])
        self.assertEqual(observations[2][2],[])  # ended hold no longer reaches this halo
        self.assertEqual(out['legality'],[])

    def test_failed_core_is_not_skipped_or_replaced(self):
        s=build_request(synthetic_mania(),kind='from_scratch',seed=0,audio_id='a',audio_duration_us=DURATION)
        def bad(*args,**kw):
            raise F.SkipExample('known events exceed bound')
        result=generate_song(SimpleNamespace(c=SimpleNamespace(phira_location=False)),s,None,core_generator=bad)
        self.assertFalse(result['complete'])
        self.assertEqual(result['failure']['c0_us'],0)

    def test_hold_decoder_respects_future_observed_tap(self):
        model=P1Model(CFG).eval()
        # Always choose latest allowed endpoint, to exercise the upper bound rather than chance.
        def latest(logits,mask,*args):
            return int(torch.where(mask)[0][-1])
        with patch('beatfcn.model.sample._pick',side_effect=latest):
            end=_hold_end(model,torch.zeros(1,CFG.gru),song(0),1_000_000,{'lane':0},[],[(0,DURATION)],
                          None,1.,1.,True,'cpu',[(1_500_000,0,-1)])
        self.assertLessEqual(end,1_500_000)
        self.assertGreater(end,1_000_000)

    def test_copy_time_marks_observed_lane_busy(self):
        s=build_request(synthetic_mania(),kind='from_scratch',seed=0,audio_id='a',audio_duration_us=DURATION)
        s=replace(s,request={**s.request,'observed_events':[ev('locked',1_000_000,lane=0).to_json()]},
                  hidden=(),model_view=ModelView('mania_4k',(),(),()))
        ex=F.build_core(s,(),0,DURATION)
        holds,runs,heads=core_context(s,ex,{})
        seen=[]
        def inner(*args):
            seen.append(args[6].copy())  # busy lanes (after model, bt, h, pa, t_rel, right_new)
            return [{'type':0,'lane':1,'line':-1,'side':-1,'x':0.}]
        with patch('beatfcn.model.sample._pick',side_effect=[F.CORE_US//F.CELL_US+1,0]), \
             patch('beatfcn.model.sample._inner',side_effect=inner):
            groups=generate_core(P1Model(CFG).eval(),ex,song(0),holds,runs,heads)
        self.assertTrue(seen[0][0])
        self.assertEqual(len(groups),1)

if __name__=='__main__':
    unittest.main()
