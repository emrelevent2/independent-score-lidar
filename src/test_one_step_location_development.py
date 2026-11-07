import json
from one_step_location_development import attach_locations


def test_point_comparison_preserves_every_detection_and_original_record():
    row=dict(samples={m:dict(score=12.,pvalue=.01,bootstrap_scores=[3.,4.,5.],primary=7.,delay=17.) for m in ['local_score','local_joint','context_max','context_transport']});row['samples']['local_score']['delay']=18.;before=json.dumps(row,sort_keys=True);point=dict(primary=7.1,delay=17.4);out=attach_locations(row,point)
    assert json.dumps(row,sort_keys=True)==before
    for m in row['samples']:assert out['samples'][m]==row['samples'][m]
    for m in ['transport_score_location','transport_one_step_location']:
        for key in ['score','pvalue','bootstrap_scores']:assert out['samples'][m][key]==row['samples']['context_transport'][key]
    assert out['samples']['transport_score_location']['delay']==18. and out['samples']['transport_one_step_location']['delay']==17.4
