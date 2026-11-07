import numpy as np
from anchored_poisson import evaluate_anchored
from real_poisson_methods_v2 import WorkingPulse,evaluate_query


def test_anchored_count_fit_with_reversed_brightness_and_correct_primary():
    p=WorkingPulse();t=np.arange(192)
    h=np.tile(np.round(400*p.derivatives(t,60.,8.)[0]+1600*p.derivatives(t,110.,8.)[0]+100/192).astype(int),(9,9,1))
    b=evaluate_query(h,p,4,4);r=evaluate_anchored(h,p,4,4,b)
    assert h[4,4].sum()>1000 and r['proposal']['raw_center_excluded'] and r['proposal']['query_primary_used']
    assert abs(r['proposal']['primary']-110)<2 and abs(r['proposal']['secondary']-60)<2
    assert r['reports']['guided']['score']>0 and r['reports']['local']==b['reports']['local']
