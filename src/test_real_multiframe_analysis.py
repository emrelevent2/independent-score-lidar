import numpy as np
from real_poisson_analysis_v2 import endpoints
from real_multiframe_analysis import block_difference


def test_interval_pair_endpoint_is_brightness_invariant_and_one_to_one():
    q=dict(windows=[[30,35],[90,95]],weak_window=0)
    assert endpoints(q,dict(primary=92.,delay=32.),True)['pair']
    assert not endpoints(q,dict(primary=32.,delay=33.),True)['pair']
    assert not endpoints(q,dict(primary=32.,delay=92.),False)['pair']


def test_known_window_duplicate_is_not_unmatched_or_physical_negative():
    q=dict(windows=[[30,40]])
    e=endpoints(q,dict(primary=33.,delay=36.),True)
    assert e['duplicate_known_window'] and not e['accepted_unmatched'] and not e['pair']
    assert endpoints(q,dict(primary=33.,delay=80.),True)['accepted_unmatched']


def test_block_resampling_preserves_pairing_and_exact_identical_difference():
    qs=[dict(row=i*16,column=0) for i in range(6)]
    es=[dict(pair=i%2==0) for i in range(6)]
    c=block_difference(qs,es,es,'pair',reps=100)
    assert c['difference']==0 and c['interval_95']==[0.,0.] and c['sampled_blocks']==6
    other=[dict(pair=True) for _ in range(6)]
    c=block_difference(qs,other,es,'pair',reps=100)
    assert np.isclose(c['difference'],.5) and c['interval_95'][0]>=0
