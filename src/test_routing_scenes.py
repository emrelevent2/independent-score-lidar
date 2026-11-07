import numpy as np
from routing_scenes import make_routing_case
from guided_confirmatory_scenes import make_case


def test_coherent_generator_matches_frozen_scene_generator_exactly():
    c=dict(scene_seed=1419000,noise_seed=2419000,kind='positive',secondary=6.,separation_family=1,support_family='coherent')
    h,m,_=make_routing_case(c);expected,metadata,_=make_case(c['scene_seed'],c['noise_seed'],'positive',6.,separation_range=(48.,80.))
    assert np.array_equal(h,expected) and m==metadata


def test_new_support_families_have_valid_physical_truth_and_repeatable_noise():
    for family in ['singleton','filament','crossing_step']:
        c=dict(scene_seed=1419100,noise_seed=2419100,kind='positive',secondary=6.,separation_family=0,support_family=family)
        h,m,t=make_routing_case(c);again,metadata,_=make_routing_case(c)
        assert np.array_equal(h,again) and m==metadata
        assert h.shape==(16,16,256) and np.all(h>=0) and np.all(h.sum(-1)<=1000)
        assert t['secondary_mask'][8,8] and m['secondary_delay']==t['secondary_delay'][8,8]
        if family=='singleton':assert t['secondary_mask'].sum()==1
        if family=='filament':assert 1<t['secondary_mask'].sum()<40
        if family=='crossing_step':
            separation=t['secondary_delay']-t['primary_delay']
            assert t['secondary_mask'].all() and np.ptp(separation)>5.
