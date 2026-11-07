import numpy as np
from confirmatory_scenes import make_case as original
from guided_confirmatory_scenes import make_case


def test_default_observations_match_previously_validated_generator_exactly():
    for kind in ['null','positive']:
        a,ma,_=original(619904,719904,kind,6.)
        b,mb,_=make_case(619904,719904,kind,6.)
        assert np.array_equal(a,b)
        assert all(ma[k]==mb[k] for k in ma)


def test_wide_return_family_is_inside_sensor_window_and_preserves_null_labels():
    for i in range(20):
        for kind in ['null','positive']:
            h,m,truth=make_case(619910+i,719910+i,kind,6.,separation_range=(48.,80.))
            assert np.all(h.sum(-1)<=1000) and h.shape==(16,16,256)
            assert m['positive']==(kind=='positive')
            assert np.max(truth['secondary_delay'])<240
            if kind=='positive':assert 48<=m['secondary_delay']-m['primary_delay']<=80
