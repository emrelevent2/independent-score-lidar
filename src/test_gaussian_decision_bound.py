import numpy as np
from flint import arb
from gaussian_decision_bound import decision_bound,window_contract,checked_witness
from envelope_evidence import exact_gaussian,certified_log_likelihood,_observations
from pilot import first_detection_probabilities


def rates(a,b,d,w,bins=64):
    return [(arb(float(a))*g+arb(float(b))/bins)/1000 for g in exact_gaussian(bins,d,w,False)]


def sample():
    h=np.random.default_rng(55693).multinomial(1000,first_detection_probabilities([float(x) for x in rates(18.,8.,30.3,5.1)]))[:-1]
    return h


def test_both_stopped_backends_cover_full_null_including_infinite_domain_tails():
    h=sample();level=float(certified_log_likelihood(h,rates(18.,8.,30.3,5.1),1000,'first'))+5
    for backend in ['ordinary','contracted']:
        b=decision_bound(h,level,backend,initial_cells=4,max_boxes=48)
        assert b['upper'] is not None
        for a in [0.,18.,100.,1000.]:
            for bg in [0.01,8.,80.,1000.]:
                for d in [0.,30.3,63.]:
                    ll=certified_log_likelihood(h,rates(a,bg,d,5.1),1000,'first',upper=True)
                    assert b['upper']>=ll


def test_new_geometry_contractor_keeps_an_above_level_physical_point():
    from query_likelihood_contractor import zero_background_cap
    from coupled_corner_profile import amplitude_cap
    h=sample();_,remaining,_,sat=_observations(h,1000,'first');level=certified_log_likelihood(h,rates(18.,8.,30.3,5.1),1000,'first')-10
    A=amplitude_cap(h,1000,0,1,level)['cap'];B=zero_background_cap(h,remaining,1000,sat,0.,level)['background_cap']
    info=window_contract(h,remaining,1000,A,B,(22,40),level,30.2,30.4,5.0,5.2)
    assert not info['empty'] and info['amplitude_range'][0]<=18<=info['amplitude_range'][1] and B>=8


def test_null_witness_is_recertified_instead_of_trusting_supplied_value():
    h=sample();w=dict(parameters=[100.,8.,30.3,5.1],lower=999999.)
    value=checked_witness(h,1000,w,(4.5,5.5));assert value<0
    result=decision_bound(h,value-1,'ordinary',witness=w)
    assert result['upper'] is None and result['feasible_lower']==value and result['boxes_evaluated']==0
