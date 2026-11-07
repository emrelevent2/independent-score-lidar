import numpy as np
from flint import arb
from pilot import first_detection_probabilities
from envelope_evidence import exact_gaussian,certified_log_likelihood,certified_mixture_likelihood,robust_mixture_evidence
from adaptive_evidence import mixture_interval,feasible_local_null,adaptive_evidence


def test_mixture_interval_encloses_independently_computed_component_average():
    h=np.zeros(64,int);h[[20,21,35]]=[15,12,4]
    means=[(a*np.asarray([float(x) for x in exact_gaussian(64,20.3,5.2)])+12/64)/1000 for a in [20.,30.,50.]]
    interval=mixture_interval(h,means)
    lower=certified_mixture_likelihood(h,means)
    assert interval['lower']<=lower+1e-10 and interval['upper']>=lower-1e-10
    assert interval['upper']-interval['lower']<1e-9


def test_local_witness_is_feasible_and_checked_against_its_actual_parameters():
    rng=np.random.default_rng(619901);g=np.asarray([float(x) for x in exact_gaussian(64,30.37,5.2)])
    h=rng.multinomial(1000,first_detection_probabilities((40*g+15/64)/1000))[:-1]
    result=feasible_local_null(h);a,b,d,w=result['parameters']
    rates=[(arb(a)*x+arb(b)/64)/1000 for x in exact_gaussian(64,d,w)]
    actual=certified_log_likelihood(h,rates,1000,'first',upper=True)
    assert result['lower']<=actual and actual-result['lower']<1e-9


def test_empty_observation_is_certifiably_rejected_without_global_boxes():
    h=np.zeros(64,int);means=[np.full(64,30/64/1000)]
    result=adaptive_evidence(h,means,initial_cells=8,max_boxes=64)
    assert not result['declared'] and result['decision']=='rejected_by_upper_evidence'
    assert result['boxes_evaluated']==0 and result['log_e_upper']<result['log_threshold_lower']
    full=robust_mixture_evidence(h,means,initial_cells=8,max_boxes=64,normalized_envelope=False)
    assert not full['declared']


def test_accepted_two_return_observation_has_complete_cover_certificate():
    rng=np.random.default_rng(619902)
    p=np.asarray([float(x) for x in exact_gaussian(128,40.3,5.2)])
    s=np.asarray([float(x) for x in exact_gaussian(128,85.2,5.2)])
    means=(70*p+35*s+8/128)/1000
    h=rng.multinomial(1000,first_detection_probabilities(means))[:-1]
    result=adaptive_evidence(h,[means],initial_cells=8,max_boxes=512)
    assert result['declared'] and result['log_e_lower']>=result['log_threshold_upper']
    assert result['log_e_lower']<=result['log_e_upper']
    for d in [0.,20.7,40.3,60.1,85.2,127.]:
        rates=[(60*g+arb(20)/128)/1000 for g in exact_gaussian(128,d,5.2)]
        actual=certified_log_likelihood(h,rates,1000,'first',upper=True)
        assert result['null_upper']>=actual-1e-10
