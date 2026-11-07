"""Meaningful mathematical checks independent of the optimization path."""
import numpy as np
from flint import arb
from pilot import first_detection_probabilities,simulate
from envelope_evidence import (gaussian_envelope,exact_gaussian,outward_float,
    relaxed_value_gradient,bound_box,continuous_null_bound,certified_log_likelihood,
    independent_alternative,_observations)


def test_gaussian_envelopes_cover_continuous_delays_widths_and_edges():
    for d0,d1,f0,f1 in [(0.,1.2,3.5,7.),(19.5,20.5,4.5,5.5),(62.8,63.,4.,6.)]:
        lo,hi=gaussian_envelope(64,d0,d1,f0,f1)
        for d in np.linspace(d0,d1,9):
            for f in np.linspace(f0,f1,7):
                actual=exact_gaussian(64,float(d),float(f))
                assert all(arb(float(a))<=g and arb(float(b))>=g for a,b,g in zip(lo,hi,actual))


def test_outward_conversion_covers_nonrepresentable_values():
    for value in [arb(2).log(),arb(1)/3,-arb(2).sqrt(),arb('1e-400')]:
        lower=outward_float(value,False);upper=outward_float(value,True)
        assert arb(lower)<=value and arb(upper)>=value


def test_relaxed_gradient_matches_finite_differences_both_models():
    rng=np.random.default_rng(14001)
    pulse=np.asarray([float(x) for x in exact_gaussian(64,20.3,5.2)])
    lo,hi=gaussian_envelope(64,19.8,20.8,4.8,5.5)
    for family in ['first','poisson']:
        mean=30*pulse+20/64
        h=(rng.multinomial(1000,first_detection_probabilities(mean/1000))[:-1]
           if family=='first' else rng.poisson(mean))
        counts,fail,scale,_=_observations(h,1000,family)
        theta=np.array([23.1,18.7])
        _,gradient=relaxed_value_gradient(theta,counts,fail,lo,hi,scale,family)
        for i in range(2):
            eps=np.eye(2)[i]*1e-4
            plus=relaxed_value_gradient(theta+eps,counts,fail,lo,hi,scale,family)[0]
            minus=relaxed_value_gradient(theta-eps,counts,fail,lo,hi,scale,family)[0]
            assert np.isclose(gradient[i],(plus-minus)/2e-4,rtol=1e-5,atol=2e-7)


def test_certified_box_dominates_dense_feasible_models_and_concavity():
    rng=np.random.default_rng(14002)
    pulse=np.asarray([float(x) for x in exact_gaussian(64,20.3,5.2)])
    for family in ['first','poisson']:
        mean=30*pulse+20/64
        h=(rng.multinomial(1000,first_detection_probabilities(mean/1000))[:-1]
           if family=='first' else rng.poisson(mean))
        counts,fail,scale,saturated=_observations(h,1000,family)
        box=bound_box(counts,fail,scale,family,19.5,20.5,4.5,5.5,saturated)
        assert box.certified
        for d in [19.5,19.83,20.3,20.5]:
            for width in [4.5,5.,5.5]:
                g=exact_gaussian(64,d,width)
                for a,b in [(0.,40.),(20.,10.),(30.,20.),(80.,.5),(500.,200.)]:
                    rates=[(arb(a)*q+arb(b)/64)/scale for q in g]
                    ll=certified_log_likelihood(h,rates,1000,family,upper=True)
                    assert box.upper>=ll-1e-10
        lo,hi=gaussian_envelope(64,19.5,20.5,4.5,5.5)
        for _ in range(20):
            a=rng.uniform(1,100,2);b=rng.uniform(1,100,2);weight=rng.random()
            f=lambda x:relaxed_value_gradient(x,counts,fail,lo,hi,scale,family)[0]
            assert f(weight*a+(1-weight)*b)>=weight*f(a)+(1-weight)*f(b)-1e-9


def test_complete_cover_early_stopping_remains_above_true_likelihood():
    rng=np.random.default_rng(14003)
    g=exact_gaussian(64,31.37,5.25)
    rates=[(30*q+arb(20)/64)/1000 for q in g]
    h=rng.multinomial(1000,first_detection_probabilities(np.asarray([float(x) for x in rates])))[:-1]
    actual=certified_log_likelihood(h,rates,1000,upper=True)
    for budget in [8,24,64]:
        bound=continuous_null_bound(h,fwhm_range=(4.5,5.5),initial_cells=8,max_boxes=budget)
        assert bound['upper']>=actual-1e-10
        assert bound['gap']>=0 and bound['boxes_evaluated']<=budget


def test_direct_donor_convolution_excludes_center_exactly():
    h,_,_=simulate(14004,size=12)
    altered=h.copy();altered[6,6]=0;altered[6,6,170]=700
    a,p,s,v=independent_alternative(h,size=9)
    b,p2,s2,v2=independent_alternative(altered,size=9)
    assert np.array_equal(a[6,6],b[6,6])
    assert p[6,6]==p2[6,6] and s[6,6]==s2[6,6] and v[6,6]==v2[6,6]


def test_continuous_alternative_parameters_exclude_center_exactly():
    from continuous_alternative import alternative_at
    h,_,_=simulate(14005,size=12,offset=.37,pulse_fwhm=5.2)
    altered=h.copy();altered[6,6]=0;altered[6,6,170]=700
    a,parameters=alternative_at(h,6,6)
    b,parameters2=alternative_at(altered,6,6)
    assert np.array_equal(a,b) and parameters==parameters2
    assert 4.5<=parameters['fitted_fwhm']<=5.5


def test_peak_parameterization_covers_the_normalized_physical_null():
    rng=np.random.default_rng(14006)
    pulse=exact_gaussian(64,31.37,5.25)
    rates=[(arb(80)*g+arb(20)/64)/1000 for g in pulse]
    h=rng.multinomial(1000,first_detection_probabilities(np.array([float(x) for x in rates])))[:-1]
    actual=certified_log_likelihood(h,rates,1000,upper=True)
    result=continuous_null_bound(h,fwhm_range=(4.5,5.5),initial_cells=8,max_boxes=64,normalized_envelope=False)
    assert result['upper']>=actual-1e-10
    assert result['parameterization']=='peak amplitude'


def test_certified_mixture_is_probability_average_not_fitted_maximum():
    from envelope_evidence import certified_mixture_likelihood
    from scipy.special import logsumexp
    from likelihood_evidence import log_likelihood
    rng=np.random.default_rng(14007)
    a=np.full(32,.001);b=np.full(32,.003)
    h=rng.multinomial(1000,first_detection_probabilities(a))[:-1]
    actual=logsumexp([log_likelihood(h,1000,a),log_likelihood(h,1000,b)])-np.log(2)
    lower=certified_mixture_likelihood(h,[a,b])
    assert lower<=actual+1e-10 and actual-lower<1e-8
    repeated=certified_mixture_likelihood(h,[a,a,a,a])
    assert np.isclose(repeated,log_likelihood(h,1000,a),atol=1e-9)
