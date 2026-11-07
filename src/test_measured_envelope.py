import numpy as np
from flint import arb
from measured_envelope import MeasuredResponse,measured_box,measured_null_bound,measured_alternative_at,measured_evidence
from envelope_evidence import certified_log_likelihood,_observations


def independent_pulse(irf,bins,delay,stretch):
    samples=[arb(0)]+[arb(float(x)) for x in irf]+[arb(0)];peak=int(np.argmax(irf))+1
    out=[]
    for t in range(bins):
        u=(arb(t)-arb(float(delay)))/arb(float(stretch))+peak
        # Test points are deliberately away from uncertain branch boundaries.
        j=int(np.floor(float(u)))
        if j<0 or j>=len(samples)-1:out.append(arb(0))
        else:out.append(samples[j]+(u-j)*(samples[j+1]-samples[j]))
    return out


def test_nonsmooth_multilobed_response_envelopes_cover_shifts_stretches_and_edges():
    irf=np.array([.3,1.,4.,.1,2.,.7,.4]);r=MeasuredResponse(irf)
    for d0,d1,s0,s1 in [(0.,1.2,.85,1.15),(25.1,25.8,.91,1.09),(61.8,63.,.9,1.1)]:
        lo,hi=r.envelope(64,d0,d1,s0,s1,.07)
        for d in np.linspace(d0+.01,d1-.01,6):
            for s in np.linspace(s0+.003,s1-.003,5):
                actual=independent_pulse(irf,64,float(d),float(s))
                assert all(arb(float(a))<=g*(arb(1)-arb(.07)) and arb(float(b))>=g*(arb(1)+arb(.07)) for a,b,g in zip(lo,hi,actual))


def test_zero_padding_is_continuous_at_both_measured_support_boundaries():
    r=MeasuredResponse([.3,1.,.4]);a=r.pulse(32,12.,1.)
    assert a[10]==0 and a[11]==.3 and a[14]==0
    assert np.max(np.abs(a-r.pulse(32,12.+1e-8)))<1e-7
    exact=r.exact_pulse(32,12.,1.)
    assert all(g.contains(float(x)) for g,x in zip(exact,a))


def test_calibration_knot_intervals_cover_independent_admissible_response_curves():
    nominal=np.array([.2,.6,1.,.3,.1]);low=np.maximum(nominal-.15,0);high=nominal+.15
    r=MeasuredResponse(nominal,low,high);rng=np.random.default_rng(61004)
    lo,hi=r.envelope(96,40.1,42.8,.9,1.1,0.)
    for _ in range(30):
        curve=rng.uniform(low,high);d=rng.uniform(40.1,42.8);s=rng.uniform(.9,1.1)
        # All allowed curves preserve this nominal peak's index in this test.
        pulse=independent_pulse(curve,96,d,s)
        assert all(arb(float(a))<=g and arb(float(b))>=g for a,b,g in zip(lo,hi,pulse))


def test_box_bounds_dominate_feasible_shape_band_models_both_families():
    irf=[.1,.5,2.,4.,2.1,.4];r=MeasuredResponse(irf);rng=np.random.default_rng(61001)
    h=rng.poisson(8*r.pulse(64,30.37,1.02)+15/64)
    for family,cycles in [('poisson',None),('first',1000)]:
        _,_,_,sat=_observations(h,cycles,family)
        box=measured_box(h,r,29.5,31.2,.9,1.1,sat,.05,family,cycles)
        for d in [29.53,30.37,31.13]:
            for s in [.913,1.023,1.087]:
                pulse=independent_pulse(irf,64,d,s)
                for a,b in [(0.,40.),(8.,15.),(30.,1.),(100.,500.)]:
                    rates=[(a*g*(arb(1)+arb(.05)*int((-1)**t))+arb(b)/64)/(1000 if cycles else 1) for t,g in enumerate(pulse)]
                    ll=certified_log_likelihood(h,rates,cycles,family,upper=True)
                    assert box.upper>=ll-1e-10


def test_early_stopped_complete_cover_bounds_true_measured_pulse_likelihood():
    r=MeasuredResponse([.1,.5,2.,4.,2.1,.4]);rng=np.random.default_rng(61002)
    h=rng.poisson(8*r.pulse(64,30.37,1.02)+15/64)
    means=[8*g+arb(15)/64 for g in r.exact_pulse(64,30.37,1.02)]
    truth=certified_log_likelihood(h,means,family='poisson',upper=True)
    for budget in [8,32]:
        b=measured_null_bound(h,r,initial_cells=8,max_boxes=budget)
        assert b['upper']>=truth-1e-10 and b['gap']>=0 and b['boxes_evaluated']<=budget


def test_measured_alternative_exactly_excludes_query_and_has_proper_component_means():
    r=MeasuredResponse([.1,.5,2.,4.,2.1,.4]);rng=np.random.default_rng(61003)
    mean=20*r.pulse(64,21.3,1.02,True)+6*r.pulse(64,40.2,1.02,True)+15/64
    h=rng.poisson(np.broadcast_to(mean,(12,12,64)))
    a,p=measured_alternative_at(h,r,6,6)
    changed=h.copy();changed[6,6]=0;changed[6,6,50]=1000
    b,q=measured_alternative_at(changed,r,6,6)
    assert p==q and all(np.array_equal(x,y) for x,y in zip(a,b))
    assert p['donors']==80 and len(a)==4 and all(np.all(x>0) for x in a)
    result=measured_evidence(h[6,6],a,r,initial_cells=8,max_boxes=32)
    assert result['alpha']==.01 and result['log_threshold_upper']>=np.log(100)
    assert result['declared']==(result['log_e_lower']>=result['log_threshold_upper'])
