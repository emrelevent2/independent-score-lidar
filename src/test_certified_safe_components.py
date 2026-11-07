from fractions import Fraction
import numpy as np
from flint import arb,ctx
from certified_safe_components import exact_safe_statistics,periodic_envelope,periodic_point_rates,mixture_log_lower,log_likelihood_interval
from safe_deadtime_exposure import safe_statistics


def test_exact_tick_exposures_equal_rational_hand_example_and_unscored_resets():
    ctx.prec=128;s=exact_safe_statistics([1.,3.,7.5],3,4,3.)
    np.testing.assert_array_equal(s['counts'],[0,1,0,1]);np.testing.assert_array_equal(s['eligible'],[True,False,True])
    assert [Fraction(v,s['denominator']) for v in s['exposure_ticks']]==[Fraction(1),Fraction(0),Fraction(3,2),Fraction(3,2)]


def test_exact_recovery_boundary_does_not_use_rounded_double_sum():
    ctx.prec=128;previous=0.1;lag=.2;rounded=previous+lag
    # Exact dyadic .1+.2 is BELOW its IEEE rounded sum: true tiny interval
    # must be retained and scored,not lost to cancellation.
    s=exact_safe_statistics([previous,rounded],1,2,lag)
    assert bool(s['eligible'][1])
    truth=Fraction(rounded)-Fraction(previous)-Fraction(lag)
    assert truth>0
    assert sum(s['exposure_ticks'])/s['denominator']>0


def test_exposure_endpoints_contain_independent_fraction_bin_intersections():
    ctx.prec=128;t=[.1,.31,1.7,2.21,5.333];lag=.27;k=4;cycles=2;s=exact_safe_statistics(t,cycles,k,lag)
    exposure=[Fraction(0) for _ in range(k)];left=Fraction(0)
    for right in [*map(Fraction,t),Fraction(cycles*k)]:
        if right>left:
            for j in range(left.numerator//left.denominator,-((-right.numerator)//right.denominator)):
                exposure[j%k]+=max(Fraction(0),min(right,Fraction(j+1))-max(left,Fraction(j)))
        left=right+Fraction(lag)
    assert exposure==[Fraction(v,s['denominator']) for v in s['exposure_ticks']]
    for lo,e,hi in zip(s['exposure_lower'],exposure,s['exposure_upper']):assert Fraction(float(lo))<=e<=Fraction(float(hi))


def test_periodic_delay_width_envelopes_cover_exact_wrapped_point_rates():
    ctx.prec=128;k=16
    for box in [(0.,.4,1.4,1.6),(5.1,6.2,1.5,1.5),(15.7,16.,1.4,1.6)]:
        lo,hi=periodic_envelope(k,*box)
        for d in np.linspace(box[0],box[1],4):
            for sigma in [box[2],box[3]]:
                rates=periodic_point_rates(k,1.,float(d),0.,sigma)
                for l,u,v in zip(lo,hi,rates):assert arb(float(l))<=v and arb(float(u))>=v


def test_point_wraps_and_full_period_probability_mass_are_enclosed():
    ctx.prec=128
    for d in [0.,3.75,16.]:
        r=periodic_point_rates(16,1.,d,0.,1.5)
        assert sum(r,arb(0)).contains(arb(1))
    a=periodic_point_rates(16,1.,0.,0.,1.5);b=periodic_point_rates(16,1.,16.,0.,1.5)
    assert all(x.overlaps(y) for x,y in zip(a,b))


def test_mixture_keeps_original_weights_and_likelihood_rounds_outward():
    ctx.prec=128;s=exact_safe_statistics([.1,2.3,5.2],2,4,.7)
    vectors=[np.array([.1,.2,.3,.4]),np.array([.4,.3,.2,.1])];weights=[Fraction(1,4),Fraction(3,4)]
    v=mixture_log_lower(s,vectors,weights);truth=(log_likelihood_interval(s,vectors[0]).exp()/4+3*log_likelihood_interval(s,vectors[1]).exp()/4).log()
    assert arb(v)<=truth
