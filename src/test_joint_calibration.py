from fractions import Fraction
from math import comb,factorial
import numpy as np
from flint import arb
from pilot import first_detection_probabilities
from envelope_evidence import exact_gaussian,certified_log_likelihood,_observations
from joint_calibration_model import split_cycles,aggregate,calibration_basis,query_basis,fit_training,EDGES
from linear_envelope_profile import profile_upper,value_gradient
from joint_calibration_envelope import joint_null_bound


def multinomial_mass(h,p):
    n=sum(h);coef=factorial(n)
    for k in h:coef//=factorial(k)
    value=Fraction(coef)
    for k,x in zip(h,p):value*=x**k
    return value


def test_fixed_cycle_split_exact_joint_law_including_no_event():
    # Exact small-outcome enumeration proves the multinomial/hypergeometric
    # factorization; independence is marginal, not conditional on parent counts.
    p=[Fraction(1,5),Fraction(1,4),Fraction(11,20)]
    vectors=[(a,b,3-a-b) for a in range(4) for b in range(4-a)]
    for a in vectors:
        for b in vectors:
            total=tuple(x+y for x,y in zip(a,b));hyper=Fraction(np.prod([comb(t,x) for t,x in zip(total,a)]),comb(6,3))
            assert multinomial_mass(total,p)*hyper==multinomial_mass(a,p)*multinomial_mass(b,p)
    h=np.zeros(256,dtype=int);h[30]=200;h[47]=30;h[200]=10
    a,b,n1,n2=split_cycles(h,1001,7515);assert n1==500 and n2==501 and np.array_equal(a+b,h)
    assert a.sum()<=n1 and b.sum()<=n2 and np.array_equal(split_cycles(h,1001,7515)[0],a)


def test_coarse_hazard_model_is_the_exact_aggregation_of_first_events():
    rates=np.full(256,.0002);rates[30]=.04;rates[48]=.005
    fine=first_detection_probabilities(rates);coarse=first_detection_probabilities(aggregate(rates))
    assert np.allclose(coarse,np.r_[aggregate(fine[:-1]),fine[-1]],rtol=1e-13,atol=1e-15)


def test_normalized_calibration_envelope_preserves_shared_lobe_ratio():
    lo,hi=calibration_basis(.03125,.125)
    for d in [31.5,31.73,32.5]:
        for w in [4.5,5.13,5.5]:
            g=exact_gaussian(256,d,w,False);side=exact_gaussian(272,arb(d)+16,w,False)[:256]
            a=sum(g[27:38],arb(0))
            for eta in [.03125,.071,.125]:
                for j,(start,stop) in enumerate(zip(EDGES[:-1],EDGES[1:])):
                    mass=sum((x+arb(eta)*y for x,y in zip(g[start:stop],side[start:stop])),arb(0))/a
                    assert mass>=arb(float(lo[j,0])) and mass<=arb(float(hi[j,0]))
                    assert arb(float(lo[j,1]))==arb(stop-start)/256==arb(float(hi[j,1]))


def test_generic_profile_gradient_and_upper_bound_with_unequal_background_bins():
    h=np.array([2,20,1,4,2,9]);n=1000
    lo,hi=calibration_basis(.03125,.125);counts,remaining,_,_=_observations(h,n,'first')
    x=np.array([25.,15.]);v,g=value_gradient(x,counts,remaining,lo,hi,n)
    for j in range(2):
        e=np.zeros(2);e[j]=1e-4
        fd=(value_gradient(x+e,counts,remaining,lo,hi,n)[0]-value_gradient(x-e,counts,remaining,lo,hi,n)[0])/(2e-4)
        assert abs(fd-g[j])<1e-6
    bound=profile_upper(h,lo,hi,n,x)
    assert bound['tangent_certified']
    for theta in [[0.,30.],[30.,0.1],[70.,50.]]:
        # Every coefficient vector in the enclosure belongs to the relaxation.
        rates=[(arb(float(a))*arb(theta[0])+arb(float(b))*arb(theta[1]))/n for a,b in (lo+hi)/2]
        assert certified_log_likelihood(h,rates,n,'first',upper=True)<=bound['upper']


def test_early_stopped_joint_cover_contains_a_true_shared_response_likelihood():
    rng=np.random.default_rng(7516);eta=.0625
    cg=exact_gaussian(256,31.73,5.13,False);cs=exact_gaussian(272,arb(31.73)+16,5.13,False)[:256]
    cr=[arb(.01)*(x+arb(eta)*y)+arb(.005)/256 for x,y in zip(cg,cs)]
    ch=rng.multinomial(500,first_detection_probabilities([float(x) for x in cr]))[:-1]
    qg=exact_gaussian(64,31.73,5.13,False);qs=exact_gaussian(80,arb(31.73)+16,5.13,False)[:64]
    qr=[(arb(10)*(x+arb(eta)*y)+arb(5)/64)/1000 for x,y in zip(qg,qs)]
    qh=rng.multinomial(1000,first_detection_probabilities([float(x) for x in qr]))[:-1]
    coarse_rates=[sum(cr[a:b],arb(0)) for a,b in zip(EDGES[:-1],EDGES[1:])]
    actual=arb(certified_log_likelihood(aggregate(ch),coarse_rates,500,'first',upper=True))+arb(certified_log_likelihood(qh,qr,1000,'first',upper=True))
    b=joint_null_bound(aggregate(ch),qh,500,initial_eta=2,initial_delay=4,max_boxes=16)
    assert arb(b['upper'])>=actual


def test_training_fit_is_a_proper_point_predictor_from_training_data_only():
    rng=np.random.default_rng(7517);t=np.arange(256);rates=.01*np.exp(-4*np.log(2)*((t-32.1)/5.1)**2)+.005/256
    h=rng.multinomial(5000,first_detection_probabilities(rates))[:-1]
    q=fit_training(h,5000)
    assert q['query_excluded'] and q['scored_calibration_excluded'] and 0<=q['eta']<=.25
    assert np.all(np.array(q['rates'])>0) and abs(first_detection_probabilities(q['rates']).sum()-1)<1e-12
