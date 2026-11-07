from fractions import Fraction
import numpy as np
from shoulder_alternative import donor_proposal,components,numeric_and_sparse,proposal_from_pool,pulse
from envelope_evidence import certified_log_likelihood
from coverage_alternative import weighted_log_lower
from pilot import first_detection_probabilities


def test_raw_query_exclusion_and_proper_irf_mixture():
    rng=np.random.default_rng(7313);t=np.arange(256)
    g=np.exp(-4*np.log(2)*((t-80.3)/5.1)**2);g/=g.sum()
    rate=(100*g+12*np.roll(g,16)+10/256)/1000
    h=rng.multinomial(1000,first_detection_probabilities(rate),size=(9,9))[...,:-1]
    q=donor_proposal(h);changed=h.copy();changed[4,4]=0;changed[4,4,250]=1000
    assert donor_proposal(changed)==q
    model=components(q);assert sum(model['weights'],Fraction(0))==1
    assert model['raw_center_excluded'] and np.all(model['rates']>0)
    assert all(w>0 for w in model['weights'])
    for r in model['rates'][[0,1395,-1]]:
        assert abs(first_detection_probabilities(r).sum()-1)<1e-12


def test_sparse_original_mass_is_not_renormalized():
    # Small actual response mixture: include three distinct spatial blocks,
    # two global delays, and arbitrary exact original rational weights.
    model=dict(rates=np.array([[.01,.02,.03],[.03,.01,.02],[.01,.04,.01],
                               [.02,.02,.02],[.03,.02,.01],[.04,.01,.01]]),
               weights=[Fraction(1,6)]*6,spatial=np.array([True,False,False,True,False,False]),
               locations=np.array([1.,0.,2.,1.,0.,2.]),global_delays=np.array([0.,2.]))
    h=np.array([2,3,1]);r=numeric_and_sparse(h,model,100)
    ell={i:certified_log_likelihood(h,rate,100,'first') for i,rate in enumerate(model['rates'])}
    full=weighted_log_lower(ell,model['weights'],range(6))
    assert r['original_weights_retained'] and r['lower']<=r['numeric_full']+1e-12
    assert r['lower']==full  # Here the small global bank is all evaluated.


def test_calibration_cap_prevents_absorbing_a_near_object_into_device_lobe():
    t=np.arange(256);d=80.3
    pool=(300*pulse(t,d,5.,0.)+32*pulse(t,d+16,5.,0.)+10/256)/1000
    q=proposal_from_pool(pool,cap=.01)
    assert 0<=q['eta_donor']<=.01 and abs(q['secondary']-(d+16))<2
    q['effective_donors']=20.
    m=components(q,cap=.01)
    assert sum(m['weights'],Fraction(0))==1 and m['raw_center_excluded']
