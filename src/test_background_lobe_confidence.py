from fractions import Fraction
import numpy as np
import pytest
from flint import arb
from background_lobe_confidence import shape_bounds,bound_from_hazards,lobe_confidence
from calibration_confidence_v2 import lobe_confidence as old_confidence,binomial_cdf
from envelope_evidence import exact_gaussian,outward_float
from pilot import first_detection_probabilities


def test_shape_enclosures_and_algebra_cover_continuous_geometry():
    box=shape_bounds()
    for d in [31.5,31.73,32.5]:
        for w in [4.5,5.13,5.5]:
            g=exact_gaussian(256,d,w,False);side=exact_gaussian(272,arb(d)+16,w,False)[:256]
            assert sum(g[27:38],arb(0))>=arb(box['main_mass_lower'])
            dark=sum((x+arb(.25)*y for x,y in zip(g[128:],side[128:])),arb(0))
            assert dark<=arb(box['dark_response_mass_upper'])
            for a in [.00001,.01,1.]:
                for b in [0.,.005,.2]:
                    for eta in [0.,.03125,.125,.25]:
                        rates=[arb(a)*(x+arb(eta)*y)+arb(b)/256 for x,y in zip(g,side)]
                        main=sum(rates[27:38],arb(0));lobe=sum(rates[43:54],arb(0));dark=sum(rates[128:],arb(0))
                        result=bound_from_hazards(outward_float(main,False),outward_float(main),outward_float(lobe),
                            max(0.,outward_float(dark,False)),outward_float(dark),box)
                        assert arb(result['eta_upper'])>=arb(eta)


def test_background_subtraction_tightens_a_known_background_example():
    t=np.arange(256);rate=.01*np.exp(-4*np.log(2)*((t-32.)/5.)**2)+.005/256
    h=np.rint(100000*first_detection_probabilities(rate)[:-1]).astype(int)
    old=old_confidence(h,100000);new=lobe_confidence(h,100000)
    assert new['informative'] and 0<new['eta_upper']<old['eta_upper']
    assert new['background_per_bin_lower']>0
    assert Fraction(new['one_sided_budget'])*new['endpoint_count']==Fraction(new['delta'])
    budget=arb(1)/5000
    for name,p in new['probability_endpoints'].items():
        window=name.split('_')[0];k,n=new['window_counts'][window]
        tail=binomial_cdf(k,n,p) if name.endswith('upper') else 1-binomial_cdf(k-1,n,p)
        assert tail<=budget


def test_uninformative_and_inconsistent_endpoints_fall_back():
    h=np.zeros(256,dtype=int)
    assert lobe_confidence(h,0)['eta_upper']==.25
    assert lobe_confidence(h,1000)['eta_upper']==.25
    box=shape_bounds()
    # Signal lower bound nonpositive: no ratio is authorized.
    assert bound_from_hazards(.001,.002,.0001,.0001,.1,box)['eta_upper']==.25
    # Background lower > lobe upper is inconsistent; retain the global cap.
    assert bound_from_hazards(.1,.2,0.,.1,.1,box)['eta_upper']==.25
    with pytest.raises(ValueError):lobe_confidence(h,-1)
