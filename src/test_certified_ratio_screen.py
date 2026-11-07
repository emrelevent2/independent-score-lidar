from fractions import Fraction
import numpy as np
from certified_ratio_screen import sparse_numerator_upper,calibration_box_witness,average_log
from coverage_alternative import weighted_log_lower
from envelope_evidence import certified_log_likelihood
from conditional_calibration import calibration_upper


def test_sparse_numerator_bounds_keep_original_mass():
    rates=np.array([[.01,.02,.03],[.03,.01,.02],[.02,.02,.02]])
    model=dict(rates=rates,weights=[Fraction(1,3)]*3);h=np.array([2,1,3])
    indices=[0,2]
    terms={i:certified_log_likelihood(h,rates[i],100,'first') for i in indices}
    lower=weighted_log_lower(terms,model['weights'],indices)
    upper=sparse_numerator_upper(h,model,indices,100)
    assert lower<=upper and upper-lower<1e-10
    assert average_log([lower,lower])<=lower<=average_log([upper,upper],True)


def test_feasible_calibration_witness_is_below_relaxed_null_profile():
    cf=dict(informative=True,signal_main_lower=.01,signal_main_upper=.07,
            background_lower=.000001,background_upper=.0001,
            forward_leakage_lower=0.,forward_leakage_upper=.00001)
    for k,r in [(0,0),(0,20),(3,20),(20,20)]:
        for eta in [0.,.073,.25]:
            lower=calibration_box_witness(k,r,eta,cf)['lower']
            upper=calibration_upper(k,r,eta,eta,cf)['upper']
            assert lower<=upper
            loose={'informative':False}
            assert calibration_box_witness(k,r,eta,loose)['lower']<=calibration_upper(k,r,eta,eta,loose)['upper']
