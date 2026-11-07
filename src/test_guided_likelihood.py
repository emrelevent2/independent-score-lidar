import numpy as np
from pilot import first_detection_probabilities
from likelihood_baseline import pulse_derivatives
from guided_likelihood import guided_likelihood_report


def test_guided_likelihood_can_refine_an_independent_secondary_proposal():
    t=np.arange(256);rng=np.random.default_rng(619903)
    rates=(200*pulse_derivatives(t,80.3,5.2)[0]+70*pulse_derivatives(t,100.4,5.2)[0]+15/256)/1000
    h=rng.multinomial(1000,first_detection_probabilities(rates))[:-1]
    r=guided_likelihood_report(h,101.2)
    assert r['score']>20 and abs(r['delay']-100.4)<1
    assert r['calibration_required'] and r['alternative_log_likelihood']>r['null_log_likelihood_lower']
