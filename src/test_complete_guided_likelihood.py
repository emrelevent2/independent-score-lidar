import numpy as np
from likelihood_baseline import pulse_derivatives
from guided_likelihood import guided_likelihood_report
from complete_guided_likelihood import complete_guided_likelihood


def test_stronger_control_handles_a_secondary_that_happens_to_be_brighter():
    t=np.arange(256);h=np.round(15*pulse_derivatives(t,80.,5.)[0]+30*pulse_derivatives(t,105.,5.)[0]).astype(int)
    simple=guided_likelihood_report(h,105.)
    strong=complete_guided_likelihood(h,80.,105.)
    assert strong['score']>simple['score']+10
    assert abs(strong['primary']-80)<1 and abs(strong['delay']-105)<1
    assert strong['primary_range']==[0,255]
