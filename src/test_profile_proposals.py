import numpy as np
from likelihood_baseline import pulse_derivatives
from profile_proposals import profile_at_proposals


def test_secondary_localization_is_invariant_to_queried_peak_displacement():
    t=np.arange(256);a=np.round(100*pulse_derivatives(t,80.,5.)[0]+20*pulse_derivatives(t,105.,5.)[0]).astype(int)
    b=np.round(100*pulse_derivatives(t,80.,5.)[0]+20*pulse_derivatives(t,106.,5.)[0]).astype(int)
    for mode in ['fixed_secondary','fixed_both']:
        x=profile_at_proposals(a,80.,105.,mode);y=profile_at_proposals(b,80.,105.,mode)
        assert x['delay']==105. and y['delay']==105.
        assert x['score']>5 and y['score']>5 and x['score']!=y['score']


def test_fixed_secondary_profile_recovers_primary_despite_brightness_reversal():
    t=np.arange(256);h=np.round(15*pulse_derivatives(t,80.,5.)[0]+30*pulse_derivatives(t,105.,5.)[0]).astype(int)
    x=profile_at_proposals(h,80.,105.,'fixed_secondary')
    assert abs(x['primary']-80.)<1 and x['delay']==105. and x['score']>10
