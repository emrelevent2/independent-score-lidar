import numpy as np
from real_primary_bank import donor_bank,bank_report
from real_poisson_methods_v2 import WorkingPulse


def test_every_bank_parameter_excludes_raw_query_before_proposal_processing():
    rng=np.random.default_rng(15);t=np.arange(64)
    mean=.01+1.2*np.exp(-.5*((t-15)/2.2)**2)+.5*np.exp(-.5*((t-42)/2.2)**2)
    h=rng.poisson(np.broadcast_to(mean,(16,16,64)));pulse=WorkingPulse('gaussian')
    a=donor_bank(h,pulse,8,8);h[8,8]=rng.integers(0,10000,size=64);b=donor_bank(h,pulse,8,8)
    assert a==b and 1<=len(a)<=3 and all(q['raw_center_excluded'] for q in a)


def test_bank_numerical_score_has_explicit_empirical_scope():
    h=np.zeros((16,16,64),int);h[:,:,15]=8;h[:,:,42]=4
    r=bank_report(h,WorkingPulse('gaussian'),8,8,-50.)
    assert np.isfinite(r['score']) and r['bank_size']>=1
    assert r['calibration_required'] and not r['analytic_guarantee']
