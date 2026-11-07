import numpy as np
from real_poisson_methods_v2 import WorkingPulse,donor_proposal,local_report


def test_empirical_pulse_derivatives_match_finite_differences():
    profile=np.exp(-.5*((np.arange(-24,25)-1)/3.)**2)+.2*np.exp(-.5*((np.arange(-24,25)-6)/4.)**2)
    p=WorkingPulse('empirical',profile);t=np.arange(128);d=60.234;w=8.137;eps=1e-6
    g,dg,wg=p.derivatives(t,d,w)
    numerical_d=(p.derivatives(t,d+eps,w)[0]-p.derivatives(t,d-eps,w)[0])/(2*eps)
    numerical_w=(p.derivatives(t,d,w+eps)[0]-p.derivatives(t,d,w-eps)[0])/(2*eps)
    assert abs(g.sum()-1)<1e-12
    assert np.allclose(dg,numerical_d,rtol=1e-5,atol=1e-8) and np.allclose(wg,numerical_w,rtol=1e-5,atol=1e-8)


def test_high_count_poisson_fixture_has_no_invented_cycle_limit():
    p=WorkingPulse();t=np.arange(192)
    h=np.round(4000*p.derivatives(t,60.,8.)[0]+800*p.derivatives(t,110.,8.)[0]+1000/192).astype(int)
    assert h.sum()>1000
    r=local_report(h,p)
    assert r['score']>0 and abs(r['delay']-110)<1 and abs(r['primary']-60)<1


def test_all_poisson_donor_fields_exactly_exclude_raw_query():
    p=WorkingPulse();t=np.arange(128)
    h=np.tile(np.round(80*p.derivatives(t,40.,8.)[0]+20*p.derivatives(t,80.,8.)[0]+20/128).astype(int),(9,9,1))
    before=donor_proposal(h,p,4,4);h[4,4]=0;h[4,4,100]=20000
    after=donor_proposal(h,p,4,4)
    assert before==after and before['raw_center_excluded']


def test_widely_separated_donor_primaries_do_not_underflow_all_weights():
    p=WorkingPulse();t=np.arange(672);h=np.zeros((9,9,672),int)
    for r in range(9):
        for c in range(9):h[r,c]=np.round(80*p.derivatives(t,80 if c<4 else 500,8.)[0]).astype(int)
    near=[(r,c) for r in range(3,6) for c in range(3,6) if (r,c)!=(4,4)]
    for i,(r,c) in enumerate(near):h[r,c]=np.round(80*p.derivatives(t,80 if i<4 else 500,8.)[0]).astype(int)
    q=donor_proposal(h,p,4,4)
    assert np.isfinite([q['primary'],q['secondary'],q['support_proxy'],q['effective_donors']]).all()
    assert q['effective_donors']>=1 and q['raw_center_excluded']
