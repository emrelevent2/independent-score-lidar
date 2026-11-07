"""Empirical local/spatial routing. No conditional e-value claim.

The support proxy uses donors only. It is a variance-inspired heuristic,
not a confidence bound. Entire score/report policies require fresh null
calibration, including any query-dependent choice between candidate heads.
"""
import numpy as np


def donor_support(q,bins=256,cycles=1000):
    a=float(q['primary_incident']);c=float(q['secondary_incident'])
    b=float(q['background_incident']);w=float(q['fitted_fwhm']);neff=float(q['effective_donors'])
    if min(a,c,b)<0 or w<=0 or neff<=0 or not np.all(np.isfinite([a,c,b,w,neff])):
        raise ValueError('Invalid donor parameters')
    t=np.arange(bins);g=np.exp(-4*np.log(2)*((t-q['secondary'])/w)**2);g/=g.sum()
    # Approximate amplitude noise after averaging independent donor pixels.
    # This deliberately remains a routing feature: fitting uncertainty,
    # heterogeneous donors and exact first-event covariance are not bounded.
    variance=(c+b/(bins*np.sum(g*g)))*np.exp((a+b*q['secondary']/bins)/cycles)/neff
    return float(c/np.sqrt(max(variance,1e-12)))


def route_report(proposal,spatial,local,normalizers,policy,gate=2.5):
    ss=float(spatial['score'])/normalizers['spatial'];ls=float(local['score'])/normalizers['local']
    support=donor_support(proposal)
    if policy=='spatial':score=ss;head='spatial'
    elif policy=='local':score=ls;head='local'
    elif policy=='route':
        head='spatial' if support>=gate else 'local';score=ss if head=='spatial' else ls
    elif policy in ['support_union','aligned_union']:
        score=max(ss,ls);aligned=abs(local['delay']-proposal['secondary'])<=2.
        use_spatial=support>=gate or (policy=='aligned_union' and aligned)
        head='spatial' if use_spatial else 'local'
    elif policy=='naive_union':
        score=max(ss,ls);head='spatial' if ss>=ls else 'local'
    else:raise ValueError('Unknown policy')
    r=spatial if head=='spatial' else local
    return dict(score=float(score),delay=float(proposal['secondary'] if head=='spatial' else r['delay']),
        primary=float(r['primary']),head=head,support_proxy=support,policy=policy,gate=float(gate),
        location_source='center-excluded donors' if head=='spatial' else 'queried local likelihood fit',
        calibration_required=True)
