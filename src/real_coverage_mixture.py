"""Working-Poisson transfer of the fixed spatial/full-delay mixture.

The distribution and weights use donors only. Its numerical score subtracts
a local numerical null fit; it is NOT a certified e-value or sensor model.
"""
from fractions import Fraction
import numpy as np
from scipy.special import logsumexp


def components(q,bins,amplitudes=(1.,2.,4.,8.,16.,32.)):
    if not q['raw_center_excluded']:raise ValueError('Query must be excluded')
    t=np.arange(bins);w=q['width'];d=q['primary']
    if not 3<=w<=16 or not 0<=d<bins:raise ValueError('Invalid donor fit')
    if min(q[k] for k in ['primary_count','secondary_count','background_count'])<0:raise ValueError('Negative intensity')
    def pulse(delays):
        raw=np.exp(-4*np.log(2)*((t[None,:]-np.asarray(delays)[:,None])/w)**2)
        return raw/raw.sum(1,keepdims=True)
    base=q['primary_count']*pulse([d])[0]+q['background_count']/bins
    spatial=base[None,:]+(q['secondary_count']*np.array([.5,1.,2.,4.]))[:,None]*pulse([q['secondary']])
    delays=t[(t>=4)&(t<bins-4)&(abs(t-d)>=8)]
    a=np.asarray(amplitudes,float)
    if not len(delays) or np.any(a<=0) or not np.isfinite(a).all():raise ValueError('Invalid fixed grid')
    global_rates=(base[None,None,:]+a[None,:,None]*pulse(delays)[:,None,:]).reshape(-1,bins)
    rho=Fraction(9,10) if q['support_proxy']>=3.5 else Fraction(1,10)
    weights=[rho/4]*4+[(1-rho)/len(global_rates)]*len(global_rates)
    assert sum(weights,Fraction(0))==1
    return dict(means=np.maximum(np.vstack([spatial,global_rates]),1e-12),weights=weights,
        locations=np.r_[np.full(4,q['secondary']),np.repeat(delays,len(a))],global_delays=delays,amplitudes=a,spatial_mass=str(rho))


def report(h,q,null_log_likelihood):
    h=np.asarray(h);m=components(q,len(h));rates=m['means']
    if h.ndim!=1 or np.any(h<0) or not np.isfinite(h).all():raise ValueError('Invalid histogram')
    likelihood=np.log(rates)@h-rates.sum(1)
    weighted=likelihood+np.log([float(w) for w in m['weights']]);full=float(logsumexp(weighted))
    spatial=float(logsumexp(weighted[:4]))
    masses=logsumexp(weighted[4:].reshape(len(m['global_delays']),len(m['amplitudes'])),axis=1)
    index=int(np.argmax(masses))
    if spatial>=masses[index]:delay=q['secondary'];head='spatial'
    else:
        mode=m['global_delays'][index];near=abs(m['global_delays']-mode)<=2
        delay=float(np.exp(masses[near]-logsumexp(masses[near]))@m['global_delays'][near]);head='global'
    return dict(score=full-null_log_likelihood,primary=q['primary'],delay=float(delay),head=head,
        log_mixture=full,components=len(m['weights']),spatial_mass=m['spatial_mass'],
        calibration_required=True,analytic_guarantee=False)
