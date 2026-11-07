"""Predict from donors plus a disjoint Poisson-thinned query training part.

Independence is marginal under an ideal Poisson law, not conditional on the
observed parent count or established for this sensor. Numerical null scores
remain empirical; this module makes no analytic sensor-error claim.
"""
from fractions import Fraction
import numpy as np
from scipy.special import logsumexp

POLICIES=['donor','query_delay','query_nuisance','shrunken_nuisance']


def split_query(h,fraction,seed):
    h=np.asarray(h)
    if h.ndim!=1 or np.any(h<0) or not np.isfinite(h).all() or not np.all(h==np.floor(h)):raise ValueError('Integer count vector required')
    if not 0<fraction<1:raise ValueError('Nonempty exposure fractions required')
    a=np.random.default_rng(seed).binomial(h.astype(np.int64),fraction)
    return a,h.astype(np.int64)-a


def predict(q,training_fit,fraction,policy,pseudocount=2.):
    if policy not in POLICIES or not q['raw_center_excluded']:raise ValueError('Invalid predictor source')
    test=1-fraction;used=policy!='donor' and training_fit is not None
    result=dict(q,primary_count=q['primary_count']*test,secondary_count=q['secondary_count']*test,
        background_count=q['background_count']*test,scored_counts_excluded=True,query_training_used=used,
        raw_center_excluded=not used,policy=policy,training_fraction=fraction)
    if not used:return result
    a,b,d,w=training_fit['parameters'];result['primary']=d
    if policy=='query_nuisance':
        result.update(primary_count=a*test/fraction,background_count=b*test/fraction,width=w)
    elif policy=='shrunken_nuisance':
        # A fixed two-pseudocount shrink estimate, not an exact posterior:
        # fitted Poisson signal/background allocations are not observed labels.
        result.update(primary_count=test*(a+pseudocount)/(fraction+pseudocount/max(q['primary_count'],1.)),
            background_count=test*(b+pseudocount)/(fraction+pseudocount/max(q['background_count'],1.)),
            width=(a*w+pseudocount*q['width'])/(a+pseudocount))
    return result


def components(q,bins,amplitudes):
    if not q['scored_counts_excluded']:raise ValueError('Scored counts may not define the predictor')
    t=np.arange(bins);d=q['primary'];w=q['width']
    if not 0<=d<bins or not 3<=w<=16:raise ValueError('Invalid predictive geometry')
    if min(q[k] for k in ['primary_count','secondary_count','background_count'])<0:raise ValueError('Negative count mean')
    def pulse(delays):
        raw=np.exp(-4*np.log(2)*((t[None,:]-np.asarray(delays)[:,None])/w)**2)
        return raw/raw.sum(1,keepdims=True)
    base=q['primary_count']*pulse([d])[0]+q['background_count']/bins
    spatial=base[None,:]+q['secondary_count']*np.array([.5,1.,2.,4.])[:,None]*pulse([q['secondary']])
    delays=t[(t>=4)&(t<bins-4)&(abs(t-d)>=8)];a=np.asarray(amplitudes,float)
    if not len(a) or np.any(a<=0) or not np.isfinite(a).all():raise ValueError('Invalid fixed amplitude grid')
    dense=(base[None,None,:]+a[None,:,None]*pulse(delays)[:,None,:]).reshape(-1,bins)
    # Preserve the original donor support feature for all ablations.
    rho=Fraction(9,10) if q['support_proxy']>=3.5 else Fraction(1,10)
    weights=[rho/4]*4+[(1-rho)/len(dense)]*len(dense);assert sum(weights,Fraction(0))==1
    return dict(means=np.maximum(np.vstack([spatial,dense]),1e-12),weights=weights,global_delays=delays,
        amplitudes=a,spatial_mass=str(rho))


def score(heldout,q,null_log_likelihood,amplitudes):
    h=np.asarray(heldout);m=components(q,len(h),amplitudes);rates=m['means']
    ll=np.log(rates)@h-rates.sum(1)+np.log([float(w) for w in m['weights']]);full=float(logsumexp(ll))
    spatial=float(logsumexp(ll[:4]));glob=logsumexp(ll[4:].reshape(len(m['global_delays']),len(m['amplitudes'])),axis=1)
    j=int(np.argmax(glob))
    if spatial>=glob[j]:delay=q['secondary'];head='spatial'
    else:
        mode=m['global_delays'][j];near=abs(m['global_delays']-mode)<=2
        delay=float(np.exp(glob[near]-logsumexp(glob[near]))@m['global_delays'][near]);head='global'
    return dict(score=full-null_log_likelihood,primary=q['primary'],delay=float(delay),head=head,
        components=len(m['weights']),spatial_mass=m['spatial_mass'],query_training_used=q['query_training_used'],
        scored_counts_excluded=True,calibration_required=True,analytic_guarantee=False)
