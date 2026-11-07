"""Exact finite-clock hidden-renewal observation kernel,ordinary arithmetic.

An event at tickt blocks the NEXT Dd detector ticks and NEXT De electronics
ticks. Hidden avalanches restart only detector recovery. Input Bernoulli
probabilities are independent per clocktick,periodic,and known to this
kernel. This is a distinct clock model,NOT an exact continuous-time likelihood.
Forward state summation/renewal arguments are established;no priority claim.
No parameter estimator or demonstrated lidar utility is implemented here.
"""
import numpy as np
from deadtime_path_bootstrap import njit


@njit(cache=True)
def build_short_kernel(p,dd,de):
    k=len(p)
    if dd==0 or dd>=de:
        start=max(dd,de)+1
        return np.zeros((k,start)),np.ones(k),start
    start=de+dd+1;hidden=np.zeros((de+1,k));short=np.zeros((k,start));tail=np.empty(k)
    for phase in range(k):
        window=0.
        for t in range(1,de+1):
            active=0. if t<=dd else 1.-window
            if active< -1e-12 or active>1+1e-12:raise ArithmeticError('Invalid hidden detector availability')
            active=max(0.,min(1.,active));hidden[t,phase]=active*p[(phase+t)%k];window+=hidden[t,phase]
            old=t-dd
            if old>=1:window-=hidden[old,phase]
        live=max(0.,1.-window)
        for t in range(de+1,start):
            probability=p[(phase+t)%k];short[phase,t]=live*probability;live=live*(1.-probability)+hidden[t-dd,phase]
        tail[phase]=live
    return short,tail,start


def transition_kernel(probabilities,detector_dead_ticks,electronics_dead_ticks):
    p=np.asarray(probabilities,float);dd=int(detector_dead_ticks);de=int(electronics_dead_ticks)
    if p.ndim!=1 or not len(p) or np.any(~np.isfinite(p)) or np.any(p<0) or np.any(p>=1) or not np.any(p>0):raise ValueError('Finite periodic probabilities in[0,1),some positive,required')
    if dd!=detector_dead_ticks or de!=electronics_dead_ticks or min(dd,de)<0:raise ValueError('Nonnegative integer recovery counts required')
    short,tail,start=build_short_kernel(p,dd,de);hazard=-np.log1p(-p);remaining=np.empty_like(short);remaining[:,-1]=tail
    if start>1:remaining[:,:-1]=np.cumsum(short[:,::-1],axis=1)[:,::-1][:,1:]+tail[:,None]
    assert np.max(np.abs(remaining[:,0]-1))<2e-12;remaining[:,0]=1.
    return dict(probabilities=p.copy(),short=short,live_at_tail=tail,tail_start=int(start),hazard_prefix=np.r_[0.,np.cumsum(hazard)],hazard_period=float(sum(hazard)),short_cdf=np.cumsum(short,axis=1),short_survival=remaining)


def hazard_prefix(table,indices):
    x=np.asarray(indices,dtype=np.int64);k=len(table['probabilities']);return (x//k)*table['hazard_period']+table['hazard_prefix'][x%k]


def integer_arrays(phase,slots):
    a,b=np.broadcast_arrays(np.asarray(phase),np.asarray(slots));ia=a.astype(np.int64);ib=b.astype(np.int64)
    if np.any(a!=ia) or np.any(b!=ib) or np.any(ib<0):raise ValueError('Integer phases and nonnegative clock gaps required')
    return ia,ib


def log_transition_mass(table,previous_phase,gap_ticks):
    phase,gap=integer_arrays(previous_phase,gap_ticks);shape=gap.shape;phase=phase.reshape(-1)%len(table['probabilities']);gap=gap.reshape(-1);out=np.full(len(gap),-np.inf);start=table['tail_start'];short=gap<start
    with np.errstate(divide='ignore'):
        out[short]=np.log(table['short'][phase[short],gap[short]])
        tail=~short;u=phase[tail];v=gap[tail];out[tail]=np.log(table['live_at_tail'][u])-hazard_prefix(table,u+v)+hazard_prefix(table,u+start)+np.log(table['probabilities'][(u+v)%len(table['probabilities'])])
    out=out.reshape(shape);return float(out) if out.ndim==0 else out


def survival_after(table,previous_phase,ticks):
    """P(next observed gap>ticks),conditional on recorded hit at tick0."""
    phase,ticks=integer_arrays(previous_phase,ticks);shape=ticks.shape;phase=phase.reshape(-1)%len(table['probabilities']);ticks=ticks.reshape(-1);out=np.ones(len(ticks));start=table['tail_start'];short=ticks<start;out[short]=table['short_survival'][phase[short],ticks[short]];u=phase[~short];v=ticks[~short];out[~short]=table['live_at_tail'][u]*np.exp(-hazard_prefix(table,u+v+1)+hazard_prefix(table,u+start));out=out.reshape(shape);return float(out) if out.ndim==0 else out


def log_survival_after(table,previous_phase,ticks):
    phase,ticks=integer_arrays(previous_phase,ticks);shape=ticks.shape;phase=phase.reshape(-1)%len(table['probabilities']);ticks=ticks.reshape(-1);out=np.empty(len(ticks));start=table['tail_start'];short=ticks<start
    with np.errstate(divide='ignore'):
        out[short]=np.log(table['short_survival'][phase[short],ticks[short]]);u=phase[~short];v=ticks[~short];out[~short]=np.log(table['live_at_tail'][u])-hazard_prefix(table,u+v+1)+hazard_prefix(table,u+start)
    out=out.reshape(shape);return float(out) if out.ndim==0 else out


def full_log_likelihood(table,observed_ticks,total_ticks):
    """All observed records plus first-arrival and terminal-survival factors.

    Detector/electronics initially active at tick0. Unrecorded avalanches
    have been summed out,not treated as absent. No random-exposure shortcut.
    """
    observed=np.asarray(observed_ticks);t=observed.astype(np.int64);total=int(total_ticks)
    if total!=total_ticks or total<0 or t.ndim!=1 or np.any(observed!=t) or np.any(np.diff(t)<=0) or np.any(t<0) or np.any(t>=total):raise ValueError('Invalid clock record/horizon')
    if not len(t):return -float(hazard_prefix(table,total))
    k=len(table['probabilities'])
    with np.errstate(divide='ignore'):first=float(np.log(table['probabilities'][t[0]%k])-hazard_prefix(table,t[0]));tail=log_survival_after(table,int(t[-1]%k),total-1-int(t[-1]))
    transitions=log_transition_mass(table,t[:-1]%k,np.diff(t)) if len(t)>1 else np.array([])
    return first+float(sum(transitions))+tail
