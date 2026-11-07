"""Forward sensitivities for the full finite-clock observation likelihood.

Exact chain rule for the stated clock model in ordinary floating arithmetic;
not a continuous-time/interval certificate or novel differentiation principle.
"""
import numpy as np
from deadtime_path_bootstrap import njit
from clock_dual_transition import transition_kernel,integer_arrays,full_log_likelihood


@njit(cache=True)
def short_sensitivities(p,dp,dd,de):
    k,dimension=dp.shape;start=max(dd,de)+1 if dd==0 or dd>=de else de+dd+1;short_gradient=np.zeros((k,start,dimension));tail_gradient=np.zeros((k,dimension))
    if dd==0 or dd>=de:return short_gradient,tail_gradient
    hidden=np.zeros((de+1,k));dhidden=np.zeros((de+1,k,dimension))
    for phase in range(k):
        window=0.;dwindow=np.zeros(dimension)
        for t in range(1,de+1):
            active=0. if t<=dd else 1-window;dactive=np.zeros(dimension) if t<=dd else -dwindow.copy();index=(phase+t)%k
            if active<0 or active>1:active=max(0.,min(1.,active));dactive=np.zeros(dimension)
            hidden[t,phase]=active*p[index];dhidden[t,phase]=dactive*p[index]+active*dp[index];window+=hidden[t,phase];dwindow+=dhidden[t,phase];old=t-dd
            if old>=1:window-=hidden[old,phase];dwindow-=dhidden[old,phase]
        live=max(0.,1-window);dlive=-dwindow
        for t in range(de+1,start):
            index=(phase+t)%k;short_gradient[phase,t]=dlive*p[index]+live*dp[index];dlive=dlive*(1-p[index])-live*dp[index]+dhidden[t-dd,phase];live=live*(1-p[index])+hidden[t-dd,phase]
        tail_gradient[phase]=dlive
    return short_gradient,tail_gradient


def kernel_with_gradient(probabilities,probability_gradient,dd,de):
    p=np.asarray(probabilities,float);dp=np.asarray(probability_gradient,float)
    if dp.ndim!=2 or dp.shape[0]!=len(p) or dp.shape[1]<1 or np.any(~np.isfinite(dp)) or np.any(p<=0):raise ValueError('Positive interior probabilities and finiteKxd Jacobian required')
    table=transition_kernel(p,dd,de);gs,gt=short_sensitivities(p,dp,int(dd),int(de));gradient_hazard=dp/(1-p[:,None]);prefix=np.vstack([np.zeros(dp.shape[1]),np.cumsum(gradient_hazard,axis=0)]);remaining=np.empty_like(gs);remaining[:,-1]=gt
    if table['tail_start']>1:remaining[:,:-1]=np.cumsum(gs[:,::-1],axis=1)[:,::-1][:,1:]+gt[:,None,:]
    remaining[:,0]=0.
    return dict(table,probability_gradient=dp.copy(),short_gradient=gs,tail_gradient=gt,short_survival_gradient=remaining,hazard_prefix_gradient=prefix,hazard_period_gradient=prefix[-1])


def prefix_gradient(table,indices):
    x=np.asarray(indices,dtype=np.int64);k=len(table['probabilities']);return (x//k)[...,None]*table['hazard_period_gradient']+table['hazard_prefix_gradient'][x%k]


def transition_log_gradient(table,previous_phase,gaps):
    phase,gap=integer_arrays(previous_phase,gaps);shape=gap.shape;phase=phase.reshape(-1)%len(table['probabilities']);gap=gap.reshape(-1);dimension=table['probability_gradient'].shape[1];out=np.empty((len(gap),dimension));start=table['tail_start'];short=gap<start
    if np.any(table['short'][phase[short],gap[short]]<=0):raise ValueError('Impossible transition has no finite log gradient')
    out[short]=table['short_gradient'][phase[short],gap[short]]/table['short'][phase[short],gap[short],None];u=phase[~short];v=gap[~short];index=(u+v)%len(table['probabilities']);out[~short]=table['tail_gradient'][u]/table['live_at_tail'][u,None]-prefix_gradient(table,u+v)+prefix_gradient(table,u+start)+table['probability_gradient'][index]/table['probabilities'][index,None]
    return out.reshape(shape+(dimension,))


def survival_log_gradient(table,previous_phase,ticks):
    phase,ticks=integer_arrays(previous_phase,ticks);shape=ticks.shape;phase=phase.reshape(-1)%len(table['probabilities']);ticks=ticks.reshape(-1);dimension=table['probability_gradient'].shape[1];out=np.empty((len(ticks),dimension));start=table['tail_start'];short=ticks<start;out[short]=table['short_survival_gradient'][phase[short],ticks[short]]/table['short_survival'][phase[short],ticks[short],None];u=phase[~short];v=ticks[~short];out[~short]=table['tail_gradient'][u]/table['live_at_tail'][u,None]-prefix_gradient(table,u+v+1)+prefix_gradient(table,u+start);return out.reshape(shape+(dimension,))


def likelihood_and_gradient(table,observed_ticks,total_ticks):
    value=full_log_likelihood(table,observed_ticks,total_ticks);t=np.asarray(observed_ticks,dtype=np.int64)
    if not len(t):return value,-prefix_gradient(table,total_ticks)
    k=len(table['probabilities']);gradient=table['probability_gradient'][t[0]%k]/table['probabilities'][t[0]%k]-prefix_gradient(table,t[0]);gradient+=np.sum(transition_log_gradient(table,t[:-1]%k,np.diff(t)),axis=0);gradient+=survival_log_gradient(table,int(t[-1]%k),total_ticks-1-int(t[-1]));return value,gradient
