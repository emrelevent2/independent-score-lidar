"""Periodic Gaussian-family full and censored factors for the CLOCK model.

The full factor marginalizes hidden detector avalanches. The safe factor
uses predictable known-active clock ticks. Random activation counts do not
imply independent binomial counts conditional on those activation counts.
No continuous-time,efficiency,or empirical-improvement claim is made here.
"""
import numpy as np
from deadtime_scan import pulse,likelihood_parts
from clock_dual_gradient import kernel_with_gradient,likelihood_and_gradient


def rates_and_design(theta,bins,sigma):
    x=np.asarray(theta,float)
    if len(x) not in [3,5] or sigma<=0 or x[0]<0 or x[2]<=0 or (len(x)==5 and x[3]<0):raise ValueError('Three/five physical parameters and positive background/width required')
    a,t,b=x[:3];f,df=pulse(t,bins,sigma);rate=a*f+b/bins;columns=[f,a*df,np.ones(bins)/bins]
    if len(x)==5:
        c,d=x[3:];g,dg=pulse(d,bins,sigma);rate+=c*g;columns.extend([g,c*dg])
    return rate,np.column_stack(columns)


def clock_rates(theta,bins,sigma,subdivision):
    rate,design=rates_and_design(theta,bins,sigma);sub=int(subdivision)
    if sub!=subdivision or sub<1:raise ValueError('Positive integer clock subdivision required')
    # Refinement preserves the ORIGINAL piecewise-constant physical intensity;
    # it does not silently substitute a finer smooth-Gaussian model.
    return np.repeat(rate/sub,sub),np.repeat(design/sub,sub,axis=0)


def full_factor(theta,observed,total,bins,sigma,dd,de,subdivision=1):
    rate,design=clock_rates(theta,bins,sigma,subdivision);p=-np.expm1(-rate);dp=np.exp(-rate)[:,None]*design;table=kernel_with_gradient(p,dp,dd,de);return likelihood_and_gradient(table,observed,total)


def safe_statistics(observed,total,bins,dd,de):
    t=np.asarray(observed);ticks=t.astype(np.int64)
    if bins<1 or int(total)!=total or total<0 or np.any(t!=ticks) or ticks.ndim!=1 or np.any(np.diff(ticks)<=0) or np.any(ticks<0) or np.any(ticks>=total):raise ValueError('Invalid clock record')
    rawdd,rawde=dd,de;dd=int(dd);de=int(de)
    if min(dd,de)<0 or dd!=rawdd or de!=rawde:raise ValueError('Nonnegative integer recovery counts required')
    lag=dd+1 if dd>=de else dd+de+1;s=np.zeros(bins,dtype=np.int64);activation=np.zeros(bins,dtype=np.int64);phases=np.arange(bins);lo=0
    def prefix(end):return end//bins+(phases<end%bins)
    for event in ticks:
        if event>=lo:activation+=prefix(int(event)+1)-prefix(lo);s[event%bins]+=1
        # Unscored events still restart the known-active mask.
        lo=int(event)+lag
    if lo<total:activation+=prefix(total)-prefix(lo)
    assert np.all(activation>=s)
    return dict(counts=s,activations=activation,safe_lag_ticks=lag)


def safe_factor(theta,statistics,bins,sigma,subdivision=1):
    rate,design=clock_rates(theta,bins,sigma,subdivision);value,score,_=likelihood_parts(statistics['counts'],statistics['activations'],rate,'ded');return value,design.T@score
