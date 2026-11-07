"""Projected full-observation score using an empirical OPG information matrix.

The nuisance pilot is the query's censored M-estimate. Orthogonal projection
removes its first-order nuisance score. Per-factor OPG is an empirical
information approximation;neither it nor plug-in bootstrap gives a finite
uniform error guarantee. Established score/OPG/projection ingredients.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from deadtime_scan import pulse,corr
from deadtime_path_bootstrap import fine_scan
from clock_pulse_likelihood import rates_and_design
import ct_dual_fast as kernel


@lru_cache(maxsize=16)
def templates(bins,sigma,subdivision=4):
    grid=np.arange(bins*subdivision)/subdivision;bank=np.array([pulse(v,bins,sigma)[0] for v in grid]);return grid,bank


def projected(gradient,information,nuisance_design,theta,sigma):
    k=len(gradient);grid,bank=templates(k,sigma);j=nuisance_design;hj=information@j;inverse=np.linalg.pinv(j.T@hj,rcond=1e-10);cross=bank@hj;raw=bank@gradient-cross@inverse@(j.T@gradient);variance=np.einsum('mi,ij,mj->m',bank,information,bank)-np.einsum('mi,ij,mj->m',cross,inverse,cross);distance=np.abs((grid-theta[1]+k/2)%k-k/2);allowed=(distance>=2.5*sigma)&(variance>1e-8);scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(variance,1e-8),-np.inf);best=int(np.argmax(scores))
    return dict(score=float(scores[best]),delay=float(grid[best]),primary=float(theta[1]),scores=scores,grid=grid,variance=variance,nuisance_information=j.T@hj)


def statistic(counts,exposure,sigma,fit,geometry):
    if geometry['td']>=geometry['te']:
        old=fine_scan(counts,exposure,sigma,fit=fit);return {key:old[key] for key in ['score','primary','delay']},dict(mode='EXACT_SAFE_PASSTHROUGH')
    rate,j=rates_and_design(fit['theta'],len(counts),sigma);value,gradient,terms,pieces=kernel.full_factor(rate,np.eye(len(rate)),geometry,return_pieces=True);information=pieces.T@pieces;r=projected(gradient,information,j,fit['theta'],sigma)
    return {key:r[key] for key in ['score','primary','delay']},dict(mode='FULL_RATE_SCORE_OPG',full_loglik=float(value),null_score_norm=float(np.linalg.norm(j.T@gradient)),nuisance_information_rank=int(np.linalg.matrix_rank(r['nuisance_information'])),observed_factors=len(terms))


def fit_full_null(counts,exposure,sigma,pilot,geometry):
    """Three-start numerical full-query null fit for shared calibration only.

    Not globally certified. Starting peaks use observed counts/risk only.
    The statistic still evaluates its censored pilot identically in each draw.
    """
    k=len(counts);theta=np.asarray(pilot['theta']);rate,j=rates_and_design(theta,k,sigma);initial_value=kernel.full_factor(rate,j,geometry)[0]
    if geometry['td']>=geometry['te']:return dict(theta=theta.copy(),loglik=initial_value,optimizer_success=pilot['optimizer_success'],failures=0,mode='IDENTICAL_SAFE_NULL')
    ratio=counts/np.maximum(exposure,1e-9);smooth=corr(ratio,pulse(0,k,sigma)[0]);starts=[float(theta[1])]
    for index in np.argsort(smooth)[::-1]:
        if all(abs((index-v+k/2)%k-k/2)>3*sigma for v in starts):starts.append(float(index))
        if len(starts)==3:break
    scale=max(1.,sum(counts));fits=[]
    def objective(x):
        rate,j=rates_and_design(x,k,sigma);value,gradient=kernel.full_factor(rate,j,geometry);return -value/scale,-gradient/scale
    for centre in starts:
        start=theta.copy();start[1]=centre;result=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=[(0,20),(centre-3*sigma,centre+3*sigma),(1e-6,30)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=120,maxls=30));q=result.x.copy();q[1]%=k;fits.append(dict(theta=q,loglik=-float(result.fun)*scale,optimizer_success=bool(result.success),evaluations=int(result.nfev)))
    fits.append(dict(theta=theta.copy(),loglik=initial_value,optimizer_success=bool(pilot['optimizer_success']),evaluations=0));best=max(fits,key=lambda x:x['loglik']);return dict(best,failures=sum(not x['optimizer_success'] for x in fits),mode='THREE_START_FULL_NULL',starts=starts)
