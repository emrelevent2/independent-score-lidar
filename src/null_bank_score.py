"""Frozen-weight and empirical-OPG ablations with the SAME full null refit.

Independent simulation weights and likelihood projection are established
ingredients. The bank is a surrogate and all adaptive steps need calibration.
No efficacy/finite-composite-null/new-theorem guarantee is implied.
"""
import numpy as np
from scipy.optimize import minimize
import ct_dual_fast as forward
import ct_dual_reverse as reverse
from ct_full_score import projected
from clock_pulse_likelihood import rates_and_design
from deadtime_path_bootstrap import fine_scan

METHODS=['full_opg_safe','full_opg_refit','full_bank_safe','full_bank_refit']


def primary_statistic(counts,exposure,sigma,pilot,geometry,bank,regime):
    full=refit_full_null(counts,sigma,pilot,geometry)
    if geometry['td']>=geometry['te']:
        v=fine_scan(counts,exposure,sigma,fit=pilot);return {k:v[k] for k in ['score','primary','delay']},full,None
    rate,j=rates_and_design(full['theta'],len(counts),sigma);ll,gradient=reverse.full_factor(rate,np.eye(len(counts)),geometry);h,flags=bank.lookup(full['theta'],regime);v=projected(gradient,h,j,full['theta'],sigma);return {k:v[k] for k in ['score','primary','delay']},full,flags


def refit_full_null(counts,sigma,pilot,geometry):
    theta=np.asarray(pilot['theta'],float);rate,j=rates_and_design(theta,len(counts),sigma);initial_value=forward.full_factor(rate,j,geometry)[0]
    if geometry['td']>=geometry['te']:return dict(theta=theta.copy(),loglik=initial_value,optimizer_success=bool(pilot['optimizer_success']),evaluations=0,mode='IDENTICAL_SAFE_NULL')
    centre=float(theta[1]);scale=max(1.,sum(counts))
    def objective(x):
        rate,j=rates_and_design(x,len(counts),sigma);value,gradient=forward.full_factor(rate,j,geometry);return -value/scale,-gradient/scale
    result=minimize(objective,theta.copy(),jac=True,method='L-BFGS-B',bounds=[(0,20),(centre-3*sigma,centre+3*sigma),(1e-6,30)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=120,maxls=30));q=result.x.copy();value=-float(result.fun)*scale
    if value<initial_value:q=theta.copy();value=initial_value
    q[1]%=len(counts);return dict(theta=q,loglik=float(value),initial_loglik=initial_value,optimizer_success=bool(result.success),evaluations=int(result.nfev),mode='ONE_WARM_START_FULL_NULL')


def statistics(counts,exposure,sigma,pilot,geometry,bank,regime):
    if geometry['td']>=geometry['te']:
        old=fine_scan(counts,exposure,sigma,fit=pilot);v={k:old[k] for k in ['score','primary','delay']};return dict(samples={m:dict(v) for m in METHODS},full_null=refit_full_null(counts,sigma,pilot,geometry),bank_flags=dict(safe=None,refit=None),mode='EXACT_SAFE_PASSTHROUGH')
    full=refit_full_null(counts,sigma,pilot,geometry);out={};flags={};k=len(counts)
    for name,theta in [('safe',np.asarray(pilot['theta'])),('refit',full['theta'])]:
        rate,j=rates_and_design(theta,k,sigma);ll,gradient,terms,pieces=reverse.full_factor(rate,np.eye(k),geometry,return_pieces=True);hb,flags[name]=bank.lookup(theta,regime)
        for kind,h in [('opg',pieces.T@pieces),('bank',hb)]:
            v=projected(gradient,h,j,theta,sigma);out['full_'+kind+'_'+name]={key:v[key] for key in ['score','primary','delay']}
    return dict(samples=out,full_null=full,bank_flags=flags,mode='FULL_CONTINUOUS_TWO_BY_TWO_ABLATION')


def full_likelihood_ratio(counts,sigma,null_fit,safe_alternatives,geometry):
    """Practical full-law control:6feasible starts evaluated,one refined.

    Preplanned stronger control. Not globally optimized. Every start selection
    and full null/alternative refit must repeat inside calibration paths.
    """
    k=len(counts);scale=max(1.,sum(counts));a,t,b=np.asarray(null_fit['theta']);candidates=[]
    for index,saved in enumerate(safe_alternatives[:6]):
        x=np.asarray(saved['theta'],float).copy();x[1]+=k*round((t-x[1])/k);x[1]=np.clip(x[1],t-3*sigma,t+3*sigma);rate,j=rates_and_design(x,k,sigma);ll=forward.full_factor(rate,j,geometry)[0];candidates.append(dict(theta=x,loglik=float(ll),source_index=index))
    best=max(candidates,key=lambda z:z['loglik']);centre=float(best['theta'][4]);bounds=[(0,20),(t-3*sigma,t+3*sigma),(1e-6,30),(0,20),(centre-3*sigma,centre+3*sigma)]
    def objective(x):
        rate,j=rates_and_design(x,k,sigma);value,gradient=forward.full_factor(rate,j,geometry);return -value/scale,-gradient/scale
    result=minimize(objective,best['theta'],jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-11,gtol=1e-7,maxiter=160,maxls=30));ll=-float(result.fun)*scale;x=result.x.copy()
    if ll<best['loglik']:x=best['theta'];ll=best['loglik']
    if ll<null_fit['loglik']:x=np.array([a,t,b,0.,centre]);ll=null_fit['loglik']
    x[1]%=k;x[4]%=k;primary,secondary=(x[1],x[4]) if x[0]>=x[3] else (x[4],x[1])
    return dict(score=max(0.,2*(ll-null_fit['loglik'])),primary=float(primary),delay=float(secondary),theta=x.tolist(),loglik=float(ll),optimizer_success=bool(result.success),evaluations=int(result.nfev),selected_safe_component=best['source_index'],all6_full_start_logliks=[x['loglik'] for x in candidates])
