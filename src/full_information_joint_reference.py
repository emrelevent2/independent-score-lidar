"""Stronger equal-information full-law reference,without changing the primary.

Keeps the legacy6-safe-start reference as a feasible solution. Adds6full-score
starts and refines the top2score starts plus the best new initial likelihood.
No global optimality is claimed. Every adaptive step requires recalibration.
"""
import numpy as np
from scipy.optimize import minimize
import ct_dual_fast as forward
import ct_dual_reverse as reverse
from ct_full_score import projected
from clock_pulse_likelihood import rates_and_design
from null_bank_score import full_likelihood_ratio


def stronger_joint(counts,sigma,null_fit,safe_alternatives,geometry,bank,regime,legacy=None):
    if legacy is None:legacy=full_likelihood_ratio(counts,sigma,null_fit,safe_alternatives,geometry)
    if geometry['td']>=geometry['te']:return dict(legacy,mode='EXACT_NO_HIDDEN_LEGACY_EQUIVALENCE',full_score_starts=[],additional_refinements=[])
    k=len(counts);a,t,b=np.asarray(null_fit['theta']);rate,j=rates_and_design(null_fit['theta'],k,sigma);ll,gradient=reverse.full_factor(rate,np.eye(k),geometry);h,flags=bank.lookup(null_fit['theta'],regime);scan=projected(gradient,h,j,null_fit['theta'],sigma);starts=[]
    for index in np.argsort(scan['scores'])[::-1]:
        if not np.isfinite(scan['scores'][index]):continue
        centre=float(scan['grid'][index])
        if all(abs((centre-v+k/2)%k-k/2)>2*sigma for v in starts):starts.append(centre)
        if len(starts)==6:break
    initial=[]
    for centre in starts:
        x=np.array([a,t,b,.02,centre]);rate,j=rates_and_design(x,k,sigma);value=forward.full_factor(rate,j,geometry)[0];initial.append(dict(theta=x,loglik=value,centre=centre))
    best_initial=int(np.argmax([r['loglik'] for r in initial]));selected=sorted(set([0,min(1,len(initial)-1),best_initial]));scale=max(1.,sum(counts));fits=[]
    def objective(x):
        rate,j=rates_and_design(x,k,sigma);value,gradient=forward.full_factor(rate,j,geometry);return -value/scale,-gradient/scale
    best=dict(legacy)
    for index in selected:
        start=initial[index];centre=start['centre'];result=minimize(objective,start['theta'],jac=True,method='L-BFGS-B',bounds=[(0,20),(t-3*sigma,t+3*sigma),(1e-6,30),(0,20),(centre-3*sigma,centre+3*sigma)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=160,maxls=30));x=result.x.copy();value=-float(result.fun)*scale
        if value<start['loglik']:x=start['theta'].copy();value=start['loglik']
        x[1]%=k;x[4]%=k;primary,secondary=(x[1],x[4]) if x[0]>=x[3] else (x[4],x[1]);fit=dict(theta=x.tolist(),loglik=float(value),primary=float(primary),delay=float(secondary),optimizer_success=bool(result.success),evaluations=int(result.nfev),full_score_start_index=index,start=float(centre));fits.append(fit)
        if value>best['loglik']:best.update({key:fit[key] for key in ['theta','loglik','primary','delay','optimizer_success','evaluations']})
    # All unrefined full-score starts remain feasible points too.
    for r in initial:
        if r['loglik']>best['loglik']:
            x=r['theta'];best.update(theta=x.tolist(),loglik=float(r['loglik']),primary=float(x[1]),delay=float(x[4]),optimizer_success=True,evaluations=0)
    best.update(score=max(0.,2*(best['loglik']-null_fit['loglik'])),mode='LEGACY_PLUS_FULL_INFORMATION_STARTS',full_score_starts=starts,full_score_initial_logliks=[float(r['loglik']) for r in initial],additional_refinements=fits,bank_flags=flags);assert best['loglik']>=legacy['loglik']-1e-10;return best
