"""Classical two-return partial-likelihood fits with an optional compiled kernel.

This implements established numerical MLE/GLRT ingredients,not a new theorem.
Fits are numerical local maxima; no global or finite-sample test guarantee.
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from deadtime_path_bootstrap import njit


@njit(cache=True)
def gaussian_bin(phase,centre,bins,sigma):
    centre=centre%bins;value=0.;derivative=0.;root2=math.sqrt(2.)*sigma;scale=1/(math.sqrt(2*math.pi)*sigma)
    for image in range(-2,3):
        low=phase-centre+image*bins;high=low+1
        if low>=0:mass=.5*(math.erfc(low/root2)-math.erfc(high/root2))
        elif high<=0:mass=.5*(math.erfc(-high/root2)-math.erfc(-low/root2))
        else:mass=1-.5*(math.erfc(-low/root2)+math.erfc(high/root2))
        value+=mass;derivative+=scale*(math.exp(-.5*(low/sigma)**2)-math.exp(-.5*(high/sigma)**2))
    return value,derivative


@njit(cache=True)
def objective(theta,counts,exposure,sigma):
    a,t,b,c,d=theta;k=len(counts);gradient=np.zeros(5);ll=0.;scale=max(1.,np.sum(counts))
    for j in range(k):
        f,df=gaussian_bin(j,t,k,sigma);g,dg=gaussian_bin(j,d,k,sigma);rate=max(a*f+b/k+c*g,1e-14);ll+=counts[j]*math.log(rate)-exposure[j]*rate;u=counts[j]/rate-exposure[j]
        gradient[0]-=u*f/scale;gradient[1]-=u*a*df/scale;gradient[2]-=u/k/scale;gradient[3]-=u*g/scale;gradient[4]-=u*c*dg/scale
    return -ll/scale,gradient


def fit_at_start(counts,exposure,sigma,null_fit,secondary):
    a,t,b=np.asarray(null_fit['theta']);initial=np.array([a,t,b,.02,secondary]);scale=max(1.,sum(counts));result=minimize(objective,initial,args=(np.asarray(counts,dtype=float),np.asarray(exposure,dtype=float),float(sigma)),jac=True,method='L-BFGS-B',bounds=[(0,20),(t-3*sigma,t+3*sigma),(1e-6,30),(0,20),(secondary-3*sigma,secondary+3*sigma)],options=dict(ftol=1e-11,gtol=1e-7,maxiter=160))
    # The nested null is a feasible alternative, even if numerical optimization
    # from this particular two-return start fails to improve it.
    theta=result.x.copy();ll=-float(result.fun)*scale
    if ll<null_fit['loglik']:theta=np.array([a,t,b,0.,secondary]);ll=float(null_fit['loglik'])
    theta[1]%=len(counts);theta[4]%=len(counts);primary,second=theta[1],theta[4]
    if theta[3]>theta[0]:primary,second=second,primary
    return dict(loglik=ll,theta=theta.tolist(),primary=float(primary),delay=float(second),optimizer_success=bool(result.success),evaluations=int(result.nfev),start=float(secondary))


def local_starts(counts,exposure,sigma,null_fit):
    # Full quarter-bin score grid is used only to propose numerical starts.
    # All selection and refinements must repeat inside every bootstrap draw.
    from deadtime_path_bootstrap import fine_kernels,fine_correlations
    from deadtime_scan import pulse
    s=np.asarray(counts,float);e=np.asarray(exposure,float);k=len(s);a,t,b=null_fit['theta'];f,df=pulse(t,k,sigma);rate=np.maximum(a*f+b/k,1e-14);u=s/rate-e;w=e/rate;j=np.column_stack([f,a*df,np.ones(k)/k]);inverse=np.linalg.pinv(j.T@(w[:,None]*j),rcond=1e-12);ff,ff2=fine_kernels(k,sigma,4);raw=fine_correlations(u,ff)[0];info=fine_correlations(w,ff2)[0];cross=np.stack([fine_correlations(w*j[:,i],ff)[0] for i in range(3)],axis=1);raw-=cross@inverse@(j.T@u);info-=np.einsum('mi,ij,mj->m',cross,inverse,cross);grid=np.arange(4*k)/4;distance=abs((grid-t+k/2)%k-k/2);scores=np.where((distance>=2.5*sigma)&(info>1e-8),np.maximum(raw,0)**2/np.maximum(info,1e-8),-np.inf);starts=[]
    for idx in np.argsort(scores)[::-1]:
        if not np.isfinite(scores[idx]):continue
        centre=float(grid[idx])
        if all(abs((centre-v+k/2)%k-k/2)>2*sigma for v in starts):starts.append(centre)
        if len(starts)==6:break
    return starts


def reports(counts,exposure,sigma,null_fit,context):
    k=len(counts);starts=local_starts(counts,exposure,sigma,null_fit);fits=[fit_at_start(counts,exposure,sigma,null_fit,v) for v in starts];assert fits
    local=max(fits,key=lambda z:z['loglik']);supported=context['two_pulse_loglr']>20
    if supported:
        relative=(null_fit['theta'][1]+context['secondary']-context['primary'])%k
        extra=[fit_at_start(counts,exposure,sigma,null_fit,v) for v in [context['secondary'],relative]]
        allfits=fits+extra;best=max(allfits,key=lambda z:z['loglik']);mixture_ll=float(logsumexp([local['loglik']+math.log(.5),extra[0]['loglik']+math.log(.25),extra[1]['loglik']+math.log(.25)]))
    else:allfits=fits;best=local;mixture_ll=local['loglik']
    base=float(null_fit['loglik']);out={}
    for name,ll,loc in [('local_joint',local['loglik'],local),('context_max',best['loglik'],best),('context_transport',mixture_ll,best)]:out[name]=dict(score=max(0.,2*(ll-base)),primary=loc['primary'],delay=loc['delay'])
    return dict(samples=out,fits=allfits,context_supported=bool(supported))
