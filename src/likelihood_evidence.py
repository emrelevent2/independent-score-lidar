"""Prototype universal likelihood evidence under a discrete-delay null.

The general e-value construction is established universal inference. Our
implementation uses a tangent upper bound for each concave single-return
likelihood, then takes the maximum bound over ALL discrete primary delays.
Alternative parameters must be independent of the tested histogram.

The analytic upper-bound certificate is evaluated in floating point. This is
not interval-arithmetic certification. Broad empirical performance is untested.
"""
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import xlogy
from scipy.ndimage import uniform_filter
from pilot import coates
from crossfit import propose


def gaussian_templates(bins=256, fwhm=5.):
    t=np.arange(bins)
    sigma=fwhm/(2*np.sqrt(2*np.log(2)))
    g=np.exp(-.5*((t[None,:]-t[:,None])/sigma)**2)
    return g/g.sum(-1,keepdims=True)


def risk_counts(h,cycles):
    h=np.asarray(h,dtype=float)
    if h.ndim!=1 or np.any(h<0) or h.sum()>cycles:
        raise ValueError('Invalid one-pixel first-detection histogram')
    return cycles-np.cumsum(h)


def log_likelihood(h,cycles,incident):
    lam=np.asarray(incident,dtype=float)
    if lam.shape!=np.asarray(h).shape or np.any(lam<0):
        raise ValueError('Invalid incident rate')
    failures=risk_counts(h,cycles)
    return float(np.sum(xlogy(h,-np.expm1(-lam))-failures*lam))


def single_delay_bound(h,cycles,pulse,initial=None):
    """Global tangent bound for nonnegative amplitude/background at ONE delay.

    theta contains total incident signal and background counts per acquisition.
    For a concave ell, ell(u)<=ell(theta)+grad(theta).(u-theta).
    If grad<=0 componentwise, its supremum on u>=0 is ell-grad.theta.
    This remains an upper bound even if numerical optimization is imperfect.
    """
    h=np.asarray(h,float)
    fail=risk_counts(h,cycles)
    design=np.stack([pulse,np.ones_like(pulse)/len(pulse)],axis=-1)/cycles
    if initial is None:
        initial=np.array([max(h.sum()*.7,1.),max(h.sum()*.3,1.)])

    def value_grad(theta):
        lam=design@theta
        # A strictly positive background ensures finite derivatives.
        ell=np.sum(xlogy(h,-np.expm1(-lam))-fail*lam)
        grad=design.T@(h/np.expm1(lam)-fail)
        return -float(ell),-grad

    fit=minimize(value_grad,initial,jac=True,method='L-BFGS-B',
                 bounds=[(0,None),(1e-10,None)],
                 options={'ftol':1e-13,'gtol':1e-9,'maxiter':150})
    theta=np.maximum(fit.x,[0.,1e-10])
    # Increasing all positive rates decreases every component of grad(ell)
    # because all design entries are nonnegative. Find a feasible tangent.
    for k in range(30):
        scale=1.+1e-7*(2.**k)
        point=theta*scale
        negative_ell,negative_grad=value_grad(point)
        grad=-negative_grad
        if np.all(grad<=-1e-12):
            ell=-negative_ell
            upper=min(0.,ell-float(grad@point)+1e-8)
            return dict(upper=upper,lower=-float(fit.fun),
                        certificate_gap=upper+float(fit.fun),
                        tangent_gradient=grad,parameters=point,optimizer_success=bool(fit.success))
    # Every hazard log likelihood is <=0. This fallback sacrifices power but
    # retains a valid analytic upper bound without relying on solver success.
    return dict(upper=0.,lower=-float(fit.fun),certificate_gap=float(fit.fun),
                tangent_gradient=None,parameters=theta,optimizer_success=bool(fit.success))


def null_upper_bound(h,cycles,templates=None):
    if templates is None:templates=gaussian_templates(len(h))
    bounds=[single_delay_bound(h,cycles,g) for g in templates]
    upper=np.asarray([b['upper'] for b in bounds])
    lower=np.asarray([b['lower'] for b in bounds])
    winner=int(np.argmax(upper))
    return dict(upper=float(upper.max()),best_feasible=float(lower.max()),
                global_gap=float(upper.max()-lower.max()),delay_at_upper=winner,
                failed_optimizers=sum(not b['optimizer_success'] for b in bounds),
                fallback_count=sum(b['tangent_gradient'] is None for b in bounds))


def neighbor_alternative(histograms,cycles=1000,size=9):
    """All alternative parameters, including nuisances, use ONLY neighbors."""
    if size%2!=1 or size<3 or size>min(histograms.shape[:2]):
        raise ValueError('Neighborhood must contain distinct donors')
    x=coates(histograms,cycles)
    k=size*size
    pooled=(uniform_filter(x,(size,size,1),mode='wrap')*k-x)/(k-1)
    pooled=np.maximum(pooled,0.)
    p,s,valid=propose(pooled)
    g=gaussian_templates(x.shape[-1])
    alt=np.empty_like(x)
    for y in range(x.shape[0]):
        for col in range(x.shape[1]):
            design=np.stack([g[p[y,col]],g[s[y,col]],np.ones(x.shape[-1])/x.shape[-1]],axis=-1)
            coeff,_=nnls(design,pooled[y,col])
            alt[y,col]=np.maximum(design@coeff,1e-12)
    return alt,p,s,valid


def evaluate_evidence(h,alternative,cycles=1000,templates=None):
    null=null_upper_bound(h,cycles,templates)
    alternative_log_likelihood=log_likelihood(h,cycles,alternative)
    return dict(log_e=alternative_log_likelihood-null['upper'],
                alternative_log_likelihood=alternative_log_likelihood,**null)
