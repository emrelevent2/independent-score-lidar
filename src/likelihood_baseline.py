"""Conventional local likelihood-improvement control; empirically calibrated.

Nonlinear fits are multistart numerical estimates, not certified global maxima.
The alternative uses the tested histogram, so this is NOT an e-value.
"""
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import minimize,nnls,least_squares
from scipy.special import xlogy
from screen import detect_scores


def pulse_derivatives(t,delay,width):
    c=4*np.log(2);dt=t-delay
    g=np.exp(-c*(dt/width)**2);g/=g.sum()
    a=2*c*dt/width**2;b=2*c*dt*dt/width**3
    return g,g*(a-np.sum(g*a)),g*(b-np.sum(g*b))


def fitted_flux_report(flux,cycles=1000):
    f=np.asarray(flux,float);t=np.arange(len(f));z=detect_scores(f[None,None])[0,0]
    secondary=int(np.argmax(z));valid=bool(np.isfinite(z[secondary]))
    template=np.exp(-4*np.log(2)*(np.arange(-10,11)/5)**2);template/=template.sum()
    primary=int(np.argmax(convolve1d(f,template,mode='constant')))
    g1=pulse_derivatives(t,primary,5.)[0];g2=pulse_derivatives(t,secondary,5.)[0]
    design=np.stack([g1,g2,np.ones(len(t))/len(t)],axis=-1)
    coefficients,_=nnls(design,np.maximum(f,0)*cycles)
    def model(q):
        a,c,b,p,s,w=q
        return a*pulse_derivatives(t,p,w)[0]+c*pulse_derivatives(t,s,w)[0]+b/len(t)
    initial=np.r_[coefficients,primary,secondary,5.]
    lower=np.array([0.,0.,1e-10,max(0,primary-1.5),max(0,secondary-1.5),4.5])
    upper=np.array([np.inf,np.inf,np.inf,min(len(t)-1,primary+1.5),min(len(t)-1,secondary+1.5),5.5])
    initial=np.minimum(np.maximum(initial,lower+1e-9),upper-1e-9)
    fit=least_squares(lambda q:model(q)-f*cycles,initial,bounds=(lower,upper),max_nfev=120,
                      ftol=1e-10,xtol=1e-10,gtol=1e-10)
    return dict(score=float(fit.x[1]) if valid else -np.inf,
                delay=float(fit.x[4]),primary=float(fit.x[3]),
                parameters=fit.x.tolist(),optimizer_success=bool(fit.success))


def local_likelihood_report(h,cycles=1000):
    h=np.asarray(h,float);t=np.arange(len(h));failures=cycles-np.cumsum(h)
    if h.sum()>cycles or np.any(h<0):raise ValueError('Invalid histogram')
    template=np.exp(-4*np.log(2)*(np.arange(-10,11)/5)**2);template/=template.sum()
    matched=convolve1d(h,template,mode='constant')
    maxima=(matched>=np.roll(matched,1))&(matched>np.roll(matched,-1))
    primary_starts=np.argsort(np.where(maxima,matched,-np.inf))[-3:][::-1]

    def objective(q,two=False):
        if two:a,c,b,d1,d2,w=q
        else:a,b,d1,w=q
        g1,dg1,wg1=pulse_derivatives(t,d1,w)
        mean=a*g1+b/len(t)
        jac=[g1,np.ones(len(t))/len(t),a*dg1,a*wg1]
        if two:
            g2,dg2,wg2=pulse_derivatives(t,d2,w)
            mean+=c*g2
            jac=[g1,g2,np.ones(len(t))/len(t),a*dg1,c*dg2,a*wg1+c*wg2]
        rate=mean/cycles
        ell=np.sum(xlogy(h,-np.expm1(-rate))-failures*rate)
        with np.errstate(over='ignore'):derivative=h/np.expm1(rate)-failures
        gradient=np.asarray([np.sum(derivative*j/cycles) for j in jac])
        return -float(ell),-gradient

    best_null=None;null_fits=[]
    for p in primary_starts:
        design=np.stack([pulse_derivatives(t,p,5.)[0],np.ones(len(t))/len(t)],axis=-1)
        coeff,_=nnls(design,h)
        initial=np.r_[coeff[0],max(coeff[1],1e-8),p,5.]
        fit=minimize(objective,initial,jac=True,method='L-BFGS-B',
            bounds=[(0,None),(1e-10,None),(max(0,p-2.),min(len(t)-1,p+2.)),(4.5,5.5)],
            options={'ftol':1e-12,'gtol':1e-8,'maxiter':200,'maxls':40})
        null_fits.append(fit)
        if best_null is None or fit.fun<best_null.fun:best_null=fit
    a,b,d1,w=best_null.x
    residual=h-(a*pulse_derivatives(t,d1,w)[0]+b/len(t))
    score=convolve1d(residual,template,mode='constant')
    valid=(np.abs(t-d1)>=8)&(t>=10)&(t<len(t)-10)
    peaks=(score>=np.roll(score,1))&(score>np.roll(score,-1))&valid
    starts=np.argsort(np.where(peaks,score,-np.inf))[-3:][::-1]
    fits=[]
    for s in starts:
        initial=[a,max(1.,float(score[s])/np.sum(template**2)),max(b,1e-8),d1,s,w]
        fit=minimize(lambda q:objective(q,True),initial,jac=True,method='L-BFGS-B',
            bounds=[(0,None),(0,None),(1e-10,None),(max(0,d1-2),min(len(t)-1,d1+2)),
                    (max(0,s-2.),min(len(t)-1,s+2.)),(4.5,5.5)],
            options={'ftol':1e-12,'gtol':1e-8,'maxiter':200,'maxls':40})
        fits.append(fit)
    best=min(fits,key=lambda f:f.fun)
    return dict(score=max(0.,float(best_null.fun-best.fun)),delay=float(best.x[4]),
                null_log_likelihood=-float(best_null.fun),alternative_log_likelihood=-float(best.fun),
                primary=float(best.x[3]),parameters=best.x.tolist(),
                optimizer_failures=sum(not f.success for f in null_fits+fits))
