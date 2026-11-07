"""Development: spatially local, foreground-compatible donor distributions.

All raw center data are removed BEFORE any correction, depth estimate or weight.
Mixtures remain proper distributions. Posterior component choice is for the
reported location only, never a replacement of the mixture likelihood.
"""
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import least_squares,nnls
from pilot import coates
from crossfit import propose
from likelihood_evidence import gaussian_templates
from continuous_alternative import alternative_at,amplitude_components


def fit_pool(pooled,cycles=1000):
    p,s,valid=propose(pooled[None,None,:]);p,s,valid=int(p[0,0]),int(s[0,0]),bool(valid[0,0])
    observed=pooled*cycles;t=np.arange(len(pooled));g=gaussian_templates(len(t))
    coefficients,_=nnls(np.stack([g[p],g[s],np.ones(len(t))/len(t)],axis=-1),observed)
    def pulse(center,width):
        x=np.exp(-4*np.log(2)*((t-center)/width)**2);return x/x.sum()
    def model(q):
        a,c,b,d1,d2,w=q;return a*pulse(d1,w)+c*pulse(d2,w)+b/len(t)
    lower=np.array([0.,0.,1e-10,max(0,p-1.5),max(0,s-1.5),4.5])
    upper=np.array([np.inf,np.inf,np.inf,min(len(t)-1,p+1.5),min(len(t)-1,s+1.5),5.5])
    initial=np.minimum(np.maximum(np.r_[coefficients,p,s,5.],lower+1e-9),upper-1e-9)
    fit=least_squares(lambda q:model(q)-observed,initial,bounds=(lower,upper),
        ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=120)
    a,c,b,d1,d2,w=fit.x
    return dict(primary=float(d1),secondary=float(d2),primary_incident=float(a),secondary_incident=float(c),
        background_incident=float(b),fitted_fwhm=float(w),valid=valid,optimizer_success=bool(fit.success))


def guided_pool(h,row,col,spatial_sigma=1.5,depth_sigma=2.,size=9,cycles=1000):
    radius=size//2
    if not (radius<=row<h.shape[0]-radius and radius<=col<h.shape[1]-radius):raise ValueError('Full donor window required')
    dy,dx=np.mgrid[-radius:radius+1,-radius:radius+1];keep=(dy!=0)|(dx!=0)
    donors=h[row-radius:row+radius+1,col-radius:col+radius+1][keep]
    flux=coates(donors,cycles);dy,dx=dy[keep],dx[keep]
    t=np.arange(-10,11);template=np.exp(-4*np.log(2)*(t/5.)**2);template/=template.sum()
    primary=np.argmax(convolve1d(flux,template,axis=-1,mode='constant'),axis=-1)
    near=(np.abs(dy)<=1)&(np.abs(dx)<=1)
    anchor=float(np.median(primary[near]))
    weights=np.exp(-.5*((dy*dy+dx*dx)/spatial_sigma**2+(primary-anchor)**2/depth_sigma**2))
    weights/=weights.sum()
    return weights@flux,dict(anchor=anchor,effective_donors=float(1/np.sum(weights*weights)),
        spatial_sigma=spatial_sigma,depth_sigma=depth_sigma,raw_center_excluded=True)


def candidates(h,row=8,col=8,cycles=1000):
    groups={}
    for size,label in [(9,'old9'),(3,'local3')]:
        _,p=alternative_at(h,row,col,cycles=cycles,size=size)
        groups[label]=dict(parameters=[p],means=amplitude_components(p,cycles=cycles),
            component_delays=[p['secondary']]*4,valid=p['valid'])
    for sigma,label in [(1.5,'guided15'),(2.5,'guided25')]:
        pool,weights=guided_pool(h,row,col,spatial_sigma=sigma,cycles=cycles)
        p=fit_pool(pool,cycles);p.update(weights)
        groups[label]=dict(parameters=[p],means=amplitude_components(p,cycles=cycles),
            component_delays=[p['secondary']]*4,valid=p['valid'])
    for labels,name in [(('local3','guided15'),'mixture_local_guided'),(('local3','guided25','old9'),'mixture_three')]:
        parts=[groups[x] for x in labels]
        groups[name]=dict(parameters=sum([x['parameters'] for x in parts],[]),means=sum([x['means'] for x in parts],[]),
            component_delays=sum([x['component_delays'] for x in parts],[]),valid=any(x['valid'] for x in parts))
    return groups
