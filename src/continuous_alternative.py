"""Fit a continuous two-return alternative using donor measurements only."""
import numpy as np
from scipy.optimize import least_squares,nnls
from scipy.ndimage import convolve
from pilot import coates
from crossfit import propose
from likelihood_evidence import gaussian_templates


def alternative_at(histograms,row,col,cycles=1000,size=9,fwhm_range=(4.5,5.5)):
    if size<3 or size%2!=1 or size>min(histograms.shape[:2]):raise ValueError('Invalid donor support')
    x=coates(histograms,cycles)
    weights=np.ones((size,size,1))/(size*size-1);weights[size//2,size//2,0]=0.
    pooled=convolve(x,weights,mode='wrap')
    p,s,valid=propose(pooled)
    p,s,valid=int(p[row,col]),int(s[row,col]),bool(valid[row,col])
    observed=pooled[row,col]*cycles
    t=np.arange(len(observed));g=gaussian_templates(len(observed))
    design=np.stack([g[p],g[s],np.ones(len(t))/len(t)],axis=-1)
    coeff,_=nnls(design,observed)
    def pulse(delay,width):
        z=np.exp(-4*np.log(2)*((t-delay)/width)**2)
        return z/z.sum()
    def model(theta):
        a,c,b,d1,d2,w=theta
        return a*pulse(d1,w)+c*pulse(d2,w)+b/len(t)
    theta=np.r_[coeff,p,s,np.mean(fwhm_range)]
    lower=[0.,0.,1e-10,max(0.,p-1.5),max(0.,s-1.5),fwhm_range[0]]
    upper=[np.inf,np.inf,np.inf,min(len(t)-1.,p+1.5),min(len(t)-1.,s+1.5),fwhm_range[1]]
    if fwhm_range[0]>=fwhm_range[1]:raise ValueError('A nonzero width interval is required')
    theta=np.maximum(theta,np.asarray(lower)+1e-9)
    theta=np.minimum(theta,np.asarray(upper)-1e-9)
    fit=least_squares(lambda q:model(q)-observed,theta,bounds=(lower,upper),
                      ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=120)
    rates=np.maximum(model(fit.x)/cycles,1e-12)
    return rates,dict(primary=float(fit.x[3]),secondary=float(fit.x[4]),
                      fitted_fwhm=float(fit.x[5]),valid=valid,
                      primary_incident=float(fit.x[0]),secondary_incident=float(fit.x[1]),
                      background_incident=float(fit.x[2]),
                      optimizer_success=bool(fit.success),donor_size=size)


def amplitude_components(parameters,bins=256,cycles=1000,factors=(.5,1.,2.,4.)):
    t=np.arange(bins);w=parameters['fitted_fwhm']
    def pulse(center):
        g=np.exp(-4*np.log(2)*((t-center)/w)**2);return g/g.sum()
    primary=parameters['primary_incident']*pulse(parameters['primary'])
    secondary=parameters['secondary_incident']*pulse(parameters['secondary'])
    background=parameters['background_incident']/bins
    return [np.maximum((primary+factor*secondary+background)/cycles,1e-12) for factor in factors]
