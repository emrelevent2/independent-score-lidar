"""Donor-only physical-response proposals and proper full-delay mixtures."""
from fractions import Fraction
import numpy as np
from scipy.optimize import least_squares,nnls
from scipy.ndimage import convolve1d
from scipy.special import logsumexp
from guided_alternatives import guided_pool
from support_routing import donor_support
from coverage_alternative import weighted_log_lower
from envelope_evidence import certified_log_likelihood


def pulse(t,d,w,eta,shift=16):
    g=np.exp(-4*np.log(2)*((t-d)/w)**2)
    side=np.exp(-4*np.log(2)*((t-(d+shift))/w)**2)
    return (g+eta*side)/g.sum()  # Amplitude counts the main lobe, not both lobes.


def proposal_from_pool(pool,cycles=1000,cap=.25,shift=16):
    observed=np.asarray(pool)*cycles;t=np.arange(len(pool));bins=len(t)
    kernel=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2)
    p=int(np.argmax(convolve1d(observed,kernel,mode='constant')))
    def single(x):
        u,v,b,d,w=x
        g=pulse(t,d,w,0.,shift);gs=pulse(t,d,w,cap,shift)
        return u*g+v*gs+b/bins
    design=np.stack([pulse(t,p,5.,0.,shift),pulse(t,p,5.,cap,shift),np.ones(bins)/bins],axis=1)
    coef,_=nnls(design,observed)
    lower=np.array([0.,0.,1e-10,max(0.,p-1.5),4.5]);upper=np.array([np.inf]*3+[min(bins-1.,p+1.5),5.5])
    x=np.minimum(np.maximum(np.r_[coef,p,5.],lower+1e-9),upper-1e-9)
    first=least_squares(lambda z:single(z)-observed,x,bounds=(lower,upper),max_nfev=120,
                        ftol=1e-10,xtol=1e-10,gtol=1e-10)
    u,v,b,d,w=first.x;eta=min(cap,max(0.,cap*v/(u+v))) if u+v>0 else 0.
    residual=np.maximum(observed-single(first.x),0.)
    score=convolve1d(residual,kernel,mode='constant')
    score[(np.abs(t-d)<8)|(t<4)|(t>=bins-4)]=-np.inf;s=int(np.argmax(score))
    def two(z):
        a,c,b,d1,d2,w=z
        return a*pulse(t,d1,w,eta,shift)+c*pulse(t,d2,w,eta,shift)+b/bins
    design=np.stack([pulse(t,d,w,eta,shift),pulse(t,s,w,eta,shift),np.ones(bins)/bins],axis=1)
    coef,_=nnls(design,observed)
    low=np.array([0.,0.,1e-10,max(0.,d-1.5),max(0.,s-1.5),4.5])
    high=np.array([np.inf]*3+[min(bins-1.,d+1.5),min(bins-1.,s+1.5),5.5])
    x=np.minimum(np.maximum(np.r_[coef,d,s,w],low+1e-9),high-1e-9)
    fit=least_squares(lambda z:two(z)-observed,x,bounds=(low,high),max_nfev=120,
                      ftol=1e-10,xtol=1e-10,gtol=1e-10)
    a,c,b,d1,d2,w=fit.x
    return dict(primary=float(d1),secondary=float(d2),primary_incident=float(a),
                secondary_incident=float(c),background_incident=float(b),fitted_fwhm=float(w),
                eta_donor=float(eta),raw_center_excluded=True,valid=True,
                optimizer_success=bool(first.success and fit.success))


def donor_proposal(h,row=4,col=4,cycles=1000,cap=.25,shift=16):
    pool,meta=guided_pool(h,row,col,cycles=cycles)
    q=proposal_from_pool(pool,cycles,cap,shift);q.update(meta)
    return q


def components(q,bins=256,cycles=1000,cap=.25,shift=16):
    t=np.arange(bins);w=q['fitted_fwhm'];d=q['primary']
    if not q['raw_center_excluded']:raise ValueError('Donor-only proposal required')
    if not(0<=d<=bins-1 and 4.5<=w<=5.5 and 0<=q['eta_donor']<=cap):raise ValueError('Invalid donor geometry')
    if min(q[k] for k in ['primary_incident','secondary_incident','background_incident'])<0:raise ValueError('Negative amplitude')
    delays=t[(t>=4)&(t<bins-4)&(np.abs(t-d)>=8)]
    amp=np.array([1.,2.,4.,8.,16.,32.]);spatial_mass=Fraction(9,10) if donor_support(q,bins,cycles)>=3.5 else Fraction(1,10)
    blocks=[];weights=[];locations=[];is_spatial=[]
    for eta in [0.,q['eta_donor'],float(cap)]:
        base=q['primary_incident']*pulse(t,d,w,eta,shift)+q['background_incident']/bins
        spatial=np.array([base+f*q['secondary_incident']*pulse(t,q['secondary'],w,eta,shift) for f in [.5,1.,2.,4.]])
        global_pulse=np.array([pulse(t,x,w,eta,shift) for x in delays])
        dense=(base[None,None,:]+global_pulse[:,None,:]*amp[None,:,None]).reshape(-1,bins)
        blocks.extend([spatial,dense])
        weights.extend([spatial_mass/12]*4+[(1-spatial_mass)/(3*len(dense))]*len(dense))
        locations.extend([q['secondary']]*4+list(np.repeat(delays,len(amp))))
        is_spatial.extend([True]*4+[False]*len(dense))
    assert sum(weights,Fraction(0))==1
    return dict(rates=np.maximum(np.vstack(blocks)/cycles,1e-12),weights=weights,
                locations=np.array(locations),spatial=np.array(is_spatial),global_delays=delays,
                raw_center_excluded=True,spatial_mass=str(spatial_mass))


def numeric_and_sparse(h,models,cycles=1000):
    rates=models['rates'];h=np.asarray(h);remaining=cycles-np.cumsum(h)
    ell=np.log(-np.expm1(-rates))@h-rates@remaining
    weighted=ell+np.log([float(w) for w in models['weights']])
    spatial=np.flatnonzero(models['spatial']);glob=np.flatnonzero(~models['spatial'])
    indices=np.r_[spatial,glob[np.argsort(weighted[glob])[::-1][:16]]].astype(int)
    terms={int(i):certified_log_likelihood(h,rates[i],cycles,'first') for i in indices}
    numerator=weighted_log_lower(terms,models['weights'],indices)
    # Sum eta/amplitude uncertainty before selecting a global location.
    spatial_mass=float(logsumexp(weighted[spatial]));grid=models['global_delays']
    masses=np.array([logsumexp(weighted[(~models['spatial'])&(models['locations']==d)]) for d in grid])
    winner=int(np.argmax(masses));mode=float(grid[winner]);near=np.abs(grid-mode)<=2
    if spatial_mass>=masses[winner]:location=float(models['locations'][spatial[0]]);head='spatial'
    else:location=float(np.exp(masses[near]-logsumexp(masses[near]))@grid[near]);head='global'
    return dict(numeric_full=float(logsumexp(weighted)),lower=numerator,location=location,head=head,
                full_components=len(rates),evaluated_components=len(indices),indices=indices.tolist(),
                component_lower={str(k):v for k,v in terms.items()},original_weights_retained=True)
