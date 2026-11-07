"""Retain donor primary modes instead of committing to a single anchor.

Development transfer method. All bank parameters/weights are donor-only;
the numerical null denominator still requires empirical calibration.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares,nnls
from scipy.special import logsumexp
from real_poisson_methods_v2 import _matched,_peaks,_initial,WorkingPulse
from real_coverage_mixture import components


def donor_bank(h,pulse,row,col):
    if not (4<=row<h.shape[0]-4 and 4<=col<h.shape[1]-4):raise ValueError('Full donor window required')
    y,x=np.mgrid[-4:5,-4:5];keep=(x!=0)|(y!=0);x,y=x[keep],y[keep]
    donors=np.asarray(h[row-4:row+5,col-4:col+5][keep],float);bins=h.shape[-1];t=np.arange(bins)
    spatial=np.exp(-.5*(x*x+y*y)/1.5**2);spatial/=spatial.sum()
    primary=np.argmax(_matched(donors,pulse),axis=-1)
    votes=gaussian_filter1d(np.bincount(primary,weights=spatial,minlength=bins),2.,mode='constant')
    anchors=[]
    for d in _peaks(votes,n=bins):
        if all(abs(d-s)>=8 for s in anchors):anchors.append(d)
        if len(anchors)==3:break
    bank=[]
    for anchor in anchors:
        logw=-.5*((x*x+y*y)/1.5**2+(primary-anchor)**2/2.**2)
        weights=np.exp(logw-logw.max());weights/=weights.sum();pool=weights@donors
        coef,_=nnls(np.stack([pulse.derivatives(t,anchor,8.)[0],np.ones(bins)/bins],axis=-1),pool)
        residual=pool-coef[0]*pulse.derivatives(t,anchor,8.)[0]-coef[1]/bins
        score=_matched(residual,pulse);allowed=(abs(t-anchor)>=8)&(t>=4)&(t<bins-4)
        secondary=int(np.argmax(np.where(allowed,score,-np.inf)))
        initial=_initial(pool,pulse,anchor,8.,secondary)
        lower=np.array([0.,0.,1e-9,max(0,anchor-2.),max(0,secondary-2.),3.])
        upper=np.array([np.inf,np.inf,np.inf,min(bins-1,anchor+2.),min(bins-1,secondary+2.),16.])
        initial=np.minimum(np.maximum(initial,lower+1e-8),upper-1e-8)
        def model(q):
            a,c,b,d1,d2,w=q;return a*pulse.derivatives(t,d1,w)[0]+c*pulse.derivatives(t,d2,w)[0]+b/bins
        fit=least_squares(lambda q:model(q)-pool,initial,bounds=(lower,upper),max_nfev=140,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        a,c,b,d1,d2,w=fit.x;g=pulse.derivatives(t,d2,w)[0];neff=float(1/(weights@weights))
        v=(c+b/(bins*np.sum(g*g)))/neff
        bank.append(dict(primary=float(d1),secondary=float(d2),primary_count=float(a),secondary_count=float(c),background_count=float(b),
            width=float(w),effective_donors=neff,support_proxy=float(c/np.sqrt(max(v,1e-12))),raw_center_excluded=True,
            optimizer_success=bool(fit.success),anchor=int(anchor)))
    return bank


def bank_report(h,pulse,row,col,null_log_likelihood):
    bank=donor_bank(h,pulse,row,col);query=np.asarray(h[row,col]);logs=[];candidates=[];count=0
    for index,q in enumerate(bank):
        m=components(q,len(query));rate=m['means'];weighted=np.log(rate)@query-rate.sum(1)+np.log([float(w) for w in m['weights']])-np.log(len(bank))
        logs.append(float(logsumexp(weighted)));count+=len(weighted)
        candidates.append(dict(log_mass=float(logsumexp(weighted[:4])),primary=q['primary'],delay=q['secondary'],head='spatial',bank=index))
        global_mass=logsumexp(weighted[4:].reshape(len(m['global_delays']),len(m['amplitudes'])),axis=1)
        j=int(np.argmax(global_mass));mode=m['global_delays'][j];near=abs(m['global_delays']-mode)<=2
        delay=float(np.exp(global_mass[near]-logsumexp(global_mass[near]))@m['global_delays'][near])
        candidates.append(dict(log_mass=float(global_mass[j]),primary=q['primary'],delay=delay,head='global',bank=index))
    full=float(logsumexp(logs));chosen=max(candidates,key=lambda r:r['log_mass'])
    return dict(score=full-null_log_likelihood,primary=chosen['primary'],delay=chosen['delay'],head=chosen['head'],selected_bank=chosen['bank'],
        log_mixture=full,bank_size=len(bank),components=count,donor_proposals=bank,calibration_required=True,analytic_guarantee=False)
