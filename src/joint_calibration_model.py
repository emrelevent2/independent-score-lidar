"""Independent fixed-cycle calibration splits and proper predictive densities."""
import numpy as np
from scipy.optimize import minimize,nnls
from scipy.special import xlogy
from flint import arb
from pilot import coates
from shoulder_alternative import donor_proposal,proposal_from_pool
from envelope_evidence import gaussian_envelope,exact_gaussian,outward_float

EDGES=(0,27,38,43,54,128,256)


def split_cycles(h,cycles,seed):
    h=np.asarray(h)
    if h.shape!=(256,) or np.any(h<0) or np.any(h!=np.floor(h)) or h.sum()>cycles or int(cycles)!=cycles or cycles<2:
        raise ValueError('Valid first-event calibration with known cycles required')
    full=np.r_[h.astype(np.int64),int(cycles-h.sum())]
    n1=int(cycles)//2;n2=int(cycles)-n1
    first=np.random.default_rng(seed).multivariate_hypergeometric(full,n1)
    second=full-first
    # Including the no-event category is essential for a fixed-cycle split.
    assert first.sum()==n1 and second.sum()==n2
    return first[:-1],second[:-1],n1,n2


def aggregate(values):
    if len(values)!=256:raise ValueError('256-bin calibration required')
    return np.array([np.sum(values[a:b]) for a,b in zip(EDGES[:-1],EDGES[1:])])


def fit_training(h,cycles,cap=.25):
    """A numerical MLE proposal trained on ONE half only; not a certificate."""
    h=np.asarray(h);t=np.arange(256);remaining=cycles-np.cumsum(h);k=4*np.log(2)
    def pulse(d,w):
        g=np.exp(-k*((t-d)/w)**2);s=np.exp(-k*((t-(d+16))/w)**2)
        return g,s
    g,s=pulse(32.,5.);design=np.stack([g,g+cap*s,np.ones(256)/256],axis=1)
    coef,_=nnls(design,coates(h,cycles)*cycles)
    initial=np.r_[coef,32.,5.];initial[2]=max(initial[2],1e-8)
    def loss(x):
        u,v,b,d,w=x;g,s=pulse(d,w)
        rate=((u+v)*g+cap*v*s+b/256)/cycles
        value=np.sum(xlogy(h,-np.expm1(-rate))-remaining*rate)
        with np.errstate(over='ignore'):
            derivative=h/np.expm1(rate)-remaining
        dg=2*k*(t-d)/w**2*g;ds=2*k*(t-(d+16))/w**2*s
        wg=2*k*(t-d)**2/w**3*g;ws=2*k*(t-(d+16))**2/w**3*s
        jac=[g,g+cap*s,np.ones(256)/256,(u+v)*dg+cap*v*ds,(u+v)*wg+cap*v*ws]
        grad=np.array([np.sum(derivative*z/cycles) for z in jac])
        return -float(value),-grad
    fit=minimize(loss,initial,jac=True,method='L-BFGS-B',bounds=[(0,None),(0,None),(1e-10,None),(31.5,32.5),(4.5,5.5)],
                 options={'ftol':1e-12,'gtol':1e-7,'maxiter':300})
    x=fit.x if np.all(np.isfinite(fit.x)) else initial
    u,v,b,d,w=map(float,x);eta=min(cap,max(0.,cap*v/(u+v))) if u+v>0 else 0.
    g,s=pulse(d,w);rates=np.maximum(((u+v)*g+cap*v*s+b/256)/cycles,1e-12)
    return dict(eta=eta,parameters=[u,v,b,d,w],rates=rates.tolist(),optimizer_success=bool(fit.success),
                training_cycles=int(cycles),scored_calibration_excluded=True,query_excluded=True)


def calibration_basis(eta0,eta1,cap=.25):
    """Enclose normalized window masses for all trusted calibration d/w.

    The main window's unshifted mass and lobe window's shifted mass are
    IDENTICAL. Normalize the free amplitude by that mass to avoid a spurious
    width/phase normalization penalty. Off-window leakage is retained.
    """
    if not 0<=eta0<=eta1<=cap:raise ValueError('Invalid shared response interval')
    lo,hi=gaussian_envelope(256,31.5,32.5,4.5,5.5,False)
    sl,sh=gaussian_envelope(272,47.5,48.5,4.5,5.5,False)
    al=sum((arb(float(x)) for x in lo[27:38]),arb(0));au=sum((arb(float(x)) for x in hi[27:38]),arb(0))
    lower=[];upper=[]
    for j,(start,stop) in enumerate(zip(EDGES[:-1],EDGES[1:])):
        gl=sum((arb(float(x)) for x in lo[start:stop]),arb(0))/au
        gu=sum((arb(float(x)) for x in hi[start:stop]),arb(0))/al
        spl=sum((arb(float(x)) for x in sl[start:stop]),arb(0))/au
        spu=sum((arb(float(x)) for x in sh[start:stop]),arb(0))/al
        if j==1:gl=gu=arb(1)
        if j==3:spl=spu=arb(1)
        lower.append([max(0.,outward_float(gl+arb(float(eta0))*spl,False)),(stop-start)/256])
        upper.append([outward_float(gu+arb(float(eta1))*spu),(stop-start)/256])
    return np.array(lower),np.array(upper)


def query_basis(bins,d0,d1,w0,w1,eta0,eta1,shift=16):
    lo,hi=gaussian_envelope(bins,d0,d1,w0,w1,False)
    sd0=max(0.,outward_float(arb(float(d0))+shift,False))
    sd1=min(float(bins+shift-1),outward_float(arb(float(d1))+shift))
    sl,sh=gaussian_envelope(bins+shift,sd0,sd1,w0,w1,False)
    lower=[max(0.,outward_float(arb(float(x))+arb(float(eta0))*arb(float(y)),False)) for x,y in zip(lo,sl[:bins])]
    upper=[outward_float(arb(float(x))+arb(float(eta1))*arb(float(y))) for x,y in zip(hi,sh[:bins])]
    bl=outward_float(arb(1)/bins,False);bu=outward_float(arb(1)/bins)
    return np.column_stack([lower,np.full(bins,bl)]),np.column_stack([upper,np.full(bins,bu)])
