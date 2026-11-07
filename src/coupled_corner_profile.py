"""Checked tangents preserving the shared optical response product v=a*eta.

Query clipped-rate likelihood is concave in(a,v,b); calibration profile is
concave in eta. Their tangents majorize the sum by C+ga*a+gv*a*eta+gb*b+ge*eta.
If gb<=0,its maximum on an amplitude/eta rectangle and b>=0 is attained at
one of FOUR rectangle corners. Standard concavity/bilinear bounding,not a
new general optimization theorem. A separately checked amplitude tail keeps
the original unbounded physical amplitude domain covered.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from flint import arb
from envelope_evidence import gaussian_envelope,_observations,outward_float
from conditional_calibration import saturated_binomial,calibration_upper
from linear_envelope_profile import value_gradient


@lru_cache(maxsize=16384)
def query_basis(bins,d0,d1,w0,w1):
    lo,hi=gaussian_envelope(bins,d0,d1,w0,w1,False)
    sd0=max(0.,outward_float(arb(d0)+16,False));sd1=min(float(bins+15),outward_float(arb(d1)+16))
    sl,sh=gaussian_envelope(bins+16,sd0,sd1,w0,w1,False)
    bl=max(0.,outward_float(arb(1)/bins,False));bu=outward_float(arb(1)/bins)
    return np.column_stack([lo,sl[:bins],np.full(bins,bl)]),np.column_stack([hi,sh[:bins],np.full(bins,bu)])


def query_tangent(theta,h,remaining,lo,hi,cycles):
    x=[arb(float(v)) for v in theta];g=[arb(0) for _ in x];phi=arb(0)
    for count,fail,lp,up in zip(h,remaining,lo,hi):
        count,fail=int(count),int(fail)
        dl=[arb(float(v))/cycles for v in lp];du=[arb(float(v))/cycles for v in up]
        lower=sum((a*b for a,b in zip(x,dl)),arb(0));upper=sum((a*b for a,b in zip(x,du)),arb(0))
        if count==0 and fail==0:continue
        if fail==0:rate,active=upper,du
        else:
            mode=(arb(count)/fail).log1p() if count else arb(0)
            if mode<lower:rate,active=lower,dl
            elif mode>upper:rate,active=upper,du
            elif mode>=lower and mode<=upper:rate,active=mode,None
            else:return None
        if count:phi+=count*(-(-rate).expm1()).log()
        phi-=fail*rate
        derivative=arb(count)/rate.expm1()-fail if count else arb(-fail)
        if active is not None:
            for j in range(3):g[j]+=derivative*active[j]
    return phi,g


def calibration_tangent(eta,k,r,cf):
    """Concave value and supergradient of the profiled affine hazard interval."""
    k,r=int(k),int(r);fail=r-k
    if not(0<=k<=r and 0<=eta<=.25):raise ValueError('Invalid calibration point')
    if not cf['informative'] or r==0:return saturated_binomial(k,r),arb(0)
    e=arb(float(eta));low=arb(cf['signal_main_lower'])*(e+arb(cf['forward_leakage_lower']))+11*arb(cf['background_lower'])
    high=arb(cf['signal_main_upper'])*(e+arb(cf['forward_leakage_upper']))+11*arb(cf['background_upper'])
    if fail==0:rate,slope=high,arb(cf['signal_main_upper'])
    else:
        mode=(arb(k)/fail).log1p() if k else arb(0)
        if mode<low:rate,slope=low,arb(cf['signal_main_lower'])
        elif mode>high:rate,slope=high,arb(cf['signal_main_upper'])
        elif mode>=low and mode<=high:return saturated_binomial(k,r),arb(0)
        else:return saturated_binomial(k,r),arb(0) # global constant majorant
    phi=-(r-k)*rate
    if k:phi+=k*(-(-rate).expm1()).log()
    derivative=(arb(k)/rate.expm1()-fail if k else arb(-fail))*slope
    return phi,derivative


def calibration_value_gradient(eta,k,r,cf):
    if not cf['informative'] or not r:return float(saturated_binomial(k,r)),0.
    lo=cf['signal_main_lower']*(eta+cf['forward_leakage_lower'])+11*cf['background_lower']
    hi=cf['signal_main_upper']*(eta+cf['forward_leakage_upper'])+11*cf['background_upper'];fail=r-k
    mode=np.log1p(k/fail) if fail else np.inf;rate=np.clip(mode,lo,hi)
    value=(k*np.log(-np.expm1(-rate)) if k else 0.)-fail*rate
    slope=cf['signal_main_lower'] if mode<lo else cf['signal_main_upper'] if mode>hi else 0.
    derivative=(k/np.expm1(rate)-fail if k and rate>0 else -fail)*slope
    return float(value),float(derivative)


def checked_corner_upper(theta,eta,h,remaining,lo,hi,cycles,k,r,cf,a0,a1,e0,e1):
    t=query_tangent(theta,h,remaining,lo,hi,cycles)
    if t is None:return None
    phi,g=t;cp,ge=calibration_tangent(eta,k,r,cf);ga,gv,gb=g
    if not gb<=0:return None
    C=phi+cp-sum((z*arb(float(x)) for z,x in zip(g,theta)),arb(0))-ge*arb(float(eta))
    values=[C+ga*arb(float(a))+gv*arb(float(a))*arb(float(e))+ge*arb(float(e)) for a in [a0,a1] for e in [e0,e1]]
    return max(outward_float(v) for v in values)


def corner_profile(h,k,r,cf,a0,a1,e0,e1,d0,d1,w0,w1,cycles=1000):
    if not(0<=a0<a1 and 0<=e0<e1<=.25):raise ValueError('Valid amplitude/eta rectangle required')
    h,remaining,_,sat=_observations(h,cycles,'first');lo,hi=query_basis(len(h),d0,d1,w0,w1)
    fallback=outward_float(arb(sat)+arb(calibration_upper(k,r,e0,e1,cf)['upper']));span=e1-e0
    def objective(z):
        a,b,u=z;eta=e0+span*u;theta=[a,a*eta,b]
        q,g=value_gradient(theta,h,remaining,lo,hi,cycles);c,gc=calibration_value_gradient(eta,k,r,cf)
        return -(q+c),-np.array([g[0]+eta*g[1],g[2],(a*g[1]+gc)*span])
    a=np.clip(max(float(h.sum())*.7,1.)/(5*1.0645),a0,a1);b=max(float(h.sum())*.3,1.)
    fit=minimize(objective,[a,b,.5],jac=True,method='L-BFGS-B',bounds=[(a0,a1),(1e-10,None),(0,1)],options={'maxiter':160,'ftol':1e-12,'gtol':1e-8})
    z=fit.x if np.all(np.isfinite(fit.x)) else np.array([a,b,.5]);eta=float(np.clip(e0+span*z[2],e0,e1));theta=[float(np.clip(z[0],a0,a1)),float(np.clip(z[0],a0,a1))*eta,max(float(z[1]),1e-10)]
    certificate=None
    for j in range(24):
        candidate=[theta[0],theta[1],theta[2]*(1+1e-6*2**j)]
        certificate=checked_corner_upper(candidate,eta,h,remaining,lo,hi,cycles,k,r,cf,a0,a1,e0,e1)
        if certificate is not None:break
    return dict(upper=fallback if certificate is None else min(fallback,certificate),tangent_certified=certificate is not None,optimizer_success=bool(fit.success),
        fitted_amplitude=theta[0],fitted_eta=eta,calibration_profile='concave affine-hazard interval',scope='Four-corner tangent upper on a bounded amplitude/eta rectangle; background unbounded nonnegative')


def gaussian_mass_lower(bins,min_width=4.5):
    # A closest integer bin is <=.5away. At leastfloor((T-1)/2)further bins
    # exist in one direction,with distances <=j+.5. This covers ANY delay.
    k=4*arb(2).log()
    return sum(((-k*((arb(j)+arb(1)/2)/arb(float(min_width)))**2).exp() for j in range((bins-1)//2+1)),arb(0))


def amplitude_tail_upper(h,cycles,amplitude,cal_k,cal_r):
    h=np.asarray(h);K=int(h.sum());n0=int(cycles)-K
    if n0<=0:return float('inf')
    G=gaussian_mass_lower(len(h));S=arb(float(amplitude))*G/cycles;mode=(arb(K)/n0).log1p() if K else arb(0)
    if not S>=mode:return float('inf')
    cond=sum((int(v)*(arb(int(v))/K).log() for v in h if v),arb(0)) if K else arb(0)
    value=cond-n0*S+saturated_binomial(cal_k,cal_r)
    if K:value+=K*(-(-S).expm1()).log()
    return outward_float(value)


def amplitude_cap(h,cycles,k,r,decision_level):
    if sum(h)>=cycles:return None
    A=max(1.,2.*(sum(h)+1)/float(gaussian_mass_lower(len(h))))
    for _ in range(64):
        upper=amplitude_tail_upper(h,cycles,A,k,r)
        if upper<=decision_level:return dict(cap=float(A),tail_upper=upper,target=float(decision_level))
        A*=2
    return None
