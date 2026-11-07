"""Causally scored lobe AND subsequent dark window,with shared background.

K predictor uses only bins0..42. D predictor uses bins0..127,including K and
intervening data,but excludes D128..255. Sequential normalization remains
proper with random risk sets. Nine positive posterior-quantile points are
an explicit proper finite predictive mixture,not exact Bayesian integration.
"""
from fractions import Fraction
import numpy as np
from scipy.stats import beta as beta_distribution
from flint import arb
from causal_calibration import prefix_features
from physical_calibration_predictive import fixed_hazard_log
from conditional_calibration import saturated_binomial
from joint_calibration_model import calibration_basis
from linear_envelope_profile import value_gradient
from coupled_corner_profile import query_tangent
from scipy.optimize import minimize
from envelope_evidence import outward_float


def background_grid(past,cycles,start,stop):
    h=np.asarray(past)
    if h.shape!=(stop,) or np.any(h<0) or np.any(h!=np.floor(h)) or h.sum()>cycles or not 0<=start<stop:raise ValueError('Only the complete past up to training-window end is admissible')
    k=int(h[start:stop].sum());remaining=cycles-np.cumsum(h);f=int(remaining[start:stop].sum())
    p=beta_distribution.ppf((np.arange(9)+.5)/9,k+.5,f+.5)
    if not np.all(np.isfinite(p)) or np.any(p<=0) or np.any(p>=1):raise ArithmeticError('Invalid predictive beta-grid proposal')
    return dict(hazards=(-np.log1p(-p)).tolist(),weights=['1/9']*9,training_count=k,training_failures=f,training_bins=[start,stop-1],scope='Explicit proper finite mixture,not exact posterior quadrature')


def mix(components):
    assert sum((Fraction(c['weight']) for c in components),Fraction(0))==1
    def endpoint(key,up):
        terms=[arb(c['log_probability'][key])+(arb(Fraction(c['weight']).numerator)/Fraction(c['weight']).denominator).log() for c in components]
        shift=max(float(x) for x in terms);v=arb(shift)+sum(((x-shift).exp() for x in terms),arb(0)).log();return outward_float(v,up)
    return dict(lower=endpoint('lower',False),upper=endpoint('upper',True),components=components,binomial_constant_omitted=True)


def score_lobe(prefix,cycles,donor_eta,k):
    f=prefix_features(prefix,cycles,donor_eta);R=f['scored_lobe_risk'];bg=background_grid(np.asarray(prefix)[:20],cycles,0,20)
    etas=[(0.,Fraction(1,2)),(donor_eta,Fraction(1,4))]+[((2*j+1)/64.,Fraction(1,32)) for j in range(8)]
    components=[]
    for eta,w in etas:
        for beta in bg['hazards']:
            H=arb(f['main_signal'])*(arb(float(eta))+arb(f['forward_leakage']))+11*arb(beta)
            components.append(dict(eta=float(eta),background_hazard=beta,weight=str(w/9),log_probability=fixed_hazard_log(k,R,H)))
    return dict(**mix(components),prefix_features=f,background_predictor=bg,count=int(k),risk=R,training_bins=[0,42],scored_bins=[43,53])


def score_dark(past,cycles,d):
    h=np.asarray(past)
    if h.shape!=(128,):raise ValueError('Only the complete earlier128bins may train dark prediction')
    bg=background_grid(h,cycles,64,128);R=int(cycles-h.sum())
    components=[dict(background_hazard=b,weight='1/9',log_probability=fixed_hazard_log(d,R,128*arb(b))) for b in bg['hazards']]
    return dict(**mix(components),background_predictor=bg,count=int(d),risk=R,training_bins=[0,127],scored_bins=[128,255])


def joint_numerator(lobe,dark):
    return dict(lower=outward_float(arb(lobe['lower'])+arb(dark['lower']),False),upper=outward_float(arb(lobe['upper'])+arb(dark['upper'])),scope='Sequential QK(K|past)*QD(D|laterpast,K),matching conditional risk sets;both binomial constants omitted')


def two_window_upper(k,r,d,rd,eta0,eta1,cf):
    counts=np.array([k,d],int);remaining=np.array([r-k,rd-d],int)
    if np.any(counts<0) or np.any(remaining<0) or not(0<=eta0<=eta1<=.25):raise ValueError('Invalid two-window counts/eta')
    saturated=outward_float(saturated_binomial(k,r)+saturated_binomial(d,rd))
    if not cf['informative']:return dict(upper=saturated,profile_certified=False,bound_kind='two-saturated-binomials')
    lower,upper=calibration_basis(eta0,eta1)
    lo=np.array([[max(0.,outward_float(arb(eta0)+arb(cf['forward_leakage_lower']),False)),0.,11.],
                 [lower[5,0],0.,128.]])
    hi=np.array([[outward_float(arb(eta1)+arb(cf['forward_leakage_upper'])),0.,11.],
                 [upper[5,0],0.,128.]])
    sl,su=cf['signal_main_lower'],cf['signal_main_upper'];bl,bu=cf['background_lower'],cf['background_upper']
    ss=max((sl+su)/2,1e-12);bs=max((bl+bu)/2,1e-12)
    def objective(z):
        theta=[max(0.,z[0]*ss),0.,max(0.,z[1]*bs)];v,g=value_gradient(theta,counts,remaining,lo,hi,1)
        return -v,-np.array([g[0]*ss,g[2]*bs])
    fit=minimize(objective,[1.,1.],jac=True,method='L-BFGS-B',bounds=[(sl/ss,su/ss),(bl/bs,bu/bs)],options={'maxiter':160,'ftol':1e-12,'gtol':1e-8})
    z=fit.x if np.all(np.isfinite(fit.x)) else [1.,1.];theta=[max(0.,float(z[0]*ss)),0.,max(0.,float(z[1]*bs))]
    t=query_tangent(theta,counts,remaining,lo,hi,1)
    if t is None:return dict(upper=saturated,profile_certified=False,bound_kind='two-saturated-binomials',optimizer_success=bool(fit.success))
    value,g=t;C=value-g[0]*arb(theta[0])-g[2]*arb(theta[2])
    result=max(outward_float(C+g[0]*arb(s)+g[2]*arb(b)) for s in [sl,su] for b in [bl,bu])
    return dict(upper=min(saturated,result),profile_certified=True,bound_kind='shared-signal-background-two-binomial-box-profile',optimizer_success=bool(fit.success))
