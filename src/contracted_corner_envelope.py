"""Complete threshold-adapted cover with query-likelihood domain reduction."""
import heapq
import numpy as np
from scipy.optimize import minimize
from flint import arb
from coupled_corner_envelope import Cell
from coupled_corner_profile import query_basis,query_tangent,calibration_tangent,calibration_value_gradient,amplitude_cap
from query_likelihood_contractor import zero_background_cap,signal_window,geometry_contract,contract_amplitude
from joint_plateau_envelope import calibration_flat
from conditional_calibration import calibration_upper,saturated_binomial
from linear_envelope_profile import value_gradient
from envelope_evidence import _observations,outward_float


def bounded_profile(h,remaining,k,r,cf,cycles,lo,hi,a0,a1,e0,e1,B):
    span=e1-e0
    def objective(z):
        a,b,u=z;eta=e0+span*u;q,g=value_gradient([a,a*eta,b],h,remaining,lo,hi,cycles);c,gc=calibration_value_gradient(eta,k,r,cf)
        return -(q+c),-np.array([g[0]+eta*g[1],g[2],(a*g[1]+gc)*span])
    a=float(np.clip(max(float(h.sum())*.7,1.)/(5*1.0645),a0,a1));b=float(np.clip(max(float(h.sum())*.3,1.),1e-10,B))
    fit=minimize(objective,[a,b,.5],jac=True,method='L-BFGS-B',bounds=[(a0,a1),(1e-10,B),(0,1)],options={'maxiter':160,'ftol':1e-12,'gtol':1e-8})
    z=fit.x if np.all(np.isfinite(fit.x)) else [a,b,.5];a=float(np.clip(z[0],a0,a1));b=float(np.clip(z[1],1e-10,B));eta=float(np.clip(e0+span*z[2],e0,e1));theta=[a,a*eta,b]
    t=query_tangent(theta,h,remaining,lo,hi,cycles)
    if t is None:return dict(upper=float('inf'),certified=False,amplitude=a,optimizer_success=bool(fit.success))
    phi,g=t;cp,ge=calibration_tangent(eta,k,r,cf);ga,gv,gb=g
    C=phi+cp-sum((v*arb(float(x)) for v,x in zip(g,theta)),arb(0))-ge*arb(eta)
    bterm=max(outward_float(arb(0)),outward_float(gb*arb(B)))
    upper=max(outward_float(C+ga*arb(float(aa))+gv*arb(float(aa))*arb(float(ee))+ge*arb(float(ee))+arb(bterm)) for aa in [a0,a1] for ee in [e0,e1])
    return dict(upper=upper,certified=True,amplitude=a,optimizer_success=bool(fit.success))


def contracted_cover(query,k,r,confidence,cycles=1000,initial_eta=8,initial_delay=16,max_boxes=1024,decision_level=None):
    h,remaining,_,sat=_observations(query,cycles,'first')
    if not(initial_eta>=1 and initial_delay>=1 and max_boxes>=initial_eta*initial_delay):raise ValueError('Invalid cover budget')
    cs=outward_float(saturated_binomial(k,r))
    if decision_level is None:decision_level=outward_float(arb(sat)+arb(cs)-10,False)
    bg=zero_background_cap(h,remaining,cycles,sat,cs,decision_level)
    if bg is None:return dict(upper=float('inf'),boxes_evaluated=0,fallback_reason='No zero-bin background contraction; no acceptance')
    if bg['entire_domain_below_level']:return dict(upper=bg['upper'],boxes_evaluated=0,entire_domain_below_level=True)
    B=bg['background_cap'];tail=amplitude_cap(h,cycles,k,r,decision_level)
    if tail is None:return dict(upper=float('inf'),boxes_evaluated=0,fallback_reason='No amplitude-tail truncation; no acceptance')
    A=tail['cap'];window=signal_window(h);cache={};queue=[];counter=pruned=profiles=fallbacks=optimizer_failures=0;splits=[0,0,0,0]
    def geometry(d0,d1,w0,w1):
        key=(d0,d1,w0,w1)
        if key not in cache:cache[key]=geometry_contract(h,remaining,cycles,k,r,decision_level,A,B,window,*key)
        return cache[key]
    def add(a0,a1,e0,e1,d0,d1,w0,w1,parent=None):
        nonlocal counter,pruned,profiles,fallbacks,optimizer_failures
        info=geometry(d0,d1,w0,w1);restricted=contract_amplitude(info,a0,a1,e0,e1,B,window[1]-window[0],len(h),cycles)
        flat=calibration_flat(k,r,e0,e1,confidence)
        if restricted is None:upper=decision_level;amplitude=(a0+a1)/2;pruned+=1
        else:
            a0,a1=restricted;lo,hi=query_basis(len(h),d0,d1,w0,w1)
            fit=bounded_profile(h,remaining,k,r,confidence,cycles,lo,hi,a0,a1,e0,e1,B);profiles+=1
            fallback=outward_float(arb(sat)+arb(calibration_upper(k,r,e0,e1,confidence)['upper']))
            upper=max(decision_level,min(fallback,fit['upper']));amplitude=fit['amplitude'];fallbacks+=not fit['certified'];optimizer_failures+=not fit['optimizer_success']
        if parent is not None:upper=min(upper,parent)
        counter+=1;cell=Cell(a0,a1,e0,e1,d0,d1,w0,w1,upper,flat,amplitude);heapq.heappush(queue,(-upper,counter,cell))
    etas=np.linspace(0.,.25,initial_eta+1);delays=np.linspace(0.,len(h)-1.,initial_delay+1)
    for e0,e1 in zip(etas[:-1],etas[1:]):
        for d0,d1 in zip(delays[:-1],delays[1:]):add(0.,A,float(e0),float(e1),float(d0),float(d1),4.5,5.5)
    while queue and counter+2<=max_boxes:
        if max(decision_level,tail['tail_upper'],-queue[0][0])<=decision_level:break
        _,_,c=heapq.heappop(queue);mean=(c.w0+c.w1)/2
        spans=[(c.a1-c.a0)/max(c.amplitude,1.),(c.e1-c.e0)/(.25/16),(c.d1-c.d0)/mean,3*(c.w1-c.w0)/mean]
        if c.flat:spans[0]=spans[1]=0.
        axis=int(np.argmax(spans));splits[axis]+=1;low=[c.a0,c.e0,c.d0,c.w0];high=[c.a1,c.e1,c.d1,c.w1];mid=(low[axis]+high[axis])/2
        if mid in (low[axis],high[axis]):heapq.heappush(queue,(-c.upper,counter,c));break
        first=list(high);first[axis]=mid;second=list(low);second[axis]=mid
        def child(l,u):add(l[0],u[0],l[1],u[1],l[2],u[2],l[3],u[3],c.upper)
        child(low,first);child(second,high)
    if not queue:raise ArithmeticError('Complete superlevel cover became empty')
    return dict(upper=max(decision_level,tail['tail_upper'],-queue[0][0]),boxes_evaluated=counter,coupled_profiles=profiles,geometry_contractors=len(cache),contractor_profiles=len(cache),pruned_boxes=pruned,
        conservative_fallbacks=fallbacks,optimizer_failures=optimizer_failures,amplitude_cap=tail,background_cap=bg,signal_window=list(window),excluded_domain_upper_floor=decision_level,splits=dict(amplitude=splits[0],eta=splits[1],delay=splits[2],width=splits[3]),
        scope='Full physical null upper: maximum of contracted superlevel cover,analytical amplitude tail,and decision-level floor for all discarded amplitude/background domains')
