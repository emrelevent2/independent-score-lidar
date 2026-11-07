"""Complete four-coordinate cover with coupled optical-response tangents."""
from dataclasses import dataclass
import heapq
import numpy as np
from flint import arb
from coupled_corner_profile import corner_profile,amplitude_cap
from joint_plateau_envelope import calibration_flat
from joint_cone_envelope import query_cone_basis
from conditional_calibration import saturated_binomial
from linear_envelope_profile import profile_upper
from envelope_evidence import _observations,outward_float


@dataclass
class Cell:
    a0:float;a1:float;e0:float;e1:float;d0:float;d1:float;w0:float;w1:float
    upper:float;flat:bool;amplitude:float


def coupled_cover(query,k,r,confidence,cycles=1000,initial_amplitude=2,initial_eta=4,initial_delay=16,max_boxes=1024,decision_level=None):
    query=np.asarray(query);_,_,_,sat=_observations(query,cycles,'first')
    if not(1<=initial_amplitude and 1<=initial_eta and 1<=initial_delay and max_boxes>=initial_amplitude*initial_eta*initial_delay):raise ValueError('Invalid complete-cover budget')
    if decision_level is None:decision_level=outward_float(arb(sat)+saturated_binomial(k,r)-10,False)
    tail=amplitude_cap(query,cycles,k,r,decision_level)
    if tail is None:return dict(upper=float('inf'),boxes_evaluated=0,fallback_reason='No finite amplitude-tail truncation found; no acceptance authorized')
    A=tail['cap'];queue=[];counter=fallbacks=optimizer_failures=flat_profiles=corner_profiles=0;splits=[0,0,0,0]
    def add(a0,a1,e0,e1,d0,d1,w0,w1,parent=None):
        nonlocal counter,fallbacks,optimizer_failures,flat_profiles,corner_profiles
        flat=calibration_flat(k,r,e0,e1,confidence)
        if flat:
            lo,hi=query_cone_basis(len(query),d0,d1,w0,w1,e0,e1)
            a=max(float(query.sum())*.35,1.)/(5*1.0645);b=max(float(query.sum())*.3,1.)
            fit=profile_upper(query,lo,hi,cycles,[a,a,b]);flat_profiles+=1
            upper=outward_float(arb(fit['upper'])+saturated_binomial(k,r));amplitude=a0+(a1-a0)/2
            certified=fit['tangent_certified']
        else:
            fit=corner_profile(query,k,r,confidence,a0,a1,e0,e1,d0,d1,w0,w1,cycles);corner_profiles+=1
            upper=fit['upper'];amplitude=fit['fitted_amplitude'];certified=fit['tangent_certified']
        fallbacks+=not certified;optimizer_failures+=not fit['optimizer_success']
        if parent is not None:upper=min(upper,parent)
        counter+=1;cell=Cell(a0,a1,e0,e1,d0,d1,w0,w1,upper,flat,amplitude);heapq.heappush(queue,(-upper,counter,cell))
    amps=np.linspace(0.,A,initial_amplitude+1);etas=np.linspace(0.,.25,initial_eta+1);delays=np.linspace(0.,len(query)-1.,initial_delay+1)
    for a0,a1 in zip(amps[:-1],amps[1:]):
        for e0,e1 in zip(etas[:-1],etas[1:]):
            for d0,d1 in zip(delays[:-1],delays[1:]):add(float(a0),float(a1),float(e0),float(e1),float(d0),float(d1),4.5,5.5)
    while queue and counter+2<=max_boxes:
        if max(tail['tail_upper'],-queue[0][0])<=decision_level:break
        _,_,c=heapq.heappop(queue);mean=(c.w0+c.w1)/2
        spans=[(c.a1-c.a0)/max(c.amplitude,1.),(c.e1-c.e0)/(.25/8),(c.d1-c.d0)/mean,3*(c.w1-c.w0)/mean]
        if c.flat:spans[0]=spans[1]=0.
        axis=int(np.argmax(spans));splits[axis]+=1
        low=[c.a0,c.e0,c.d0,c.w0];high=[c.a1,c.e1,c.d1,c.w1];mid=(low[axis]+high[axis])/2
        if mid in (low[axis],high[axis]):heapq.heappush(queue,(-c.upper,counter,c));break
        high1=list(high);high1[axis]=mid;low2=list(low);low2[axis]=mid
        def child(l,u):add(l[0],u[0],l[1],u[1],l[2],u[2],l[3],u[3],c.upper)
        child(low,high1);child(low2,high)
    if not queue:raise ArithmeticError('Complete coupled cover became empty')
    return dict(upper=max(tail['tail_upper'],-queue[0][0]),boxes_evaluated=counter,corner_profiles=corner_profiles,flat_cone_profiles=flat_profiles,
        conservative_fallbacks=fallbacks,optimizer_failures=optimizer_failures,amplitude_cap=tail,splits=dict(amplitude=splits[0],eta=splits[1],delay=splits[2],width=splits[3]),
        domain=dict(peak_amplitude=[0.,A],eta=[0.,.25],delay=[0.,float(len(query)-1)],width=[4.5,5.5],background='unbounded nonnegative',amplitude_tail='analytically included to infinity'),
        scope='Complete amplitude/eta/delay/width cover; coupled four-corner tangents or flat-calibration cone profiles; unbounded amplitude tail retained')
