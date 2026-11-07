"""Complete continuous null cover for safe piecewise-constant event rates."""
from __future__ import annotations
import heapq,math,time
import numpy as np
from flint import arb,ctx
from deadtime_scan import fit_null
from certified_safe_components import periodic_envelope,periodic_point_rates,log_likelihood_interval,outward
from continuous_rate_profile import profile_upper


def null_witness(stats,theta,sigma):
    a,d,b=np.asarray(theta,dtype=float);rates=periodic_point_rates(len(stats['counts']),a,d,b,sigma)
    return outward(log_likelihood_interval(stats,rates),False)


def bound_null(stats,sigma_range=(1.5,1.5),decision_level=None,initial_cells=8,max_boxes=192,gap=.01,fit=None):
    """Early termination preserves a complete cover,with no nuisance caps.

    A feasible-null screen only declines to certify the supplied fixed
    numerator bound. A declaration requires an actual complete-cover upper
    bound<=decision_level. No local numerical maximum replaces that bound.
    """
    if ctx.prec<128:raise RuntimeError('Arb128required')
    k=len(stats['counts']);s0,s1=map(float,sigma_range)
    if not(0<s0<=s1) or initial_cells<1 or max_boxes<initial_cells:raise ValueError('Invalid cover settings')
    if not np.any(stats['counts']):return dict(upper=0.,feasible_lower=0.,boxes=0,complete_cover=True,reason='zero_count',declared=bool(decision_level is not None and 0<=decision_level))
    midpoint=np.array([float(v.mid()) for v in stats['exposure']]);sig=(s0+s1)/2
    fit=fit_null(stats['counts'],midpoint,sig) if fit is None else fit
    witness=null_witness(stats,fit['theta'],sig);start=time.perf_counter()
    if decision_level is not None and witness>decision_level:
        return dict(upper=None,feasible_lower=witness,boxes=0,complete_cover=False,reason='feasible_null_excludes_fixed_numerator_bound',declared=False,seconds=time.perf_counter()-start)
    counter=0;heap=[]
    def cell(d0,d1,w0,w1,parent=None,initial=None):
        nonlocal counter
        lo,hi=periodic_envelope(k,d0,d1,w0,w1)
        value=profile_upper(stats['counts'],stats['exposure_lower'],lo,hi,initial=initial)
        upper=value['upper'] if parent is None else min(parent,value['upper']);counter+=1
        return (-upper,counter,(d0,d1,w0,w1),value['parameters'] if 'parameters' in value else list(fit['theta'][[0,2]]))
    edges=np.linspace(0,k,initial_cells+1)
    for a,b in zip(edges[:-1],edges[1:]):heapq.heappush(heap,cell(float(a),float(b),s0,s1,initial=fit['theta'][[0,2]]))
    reason='budget'
    while True:
        upper=-heap[0][0];assert upper>=witness
        if decision_level is not None and upper<=decision_level:reason='certified_decision';break
        if upper-witness<=gap:reason='witness_gap';break
        if counter+2>max_boxes:break
        neg,serial,(d0,d1,w0,w1),initial=heapq.heappop(heap)
        if d1-d0>=4*(w1-w0) or w0==w1:
            mid=(d0+d1)/2;rects=[(d0,mid,w0,w1),(mid,d1,w0,w1)]
        else:
            mid=(w0+w1)/2;rects=[(d0,d1,w0,mid),(d0,d1,mid,w1)]
        for rect in rects:heapq.heappush(heap,cell(*rect,parent=-neg,initial=initial))
    upper=-heap[0][0]
    return dict(upper=upper,feasible_lower=witness,boxes=counter,active_cells=len(heap),complete_cover=True,reason=reason,declared=bool(decision_level is not None and upper<=decision_level),seconds=time.perf_counter()-start,null_fit_theta=np.asarray(fit['theta']).tolist(),scope='Continuous delay/declared sigma range and ALL nonnegative amplitude/background in an idealized bin-constant intensity model. Exact recorded dyadics,known safe lag;not physical quantization/jitter/recovery robustness.')


def declare_from_fixed_mixture(stats,log_numerator_lower,**settings):
    level=outward(arb(float(log_numerator_lower))-arb(100).log(),False)
    result=bound_null(stats,decision_level=level,**settings)
    result['log_numerator_lower']=float(log_numerator_lower);result['decision_level_lower']=level
    if result['upper'] is not None:
        result['log_evidence_lower']=outward(arb(float(log_numerator_lower))-arb(result['upper']),False)
    return result
