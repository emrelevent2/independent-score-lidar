"""Decision-directed certification; never refit the alternative on the query.

A feasible local null provides a lower likelihood bound. If even the upper
evidence is below threshold, reject without global subdivision. Otherwise
refine a complete null cover until acceptance/rejection is certified or the
budget is exhausted. Child upper bounds inherit valid parent caps.
"""
import heapq
import numpy as np
from flint import arb,ctx
from scipy.ndimage import convolve1d
from scipy.optimize import minimize,nnls
from scipy.special import xlogy
from likelihood_baseline import pulse_derivatives
from envelope_evidence import (_observations,exact_gaussian,certified_log_likelihood,
    bound_box,outward_float)


def mixture_interval(h,alternatives,cycles=1000):
    counts,failures,_,_=_observations(h,cycles,'first');terms=[]
    for means in alternatives:
        means=np.asarray(means,float)
        if means.shape!=counts.shape or np.any(means<=0) or not np.all(np.isfinite(means)):
            raise ValueError('Proper full-length positive alternative required')
        term=arb(0)
        for count,fail,mean in zip(counts,failures,means):
            rate=arb(float(mean))
            if count:term+=int(count)*(-(-rate).expm1()).log()
            term-=int(fail)*rate
        terms.append(term)
    if not terms:raise ValueError('Empty alternative mixture')
    shift=max(float(x) for x in terms)
    value=arb(shift)+(sum(((x-arb(shift)).exp() for x in terms),arb(0))/len(terms)).log()
    return dict(lower=outward_float(value,False),upper=outward_float(value,True),
        component_lower=[outward_float(x,False) for x in terms])


def feasible_local_null(h,cycles=1000,fwhm_range=(4.5,5.5)):
    h=np.asarray(h);t=np.arange(len(h));failures=cycles-np.cumsum(h)
    kernel=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2);kernel/=kernel.sum()
    matched=convolve1d(h.astype(float),kernel,mode='constant')
    peaks=(matched>=np.roll(matched,1))&(matched>np.roll(matched,-1))
    starts=np.argsort(np.where(peaks,matched,-np.inf))[-3:][::-1]
    def loss(q):
        a,b,d,w=q;g,dg,wg=pulse_derivatives(t,d,w);rate=(a*g+b/len(t))/cycles
        ell=np.sum(xlogy(h,-np.expm1(-rate))-failures*rate)
        with np.errstate(over='ignore'):derivative=h/np.expm1(rate)-failures
        gradient=np.array([np.sum(derivative*x/cycles) for x in [g,np.ones(len(t))/len(t),a*dg,a*wg]])
        return -float(ell),-gradient
    fits=[]
    for d in starts:
        coeff,_=nnls(np.stack([pulse_derivatives(t,d,5.)[0],np.ones(len(t))/len(t)],axis=-1),h.astype(float))
        fit=minimize(loss,[coeff[0],max(coeff[1],1e-8),d,np.mean(fwhm_range)],jac=True,method='L-BFGS-B',
            bounds=[(0,None),(1e-10,None),(max(0,d-2.),min(len(t)-1,d+2.)),fwhm_range],
            options={'ftol':1e-12,'gtol':1e-8,'maxiter':200,'maxls':40})
        fits.append(fit)
    best=min(fits,key=lambda f:f.fun);a,b,d,w=map(float,best.x)
    if not all(np.isfinite([a,b,d,w])) or a<0 or b<0 or not 0<=d<=len(h)-1 or not fwhm_range[0]<=w<=fwhm_range[1]:
        raise RuntimeError('No feasible finite local-null witness')
    pulse=exact_gaussian(len(h),d,w,normalized=True)
    rates=[(arb(a)*g+arb(b)/len(h))/cycles for g in pulse]
    lower=certified_log_likelihood(h,rates,cycles,'first')
    return dict(lower=lower,parameters=[a,b,d,w],optimizer_failures=sum(not f.success for f in fits))


def adaptive_evidence(h,alternatives,cycles=1000,fwhm_range=(4.5,5.5),alpha=.01,
                      initial_cells=32,max_boxes=2048):
    counts,failures,scale,saturated=_observations(h,cycles,'first')
    if not 0<alpha<1 or initial_cells<1 or max_boxes<initial_cells:raise ValueError('Invalid alpha/budget')
    if not 0<fwhm_range[0]<=fwhm_range[1]:raise ValueError('Invalid width interval')
    numerator=mixture_interval(counts,alternatives,cycles)
    witness=feasible_local_null(counts,cycles,fwhm_range);best_lower=witness['lower']
    threshold=(arb(1)/arb(str(alpha))).log()
    threshold_low=outward_float(threshold,False);threshold_high=outward_float(threshold,True)
    queue=[];counter=0;fallbacks=0;optimizer_failures=0
    result=dict(alpha=alpha,numerator_lower=numerator['lower'],numerator_upper=numerator['upper'],
        component_lower=numerator['component_lower'],log_threshold_lower=threshold_low,
        log_threshold_upper=threshold_high,local_null_witness=witness,arb_precision_bits=ctx.prec,
        # This is an arbitrary score for OPTIONAL exchangeability calibration.
        # It is not the analytic e-value and must never use the Markov guarantee.
        fast_calibration_score=(numerator['lower']+numerator['upper'])/2-best_lower)
    def finish(decision,upper=None):
        log_hi=outward_float(arb(numerator['upper'])-arb(best_lower),True)
        log_lo=outward_float(arb(numerator['lower'])-arb(upper),False) if upper is not None else None
        return dict(**result,decision=decision,declared=decision=='accepted',log_e_lower=log_lo,log_e_upper=log_hi,
            null_upper=upper,null_feasible_lower=best_lower,boxes_evaluated=counter,
            conservative_fallbacks=fallbacks,box_optimizer_failures=optimizer_failures,
            gap=upper-best_lower if upper is not None else None)
    if outward_float(arb(numerator['upper'])-arb(best_lower),True)<threshold_low:
        return finish('rejected_by_upper_evidence')
    def add(d0,d1,w0,w1,parent_cap=None):
        nonlocal counter,best_lower,fallbacks,optimizer_failures
        box=bound_box(counts,failures,scale,'first',d0,d1,w0,w1,saturated,normalized=False)
        if parent_cap is not None:box.upper=min(box.upper,parent_cap)
        if box.lower>box.upper:raise RuntimeError('Certified cell upper below a feasible likelihood')
        counter+=1;best_lower=max(best_lower,box.lower);fallbacks+=not box.certified;optimizer_failures+=not box.optimizer_success
        heapq.heappush(queue,(-box.upper,counter,box))
    edges=np.linspace(0.,len(h)-1.,initial_cells+1)
    for d0,d1 in zip(edges[:-1],edges[1:]):add(float(d0),float(d1),*map(float,fwhm_range))
    while queue:
        upper=-queue[0][0]
        if upper<best_lower:raise RuntimeError('Complete-cover upper below a global feasible witness')
        if outward_float(arb(numerator['lower'])-arb(upper),False)>=threshold_high:
            return finish('accepted',upper)
        if outward_float(arb(numerator['upper'])-arb(best_lower),True)<threshold_low:
            return finish('rejected_by_upper_evidence',upper)
        if counter+2>max_boxes:return finish('budget_undecided',upper)
        _,_,box=heapq.heappop(queue)
        if box.upper<best_lower:continue
        average=(box.f0+box.f1)/2
        if (box.d1-box.d0)/average>=3*(box.f1-box.f0)/average:
            mid=(box.d0+box.d1)/2
            if mid in (box.d0,box.d1):return finish('precision_undecided',upper)
            add(box.d0,mid,box.f0,box.f1,box.upper);add(mid,box.d1,box.f0,box.f1,box.upper)
        else:
            mid=(box.f0+box.f1)/2
            if mid in (box.f0,box.f1):return finish('precision_undecided',upper)
            add(box.d0,box.d1,box.f0,mid,box.upper);add(box.d0,box.d1,mid,box.f1,box.upper)
    raise RuntimeError('Unexpected empty certified cover')
