"""Conservative standard Chernoff inversion for heterogeneous query outcomes.

Independent Bernoulli outcomes need not have equal success probabilities.
The target is their average expected success rate under the frozen design.
These are assessment bounds, not guarantees for a plug-in detector's null
calibration, field prevalence, model fidelity or future adaptive sampling.
No novelty is claimed for Chernoff/union-bound/paired discordance arguments.
"""
import math
import numbers
import numpy as np
from scipy.optimize import brentq
from scipy.special import rel_entr


def _counts(successes, count):
    if (isinstance(successes, bool) or isinstance(count, bool)
            or not isinstance(successes, numbers.Integral)
            or not isinstance(count, numbers.Integral)
            or count <= 0 or not 0 <= successes <= count):
        raise ValueError('Integer count>0 and 0<=successes<=count required')
    return int(successes), int(count)


def bernoulli_kl(q, p):
    if not (np.isfinite(q) and np.isfinite(p) and 0 <= q <= 1 and 0 <= p <= 1):
        raise ValueError('Probabilities in[0,1] required')
    return float(rel_entr(q,p) + rel_entr(1-q,1-p))


def rate_bounds(successes, count, tail_error):
    """Each one-sided noncoverage is <=tail_error under independent queries.

    Invert n*KL(k/n||p)=log(1/tail_error). The two-sided interval consequently
    has noncoverage <=2*tail_error. Boundary counts receive nonzero width.
    Rounded roots are expanded outward, rather than declared exact numerical
    certificates. See the separate proof/assumptions note.
    """
    k,n=_counts(successes,count)
    if not np.isfinite(tail_error) or not 0 < tail_error < 1:
        raise ValueError('Finite tail error in(0,1) required')
    log_error=-math.log(float(tail_error))
    if k == 0:
        lo,hi=0.,-math.expm1(-log_error/n)
    elif k == n:
        lo,hi=math.exp(-log_error/n),1.
    else:
        q=k/n
        f=lambda p:n*bernoulli_kl(q,p)-log_error
        lo=brentq(f,0.,q,xtol=np.nextafter(0.,1.),rtol=8*np.finfo(float).eps,maxiter=300)
        hi=brentq(f,q,1.,xtol=np.nextafter(0.,1.),rtol=8*np.finfo(float).eps,maxiter=300)
    # Conservative floating-point padding, not an interval-arithmetic proof.
    return [max(0.,float(lo)-1e-12),min(1.,float(hi)+1e-12)]


def paired_bounds(gains, losses, count, failure_probability=.05, contrasts=1):
    """Simultaneous two-sided differences for a PREDECLARED contrast family.

    A gain and a loss are mutually exclusive Bernoulli outcomes in each
    independent query. Bound both rates and subtract their intervals. Four
    tails per contrast receive an equal union-bound allocation. Dependence
    between methods/within a query is allowed. Different endpoint cohorts
    use their actual denominators, never a pooled artificial prevalence.
    """
    gains,count=_counts(gains,count)
    losses,_=_counts(losses,count)
    if gains+losses > count:
        raise ValueError('Discordant gains and losses must be disjoint')
    if isinstance(contrasts,bool) or not isinstance(contrasts,numbers.Integral) or contrasts < 1:
        raise ValueError('Positive integer predeclared contrast count required')
    if not np.isfinite(failure_probability) or not 0 < failure_probability < 1:
        raise ValueError('Failure probability in(0,1) required')
    delta=float(failure_probability)/(4*int(contrasts))
    g=rate_bounds(gains,count,delta);l=rate_bounds(losses,count,delta)
    return dict(gains=gains,losses=losses,count=count,estimate=(gains-losses)/count,
                lower=g[0]-l[1],upper=g[1]-l[0],gain_interval=g,loss_interval=l,
                tail_error=delta,family_failure_probability=float(failure_probability),
                predeclared_contrasts=int(contrasts),
                interpretation='Conservative bound for fixed-design average paired difference; '
                               'independent queries required. No i.i.d./normality/rare-zero bootstrap assumption.')


def primary_recall_bounds(rows, primary, controls, alpha=.01, tolerance=2.):
    """Fixed-alpha paired correct-localization gains on ALL positive queries."""
    controls=list(controls)
    if not controls or len(set(controls)) != len(controls) or primary in controls:
        raise ValueError('Unique predeclared distinct controls required')
    positives=[r for r in rows if r['case']['positive']]
    if len({r['case']['id'] for r in rows}) != len(rows) or not positives:
        raise ValueError('Independent unique query IDs and positive queries required')
    def correct(row,method):
        z=row['samples'][method]
        dist=lambda x,y:abs((x-y+32)%64-32)
        return (z['pvalue'] <= alpha and dist(z['primary'],row['truth']['primary']) <= tolerance
                and dist(z['delay'],row['truth']['secondary']) <= tolerance)
    a=np.array([correct(r,primary) for r in positives],dtype=bool)
    result={}
    for method in controls:
        b=np.array([correct(r,method) for r in positives],dtype=bool)
        result[method]=paired_bounds(int(np.count_nonzero(a&~b)),int(np.count_nonzero(~a&b)),
                                    len(a),failure_probability=.05,contrasts=len(controls))
    return result
