"""Context-only proper alternative;query data never enter its parameters."""
from fractions import Fraction
import hashlib
import numpy as np
from scipy.special import logsumexp
from deadtime_scan import pulse,scan,multistart_glrt


def build_alternative(donor_counts,donor_exposures,sigma):
    counts=np.asarray(donor_counts).sum(axis=0);exposure=np.asarray(donor_exposures).sum(axis=0);k=len(counts)
    first=scan(counts,exposure,sigma);two=multistart_glrt(counts,exposure,sigma,first);a,d,b,c,t=np.asarray(two['theta'],dtype=float)
    if c>a:a,c,d,t=c,a,t,d
    a=max(a,1e-6);b=max(b,1e-6);c=max(c,.015*a);d%=k;t%=k
    rates=[];locations=[];weights=[];kinds=[]
    f,_=pulse(d,k,sigma)
    for factor in [.5,1.,2.,4.]:
        rates.append(np.maximum(a*f+factor*c*pulse(t,k,sigma)[0]+b/k,1e-12))
        locations.append([float(d if a>=factor*c else t),float(t if a>=factor*c else d)]);weights.append(Fraction(3,16));kinds.append('spatial')
    offsets=[v for v in range(k) if min(v,k-v)>=2.5*sigma]
    for offset in offsets:
        delay=(d+offset)%k
        for fraction in [.04,.08,.16]:
            rates.append(np.maximum(a*f+fraction*a*pulse(delay,k,sigma)[0]+b/k,1e-12));locations.append([float(d),float(delay)]);weights.append(Fraction(1,4*len(offsets)*3));kinds.append('coverage')
    assert sum(weights)==1
    rates=np.asarray(rates);locations=np.asarray(locations);digest=hashlib.sha256(rates.tobytes()+locations.tobytes()+str(weights).encode()).hexdigest()
    return dict(rates=rates,locations=locations,weights=weights,log_weights=np.log([float(v) for v in weights]),kinds=kinds,sha256=digest,donor_fit=[float(a),float(d),float(b),float(c),float(t)])


def mixture_scores(counts,exposures,alternative):
    s=np.atleast_2d(counts);e=np.atleast_2d(exposures);rates=alternative['rates']
    ll=s@np.log(rates).T-e@rates.T+alternative['log_weights']
    best=np.argmax(ll,axis=1)
    return logsumexp(ll,axis=1),alternative['locations'][best],best
