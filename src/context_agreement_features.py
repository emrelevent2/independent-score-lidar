"""Context features and causal early-pilot location;no query future in pilot."""
import numpy as np
from deadtime_scan import pulse,scan,multistart_glrt
from deadtime_path_bootstrap import fine_scan,one_step
from certified_safe_components import exact_safe_statistics


def numeric_statistics(observed,cycles,bins,lag):
    exact=exact_safe_statistics(observed,cycles,bins,lag)
    return exact['counts'],np.array([float(v.mid()) for v in exact['exposure']])


def donor_features(counts,exposures,sigma):
    s=np.asarray(counts).sum(axis=0);e=np.asarray(exposures).sum(axis=0);k=len(s)
    first=scan(s,e,sigma);two=multistart_glrt(s,e,sigma,first);a,d,b,c,t=np.asarray(two['theta'],dtype=float)
    if c>a:a,c,d,t=c,a,t,d
    d%=k;t%=k;initial=np.array([max(float(a),1e-6),float(d),max(float(b),1e-6)])
    offsets=[z for z in range(k) if min(z,k-z)>=2.5*sigma]
    locations=np.array([[d,t],*[[d,(d+z)%k] for z in offsets]])
    weights=np.r_[.75,np.full(len(offsets),.25/len(offsets))]
    templates=np.array([pulse(loc[1],k,sigma)[0] for loc in locations])
    return dict(initial_theta=initial,primary=float(d),secondary=float(t),two_pulse_loglr=float(two['score']),prior_bank=(locations,weights,templates))


def pilot_location(observed,pilot_cycles,bins,lag,sigma,context):
    # Only early timestamps enter this function's fit. The start comes from
    # independent context,not the full-query fit;the single damped step is
    # merely a pilot heuristic,not a claimed standalone calibrated test.
    end=pilot_cycles*bins;cut=int(np.searchsorted(observed,end,side='left'));s,e=numeric_statistics(np.asarray(observed)[:cut],pilot_cycles,bins,lag)
    fit=one_step(s,e,context['initial_theta'],sigma,'duration')
    result=fine_scan(s,e,sigma,fit=fit)
    return result['delay'],int(s.sum())
