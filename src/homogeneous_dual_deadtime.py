"""Homogeneous renewal-overshoot kernel for two nonparalyzable components.

Known renewal reasoning,not a priority claim. A recorded avalanche at0 resets
the detector for Td and electronics for Te. Hidden avalanches restart only
the detector. The next record is the first shifted-exponential renewal past
Te. This positive-term formula includes any possible number of hidden hits.
It is a homogeneous model-validation primitive,NOT a periodic lidar solver,
not interval-certified arithmetic,and not evidence of detection improvement.
"""
import math
import numpy as np
from scipy.special import gammaln,gammainc,gammaincc,logsumexp


def validate(rate,detector_dead,electronics_dead):
    if not np.isfinite(rate+detector_dead+electronics_dead) or rate<=0 or min(detector_dead,electronics_dead)<0:raise ValueError('Finite positive rate and nonnegative recovery required')


def log_active_probability(rate,detector_dead,electronics_dead):
    """Probability detector is active at Te,conditioned on recorded hit at0."""
    validate(rate,detector_dead,electronics_dead);td=detector_dead;te=electronics_dead
    if td==0:return 0.
    if te<td:return -np.inf
    terms=[]
    for n in range(int(math.floor(te/td))):
        x=te-(n+1)*td
        if x<0:continue
        mean=rate*x
        if mean==0:
            if n==0:terms.append(0.)
        else:terms.append(-mean+n*math.log(mean)-gammaln(n+1))
    return float(logsumexp(terms)) if terms else -np.inf


def log_gap_density(gaps,rate,detector_dead,electronics_dead):
    validate(rate,detector_dead,electronics_dead);original=np.asarray(gaps,float);s=np.atleast_1d(original).reshape(-1);out=np.full(len(s),-np.inf);td=detector_dead;te=electronics_dead;lograte=math.log(rate)
    if td==0 or td>=te:
        dead=max(td,te);valid=s>=dead;out[valid]=lograte-rate*(s[valid]-dead)
    else:
        valid=s>=te;v=s[valid];parts=[lograte+log_active_probability(rate,td,te)-rate*(v-te)];lower=te-td;upper=np.minimum(te,v-td)
        for n in range(1,int(math.floor(te/td))+1):
            high=np.maximum(upper-n*td,0);low=max(lower-n*td,0);term=np.full(len(v),-np.inf);eligible=high>low
            if np.any(eligible):
                h=high[eligible]
                if low==0:log_difference=n*np.log(h)
                else:
                    with np.errstate(divide='ignore',invalid='ignore'):log_difference=n*np.log(h)+np.log(-np.expm1(n*(math.log(low)-np.log(h))))
                term[eligible]=(n+1)*lograte-rate*(v[eligible]-(n+1)*td)+log_difference-gammaln(n+1)
            parts.append(term)
        if len(v):out[valid]=logsumexp(np.vstack(parts),axis=0)
    result=out.reshape(original.shape if original.ndim else (1,));return float(result[0]) if original.ndim==0 else result


def gap_survival(gaps,rate,detector_dead,electronics_dead):
    validate(rate,detector_dead,electronics_dead);original=np.asarray(gaps,float);s=np.atleast_1d(original).reshape(-1);out=np.ones(len(s));td=detector_dead;te=electronics_dead
    if td==0 or td>=te:
        dead=max(td,te);valid=s>=dead;out[valid]=np.exp(-rate*(s[valid]-dead))
    else:
        valid=s>=te;v=s[valid];active=np.exp(log_gap_density(v,rate,td,te)-math.log(rate));inactive=np.zeros(len(v));lower=np.maximum(te-td,v-td)
        for n in range(1,int(math.floor(te/td))+1):
            high=max(rate*(te-n*td),0);low=np.maximum(rate*(lower-n*td),0);eligible=high>low
            if np.any(eligible):
                a=low[eligible];cdf_difference=gammainc(n,high)-gammainc(n,a);sf_difference=gammaincc(n,a)-gammaincc(n,high);difference=np.where(a>n,sf_difference,cdf_difference);assert np.all(difference>=-1e-14);inactive[eligible]+=np.maximum(difference,0)
        out[valid]=active+inactive
        assert np.all(out[valid]<=1+2e-12) and np.all(out[valid]>=0)
    result=out.reshape(original.shape if original.ndim else (1,));return float(result[0]) if original.ndim==0 else result
