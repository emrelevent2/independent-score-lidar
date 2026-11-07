"""Version 1 detector suite: at most one reported secondary per pixel.

Parameter choices are fixed for the next evaluation. Scores have different
scales and require method-specific calibration; none is a calibrated p-value.
"""
import numpy as np
from scipy.ndimage import convolve1d,uniform_filter
from pilot import coates
from methods import depth_guided_pool,pooled_wavelet_proposer,remaining_and_variance
from crossfit import propose,validate
from joint_detector import secondary_scores


def peak_report(flux):
    score=secondary_scores(flux)
    delay=np.argmax(score,axis=-1)
    value=np.take_along_axis(score,delay[...,None],axis=-1)[...,0]
    return value,delay


def neighbor_report(h,cycles,size,wavelet=False):
    f=pooled_wavelet_proposer(h,cycles=cycles,spatial_size=size,
                             exclude_center=True,z=1.5 if wavelet else 0.)
    p,s,valid=propose(f)
    return validate(h,cycles,p,s,valid,primary_consistency=True),s


def multiscale_neighbor_report(h,cycles=1000,sizes=(3,9)):
    """Select scale using neighbor statistics only, then validate once locally.

    The confidence used for selecting a scale is a ranking heuristic; fitted
    primary uncertainty is not included. Center-pixel measurements do not enter
    proposal generation or scale choice.
    """
    x=coates(h,cycles); _,v=remaining_and_variance(h,cycles)
    t=np.arange(-10,11); sigma=5/(2*np.sqrt(2*np.log(2)))
    g=np.exp(-.5*(t/sigma)**2); g/=g.sum(); matched=g/np.sum(g*g)
    primary=[]; secondary=[]; confidence=[]; flags=[]
    for size in sizes:
        k=size*size
        pooled=(uniform_filter(x,(size,size,1),mode='wrap')*k-x)/(k-1)
        variance=np.maximum(uniform_filter(v,(size,size,1),mode='wrap')*k-v,0)/(k-1)**2
        p,s,valid=propose(pooled)
        # Use the same actual proposed delay for the amplitude heuristic.
        # secondary_scores may exclude a different local maximum; use raw
        # matched amplitude minus its local background as a stable ranking.
        matched_flux=convolve1d(pooled,matched,axis=-1,mode='constant')
        amplitude_at=np.take_along_axis(matched_flux,s[...,None],axis=-1)[...,0]
        baseline=np.median(pooled,axis=-1)/np.sum(g*g)
        var_matched=convolve1d(variance,matched*matched,axis=-1,mode='constant')
        noise_at=np.take_along_axis(var_matched,s[...,None],axis=-1)[...,0]
        c=(amplitude_at-baseline)/np.sqrt(np.maximum(noise_at,1e-20))
        confidence.append(np.where(valid,c,-np.inf)); primary.append(p); secondary.append(s); flags.append(valid)
    idx=np.argmax(np.stack(confidence,axis=-1),axis=-1)[...,None]
    p=np.take_along_axis(np.stack(primary,axis=-1),idx,axis=-1)[...,0]
    s=np.take_along_axis(np.stack(secondary,axis=-1),idx,axis=-1)[...,0]
    valid=np.take_along_axis(np.stack(flags,axis=-1),idx,axis=-1)[...,0]
    return validate(h,cycles,p,s,valid,primary_consistency=True),s


METHOD_NAMES=('raw','box3','depth_guided_3','depth_guided_7',
              'neighbor_box_3','neighbor_box_9','neighbor_wavelet_3',
              'neighbor_wavelet_9','neighbor_multiscale')


def suite(h,cycles=1000):
    raw=coates(h,cycles)
    result={'raw':peak_report(raw),
            'box3':peak_report(uniform_filter(raw,(3,3,1),mode='wrap'))}
    for size in (3,7): result[f'depth_guided_{size}']=peak_report(depth_guided_pool(h,cycles,size))
    for size in (3,9):
        result[f'neighbor_box_{size}']=neighbor_report(h,cycles,size)
        result[f'neighbor_wavelet_{size}']=neighbor_report(h,cycles,size,True)
    result['neighbor_multiscale']=multiscale_neighbor_report(h,cycles)
    return result


def evaluate_report(score,delay,threshold,mask,target,interior,tolerance=2.):
    target=np.broadcast_to(target,np.asarray(delay).shape)
    detected=score>threshold
    correct=np.abs(delay-target)<=tolerance
    positive=mask&interior; negative=~mask&interior
    result={}
    if positive.any():
        result['recall']=float((detected&correct)[positive].mean())
        result['wrong_depth_fp']=float((detected&~correct)[positive].mean())
        good=positive&detected&correct
        result['localization_sq_error_sum']=float(np.sum((delay[good]-target[good])**2))
        result['localized_count']=int(good.sum())
    if negative.any(): result['absent_pixel_fp']=float(detected[negative].mean())
    result['reported_count']=int(detected[interior].sum())
    return result
