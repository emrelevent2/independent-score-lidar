"""Censor to known-active intervals when electronics hide SPAD avalanches.

The regeneration/censoring lag is established in High-flux single-photon
lidar (Optica2021),Section3.C,and upstream main_ranging_comparison.m:Tmin.
This module reconstructs phase-wise safe counts/durations,not a new general
censoring theorem or a full likelihood of every observed event.
"""
from __future__ import annotations
import numpy as np
from deadtime_exposure import periodic_measure
from deadtime_path_bootstrap import simulate,njit


def safe_lag(detector_dead,electronics_dead):
    if min(detector_dead,electronics_dead)<0:raise ValueError('Negative recovery time')
    return float(detector_dead if detector_dead>=electronics_dead else detector_dead+electronics_dead)


@njit(cache=True)
def observed_after_electronics(avalanches,electronics_dead):
    out=np.empty(len(avalanches));n=0;available=0.
    for t in avalanches:
        if t>=available:
            out[n]=t;n+=1;available=t+electronics_dead
    return out[:n]


def simulate_two_dead_times(rates,cycles,detector_dead,electronics_dead,seed):
    """Latent avalanches reset SPAD even when electronics cannot record them."""
    latent=simulate(rates,cycles,detector_dead,seed)
    observed=observed_after_electronics(latent,float(electronics_dead))
    return observed,latent


def safe_statistics(observed,cycles,bins,lag,previous=None):
    """Use ALL observed hits for the mask,score only hits in known-active time.

    Initial state active if previous=None;otherwise latest pre-block observed
    hit supplied. Short observed gaps are valid and their hits are unscored,
    but still restart the mask. This is crucial: dropping them would invent
    exposure. Window/units/continuous event assumptions match exposure.py.
    """
    t=np.asarray(observed,dtype=float).reshape(-1);end=float(cycles*bins)
    if np.any(np.diff(t)<=0) or np.any(t<0) or np.any(t>=end) or lag<0:
        raise ValueError('Invalid observed record/block')
    if previous is not None and previous>=0:raise ValueError('Previous must precede block')
    initial=0. if previous is None else max(0.,previous+lag)
    lo=np.r_[initial,t+lag];hi=np.r_[t,end];valid=hi>lo
    exposure=periodic_measure(lo[valid],hi[valid],bins)
    eligible=t>=lo[:-1]
    counts=np.bincount(np.floor(np.remainder(t[eligible],bins)).astype(np.int64),minlength=bins)
    if np.any(exposure< -1e-8) or np.any(exposure>cycles+1e-8):raise ArithmeticError('Invalid safe exposure')
    return counts,np.maximum(exposure,0),dict(eligible=eligible,open_starts=lo[valid],open_ends=hi[valid],observed_events=len(t),scored_events=int(eligible.sum()),safe_duration=float(exposure.sum()))
