"""Prototype: boost a LOCAL peak only when context and an early pilot support it.

This is a heuristic scalar statistic requiring joint event-path bootstrap
calibration. It is not an e-value and inherits no certified guarantee.
No study has yet established superiority of this prototype.
"""
import numpy as np


def circular_distance(a,b,bins):return np.abs((np.asarray(a)-np.asarray(b)+bins/2)%bins-bins/2)


def context_agreement_score(local_score,local_delay,pilot_delay,donor_delay,donor_two_pulse_loglr,bins,sigma,use_pilot=True):
    if bins<=0 or sigma<=0:raise ValueError('Positive phase period and width required')
    donor_strength=-np.expm1(-max(float(donor_two_pulse_loglr)-20.,0.)/20.)
    fwhm=2*np.sqrt(2*np.log(2))*sigma;bonus=2*np.log(max(bins/fwhm,1.))
    full=np.exp(-.5*(circular_distance(local_delay,donor_delay,bins)/sigma)**2)
    pilot=np.exp(-.5*(circular_distance(pilot_delay,donor_delay,bins)/(2*sigma))**2) if use_pilot else 1.
    return np.asarray(local_score)+bonus*donor_strength*full*pilot
