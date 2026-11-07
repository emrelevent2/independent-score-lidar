"""Proper alternative power allocation around a KNOWN optical response lobe.

The center is donor primary+instrument shift,never the true secondary delay.
Half the global delay prior stays uniform;half follows the physical lobe
width. Original spatial mass and rate models remain unchanged. Integer
kernel ticks make every new mixture weight an exact positive fraction.
"""
from fractions import Fraction
import numpy as np


def reweight(models,proposal,shift=16,kernel_ticks=1000000000):
    if not proposal['raw_center_excluded'] or shift<=0 or not isinstance(shift,int):raise ValueError('Independent primary and known positive instrument shift required')
    grid=np.asarray(models['global_delays']);center=float(proposal['primary'])+shift;width=float(proposal['fitted_fwhm'])
    if not 4.5<=width<=5.5:raise ValueError('Trusted width range required')
    ticks=np.rint(kernel_ticks*np.exp(-4*np.log(2)*((grid-center)/width)**2)).astype(np.int64);total=int(ticks.sum());L=len(grid)
    probabilities={float(d):Fraction(1,2*L)+(Fraction(int(t),2*total) if total else Fraction(1,2*L)) for d,t in zip(grid,ticks)}
    assert sum(probabilities.values(),Fraction(0))==1
    weights=[]
    for w,spatial,location in zip(models['weights'],models['spatial'],models['locations']):
        if not isinstance(w,Fraction):raise ValueError('Exact original prior weights required')
        weights.append(w if spatial else w*L*probabilities[float(location)])
    assert sum(weights,Fraction(0))==1 and all(w>0 for w in weights)
    new=dict(models);new['weights']=weights
    new['ambiguity_prior']=dict(center=center,fwhm=width,shift=shift,uniform_mass='1/2',response_mass='1/2',integer_kernel_ticks=int(kernel_ticks),support_preserved=True,
        scope='Alternative prior from known instrument lobe and donor-only primary;does not use true secondary delay or query observations')
    return new
