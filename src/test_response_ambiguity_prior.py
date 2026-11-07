from fractions import Fraction
import numpy as np
from response_ambiguity_prior import reweight


def test_reweight_is_proper_positive_preserves_spatial_mass_and_full_support():
    grid=np.arange(10.,50.);m=dict(global_delays=grid,locations=np.r_[33.,33.,np.repeat(grid,2)],spatial=np.r_[True,True,np.zeros(80,bool)],weights=[Fraction(1,20)]*2+[Fraction(9,800)]*80,rates=np.ones((82,64)))
    q=dict(primary=16.,fitted_fwhm=5.,raw_center_excluded=True)
    original=list(m['weights']);n=reweight(m,q)
    assert sum(n['weights'])==1 and all(w>0 for w in n['weights']) and m['weights']==original
    assert n['weights'][:2]==original[:2] and n['rates'] is m['rates']
    assert sum(n['weights'][2:])==Fraction(9,10)
    assert n['weights'][2+2*22]>original[2+2*22] # known lobe at32


def test_out_of_range_lobe_falls_back_to_uniform_without_losing_mass():
    m=dict(global_delays=np.array([4.,5.]),locations=np.array([0.,4.,5.]),spatial=np.array([True,False,False]),weights=[Fraction(1,10),Fraction(9,20),Fraction(9,20)],rates=np.ones((3,256)))
    n=reweight(m,dict(primary=250.,fitted_fwhm=5.,raw_center_excluded=True));assert n['weights']==m['weights']
