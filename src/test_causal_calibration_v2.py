from fractions import Fraction as F
from math import comb
import numpy as np
from flint import arb
from causal_calibration_v2 import score_lobe,score_dark,two_window_upper


def binom(n,k,p):return comb(n,k)*p**k*(1-p)**(n-k)


def test_causal_two_windows_with_unscored_middle_and_full_cal_query_normalize():
    N=3;total=F(0)
    for pre in range(N+1):
        R=N-pre
        for k in range(R+1):
            for mid in range(R-k+1):
                RD=R-k-mid
                for d in range(RD+1):
                    mass=binom(N,pre,F(1,5))*binom(R,k,F(1,4))*binom(R-k,mid,F(1,7))*binom(RD,d,F(1,3))
                    qk=F(pre+1,N+2);qd=F(pre+k+mid+1,N+2);qx=F(pre+k+mid+d+1,N+2)
                    for x in range(3):
                        px=binom(2,x,F(2,5));ratio=binom(R,k,qk)/binom(R,k,F(1,4))*binom(RD,d,qd)/binom(RD,d,F(1,3))*binom(2,x,qx)/px
                        total+=mass*px*ratio
    assert total==1


def test_implemented_sequential_predictors_each_sum_to_one():
    prefix=np.zeros(43,int);prefix[30:35]=[1,2,3,2,1];N=20;R=N-prefix.sum();prob=arb(0)
    for k in range(R+1):
        q=score_lobe(prefix,N,.1,k);prob+=comb(int(R),k)*arb(q['lower']).union(arb(q['upper'])).exp()
    assert prob.contains(1)
    past=np.zeros(128,int);past[:43]=prefix;past[70]=2;RD=N-past.sum();prob=arb(0)
    for d in range(RD+1):
        q=score_dark(past,N,d);prob+=comb(int(RD),d)*arb(q['lower']).union(arb(q['upper'])).exp()
    assert prob.contains(1)


def test_joint_calibration_profile_contains_shared_background_models():
    cf=dict(informative=True,signal_main_lower=.04,signal_main_upper=.06,background_lower=.00001,background_upper=.00003,forward_leakage_lower=0.,forward_leakage_upper=.000002)
    k,r,d,rd=100,20000,48,19000
    upper=two_window_upper(k,r,d,rd,.04,.14,cf);assert upper['profile_certified']
    for eta in np.linspace(.04,.14,9):
        for s in [.04,.05,.06]:
            for beta in [.00001,.00002,.00003]:
                H=arb(s)*(arb(float(eta))+arb(.000001))+11*arb(beta);HD=128*arb(beta)
                ll=k*(-(-H).expm1()).log()-(r-k)*H+d*(-(-HD).expm1()).log()-(rd-d)*HD
                assert arb(upper['upper'])>=ll
