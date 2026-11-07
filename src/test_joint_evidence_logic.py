"""Exact finite-outcome checks of known joint/averaging/risk-set principles."""
from fractions import Fraction
from math import comb,factorial


def rising(a,n):
    x=Fraction(1)
    for i in range(n):x*=a+i
    return x


def test_shared_parameter_joint_likelihood_and_twofold_mean_have_unit_bound():
    n=3
    def fold(kt,kv,kx):
        a=Fraction(2*kt+1,2);b=Fraction(2*(n-kt)+1,2)
        qcal=rising(a,kv)*rising(b,n-kv)/rising(a+b,n)
        px=min(Fraction(99,100),Fraction(1,10)+2*a/(a+b))
        qquery=px**kx*(1-px)**(n-kx)
        # Continuous shared Bernoulli null, globally profiled analytically.
        total=kv+kx;p=Fraction(total,2*n)
        denominator=p**total*(1-p)**(2*n-total)
        return qcal*qquery/denominator
    for p in [Fraction(1,5),Fraction(1,2),Fraction(4,5)]:
        expected=Fraction(0)
        for k1 in range(n+1):
            for k2 in range(n+1):
                for kx in range(n+1):
                    mass=Fraction(1)
                    for k in [k1,k2,kx]:mass*=comb(n,k)*p**k*(1-p)**(n-k)
                    expected+=mass*(fold(k1,k2,kx)+fold(k2,k1,kx))/2
        assert expected<=1


def test_binomial_risk_set_law_is_exact_after_earlier_first_detections():
    # Four disjoint first-event categories: early,lobe,late,no event.
    n=4;prob=[Fraction(1,10),Fraction(9,40),Fraction(3,20),Fraction(21,40)]
    joint={};risk={}
    for early in range(n+1):
        for lobe in range(n-early+1):
            for late in range(n-early-lobe+1):
                h=[early,lobe,late,n-early-lobe-late]
                coeff=factorial(n)
                for k in h:coeff//=factorial(k)
                mass=Fraction(coeff)
                for k,p in zip(h,prob):mass*=p**k
                r=n-early;joint[r,lobe]=joint.get((r,lobe),Fraction(0))+mass
                risk[r]=risk.get(r,Fraction(0))+mass
    for r in range(n+1):
        for k in range(r+1):
            assert joint[r,k]/risk[r]==comb(r,k)*Fraction(1,4)**k*Fraction(3,4)**(r-k)
