"""Conservative dyadic lag from exact supplied detector/electronics parameters."""
from fractions import Fraction
import math


def outward_safe_lag(detector_dead,electronics_dead):
    td=float(detector_dead);te=float(electronics_dead)
    if not math.isfinite(td) or not math.isfinite(te) or min(td,te)<0:raise ValueError('Finite nonnegative durations required')
    exact=Fraction(td) if td>=te else Fraction(td)+Fraction(te)
    lag=float(exact)
    if not math.isfinite(lag):raise ValueError('Unrepresentable recovery sum')
    if Fraction(lag)<exact:lag=math.nextafter(lag,math.inf)
    assert Fraction(lag)>=exact
    return lag
