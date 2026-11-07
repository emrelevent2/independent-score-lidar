from fractions import Fraction
import numpy as np
from outward_safety_lag import outward_safe_lag


def test_rounded_down_recovery_sum_is_replaced_by_an_upper_endpoint():
    td,te=83.7,100.;exact=Fraction(td)+Fraction(te)
    assert Fraction(td+te)<exact
    corrected=outward_safe_lag(td,te)
    assert Fraction(corrected)>=exact and corrected==np.nextafter(td+te,np.inf)


def test_all_orderings_cover_the_correct_exact_recovery_expression():
    for td,te in [(31.7,80.),(50.3,80.7),(5.3,61.2),(198.,80.),(50.,50.),(0.,80.)]:
        exact=Fraction(td) if td>=te else Fraction(td)+Fraction(te)
        assert Fraction(outward_safe_lag(td,te))>=exact
