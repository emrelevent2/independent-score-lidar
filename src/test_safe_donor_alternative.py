from fractions import Fraction
import numpy as np
from safe_donor_alternative import build_alternative,mixture_scores
from deadtime_scan import pulse


def test_context_alternative_is_proper_and_query_changes_cannot_modify_its_parameters():
    k=32;f,_=pulse(7.2,k,1.5);g,_=pulse(16.4,k,1.5);e=np.full((16,k),3000.)
    s=np.rint(e*(.8*f+.06*g+.3/k)).astype(int);q=build_alternative(s,e,1.5);before=q['rates'].copy();locations=q['locations'].copy();digest=q['sha256']
    assert sum(q['weights'])==Fraction(1) and np.all(q['rates']>0)
    assert sum(w for w,kind in zip(q['weights'],q['kinds']) if kind=='spatial')==Fraction(3,4)
    a=np.zeros(k);b=np.zeros(k);b[24]=400
    mixture_scores(a,np.full(k,100.),q);mixture_scores(b,np.full(k,100.),q)
    np.testing.assert_array_equal(q['rates'],before);np.testing.assert_array_equal(q['locations'],locations);assert q['sha256']==digest
