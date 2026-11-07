import numpy as np
from context_agreement_score import context_agreement_score


def test_no_donor_secondary_evidence_preserves_all_local_statistics_exactly():
    local=np.array([1.,14.,20.,-np.inf]);delays=np.array([0.,10.,20.,30.]);pilot=delays[::-1]
    for evidence in [0.,10.,20.]:
        actual=context_agreement_score(local,delays,pilot,12.,evidence,64,1.5)
        np.testing.assert_array_equal(actual,local)


def test_mismatched_context_cannot_boost_a_distant_peak_or_override_weak_pilot():
    a=context_agreement_score(12.,20.,20.,20.,100.,64,1.5)
    b=context_agreement_score(12.,20.,20.,30.,100.,64,1.5)
    c=context_agreement_score(12.,20.,40.,20.,100.,64,1.5)
    assert a>12 and abs(b-12)<1e-6 and abs(c-12)<1e-6
    # Circular phase geometry must not make a wrap-adjacent match disappear.
    d=context_agreement_score(12.,63.9,.1,0.,100.,64,1.5)
    assert d>12
