from guarded_real_policy import guarded_real_report
from real_poisson_analysis_v2 import route


def fixture():
    return dict(proposal=dict(primary=40.,secondary=82.,support_proxy=5.),
        reports=dict(local=dict(primary=40.,delay=85.,score=10.),guided=dict(primary=40.,delay=82.,score=6.)))


def test_real_guard_keeps_exact_normalized_local_score_and_primary():
    r=fixture();norms=dict(local=7.,guided=3.)
    base=route(r,norms,'local');new=guarded_real_report(r,norms)
    assert new['score']==base['score'] and new['primary']==base['primary'] and new['location_adjusted']
    assert new['delay']==83.5


def test_gate_uses_raw_likelihood_units_not_arbitrary_normalization():
    r=fixture();r['reports']['guided']['score']=4.
    new=guarded_real_report(r,dict(local=1000.,guided=.0001))
    assert not new['location_adjusted'] and new['delay']==85.


def test_distant_or_unsupported_donor_does_not_change_local_location():
    for key,value in [('secondary',120.),('support_proxy',1.)]:
        r=fixture();r['proposal'][key]=value;new=guarded_real_report(r,dict(local=7.,guided=3.))
        assert not new['location_adjusted'] and new['delay']==85.
