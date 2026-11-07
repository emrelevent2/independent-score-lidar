from agreement_policy import agreement_report


def fixture():
    return dict(proposal=dict(primary=101.,secondary=41.,support_proxy=8.),reports=dict(
        local=dict(primary=40.,delay=100.,score=6.),guided=dict(score=3.)))


def test_pair_association_is_role_invariant():
    r=agreement_report(fixture(),dict(local=2.,guided=1.),dict(distance=4.,bonus=1.,blend=.5))
    assert r['compatible'] and r['score']==6 and r['primary']==40.5 and r['delay']==100.5


def test_disagreeing_donor_or_weak_support_preserves_local_report():
    f=fixture();f['proposal']['primary']=130.
    r=agreement_report(f,dict(local=2.,guided=1.),dict(distance=4.,bonus=2.,blend=1.))
    assert not r['compatible'] and r['score']==3 and r['primary']==40 and r['delay']==100
    f=fixture();f['proposal']['support_proxy']=1.
    assert not agreement_report(f,dict(local=2.,guided=1.),dict(distance=4.,bonus=2.,blend=1.))['compatible']
