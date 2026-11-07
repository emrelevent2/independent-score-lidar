from guarded_localization import guarded_report


def fixture():
    q=dict(primary_incident=60.,secondary_incident=8.,background_incident=10.,fitted_fwhm=5.,effective_donors=30.,secondary=80.)
    return q,dict(score=6.),dict(score=10.,primary=40.,delay=83.)


def test_guarded_localization_preserves_score_and_averages_only_compatible_location():
    q,s,l=fixture();r=guarded_report(q,s,l,dict(distance=4.,fraction=.5,blend=.5))
    assert r['location_adjusted'] and r['delay']==81.5 and r['score']==l['score'] and r['primary']==l['primary']


def test_disagreement_or_insufficient_query_evidence_keeps_local_position():
    q,s,l=fixture();s['score']=4.
    r=guarded_report(q,s,l,dict(distance=4.,fraction=.5,blend=.5));assert not r['location_adjusted'] and r['delay']==l['delay']
    s['score']=6.;l['delay']=88.
    assert not guarded_report(q,s,l,dict(distance=4.,fraction=.5,blend=.5))['location_adjusted']
