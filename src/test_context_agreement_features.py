import numpy as np
from flint import ctx
from context_agreement_features import pilot_location,donor_features
from deadtime_scan import pulse


def test_future_query_events_cannot_change_early_pilot_fit():
    ctx.prec=128;context=dict(initial_theta=np.array([.8,7.2,.3]));early=np.array([.5,7.1,19.2,34.5,41.2]);a=np.r_[early,75.2,90.4];b=np.r_[early,65.1,65.9,70.1,80.5,95.2]
    x=pilot_location(a,2,32,1.2,1.5,context);y=pilot_location(b,2,32,1.2,1.5,context)
    assert x==y


def test_context_features_are_valid_and_offer_same_location_prior_mass():
    k=32;f,_=pulse(7.2,k,1.5);g,_=pulse(16.4,k,1.5);e=np.full((16,k),3000.);s=np.rint(e*(.8*f+.06*g+.3/k)).astype(int)
    z=donor_features(s,e,1.5);locations,w,templates=z['prior_bank'];assert z['two_pulse_loglr']>=0 and z['initial_theta'][0]>0 and z['initial_theta'][2]>0
    np.testing.assert_allclose(w.sum(),1);assert w[0]==.75 and len(locations)==len(w)==len(templates)
