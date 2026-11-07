import numpy as np
from safe_donor_alternative import build_alternative
from safe_prior_score import candidate_bank,prior_score
from deadtime_scan import pulse


def test_weighted_context_score_matches_independent_qr_projection():
    k=32;sig=1.5;theta=np.array([.8,7.2,.3]);f,d=pulse(theta[1],k,sig);g,_=pulse(16.4,k,sig);exposure=np.full((16,k),3000.);donor=np.rint(exposure*(.8*f+.06*g+.3/k)).astype(int);alt=build_alternative(donor,exposure,sig);bank=candidate_bank(alt,sig)
    locations,weights,templates=bank;np.testing.assert_allclose(weights.sum(),1.)
    assert set(locations[:,1])==set(alt['locations'][:,1])
    rng=np.random.default_rng(428);e=rng.uniform(100,300,k);lam=.8*f+.3/k;s=rng.poisson(e*(lam+.04*g));result=prior_score(s,e,theta,sig,bank)
    w=e/lam;z=np.column_stack([f,.8*d,np.ones(k)/k])*np.sqrt(w)[:,None];q,_=np.linalg.qr(z);h=templates*np.sqrt(w);residual=h-(h@q)@q.T;y=(s/lam-e)/np.sqrt(w);scores=np.maximum(residual@y,0)**2/np.sum(residual**2,axis=1)+2*np.log(weights)
    allowed=np.abs((locations[:,1]-theta[1]+k/2)%k-k/2)>=2.5*sig;scores=np.where(allowed,scores,-np.inf);best=int(np.argmax(scores))
    np.testing.assert_allclose(result['score'],scores[best],rtol=1e-10,atol=1e-10);assert result['index']==best
