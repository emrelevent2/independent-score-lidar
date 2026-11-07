import numpy as np
from ct_full_score import projected,templates,statistic,fit_full_null
from clock_pulse_likelihood import rates_and_design
from deadtime_scan import fit_null
from deadtime_path_bootstrap import fine_scan
from safe_deadtime_exposure import simulate_two_dead_times
from context_agreement_features import numeric_statistics
from ct_dual_fast import prepare_record


def test_full_projected_score_matches_independent_whitened_QR():
    rng=np.random.default_rng(728401);k=32;theta=np.array([.8,7.3,.3]);_,j=rates_and_design(theta,k,1.5);z=rng.normal(size=(80,k));residual=rng.normal(size=80);h=z.T@z;score=z.T@residual;actual=projected(score,h,j,theta,1.5);grid,g=templates(k,1.5);q,_=np.linalg.qr(z@j);design=z@g.T;design-=q@(q.T@design);raw=design.T@residual;information=np.sum(design**2,axis=0);allowed=np.abs((grid-theta[1]+k/2)%k-k/2)>=3.75;expected=np.where(allowed,np.maximum(raw,0)**2/information,-np.inf);np.testing.assert_allclose(actual['scores'],expected,atol=1e-10,rtol=1e-10)


def test_detector_dominant_statistic_and_calibration_fit_are_identical_to_safe_control():
    rate,_=rates_and_design(np.array([.8,15.3,.3]),64,1.5);obs,_=simulate_two_dead_times(rate,80,83.7,40.,930176);s,e=numeric_statistics(obs,80,64,83.7);fit=fit_null(s,e,1.5);g=prepare_record(obs,5120,64,83.7,40.);a,metadata=statistic(s,e,1.5,fit,g);b=fine_scan(s,e,1.5,fit=fit);assert a=={key:b[key] for key in ['score','primary','delay']};null=fit_full_null(s,e,1.5,fit,g);np.testing.assert_array_equal(null['theta'],fit['theta']);assert metadata['mode']=='EXACT_SAFE_PASSTHROUGH'
