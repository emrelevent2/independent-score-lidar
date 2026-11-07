import numpy as np
from scipy.ndimage import convolve1d
from pilot import first_detection_probabilities, coates, simulate
from methods import spline_filters, candidate, remaining_and_variance,pooled_wavelet_proposer
from photon_split import split_cycles
from crossfit import validate
from joint_detector import secondary_scores
from detector_suite import multiscale_neighbor_report,evaluate_report


def test_first_photon_and_coates_analytical_inverse():
    rng=np.random.default_rng(89)
    lam=rng.uniform(.001,.05,64)
    p=first_detection_probabilities(lam)
    assert abs(p.sum()-1)<1e-14
    assert np.allclose(coates(1e6*p[:-1],int(1e6)),lam)


def test_exact_filter_reconstruction_and_shift_invariance():
    x=np.random.default_rng(71).normal(size=(2,3,256))
    fs=spline_filters(4)
    reconstructed=convolve1d(x,fs[-1][0],axis=-1,mode='wrap')
    for _,d in fs: reconstructed+=convolve1d(x,d,axis=-1,mode='wrap')
    assert np.max(np.abs(x-reconstructed))<1e-14
    h,_,_=simulate(780)
    actual=candidate(np.roll(h,7,axis=-1))
    # Coates correction itself is causal and not shift invariant in pileup:
    # check the filter stage's invariance separately rather than claiming
    # complete-pipeline invariance.
    for _,d in fs:
        lhs=convolve1d(np.roll(x,7,axis=-1),d,axis=-1,mode='wrap')
        rhs=np.roll(convolve1d(x,d,axis=-1,mode='wrap'),7,axis=-1)
        assert np.allclose(lhs,rhs)
    assert np.isfinite(actual).all()


def test_effective_filter_noise_variance_monte_carlo():
    rng=np.random.default_rng(72)
    var=np.linspace(.2,2,64)
    x=rng.normal(size=(40000,64))*np.sqrt(var)
    _,d=spline_filters(3)[-1]
    measured=convolve1d(x,d,axis=-1,mode='wrap').var(0)
    predicted=convolve1d(var,d*d,axis=-1,mode='wrap')
    assert np.max(np.abs(measured/predicted-1))<.04


def test_candidate_validity_and_input_rejection():
    h,_,_=simulate(73)
    _,var=remaining_and_variance(h)
    assert np.isfinite(var).all() and np.all(var>0)
    out=candidate(h)
    assert out.shape==h.shape and np.isfinite(out).all() and np.all(out>=0)
    bad=h.copy(); bad[0,0,0]=2000
    try: candidate(bad)
    except ValueError: pass
    else: raise AssertionError('Impossible histogram accepted')


def test_cycle_splitting_preserves_counts_and_marginal_model():
    rng=np.random.default_rng(91)
    p=np.array([.1,.2,.7])
    observations=rng.multinomial(100,p,size=20000)[:,:-1]
    a,b,na,nb=split_cycles(observations,100,92)
    assert np.array_equal(a+b,observations)
    assert np.all(a.sum(-1)<=na) and np.all(b.sum(-1)<=nb)
    assert np.allclose(a.mean(0),50*p[:-1],atol=.06)
    assert np.allclose(a.var(0),50*p[:-1]*(1-p[:-1]),rtol=.04)
    # Independence is an unconditional property across random acquisitions.
    assert abs(np.corrcoef(a[:,0],b[:,0])[0,1])<.025


def test_validation_is_pixel_local_and_responds_to_fixed_delay():
    n=500; t=np.arange(256); sigma=5/(2*np.sqrt(2*np.log(2)))
    g1=np.exp(-.5*((t-80)/sigma)**2); g1/=g1.sum()
    g2=np.exp(-.5*((t-92)/sigma)**2); g2/=g2.sum()
    null=.00008+.03*g1
    alternative=null+.02*g2
    h0=n*first_detection_probabilities(null)[:-1]
    h1=n*first_detection_probabilities(alternative)[:-1]
    h=np.broadcast_to(h0,(3,3,256)).copy()
    p=np.full((3,3),80); s=np.full((3,3),92); valid=np.ones((3,3),bool)
    z0=validate(h,n,p,s,valid)
    h[1,1]=h1
    z1=validate(h,n,p,s,valid)
    assert z1[1,1]>z0[1,1]+1
    others=np.ones((3,3),bool); others[1,1]=False
    assert np.array_equal(z0[others],z1[others])
    assert np.isfinite(z1).all()


def test_joint_score_eliminates_primary_amplitude_and_background():
    t=np.arange(256); sigma=5/(2*np.sqrt(2*np.log(2)))
    g1=np.exp(-.5*((t-80)/sigma)**2); g1/=g1.sum()
    g2=np.exp(-.5*((t-92)/sigma)**2); g2/=g2.sum()
    a=.1+3*g1
    score=secondary_scores(a[None,None])
    assert np.max(score)<1e-5
    score2=secondary_scores((a+.3*g2)[None,None])
    assert abs(np.argmax(score2)-92)<=2
    assert np.max(score2)>.01


def test_neighbor_proposal_excludes_own_histogram():
    h,_,_=simulate(950)
    h2=h.copy(); h2[8,8]=0; h2[8,8,150]=700
    a=pooled_wavelet_proposer(h,cycles=1000,spatial_size=7,exclude_center=True)
    b=pooled_wavelet_proposer(h2,cycles=1000,spatial_size=7,exclude_center=True)
    assert np.allclose(a[8,8],b[8,8],rtol=0,atol=1e-14)


def test_multiscale_delay_selection_excludes_center_pixel():
    h,_,_=simulate(952)
    h2=h.copy(); h2[8,8]=0; h2[8,8,150]=700
    _,a=multiscale_neighbor_report(h)
    _,b=multiscale_neighbor_report(h2)
    assert a[8,8]==b[8,8]


def test_single_report_metrics_charge_wrong_depth():
    score=np.array([[5.,5.,5.,0.]])
    delay=np.array([[92.,120.,92.,92.]])
    mask=np.array([[True,True,False,True]])
    metrics=evaluate_report(score,delay,1.,mask,92.,np.ones_like(mask))
    assert metrics['recall']==1/3
    assert metrics['wrong_depth_fp']==1/3
    assert metrics['absent_pixel_fp']==1.
    assert metrics['reported_count']==3


def test_rbin_reader_against_official_matlab_writer():
    from read_rt3d import read_rbin
    from pathlib import Path
    from scipy.io import loadmat
    root=Path(__file__).resolve().parents[1]/'data'/'rt3d_measured'
    d=read_rbin(root/'format_fixture.rbin')
    original=loadmat(root/'format_fixture.mat')
    assert np.array_equal(d['histograms'].transpose(1,2,3,0),original['Y'])
    assert d['frames']==2 and d['scale_ratio']==2.5 and not d['per_pixel_irf']
    # The writer intentionally trims the impulse response before serialization.
    irf=original['h'].ravel()
    keep=np.flatnonzero(irf>=irf.max()*.01)
    assert np.allclose(d['irf'],irf[max(0,keep[0]-1):keep[-1]+1])


def test_hazard_likelihood_equals_multinomial_likelihood_without_coefficient():
    from likelihood_evidence import log_likelihood
    rng=np.random.default_rng(854)
    lam=rng.uniform(.001,.03,32)
    probs=first_detection_probabilities(lam)
    obs=rng.multinomial(500,probs)
    expected=float(obs@np.log(probs))
    assert np.isclose(log_likelihood(obs[:-1],500,lam),expected,atol=1e-10)


def test_concave_null_bound_dominates_parameter_grid():
    from likelihood_evidence import gaussian_templates,single_delay_bound,log_likelihood
    rng=np.random.default_rng(855)
    g=gaussian_templates(64)[20]
    lam=(30*g+20/64)/1000
    h=rng.multinomial(1000,first_detection_probabilities(lam))[:-1]
    bound=single_delay_bound(h,1000,g)
    assert bound['tangent_gradient'] is not None
    assert np.all(bound['tangent_gradient']<=0)
    assert 0<=bound['certificate_gap']<.002
    for a in [0.,1.,10.,30.,60.,300.]:
        for b in [0.,1.,10.,20.,100.,500.]:
            assert bound['upper']>=log_likelihood(h,1000,(a*g+b/64)/1000)-1e-9


def test_likelihood_numerator_has_no_center_parameter_leakage():
    from likelihood_evidence import neighbor_alternative
    h,_,_=simulate(856,size=12)
    h2=h.copy();h2[6,6]=0;h2[6,6,170]=600
    alt,p,s,_=neighbor_alternative(h,size=9)
    alt2,p2,s2,_=neighbor_alternative(h2,size=9)
    assert p[6,6]==p2[6,6] and s[6,6]==s2[6,6]
    assert np.allclose(alt[6,6],alt2[6,6],atol=1e-14,rtol=0)


def test_poisson_concave_bound_and_measured_irf_alignment():
    from poisson_evidence import irf_templates,poisson_single_bound,poisson_log_likelihood
    from scipy.ndimage import correlate1d
    irf=np.array([.1,.3,1.,.7,.4,.2,.1])
    g=irf_templates(irf,64)[20]
    response=correlate1d(g,irf,mode='constant',origin=int(np.argmax(irf))-len(irf)//2)
    assert np.argmax(response)==20
    rng=np.random.default_rng(858)
    h=rng.poisson(30*g+20/64)
    bound=poisson_single_bound(h,g)
    assert not bound['fallback']
    assert 0<=bound['upper']-bound['lower']<.002
    for a in [0.,1.,10.,30.,60.,300.]:
        for b in [0.,1.,10.,20.,100.,500.]:
            assert bound['upper']>=poisson_log_likelihood(h,a*g+b/64)-1e-9
