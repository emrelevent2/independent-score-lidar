"""Numerical/invariance checks only; no efficacy or label-selected fixtures."""
import numpy as np
import pytest
from conditional_context_scan import censored_field, weighted_scan
from deadtime_path_bootstrap import fixed_fit_scores
from deadtime_scan import pulse


def fixture():
    grid=np.arange(256)/4
    score=8+3*np.cos(grid/9)
    score[np.abs((grid-15.3+32)%64-32)<3.75]=-np.inf
    return dict(grid=grid,scores=score,primary=15.3)


def test_no_supported_context_is_exact_original_scan_and_keeps_exclusions():
    f=fixture()
    for strength in [-4.,0.,20.]:
        r=weighted_scan(f,dict(primary=15.,secondary=27.,two_pulse_loglr=strength),1.5,64)
        assert np.array_equal(r['scores'],f['scores'])
        assert r['score']==max(f['scores'])
        assert r['delay']==f['grid'][np.argmax(f['scores'])]
        assert not r['supported']
        assert np.all(r['weights'][~r['allowed']]==0)


def test_full_coverage_floor_and_score_distortion_bounds():
    f=fixture()
    r=weighted_scan(f,dict(primary=15.,secondary=27.,two_pulse_loglr=45.),1.5,64)
    n=r['allowed'].sum()
    np.testing.assert_allclose(r['weights'].sum(),1.,atol=1e-15)
    assert np.min(r['weights'][r['allowed']]) >= .5/n
    base=max(f['scores'])
    assert base+2*np.log(.5)-1e-12 <= r['score'] <= base+2*np.log(.5+.5*n)+1e-12
    assert np.array_equal(np.isfinite(r['scores']),np.isfinite(f['scores']))


def test_integer_phase_rotation_and_query_relative_transport():
    f=fixture();c=dict(primary=15.,secondary=27.,two_pulse_loglr=45.)
    a=weighted_scan(f,c,1.5,64)
    shift=7
    g=dict(grid=f['grid'],scores=np.roll(f['scores'],4*shift),primary=(f['primary']+shift)%64)
    b=weighted_scan(g,dict(c,primary=(c['primary']+shift)%64,secondary=(c['secondary']+shift)%64),1.5,64)
    np.testing.assert_allclose(b['weights'],np.roll(a['weights'],4*shift),rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(b['score'],a['score'],rtol=1e-14)
    assert b['delay']==(a['delay']+shift)%64
    assert b['transported_centre']==(a['transported_centre']+shift)%64


def test_safe_field_agrees_with_frozen_fft_scan_and_independent_dense_projection():
    rng=np.random.default_rng(305184729)
    for theta in [[.8,15.3,.3],[2.4,63.6,.8],[.25,.1,.1]]:
        s=rng.poisson(7,size=64);e=rng.uniform(100,800,64)
        r=censored_field(s,e,theta,1.5)
        maximum,location=fixed_fit_scores(s,e,theta,1.5,'duration',4)
        np.testing.assert_allclose(max(r['scores']),maximum[0],rtol=1e-11,atol=1e-9)
        assert r['grid'][np.argmax(r['scores'])]==location[0]
        a,t,b=theta;f,d=pulse(t,64,1.5);rate=a*f+b/64
        j=np.column_stack([f,a*d,np.ones(64)/64]);w=e/rate;u=s/rate-e
        v=np.linalg.pinv(j.T@(w[:,None]*j),rcond=1e-12)
        dense=[]
        for centre in r['grid']:
            q=pulse(centre,64,1.5)[0];cross=j.T@(w*q)
            raw=q@u-cross@v@(j.T@u);variance=q@(w*q)-cross@v@cross
            dense.append(max(raw,0)**2/variance if abs((centre-t+32)%64-32)>=3.75 and variance>1e-8 else -np.inf)
        np.testing.assert_allclose(r['scores'],dense,rtol=1e-10,atol=1e-7)


def test_invalid_field_and_no_admissible_centre_rejected():
    f=fixture();c=dict(primary=15.,secondary=27.,two_pulse_loglr=45.)
    with pytest.raises(ValueError): weighted_scan(dict(f,scores=np.full(256,-np.inf)),c,1.5,64)
    with pytest.raises(ValueError): weighted_scan(dict(f,scores=np.full(256,np.nan)),c,1.5,64)
    with pytest.raises(ValueError): weighted_scan(f,dict(c,secondary=np.nan),1.5,64)
