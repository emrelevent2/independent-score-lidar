import numpy as np
from clock_pulse_likelihood import rates_and_design,clock_rates,full_factor,safe_statistics,safe_factor
from test_clock_dual_transition import observe


def test_periodic_pulse_full_gradient_matches_five_parameter_finite_difference():
    k=64;sigma=1.5;x=np.array([.8,15.3,.3,.065,27.3]);rate,j=rates_and_design(x,k,sigma);p=-np.expm1(-rate);pattern=np.random.default_rng(92047031).binomial(1,np.tile(p,20));record=np.array(observe(pattern,31,79));value,gradient=full_factor(x,record,len(pattern),k,sigma,31,79);finite=[]
    for i in range(5):
        delta=np.zeros(5);delta[i]=1e-6;finite.append((full_factor(x+delta,record,len(pattern),k,sigma,31,79)[0]-full_factor(x-delta,record,len(pattern),k,sigma,31,79)[0])/2e-6)
    np.testing.assert_allclose(gradient,finite,atol=2e-6,rtol=2e-6)


def test_full_and_safe_factors_match_when_all_avalanches_are_recordable():
    k=32;sigma=1.5
    for x in [np.array([.8,7.3,.3]),np.array([.8,7.3,.3,.06,17.1])]:
        rate,_=rates_and_design(x,k,sigma);p=-np.expm1(-rate);pattern=np.random.default_rng(580021).binomial(1,np.tile(p,20));record=np.array(observe(pattern,5,2));stats=safe_statistics(record,len(pattern),k,5,2);assert stats['counts'].sum()==len(record);a,ga=full_factor(x,record,len(pattern),k,sigma,5,2);b,gb=safe_factor(x,stats,k,sigma);np.testing.assert_allclose(a,b,atol=1e-11);np.testing.assert_allclose(ga,gb,atol=1e-10)


def test_unscored_clock_events_restart_mask_and_scored_ticks_include_event():
    stats=safe_statistics(np.array([0,6,14]),24,8,2,5);assert stats['safe_lag_ticks']==8;np.testing.assert_array_equal(stats['counts'],[1,0,0,0,0,0,1,0]);np.testing.assert_array_equal(stats['activations'],[1,0,0,0,0,0,2,1])


def test_clock_refinement_preserves_original_piecewise_rate_and_physical_gradient():
    x=np.array([.8,7.3,.3,.06,17.1]);rate,j=rates_and_design(x,32,1.5);fine,jfine=clock_rates(x,32,1.5,4);np.testing.assert_allclose(fine.reshape(32,4).sum(axis=1),rate,atol=1e-15);np.testing.assert_allclose(jfine.reshape(32,4,5).sum(axis=1),j,atol=1e-15)
