import numpy as np
from null_bank_score import statistics,METHODS,refit_full_null,primary_statistic
from null_information_bank import NullInformationBank
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times
from context_agreement_features import numeric_statistics
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null
from deadtime_path_bootstrap import fine_scan
from ct_dual_reverse import prepare_record
from null_bank_score import full_likelihood_ratio
import ct_dual_reference as reference
from fast_two_return_fit import reports


def test_no_hidden_all_four_variants_equal_the_complete_existing_score_pipeline():
    rate,_=rates_and_design([.8,15.3,.3],64,1.5);obs,_=simulate_two_dead_times(rate,50,83.7,40.,319852);s,e=numeric_statistics(obs,50,64,83.7);fit=fit_null(s,e,1.5);g=prepare_record(obs,3200,64,83.7,40.);result=statistics(s,e,1.5,fit,g,None,0);old=fine_scan(s,e,1.5,fit=fit)
    for m in METHODS:assert result['samples'][m]=={k:old[k] for k in ['score','primary','delay']}
    np.testing.assert_array_equal(result['full_null']['theta'],fit['theta'])


def test_hidden_full_refit_is_feasible_nondecreasing_and_two_weight_ablations_share_fits():
    rate,_=rates_and_design([2.4,15.3,.8,.15,25.3],64,1.5);obs,_=simulate_two_dead_times(rate,80,31.7,80.,417829);s,e=numeric_statistics(obs,80,64,outward_safe_lag(31.7,80.));fit=fit_null(s,e,1.5);g=prepare_record(obs,5120,64,31.7,80.);bank=NullInformationBank(np.tile(np.eye(64),(1,2,2,4,1,1)),[[.01,20]],[[.001,30]],[0,.25,.5,.75],[2]);result=statistics(s,e,1.5,fit,g,bank,2);f=result['full_null'];assert f['loglik']>=f['initial_loglik']-1e-10 and 0<=f['theta'][0]<=20 and 0<=f['theta'][1]<64 and 1e-6<=f['theta'][2]<=30
    for m in METHODS:assert np.isfinite(result['samples'][m]['score'])
    assert result['samples']['full_opg_refit']['primary']==result['samples']['full_bank_refit']['primary']==f['theta'][1]
    assert result['samples']['full_opg_safe']['primary']==result['samples']['full_bank_safe']['primary']==fit['theta'][1]
    single,_,_=primary_statistic(s,e,1.5,fit,g,bank,2);assert single==result['samples']['full_bank_refit']


def test_practical_full_likelihood_ratio_is_nested_and_matches_reference_quadrature():
    rate,_=rates_and_design([2.4,15.3,.8,.15,25.3],64,1.5);obs,_=simulate_two_dead_times(rate,30,31.7,80.,740192);s,e=numeric_statistics(obs,30,64,outward_safe_lag(31.7,80.));pilot=fit_null(s,e,1.5);g=prepare_record(obs,1920,64,31.7,80.);null=refit_full_null(s,1.5,pilot,g);safe=reports(s,e,1.5,pilot,dict(primary=15.,secondary=25.,two_pulse_loglr=0.));result=full_likelihood_ratio(s,1.5,null,safe['fits'],g);r,j=rates_and_design(result['theta'],64,1.5);ll,_=reference.full_factor(r,j,g,32);np.testing.assert_allclose(ll,result['loglik'],atol=1e-7);assert result['loglik']>=null['loglik']-1e-10 and result['loglik']>=max(result['all6_full_start_logliks'])-1e-10;np.testing.assert_allclose(result['score'],2*(result['loglik']-null['loglik']),atol=1e-10)
