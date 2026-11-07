import numpy as np
from full_information_joint_reference_v2 import stronger_joint,reported_locations
from null_bank_score import refit_full_null,full_likelihood_ratio
from null_information_bank import NullInformationBank
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times
from context_agreement_features import numeric_statistics
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null
from fast_two_return_fit import reports
from ct_dual_reverse import prepare_record
import ct_dual_reference as reference


def test_equal_information_reference_contains_legacy_and_matches_quadrature():
    rate,_=rates_and_design([2.4,15.3,.8,.15,25.3],64,1.5);obs,_=simulate_two_dead_times(rate,30,31.7,80.,572194);s,e=numeric_statistics(obs,30,64,outward_safe_lag(31.7,80.));pilot=fit_null(s,e,1.5);g=prepare_record(obs,1920,64,31.7,80.);null=refit_full_null(s,1.5,pilot,g);six=reports(s,e,1.5,pilot,dict(primary=0.,secondary=0.,two_pulse_loglr=0.));legacy=full_likelihood_ratio(s,1.5,null,six['fits'],g);bank=NullInformationBank(np.tile(np.eye(64),(1,2,2,4,1,1)),[[.01,20]],[[.001,30]],[0,.25,.5,.75],[2]);strong=stronger_joint(s,1.5,null,six['fits'],g,bank,2,legacy);assert strong['loglik']>=legacy['loglik'] and 1<=len(strong['additional_refinements'])<=3;rate,j=rates_and_design(strong['theta'],64,1.5);value,_=reference.full_factor(rate,j,g,32);np.testing.assert_allclose(value,strong['loglik'],atol=1e-7)


def test_unoptimized_feasible_components_report_the_stronger_return_as_primary():
    # Applies to raw feasible points too,including the low-primary-amplitude
    # corner missed by the first control implementation.
    assert reported_locations([.01,15.3,.8,.02,25.3],64)==(25.3,15.3)
    assert reported_locations([.2,-1.,.8,.02,66.],64)==(63.,2.)
