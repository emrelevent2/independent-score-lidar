"""Numerical-audit checks using real continuous likelihoods, no fit mocks."""
import numpy as np
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from context_agreement_features import numeric_statistics
from deadtime_scan import fit_null
from full_information_joint_reference_v2 import reported_locations
from null_bank_control_reference import refine_component
from null_bank_score import refit_full_null
from outward_safety_lag import outward_safe_lag
from safe_deadtime_exposure import simulate_two_dead_times


def test_independent_strict_refinement_keeps_feasible_point_and_matches_likelihood():
    rate, _ = rates_and_design([2.4, 15.3, .8, .2, 25.3], 64, 1.5)
    obs, _ = simulate_two_dead_times(rate, 30, 31.7, 80., 619283)
    s, e = numeric_statistics(obs, 30, 64, outward_safe_lag(31.7, 80.))
    pilot = fit_null(s, e, 1.5)
    geo = reference.prepare_record(obs, 1920, 64, 31.7, 80.)
    null = refit_full_null(s, 1.5, pilot, geo)
    a, t, b = null['theta']
    theta = np.array([a, t, b, .02, 25.3])
    rate, j = rates_and_design(theta, 64, 1.5)
    initial = reference.full_factor(rate, j, geo, 32)[0]
    primary, delay = reported_locations(theta, 64)
    fit = dict(theta=theta, loglik=initial, primary=primary, delay=delay, optimizer_success=False)
    best, audit = refine_component(fit, 25.3, null, s, geo, 'unit_fixture')
    assert best['loglik'] >= initial and audit['improvement'] >= 0
    rate, j = rates_and_design(best['theta'], 64, 1.5)
    actual = reference.full_factor(rate, j, geo, 32)[0]
    np.testing.assert_allclose(actual, best['loglik'], atol=1e-8)
    assert tuple(best[k] for k in ['primary', 'delay']) == reported_locations(best['theta'], 64)
