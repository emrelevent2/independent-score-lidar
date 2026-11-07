import numpy as np
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from context_agreement_features import numeric_statistics
from deadtime_scan import fit_null
from null_bank_score import refit_full_null
from outward_safety_lag import outward_safe_lag
from safe_deadtime_exposure import simulate_two_dead_times
from score_peak_localization import localize


def test_full_localizer_matches_independent_quadrature_and_contains_nested_null():
    rate, _ = rates_and_design([2.4, 15.3, .8, .15, 25.3], 64, 1.5)
    obs, _ = simulate_two_dead_times(rate, 30, 31.7, 80., 214937)
    s, e = numeric_statistics(obs, 30, 64, outward_safe_lag(31.7, 80.))
    safe = fit_null(s, e, 1.5)
    geo = reference.prepare_record(obs, 1920, 64, 31.7, 80.)
    full = refit_full_null(s, 1.5, safe, geo)
    result = localize(s, e, 1.5, full, 25.25, geo)
    rate, j = rates_and_design(result['theta'], 64, 1.5)
    ll = reference.full_factor(rate, j, geo, 32)[0]
    np.testing.assert_allclose(ll, result['loglik'], atol=1e-8)
    assert ll >= full['loglik'] - 1e-8
    assert all(0 <= result[k] < 64 for k in ['primary', 'delay'])


def test_full_and_safe_localizers_coincide_when_no_avalanches_are_hidden():
    rate, _ = rates_and_design([.8, 15.3, .3, .08, 27.3], 64, 1.5)
    obs, _ = simulate_two_dead_times(rate, 50, 83.7, 40., 629731)
    s, e = numeric_statistics(obs, 50, 64, outward_safe_lag(83.7, 40.))
    safe = fit_null(s, e, 1.5)
    geo = reference.prepare_record(obs, 3200, 64, 83.7, 40.)
    full = localize(s, e, 1.5, safe, 27.25, geo)
    partial = localize(s, e, 1.5, safe, 27.25)
    np.testing.assert_allclose(full['theta'], partial['theta'], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(full['loglik'], partial['loglik'], atol=1e-7)
