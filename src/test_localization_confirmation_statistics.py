import numpy as np
from localization_confirmation_statistics import stratified_sums, wilson


def test_stratified_bootstrap_preserves_fixed_stratum_sizes_and_endpoint_pairing():
    # Constants within each stratum must remain constants after resampling;
    # independent resampling of methods would break the exact paired relation.
    a = np.array([[-1., 1.], [0., 0.], [1., -1.], [1., -1.]])
    b = np.tile([2., -2.], (7, 1))
    samples = stratified_sums([a, b], 20000, 512947)
    np.testing.assert_array_equal(samples[:, 0], -samples[:, 1])
    expected_mean = a.sum(axis=0) + b.sum(axis=0)
    expected_variance = len(a) * a.var(axis=0)
    mcse = np.sqrt(expected_variance / len(samples))
    assert np.all(np.abs(samples.mean(axis=0) - expected_mean) < 6 * mcse)
    np.testing.assert_allclose(samples.var(axis=0), expected_variance, rtol=.06)


def test_stratum_constants_are_not_mixed_between_regimes():
    samples = stratified_sums([np.tile([1., 0.], (3, 1)), np.tile([0., 1.], (11, 1))], 100, 391)
    np.testing.assert_array_equal(samples, np.tile([3., 11.], (100, 1)))


def test_zero_observed_errors_do_not_yield_a_zero_population_risk_bound():
    zero = wilson(0, 1920)
    all_events = wilson(1920, 1920)
    assert zero[0] == 0. and 0. < zero[1] < .01
    assert 0.99 < all_events[0] < 1. and all_events[1] == 1.
    np.testing.assert_allclose(zero[1], 1 - all_events[0], atol=1e-14)
