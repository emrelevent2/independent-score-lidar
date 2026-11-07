"""Prespecified paired-stratified uncertainty for FUTURE confirmation data.

Standard bootstrap and Wilson summaries, not new inference principles or
finite-sample coverage certificates. Never apply a confirmation label to a
closed development cohort merely because these calculations are available.
"""
import numpy as np


def stratified_sums(strata, replicates, seed):
    """Resample complete vectors within strata, preserving paired endpoints."""
    if not strata or replicates < 2:
        raise ValueError('Nonempty strata and at least two bootstrap replicates required')
    rng = np.random.default_rng(seed)
    dimension = np.asarray(strata[0]).shape[1]
    total = np.zeros((replicates, dimension))
    for values in strata:
        values = np.asarray(values, float)
        if values.ndim != 2 or not len(values) or values.shape[1] != dimension or not np.all(np.isfinite(values)):
            raise ValueError('Finite nonempty equally dimensioned stratum vectors required')
        states, frequencies = np.unique(values, axis=0, return_counts=True)
        weights = rng.multinomial(len(values), frequencies / len(values), size=replicates)
        total += weights @ states
    return total


def wilson(successes, count, z=1.959963984540054):
    if not 0 <= successes <= count or count <= 0:
        raise ValueError('A valid count and denominator are required')
    proportion = successes / count
    divisor = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / divisor
    half = z * np.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / divisor
    lower = 0. if successes == 0 else float(max(0., centre - half))
    upper = 1. if successes == count else float(min(1., centre + half))
    return [lower, upper]


def endpoint(row, method, alpha, tolerance):
    sample = row['samples'][method]
    declared = sample['pvalue'] <= alpha
    positive = row['case']['positive']
    distance = lambda a, b: abs((a - b + 32) % 64 - 32)
    correct = bool(positive and declared and distance(sample['primary'], row['truth']['primary']) <= tolerance
                   and distance(sample['delay'], row['truth']['secondary']) <= tolerance)
    wrong = bool(positive and declared and not correct)
    false = bool(not positive and declared)
    return np.array([int(correct), int(wrong) + int(false)], dtype=int)


def primary_effects(rows, design):
    counts = design['design']
    assert len(rows) == counts['total_cases'] == 3200
    assert sum(bool(r['case']['positive']) for r in rows) == counts['total_positive'] == 1280
    assert len(rows) - counts['total_positive'] == counts['total_null'] == 1920
    methods = ['localized_safe_score_original', 'localized_safe_score_common']
    strata, cases = {}, {}
    for row in rows:
        key = (row['case']['regime'], row['case']['family'])
        a = endpoint(row, 'localized_full_bank', counts['alpha'], counts['location_tolerance_bins'])
        vector = []
        for method in methods:
            b = endpoint(row, method, counts['alpha'], counts['location_tolerance_bins'])
            vector.extend([a[0] - b[0], b[1] - a[1]])
        strata.setdefault(key, []).append(vector)
        cases.setdefault(key, []).append(row['case'])
    assert len(strata) == 28
    for key, values in cases.items():
        assert len({bool(v['positive']) for v in values}) == 1
        expected = counts['positive_per_family_regime'] if values[0]['positive'] else counts['null_per_family_regime']
        assert len(values) == expected
    data = [np.array(strata[key], dtype=int) for key in sorted(strata)]
    denominators = np.array([1280., 3200., 1280., 3200.])
    point = sum(v.sum(axis=0) for v in data) / denominators
    uncertainty = design['uncertainty']
    distribution = stratified_sums(data, uncertainty['paired_stratified_bootstrap_replicates'], uncertainty['bootstrap_seed']) / denominators
    lower = np.quantile(distribution, .0125, axis=0)
    summaries = []
    for i, method in enumerate(methods):
        for j, effect in enumerate(['correct_detection_gain', 'total_error_reduction']):
            k = 2 * i + j
            summaries.append(dict(comparator=method, effect=effect, estimate=float(point[k]),
                                  lower_one_sided_98_75_percent=float(lower[k]), positive_lower_bound=bool(lower[k] > 0)))
    return dict(status='FUTURE_CONFIRMATION_UNCERTAINTY_CALCULATED', effects=summaries,
                all4_positive_lower_bounds=all(v['positive_lower_bound'] for v in summaries),
                scope='Pairedresamplingofcompletequeryoutcomeswithin28fixedstrata. '
                      'Approximatebootstrapcoverage,Bonferroninominal0.05across4prespecified '
                      'effects. No finitecertificate,zero-riskorcategorynoninferiorityclaim. '
                      'Point/category/full-LRrulesandallauditsremainrequiredseparately.')
