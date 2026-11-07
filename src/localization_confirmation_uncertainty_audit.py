"""Independent outcome reconstruction for the locked confirmation intervals.

Uses the shared, analytically tested resampling primitive after independently
reconstructing every paired outcome vector. It does not validate bootstrap
coverage itself, or turn a failed superiority criterion into a pass.
"""
import json
import time
import numpy as np

import localization_fresh_confirmation as study
from localization_confirmation_statistics import stratified_sums, wilson
from transported_joint_integrity import independent_endpoint


def wait_json(name):
    while True:
        try:
            return json.loads((study.OUT / name).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(2)


def main():
    start = time.perf_counter()
    p = study.check()
    complete = wait_json('completion.json')
    analysis = wait_json('analysis.json')
    assert complete['cases'] == 3200 and complete['protocol_sha256'] == study.d.sha(study.P)
    assert complete['records_sha256'] == study.d.sha(study.OUT / 'records.sqlite3')
    rows = study.read()
    assert [r['case'] for r in rows] == list(study.cases(p))
    design = json.loads(study.DESIGN.read_text())
    comparisons = ['localized_safe_score_original', 'localized_safe_score_common']
    strata = {}
    totals = {m: dict(correct=0, wrong=0, false=0, positive=0, null=0) for m in study.METHODS}
    for row in rows:
        outcomes = {m: independent_endpoint(row, m, .01) for m in study.METHODS}
        for m in study.METHODS:
            for key, value in outcomes[m].items():
                totals[m][key] += value
        a = outcomes[study.PRIMARY]
        vector = []
        for m in comparisons:
            b = outcomes[m]
            vector.extend([a['correct'] - b['correct'], b['wrong'] + b['false'] - a['wrong'] - a['false']])
        key = (row['case']['regime'], row['case']['family'])
        strata.setdefault(key, []).append(vector)
    assert len(strata) == 28 and totals == analysis['totals']
    assert totals[study.PRIMARY]['positive'] == 1280 and totals[study.PRIMARY]['null'] == 1920
    vectors = [np.array(strata[key], dtype=int) for key in sorted(strata)]
    for key, array in zip(sorted(strata), vectors):
        family = key[1]
        expected = 80 if family in p['families']['positive'] else 160
        assert len(array) == expected
    divisors = np.array([1280., 3200., 1280., 3200.])
    point = sum(array.sum(axis=0) for array in vectors) / divisors
    parameters = design['uncertainty']
    samples = stratified_sums(vectors, parameters['paired_stratified_bootstrap_replicates'],
                               parameters['bootstrap_seed']) / divisors
    lower = np.quantile(samples, .0125, axis=0)
    expected = analysis['primary_uncertainty']
    for i, effect in enumerate(expected['effects']):
        assert effect['comparator'] == comparisons[i // 2]
        assert effect['effect'] == ['correct_detection_gain', 'total_error_reduction'][i % 2]
        assert effect['estimate'] == point[i]
        assert effect['lower_one_sided_98_75_percent'] == lower[i]
        assert effect['positive_lower_bound'] == bool(lower[i] > 0)
    passed = bool(np.all(lower > 0))
    assert passed == expected['all4_positive_lower_bounds'] == analysis['primary_uncertainty_rule_passed']
    marginal = {}
    for method, values in totals.items():
        intervals = {key: wilson(values[key], values['null'] if key == 'false' else values['positive'])
                     for key in ['correct', 'wrong', 'false']}
        assert all(interval[1] > 0 for interval in intervals.values())
        marginal[method] = dict(counts=values, descriptive_Wilson95=intervals)
    assert study.d.sha(study.OUT / 'records.sqlite3') == complete['records_sha256']
    study.check()
    result = dict(status='PASS', independently_reconstructed_query_outcomes=3200,
                  reconstructed_paired_effect_vectors=3200, fixed_strata=28,
                  bootstrap_replicates=10000, independently_checked_primary_effect_summaries=4,
                  primary_effects=expected['effects'], primary_interval_rule_passed=passed,
                  marginal_counts_and_intervals=marginal, records_sha256=complete['records_sha256'],
                  auditor_source_sha256=study.d.sha(__file__), seconds=time.perf_counter() - start,
                  scope='Independentendpointvectors/counts;sharedtestedmultinomialstratum '
                        'resamplerandquantiles. Approximatecoverage,notfinitecertificate. '
                        'An auditPASScanvalidateanegativeprimaryresult. Nozero-riskclaim.')
    (study.OUT / 'uncertainty_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
