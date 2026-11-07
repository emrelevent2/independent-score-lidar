"""Independent read-only integrity audit of the conditional stronger controls.

This file does not choose a method or modify any scientific record. Run only
after the main preliminary screen passes and its two independent audits pass.
Full workers are replayed for the predeclared first donor of every family.
"""
import hashlib
import json
import sqlite3
import time

import numpy as np

import null_bank_development as d
import null_bank_equal_information_control_v2 as control
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from context_agreement_features import numeric_statistics
from deadtime_scan import fit_null
from null_bank_score import refit_full_null, full_likelihood_ratio
from full_information_joint_reference_v2 import stronger_joint, reported_locations
from outward_safety_lag import outward_safe_lag
from transported_joint_integrity import independent_endpoint


def wait_record(case):
    path = d.OUT / 'equal_information_joint_reference_v2.sqlite3'
    while True:
        if path.exists():
            with sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True) as db:
                row = db.execute('SELECT payload FROM records WHERE id=?', (case['id'],)).fetchone()
            if row:
                return json.loads(row[0])
        time.sleep(2)


def run():
    start = time.perf_counter()
    p = control.check()
    assert json.loads((d.OUT / 'analysis.json').read_text())['preliminary_rule_passed']
    assert all(json.loads((d.OUT / name).read_text())['status'] == 'PASS'
               for name in ['integrity.json', 'failure_audit.json'])
    main_rows = d.read()
    main_hash = d.sha(d.OUT / 'records.sqlite3')
    rows = []
    ranks = quadratures = whole = paths = identities = 0
    largest = 0.
    totals = {str(alpha): {m: dict(correct=0, wrong=0, false=0, positive=0, null=0)
                         for m in ['legacy', 'strong']} for alpha in [.005, .01, .02]}
    for i, main in enumerate(main_rows):
        case = main['case']
        saved = wait_record(case)
        assert saved['case'] == case
        assert saved['bootstrap_path_sha256'] == main['bootstrap_path_sha256']['common']
        rows.append(saved)
        g = p['regimes'][case['regime']]
        obs, _, _, _, _, data_hash = d.previous.old.base.data(case, p)
        assert data_hash == main['data_sha256']
        s, e = numeric_statistics(obs, g['cycles'], 64, outward_safe_lag(g['td'], g['te']))
        pilot = fit_null(s, e, 1.5)
        geo = d.kernel.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
        full = refit_full_null(s, 1.5, pilot, geo)
        assert d.json_fit(full) == main['observed_full_null']
        legacy = full_likelihood_ratio(s, 1.5, full, main['observed_alternative_fits'], geo)
        strong = stronger_joint(s, 1.5, full, main['observed_alternative_fits'],
                                geo, d.bank(), case['regime'], legacy)
        assert legacy == saved['observed_full_joint']
        assert strong == saved['observed_stronger_joint']
        assert strong['loglik'] >= legacy['loglik'] - 1e-9 >= full['loglik'] - 2e-9
        rg = reference.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
        for method, fitted in [('legacy', legacy), ('strong', strong)]:
            rate, design = rates_and_design(fitted['theta'], 64, 1.5)
            value = reference.full_factor(rate, design, rg, 32)[0]
            gap = abs(value - fitted['loglik'])
            largest = max(largest, gap)
            assert gap < 1e-6
            assert tuple(fitted[k] for k in ['primary', 'delay']) == reported_locations(fitted['theta'], 64)
            sample = saved['samples'][method]
            assert all(sample[k] == fitted[k] for k in ['score', 'primary', 'delay'])
            assert abs(sample['score'] - max(0., 2 * (value - full['loglik']))) < 2e-6
            boot = np.asarray(sample['bootstrap_scores'])
            assert len(boot) == 199 and np.all(np.isfinite(boot))
            assert sample['pvalue'] == (1 + np.count_nonzero(boot >= sample['score'])) / 200
            ranks += 1
            quadratures += 1
        assert np.all(np.array(saved['samples']['strong']['bootstrap_scores']) >=
                      np.array(saved['samples']['legacy']['bootstrap_scores']) - 1e-9)
        if g['td'] >= g['te']:
            assert saved['samples']['strong'] == saved['samples']['legacy']
            identities += 1
        augmented = dict(main, samples=dict(main['samples'], **saved['samples']))
        for alpha, methods in totals.items():
            for method in methods:
                for key, value in independent_endpoint(augmented, method, float(alpha)).items():
                    methods[method][key] += value
        if case['id'].endswith('_000'):
            replay = control.worker(main, p)
            assert {k: v for k, v in replay.items() if k != 'seconds'} == {
                k: v for k, v in saved.items() if k != 'seconds'}
            whole += 1
            paths += 199
        if (i + 1) % 32 == 0:
            print('Equal-information independent audit', i + 1, '/640', flush=True)
    while not (d.OUT / 'equal_information_joint_completion_v2.json').exists():
        time.sleep(1)
    assert rows == control.read()
    completion = json.loads((d.OUT / 'equal_information_joint_completion_v2.json').read_text())
    assert completion['reference_records_sha256'] == d.sha(d.OUT / 'equal_information_joint_reference_v2.sqlite3')
    assert main_hash == completion['main_records_sha256'] == d.sha(d.OUT / 'records.sqlite3')
    while True:
        try:
            analysis = json.loads((d.OUT / 'equal_information_joint_analysis_v2.json').read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
    assert totals['0.01'] == analysis['reference_totals']
    primary = d.totals(main_rows, .01)[d.PRIMARY]
    rules = {m: dict(at_least_correct=primary['correct'] >= v['correct'],
                     no_more_wrong=primary['wrong'] <= v['wrong'],
                     no_more_null=primary['false'] <= v['false'])
             for m, v in totals['0.01'].items()}
    assert rules == analysis['primary_rules']
    assert all(all(v.values()) for v in rules.values()) == analysis['all_point_rules_pass']
    control.check()
    result = dict(status='PASS', observed_control_fits_replayed=1280,
                  independent_quadrature_fitted_loglik_checks=quadratures,
                  max_quadrature_loglik_gap=largest, independent_rankpvalues=ranks,
                  independent_endpoints=640 * 2 * 3, exact_nohidden_identity_cases=identities,
                  whole_workers_replayed=whole, common_bootstrap_paths_replayed=paths,
                  descriptive_operating_points=totals, main_records_sha256=main_hash,
                  reference_records_sha256=completion['reference_records_sha256'],
                  auditor_source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
                  seconds=time.perf_counter() - start,
                  scope='All640observedfits,1280ranks,3840endpoints;1280independentquadraturevalues. '
                        '28wholeworkers5572paths,notall127360paths. Allwarnings requireseparateaudit. '
                        'No globalML,physical,uniformnull,efficacyorQ1certification.')
    (d.OUT / 'equal_information_joint_integrity_v2.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
