"""Independent streaming data/statistic/localization/rank/endpoint audit."""
import json
import sqlite3
import time
import numpy as np
from scipy.stats import binomtest

import localization_fresh_confirmation as study
import null_bank_development as d
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from context_agreement_features import numeric_statistics, donor_features
from deadtime_scan import fit_null
from outward_safety_lag import outward_safe_lag
from null_bank_integrity import reference_score
from audit_null_bank_validation import interpolate
from transported_joint_integrity import independent_endpoint, reference_objective


def await_row(case):
    while True:
        path = study.OUT / 'records.sqlite3'
        if path.exists():
            try:
                with sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True) as db:
                    value = db.execute('SELECT payload FROM records WHERE id=?', (case['id'],)).fetchone()
            except sqlite3.OperationalError as error:
                if not any(token in str(error).lower() for token in ['no such table', 'locked', 'unable to open']):
                    raise
                time.sleep(2)
                continue
            if value:
                return json.loads(value[0])
        time.sleep(2)


def main():
    start = time.perf_counter()
    p = study.check()
    addendum = study.DESIGN
    assert d.sha(addendum) == p['design_sha256']
    rows = []
    whole = paths = ranks = identities = bank_checks = full_score_checks = local_checks = rank_identities = 0
    max_score_gap = max_local_gap = 0.
    bankarrays = dict(np.load(d.training.OUT / 'bank.npz'))
    for i, case in enumerate(study.cases(p)):
        row = await_row(case)
        assert row['case'] == case
        rows.append(row)
        g = p['regimes'][case['regime']]
        obs, ds, de, truth, parameters, dh = d.previous.old.base.data(case, p)
        assert dh == row['data_sha256'] and truth == row['truth'] and parameters == row['generating_parameters']
        s, e = numeric_statistics(obs, g['cycles'], 64, outward_safe_lag(g['td'], g['te']))
        assert len(obs) == row['observed_events'] and int(s.sum()) == row['scored_events']
        context = donor_features(ds, de, 1.5)
        assert {k: context[k] for k in ['primary', 'secondary', 'two_pulse_loglr']} == row['context']
        pilot = fit_null(s, e, 1.5)
        assert d.json_fit(pilot) == row['safe_null']
        geo = d.kernel.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
        full = d.candidate.statistics(s, e, 1.5, pilot, geo, d.bank(), case['regime'])
        assert d.json_fit(full['full_null']) == row['observed_full_null']
        assert full['bank_flags'] == row['observed_bank_flags']
        calibration = d.previous.fit_full_null(s, e, 1.5, pilot, geo)
        assert d.json_fit(calibration) == row['calibration_null']
        old = d.previous.old.statistics(s, e, pilot, context, p['settings'])
        assert old['fits'] == row['observed_alternative_fits']
        for cal in ['common', 'original']:
            for key, name in d.BASE.items():
                assert {k: row['samples'][name + '_' + cal][k] for k in ['score', 'primary', 'delay']} == old['samples'][key]
        for method in d.candidate.METHODS:
            assert {k: row['samples'][method][k] for k in ['score', 'primary', 'delay']} == full['samples'][method]
        reproduced = study.append_localizations(row, p)
        assert reproduced['localization_fits'] == row['localization_fits']
        assert reproduced['samples'] == row['samples']
        for raw, localized in [('safe_score_common', 'localized_safe_score_common'),
                               ('safe_score_original', 'localized_safe_score_original'),
                               (d.PRIMARY, study.PRIMARY)]:
            assert all(row['samples'][raw][k] == row['samples'][localized][k]
                       for k in ['score', 'pvalue', 'bootstrap_scores'])
            rank_identities += 1
        for method in study.METHODS:
            sample = row['samples'][method]
            boot = sample['bootstrap_scores']
            assert len(boot) == 199 and np.all(np.isfinite(boot))
            assert sample['pvalue'] == (1 + sum(v >= sample['score'] for v in boot)) / 200
            ranks += 1
        rg = reference.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
        for kind, fit in row['localization_fits'].items():
            theta = fit['theta']
            if kind == 'full':
                rate, j = rates_and_design(theta, 64, 1.5)
                ll = reference.full_factor(rate, j, rg, 32)[0]
            else:
                ll = -reference_objective(np.array(theta), s, e, 1.5)[0] * max(1., sum(s))
            gap = abs(ll - fit['loglik'])
            max_local_gap = max(max_local_gap, gap)
            assert gap < 1e-6
            primary, delay = (theta[1], theta[4]) if theta[0] >= theta[3] else (theta[4], theta[1])
            assert fit['primary'] == primary and fit['delay'] == delay
            local_checks += 1
        if g['td'] >= g['te']:
            assert row['samples'][study.PRIMARY] == row['samples']['localized_safe_score_common'] == row['samples']['localized_safe_score_original']
            for method in d.candidate.METHODS:
                assert row['samples'][method] == row['samples']['safe_score_common'] == row['samples']['safe_score_original']
            identities += 1
        else:
            for theta in [pilot['theta'], full['full_null']['theta']]:
                actual, _ = d.bank().lookup(theta, case['regime'])
                np.testing.assert_allclose(actual, interpolate(bankarrays, theta, case['regime']), atol=1e-7, rtol=1e-12)
                bank_checks += 1
        if case['id'].endswith('_000'):
            replay = study.worker(case, p)
            assert {k: v for k, v in replay.items() if k != 'seconds'} == {
                k: v for k, v in row.items() if k != 'seconds'}
            whole += 1
            paths += 199 * (1 if g['td'] >= g['te'] else 2)
            if g['td'] < g['te']:
                for name, theta in [('safe', pilot['theta']), ('refit', full['full_null']['theta'])]:
                    rate, j = rates_and_design(theta, 64, 1.5)
                    _, gradient, _, pieces = reference.full_factor(rate, np.eye(64), rg, 32, True)
                    matrix = interpolate(bankarrays, theta, case['regime'])
                    for weight, h in [('opg', None), ('bank', matrix)]:
                        expected = reference_score(gradient, pieces, j, theta, h)
                        actual = full['samples']['full_' + weight + '_' + name]
                        gap = abs(expected['score'] - actual['score'])
                        max_score_gap = max(max_score_gap, gap)
                        assert gap < 1e-5 and expected['delay'] == actual['delay']
                        full_score_checks += 1
        if (i + 1) % 32 == 0:
            print('Localization independent audit', i + 1, '/3200', flush=True)
    while not (study.OUT / 'completion.json').exists():
        time.sleep(1)
    while True:
        try:
            analysis = json.loads((study.OUT / 'analysis.json').read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
    assert rows == study.read()
    completion = json.loads((study.OUT / 'completion.json').read_text())
    assert completion['records_sha256'] == d.sha(study.OUT / 'records.sqlite3')
    assert completion['protocol_sha256'] == d.sha(study.P)
    all_totals = {}
    for alpha in [.005, .01, .02]:
        t = {m: dict(correct=0, wrong=0, false=0, positive=0, null=0) for m in study.METHODS}
        for row in rows:
            for method in study.METHODS:
                for key, value in independent_endpoint(row, method, alpha).items():
                    t[method][key] += value
        assert t == analysis['descriptive_operating_points'][str(alpha)]
        all_totals[str(alpha)] = t
    t = all_totals['0.01']
    assert t == analysis['totals']
    primary = t[study.PRIMARY]
    rules, paired = {}, {}
    for method, value in t.items():
        if method == study.PRIMARY:
            continue
        rule = dict(no_more_wrong=primary['wrong'] <= value['wrong'], no_more_null=primary['false'] <= value['false'])
        if method in ['safe_score_common', 'safe_score_original', 'full_opg_safe', d.PRIMARY]:
            rule.update(strict_more_correct=primary['correct'] > value['correct'],
                        strict_fewer_errors=primary['wrong'] + primary['false'] < value['wrong'] + value['false'])
        else:
            rule['at_least_correct'] = primary['correct'] >= value['correct']
        rules[method] = rule
        gains = losses = fixes = new = 0
        for row in rows:
            a = independent_endpoint(row, study.PRIMARY, .01)
            b = independent_endpoint(row, method, .01)
            gains += a['correct'] > b['correct']
            losses += a['correct'] < b['correct']
            fixes += a['wrong'] + a['false'] < b['wrong'] + b['false']
            new += a['wrong'] + a['false'] > b['wrong'] + b['false']
        paired[method] = dict(correct_gains=gains, correct_losses=losses, error_fixes=fixes, new_errors=new,
                              correct_exact_p=binomtest(gains, gains + losses).pvalue if gains + losses else 1.,
                              error_exact_p=binomtest(fixes, fixes + new).pvalue if fixes + new else 1.)
    assert rules == analysis['preliminary_rule']
    assert all(all(v.values()) for v in rules.values()) == analysis['preliminary_point_rule_passed']
    assert analysis['preliminary_rule_passed'] == (analysis['preliminary_point_rule_passed'] and analysis['primary_uncertainty_rule_passed'])
    assert not analysis['continue_to_submission']
    study.check()
    result = dict(status='PASS', all_query_context_and_observed_statistics_replayed=3200,
                  independent_rankpvalues=ranks, independent_operatingpoint_endpoints=3200 * 15 * 3,
                  exact_detection_rank_pipeline_identity_checks=rank_identities,
                  independent_localizer_likelihood_checks=local_checks, max_localizer_loglik_gap=max_local_gap,
                  nohidden_exact_identity_cases=identities, independent_bank_interpolations=bank_checks,
                  whole_workers_replayed=whole, unique_bootstrap_paths_replayed=paths,
                  independent_quadrature_full_score_checks=full_score_checks, max_full_score_gap=max_score_gap,
                  paired=paired, records_sha256=completion['records_sha256'],
                  confirmation_design_sha256=d.sha(addendum), auditor_source_sha256=d.sha(__file__),
                  seconds=time.perf_counter() - start,
                  scope='All3200data/observedfits,48000rankpvalues,144000endpoints,6400localizer '
                        'likelihoodchecks.28wholeworkers8358paths,notall955200paths. '
                        'Everyoptimizerwarningrequiresseparatesensitivityaudit. '
                        'Noefficacy,physical,globalML,uniformnullorQ1certification.')
    (study.OUT / 'integrity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
