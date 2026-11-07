"""Warning sensitivity for the three localized reporting pipelines.

The separate parent audit must establish unchanged declarations at all three
operating points. This audit propagates warned input fits and stricter
localization refinements to the reported positions, without rewriting rows.
"""
import json
import time
import numpy as np
from scipy.optimize import minimize

import localization_fresh_confirmation as study
import null_bank_development as d
import ct_dual_reference as reference
from context_agreement_features import numeric_statistics
from deadtime_path_bootstrap import fine_scan
from full_information_joint_reference_v2 import reported_locations
from null_bank_control_reference import refine_component
from null_bank_failure_audit import refine_full, full_statistics
from one_step_location_failure_audit import stronger_null
from outward_safety_lag import outward_safe_lag
from score_peak_localization import localize
from localization_confirmation_integrity import await_row
from transported_joint_integrity import reference_objective, independent_endpoint


def refine_safe(fit, centre, null, counts, exposure):
    a, t, b = null['theta']
    scale = max(1., sum(counts))
    bounds = [(0, 20), (t - 4.5, t + 4.5), (1e-6, 30), (0, 20), (centre - 4.5, centre + 4.5)]
    initial = np.array(fit['theta'], float)
    initial[1] += 64 * round((t - initial[1]) / 64)
    initial[4] += 64 * round((centre - initial[4]) / 64)
    initial = np.clip(initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
    candidates = [fit]
    for theta in [initial, np.array([a, t, b, .02, centre])]:
        result = minimize(reference_objective, theta, args=(counts, exposure, 1.5), jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options=dict(ftol=1e-13, gtol=1e-9, maxiter=1000, maxls=60))
        q = result.x.copy()
        primary, delay = reported_locations(q, 64)
        candidates.append(dict(theta=q.tolist(), loglik=-float(result.fun) * scale,
                               primary=primary, delay=delay, optimizer_success=bool(result.success)))
    best = max(candidates, key=lambda v: v['loglik'])
    return best, dict(improvement=best['loglik'] - fit['loglik'], reference_optimizer_success=best['optimizer_success'])


def main():
    start = time.perf_counter()
    p = study.check()
    rows, refinements, changes, inputs = [], [], [], []
    checked = 0
    for case in study.cases(p):
        row = await_row(case)
        rows.append(row)
        warned = (not row['safe_null']['optimizer_success'] or not row['observed_full_null']['optimizer_success'] or
                  any(not v['optimizer_success'] for v in row['localization_fits'].values()))
        if not warned:
            continue
        checked += 1
        g = p['regimes'][case['regime']]
        obs, _, _, _, _, dh = d.previous.old.base.data(case, p)
        assert dh == row['data_sha256']
        s, e = numeric_statistics(obs, g['cycles'], 64, outward_safe_lag(g['td'], g['te']))
        pilot = row['safe_null']
        if not pilot['optimizer_success']:
            candidate = stronger_null(s, e, np.array(pilot['theta']), pilot)
            inputs.append(dict(case=case['id'], kind='safe_null', improvement=candidate['loglik'] - pilot['loglik']))
            pilot = candidate
        geo = d.kernel.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
        full = d.candidate.refit_full_null(s, 1.5, pilot, geo)
        if not row['observed_full_null']['optimizer_success'] and g['td'] < g['te']:
            full, record = refine_full(s, pilot, full, obs, g)
            inputs.append(dict(case=case['id'], kind='full_null', **record))
        safe_stat = fine_scan(s, e, 1.5, fit=pilot)
        full_stat = full_statistics(s, e, pilot, full, geo, case['regime'])[d.PRIMARY]
        safe = localize(s, e, 1.5, pilot, safe_stat['delay'])
        safe, record = refine_safe(safe, safe_stat['delay'], pilot, s, e)
        refinements.append(dict(case=case['id'], kind='safe', **record))
        if g['td'] >= g['te']:
            fitted = safe
        else:
            fitted = localize(s, e, 1.5, full, full_stat['delay'], geo)
            rg = reference.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
            fitted, record = refine_component(fitted, full_stat['delay'], full, s, rg, case['id'])
            refinements.append(dict(case=case['id'], kind='full', **record))
        hypothetical = dict(row, samples=dict(row['samples']))
        for method, fit in [('localized_safe_score_common', safe), ('localized_safe_score_original', safe),
                             (study.PRIMARY, fitted)]:
            # Parent threshold-decision stability is required below. Thus any
            # hypothetical parent p-value changes that stay within the same
            # three decision regions cannot affect this location-only check.
            hypothetical['samples'][method] = dict(row['samples'][method], primary=fit['primary'], delay=fit['delay'])
            for alpha in [.005, .01, .02]:
                if independent_endpoint(row, method, alpha) != independent_endpoint(hypothetical, method, alpha):
                    changes.append(dict(case=case['id'], method=method, alpha=alpha,
                                        frozen_primary=row['samples'][method]['primary'],
                                        frozen_delay=row['samples'][method]['delay'],
                                        hypothetical_primary=fit['primary'], hypothetical_delay=fit['delay']))
        print('Localization warning case checked', case['id'], flush=True)
    while not (study.OUT / 'parent_failure_audit.json').exists():
        time.sleep(1)
    parent = json.loads((study.OUT / 'parent_failure_audit.json').read_text())
    assert rows == study.read()
    completion = json.loads((study.OUT / 'completion.json').read_text())
    assert completion['records_sha256'] == parent['records_sha256'] == d.sha(study.OUT / 'records.sqlite3')
    source_warnings = [row['case']['id'] for row in rows if any(
        not v['optimizer_success'] for v in row['localization_fits'].values())]
    assert source_warnings == json.loads((study.OUT / 'analysis.json').read_text())['localization_warning_cases']
    passed = parent['status'] == 'PASS' and not parent['operatingpoint_endpoint_changes'] and not changes
    result = dict(status='PASS' if passed else 'MATERIAL_NUMERICAL_SENSITIVITY',
                  parent12_warning_audit_status=parent['status'], parent12_warning_draws=parent['all_warning_draws_regenerated'],
                  observed_null_or_localizer_warning_cases_checked=checked,
                  recorded_localization_warning_cases=source_warnings, stricter_localizer_refinements=refinements,
                  warned_null_input_refinements=inputs, localized_operatingpoint_changes=changes,
                  records_sha256=completion['records_sha256'], parent_failure_audit_sha256=d.sha(study.OUT / 'parent_failure_audit.json'),
                  auditor_source_sha256=d.sha(__file__), seconds=time.perf_counter() - start,
                  scope='Parent12calibration/optimizerwarningsandobservedlocalizationwarnings '
                        'separatelychecked. Unchangedrankteststatistic/pvalueidentitiesjustify '
                        'parentdeclarationstabilityforlocalizedpipelines. All15endpointschecked '
                        'at3alphas. No recordrewrite,globalML,physical,uniformriskorQ1claim.')
    (study.OUT / 'failure_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
