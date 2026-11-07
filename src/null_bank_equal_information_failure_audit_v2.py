"""Read-only sensitivity audit of every recorded stronger-control warning.

Strict quadrature refinements are hypothetical checks, never replacements in
the frozen efficacy tables. A changed decision prevents confirmation.
"""
import json
import time

import numpy as np

import null_bank_development as d
import null_bank_equal_information_control_v2 as control
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from context_agreement_features import numeric_statistics
from deadtime_scan import fit_null
from deadtime_path_bootstrap import refit_from_start, pvalue
from fast_two_return_fit import reports
from full_information_joint_reference_v2 import stronger_joint
from null_bank_control_reference import refine_component
from null_bank_score import refit_full_null, full_likelihood_ratio
from null_bank_equal_information_integrity_v2 import wait_record
from null_bank_failure_audit import refine_full, refine_alternatives
from one_step_location_failure_audit import stronger_null
from outward_safety_lag import outward_safe_lag
from safe_deadtime_exposure import simulate_two_dead_times
from transported_joint_integrity import independent_endpoint


def corrected_pair(obs, counts, exposure, pilot, six, full, legacy, strong,
                   case, g, label):
    """Propagate warning refinements through both full-law references."""
    audit = []
    safe = pilot
    if not pilot['optimizer_success']:
        safe = stronger_null(counts, exposure, pilot['theta'], pilot)
        audit.append(dict(label=label + '_safe_null',
                          improvement=safe['loglik'] - pilot['loglik']))
        six = reports(counts, exposure, 1.5, safe,
                      dict(primary=0., secondary=0., two_pulse_loglr=0.))['fits']
    six, records = refine_alternatives(six, counts, exposure, safe, case['id'], label)
    audit.extend(dict(label=label + '_safe_alternative_' + str(v['component']), **{
        k: v[k] for k in ['improvement', 'reference_optimizer_success']}) for v in records)
    geo = d.kernel.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
    current_null = refit_full_null(counts, 1.5, safe, geo) if safe is not pilot else full
    if not full['optimizer_success'] and g['td'] < g['te']:
        current_null, record = refine_full(counts, safe, current_null, obs, g)
        audit.append(dict(label=label + '_full_null', **record))
    current_legacy = full_likelihood_ratio(counts, 1.5, current_null, six, geo)
    rg = reference.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
    # Check originally failed selected fits even if upstream refinements change
    # the start selector. Retain every feasible scientific solution as well.
    for tag, fit in [('original', legacy), ('recomputed', current_legacy)]:
        if not fit['optimizer_success']:
            centre = float(six[fit['selected_safe_component']]['theta'][4])
            refined, record = refine_component(fit, centre, current_null, counts, rg,
                                                label + '_legacy_' + tag)
            audit.append(record)
            if refined['loglik'] > current_legacy['loglik']:
                current_legacy = dict(current_legacy, **refined)
    current_strong = stronger_joint(counts, 1.5, current_null, six, geo,
                                   d.bank(), case['regime'], current_legacy)
    for tag, value in [('original', strong), ('recomputed', current_strong)]:
        for index, fit in enumerate(value['additional_refinements']):
            if fit['optimizer_success']:
                continue
            refined, record = refine_component(fit, float(fit['start']), current_null,
                                                counts, rg, label + '_strong_' + tag + '_' + str(index))
            audit.append(record)
            if refined['loglik'] > current_strong['loglik']:
                current_strong = dict(current_strong, **refined)
    out = {}
    for name, fit in [('legacy', current_legacy), ('strong', current_strong)]:
        out[name] = dict(score=max(0., 2 * (fit['loglik'] - current_null['loglik'])),
                         primary=float(fit['primary']), delay=float(fit['delay']))
    return out, audit


def run():
    start = time.perf_counter()
    p = control.check()
    assert json.loads((d.OUT / 'analysis.json').read_text())['preliminary_rule_passed']
    assert all(json.loads((d.OUT / n).read_text())['status'] == 'PASS'
               for n in ['integrity.json', 'failure_audit.json'])
    main_rows = d.read()
    checked = observed = 0
    refinements, changes, rows = [], [], []
    for main in main_rows:
        case = main['case']
        row = wait_record(case)
        rows.append(row)
        warning_draws = sorted(set(w['draw'] for w in row['legacy_bootstrap_warnings'] +
                                   row['stronger_bootstrap_warnings'] +
                                   [v for v in main['bootstrap_warnings'] if v['calibration'] == 'common']))
        observed_warning = (not main['safe_null']['optimizer_success'] or
                            not main['observed_full_null']['optimizer_success'] or
                            any(not f['optimizer_success'] for f in main['observed_alternative_fits'][:6]) or
                            not row['observed_full_joint']['optimizer_success'] or
                            not row['observed_stronger_joint']['optimizer_success'] or
                            any(not f['optimizer_success'] for f in row['observed_stronger_joint']['additional_refinements']))
        if not warning_draws and not observed_warning:
            continue
        g = p['regimes'][case['regime']]
        obs, _, _, _, _, dh = d.previous.old.base.data(case, p)
        assert dh == main['data_sha256']
        lag = outward_safe_lag(g['td'], g['te'])
        s, e = numeric_statistics(obs, g['cycles'], 64, lag)
        pilot = fit_null(s, e, 1.5)
        geo = d.kernel.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
        null = refit_full_null(s, 1.5, pilot, geo)
        samples = {m: dict(v, bootstrap_scores=list(v['bootstrap_scores'])) for m, v in row['samples'].items()}
        if observed_warning:
            observed += 1
            corrected, recs = corrected_pair(obs, s, e, pilot, main['observed_alternative_fits'][:6],
                                             null, row['observed_full_joint'], row['observed_stronger_joint'],
                                             case, g, 'observed')
            refinements.extend(dict(case=case['id'], **v) for v in recs)
            for m, v in corrected.items():
                samples[m].update(v)
        theta = np.array(main['calibration_null']['theta'])
        rate = rates_and_design(theta, 64, 1.5)[0]
        for draw in warning_draws:
            bo, _ = simulate_two_dead_times(rate, g['cycles'], g['td'], g['te'],
                                            case['bootstrap_seed'] + 1009 * draw)
            bs, be = numeric_statistics(bo, g['cycles'], 64, lag)
            bf = refit_from_start(bs, be, theta, 1.5, 'duration')
            bg = d.kernel.prepare_record(bo, g['cycles'] * 64, 64, g['td'], g['te'])
            nf = refit_full_null(bs, 1.5, bf, bg)
            six = reports(bs, be, 1.5, bf, dict(primary=0., secondary=0., two_pulse_loglr=0.))['fits']
            legacy = full_likelihood_ratio(bs, 1.5, nf, six, bg)
            strong = stronger_joint(bs, 1.5, nf, six, bg, d.bank(), case['regime'], legacy)
            assert legacy['score'] == row['samples']['legacy']['bootstrap_scores'][draw]
            assert strong['score'] == row['samples']['strong']['bootstrap_scores'][draw]
            corrected, recs = corrected_pair(bo, bs, be, bf, six, nf, legacy, strong,
                                             case, g, 'draw_' + str(draw))
            refinements.extend(dict(case=case['id'], **v) for v in recs)
            for m, v in corrected.items():
                samples[m]['bootstrap_scores'][draw] = v['score']
            checked += 1
        for v in samples.values():
            v['pvalue'] = pvalue(v['score'], v['bootstrap_scores'])
        original = dict(main, samples=dict(main['samples'], **row['samples']))
        hypothetical = dict(main, samples=dict(main['samples'], **samples))
        for m in samples:
            for alpha in [.005, .01, .02]:
                if independent_endpoint(original, m, alpha) != independent_endpoint(hypothetical, m, alpha):
                    changes.append(dict(case=case['id'], method=m, alpha=alpha,
                                        frozen_p=row['samples'][m]['pvalue'], hypothetical_p=samples[m]['pvalue']))
        print('Equal-information warnings checked', case['id'], 'draws', len(warning_draws), flush=True)
    while not (d.OUT / 'equal_information_joint_completion_v2.json').exists():
        time.sleep(1)
    assert rows == control.read()
    completion = json.loads((d.OUT / 'equal_information_joint_completion_v2.json').read_text())
    assert completion['reference_records_sha256'] == d.sha(d.OUT / 'equal_information_joint_reference_v2.sqlite3')
    control.check()
    result = dict(status='PASS' if not changes else 'MATERIAL_NUMERICAL_SENSITIVITY',
                  observed_warning_cases_checked=observed, distinct_warning_draws_regenerated=checked,
                  stricter_component_refinements=len(refinements),
                  max_loglik_improvement=max([v['improvement'] for v in refinements], default=0.),
                  refinements=refinements, operatingpoint_endpoint_changes=changes,
                  reference_records_sha256=completion['reference_records_sha256'],
                  main_records_sha256=completion['main_records_sha256'], seconds=time.perf_counter() - start,
                  scope='Unionofallrecordedcontrol/commonmainwarnings. IndependentGL32strictfullfits, '
                        'NumPysafealternativesandsafenullrefinements. Changesarehypothetical, '
                        'frozenrecordsunmodified. NotglobalML,physical,uniformriskorQ1certification.')
    (d.OUT / 'equal_information_joint_failure_audit_v2.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
