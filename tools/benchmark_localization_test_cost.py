"""Prospective engineering cost of calibrated testing, no efficacy endpoints.

All original regimes and acquisition budgets. Every draw repeats the full
null refit and relevant statistic. Shared path preparation is charged to both
methods. Timings are warm single-process measurements under concurrent load.
"""
from pathlib import Path
import json
import sys
import time
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import localization_full_joint_control as control
import null_bank_development as d
import score_peak_localization_development as study
from clock_pulse_likelihood import rates_and_design
from context_agreement_features import numeric_statistics
from deadtime_scan import fit_null
from deadtime_path_bootstrap import refit_from_start
from fast_two_return_fit import reports
from full_information_joint_reference_v2 import stronger_joint
from null_bank_score import primary_statistic, refit_full_null, full_likelihood_ratio
from outward_safety_lag import outward_safe_lag
from safe_deadtime_exposure import simulate_two_dead_times
from score_peak_localization import localize

P = ROOT / 'LOCALIZATION_TEST_RUNTIME_PROTOCOL.json'
OUT = ROOT / 'results/localization_test_runtime_v1.json'


def freeze():
    control.check()
    if P.exists() or OUT.exists():
        raise FileExistsError('Engineering cost definition already exists')
    p = dict(status='FROZEN_ENGINEERING_ONLY_NO_EFFICACY_ENDPOINTS', source_sha256=d.sha(__file__),
             study_protocol_sha256=d.sha(study.P), control_manifest_sha256=d.sha(control.P),
             query_seed_base=5938172401, draw_seed_base=6048713927, bootstrap_replicates=199,
             regimes='All4originalregimes,originalnominalrateparameters,knownGaussianbinwidth1.5, '
                     'originalcycles/continuoustwo-recoveryphysics,primary15.3andnominalsecondarygap.',
             design='Fourindependentengineeringqueries,NOTfresh640cohort. One fullbank score plus '
                    'oneobservedfullalternativelocalization;B199completebootstrapscorepasses. '
                    'Reference:fullnull,6safealternativefits,legacyfullfitandadditionalfullscore '
                    'starts oneverydraw. Samequery3startfullnullgenerator/sharedphysicalpaths. '
                    'No pvalues,rejectionlabels,TP/FPorselectedoperatingpointscomputed/stored.',
             timing='Warmoneprocessunderconcurrentload. Sharedsimulation/exposure/pilot/geometry '
                    'preparationmeasuredonceandchargedtobothmethods;eachdoesownfullnullrefit. '
                    'Observedquerypreparation,generatorandlocalization/referencesincluded. '
                    'Notisolatedend-to-endlatency,multicorescaling,FPGAorutilityclaim.')
    P.write_text(json.dumps(p, indent=2))
    print('Frozen engineering test-cost protocol', d.sha(P), flush=True)


def check():
    p = json.loads(P.read_text())
    control.check()
    assert p['source_sha256'] == d.sha(__file__)
    assert p['study_protocol_sha256'] == d.sha(study.P)
    assert p['control_manifest_sha256'] == d.sha(control.P)
    return p


def reference_statistic(s, e, safe, geo, regime):
    null = refit_full_null(s, 1.5, safe, geo)
    six = reports(s, e, 1.5, safe, dict(primary=0., secondary=0., two_pulse_loglr=0.))['fits']
    legacy = full_likelihood_ratio(s, 1.5, null, six, geo)
    strong = stronger_joint(s, 1.5, null, six, geo, d.bank(), regime, legacy)
    assert strong['loglik'] >= legacy['loglik'] - 1e-9
    return null, strong


def run():
    p = check()
    source = study.check()
    assert all(os.environ.get(k) == v for k, v in source['execution']['threads'].items())
    if OUT.exists():
        raise FileExistsError('Engineering test-cost result already complete')
    groups = []
    for gi, g in enumerate(source['regimes']):
        rate = rates_and_design([g['a'], 15.3, g['b'], g['c'], 15.3 + g['gap']], 64, 1.5)[0]
        lag = outward_safe_lag(g['td'], g['te'])
        # Compile/warm the real numerical paths on a separate engineering record.
        warm, _ = simulate_two_dead_times(rate, g['cycles'], g['td'], g['te'], p['query_seed_base'] + 1000003 * gi + 17)
        ws, we = numeric_statistics(warm, g['cycles'], 64, lag)
        wf = fit_null(ws, we, 1.5)
        wg = d.kernel.prepare_record(warm, g['cycles'] * 64, 64, g['td'], g['te'])
        stat, null, _ = primary_statistic(ws, we, 1.5, wf, wg, d.bank(), gi)
        localize(ws, we, 1.5, null, stat['delay'], wg)
        reference_statistic(ws, we, wf, wg, gi)
        start = time.perf_counter()
        obs, _ = simulate_two_dead_times(rate, g['cycles'], g['td'], g['te'], p['query_seed_base'] + 1000003 * gi)
        s, e = numeric_statistics(obs, g['cycles'], 64, lag)
        safe = fit_null(s, e, 1.5)
        geo = d.kernel.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
        generator = d.previous.fit_full_null(s, e, 1.5, safe, geo)
        common_query = time.perf_counter() - start
        start = time.perf_counter()
        stat, null, _ = primary_statistic(s, e, 1.5, safe, geo, d.bank(), gi)
        localized = localize(s, e, 1.5, null, stat['delay'], geo)
        primary_observed = time.perf_counter() - start
        start = time.perf_counter()
        _, fitted = reference_statistic(s, e, safe, geo, gi)
        reference_observed = time.perf_counter() - start
        null_rate = rates_and_design(generator['theta'], 64, 1.5)[0]
        records = []
        warnings = dict(primary_null=0, reference_null=0, reference_selected=0)
        for draw in range(p['bootstrap_replicates']):
            start = time.perf_counter()
            bo, _ = simulate_two_dead_times(null_rate, g['cycles'], g['td'], g['te'],
                                            p['draw_seed_base'] + 1000003 * gi + 1009 * draw)
            bs, be = numeric_statistics(bo, g['cycles'], 64, lag)
            bf = refit_from_start(bs, be, generator['theta'], 1.5, 'duration')
            bg = d.kernel.prepare_record(bo, g['cycles'] * 64, 64, g['td'], g['te'])
            common = time.perf_counter() - start
            start = time.perf_counter()
            _, nf, _ = primary_statistic(bs, be, 1.5, bf, bg, d.bank(), gi)
            primary_seconds = time.perf_counter() - start
            start = time.perf_counter()
            rn, ref = reference_statistic(bs, be, bf, bg, gi)
            reference_seconds = time.perf_counter() - start
            assert d.json_fit(nf) == d.json_fit(rn)
            warnings['primary_null'] += not nf['optimizer_success']
            warnings['reference_null'] += not rn['optimizer_success']
            warnings['reference_selected'] += not ref['optimizer_success']
            records.append(dict(common_preparation_seconds=common,
                                primary_only_seconds=primary_seconds, reference_only_seconds=reference_seconds))
            if (draw + 1) % 25 == 0:
                print('Engineering test-cost', g['name'], draw + 1, '/199', flush=True)
        primary_total = common_query + primary_observed + sum(v['common_preparation_seconds'] + v['primary_only_seconds'] for v in records)
        reference_total = common_query + reference_observed + sum(v['common_preparation_seconds'] + v['reference_only_seconds'] for v in records)
        group = dict(regime=g['name'], bootstrap_replicates=199,
                     common_observed_query_seconds=common_query,
                     primary_observed_statistic_and_localization_seconds=primary_observed,
                     reference_observed_statistic_seconds=reference_observed,
                     primary_total_charged_seconds=primary_total, reference_total_charged_seconds=reference_total,
                     reference_to_primary_charged_ratio=reference_total / primary_total,
                     bootstrap_component_measurements=records, optimizer_warning_counts=warnings,
                     observed_localization_optimizer_success=localized['optimizer_success'])
        groups.append(group)
        print('Engineering test-cost complete', g['name'], 'chargedseconds', primary_total, reference_total, flush=True)
    check()
    OUT.write_text(json.dumps(dict(status='ENGINEERING_COST_ONLY_COMPLETE', groups=groups,
                                   protocol_sha256=d.sha(P), scope=p['timing']), indent=2))
    print('Engineering complete', OUT, flush=True)


if __name__ == '__main__':
    {'freeze': freeze, 'run': run}[sys.argv[1]]()
