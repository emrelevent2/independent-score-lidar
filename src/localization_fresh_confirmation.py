"""Locked 3200-case confirmation, conditional on all preceding development gates.

Reuses the frozen physical/statistical workers without editing them. Both
full-LR references retain separate output storage and repeat every adaptive
step. This module never marks a manuscript or the overall goal complete.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sqlite3
import sys
import time

import score_peak_localization_development as previous
import localization_full_joint_control as development_control
from localization_confirmation_statistics import primary_effects

d = previous.d
ROOT = d.ROOT
DESIGN = ROOT / 'LOCALIZATION_FRESH_CONFIRMATION_DESIGN.json'
STATISTICS = ROOT / 'LOCALIZATION_CONFIRMATION_STATISTICS_MANIFEST.json'
P = ROOT / 'LOCALIZATION_FRESH_CONFIRMATION_PROTOCOL.json'
OUT = ROOT / 'results/localization_fresh_confirmation_v1'
METHODS = previous.METHODS
PRIMARY = previous.PRIMARY
TOTAL = 3200
append_localizations = previous.append_localizations


def freeze():
    if P.exists() or (OUT / 'records.sqlite3').exists():
        raise FileExistsError('Confirmation source/protocol already frozen or data present')
    original = development_control.check()
    assert not (previous.OUT / 'analysis.json').exists(), 'Pin wrapper before fresh-development aggregate outcomes'
    design = json.loads(DESIGN.read_text())
    statistics = json.loads(STATISTICS.read_text())
    assert design['development_protocol_sha256'] == d.sha(previous.P)
    assert design['full_joint_control_manifest_sha256'] == d.sha(development_control.P)
    assert statistics['design_sha256'] == d.sha(DESIGN)
    for name, digest in statistics['sources'].items():
        assert d.sha(ROOT / name) == digest
    settings = dict(original['settings'], seed_base=design['design']['query_seed_base'],
                    positive_per_family_regime=80, null_per_family_regime=160)
    sources = ['localization_fresh_confirmation.py', 'localization_confirmation_statistics.py',
               'test_localization_confirmation_statistics.py']
    p = {key: original[key] for key in ['regimes', 'families', 'environment', 'execution']}
    p.update(status='FROZEN_BEFORE_CONFIRMATION_DATA_AND_BEFORE_DEVELOPMENT_AGGREGATE_OUTCOMES',
             settings=settings, methods=METHODS, primary_method=PRIMARY, total_cases=TOTAL,
             bootstrap_seed_offset_from_parent=design['design']['bootstrap_seed_base'] - 120000000000,
             sources={name: d.sha(ROOT / 'src' / name) for name in sources},
             design_sha256=d.sha(DESIGN), statistics_manifest_sha256=d.sha(STATISTICS),
             development_protocol_sha256=d.sha(previous.P),
             development_control_manifest_sha256=d.sha(development_control.P),
             algorithm='IDENTICALfrozen15-pipelineworkerandone-startobservedlocalization. '
                       'Onlycasecounts/queryseeds/MCseedschangeaslockeddesign. No retraining, '
                       'alpha/tolerance/IRF/recovery/acquisition/contrastchanges.',
             analysis='All15corepipelinesat3200cases. Developmentpointrulesretained, plus '
                      'all4positiveone-sided98.75percentpaired-stratifiedbootstrapbounds '
                      'againstBOTHmatchedlocalizedcensoredscores. Everymain/statistical '
                      'warningauditrequired. BOTHfull-LRreferencesonALL3200/199commonpaths '
                      'andallcontrolauditsrequiredforanysuccessclaim.',
             execution_gate='Currentfresh640developmentprimaryandallmainauditsPASS; '
                            'BOTHdevelopmentfull-LRcomparisonsandindependentauditsPASS. '
                            'Confirmationcorefirst;expensivefull-LRstageconditionalon '
                            'confirmedcorepoint+uncertainty+numericalPASS. Deferralafter '
                            'alreadyfailedconfirmationcorecannotauthorizeasuccessclaim.',
             scope='Lockedprospectiveconfirmationdesign. Coreorcontrolpointpassalone '
                   'isnotmanuscriptsubmissionreadiness,physicalvalidationorQ1certification.')
    generated = list(cases(p))
    assert len(generated) == TOTAL and sum(c['positive'] for c in generated) == 1280
    assert generated[0]['seed'] == 37419865021 and generated[0]['bootstrap_seed'] == 650000000000
    P.write_text(json.dumps(p, indent=2))
    print('Frozen future confirmation wrapper', d.sha(P), flush=True)


def check():
    development_control.check()
    p = json.loads(P.read_text())
    assert p['development_protocol_sha256'] == d.sha(previous.P)
    assert p['development_control_manifest_sha256'] == d.sha(development_control.P)
    assert p['design_sha256'] == d.sha(DESIGN)
    assert p['statistics_manifest_sha256'] == d.sha(STATISTICS)
    for name, digest in p['sources'].items():
        assert d.sha(ROOT / 'src' / name) == digest, name
    design = json.loads(DESIGN.read_text())['design']
    assert all(p['settings'][name] == design[name] for name in
               ['positive_per_family_regime', 'null_per_family_regime', 'alpha'])
    assert p['settings']['tolerance'] == design['location_tolerance_bins'] == 2
    assert p['settings']['seed_base'] == design['query_seed_base']
    assert p['settings']['bootstrap_replicates'] == design['bootstrap_replicates'] == 199
    return p


def cases(p):
    return previous.cases(p)


def worker(case, p):
    return previous.worker(case, p)


def execute(case):
    return worker(case, check())


def read():
    with sqlite3.connect(f'file:{(OUT / "records.sqlite3").as_posix()}?mode=ro', uri=True) as db:
        return [json.loads(r[0]) for r in db.execute('SELECT payload FROM records ORDER BY rowid')]


totals = previous.totals
rules = previous.rules


def verify_development_gate():
    check()
    main = previous.OUT
    completion = json.loads((main / 'completion.json').read_text())
    a = json.loads((main / 'analysis.json').read_text())
    assert completion['cases'] == 640 and a['preliminary_rule_passed']
    assert completion['protocol_sha256'] == d.sha(previous.P)
    assert completion['records_sha256'] == d.sha(main / 'records.sqlite3')
    for name in ['integrity.json', 'parent_failure_audit.json', 'failure_audit.json']:
        audit = json.loads((main / name).read_text())
        assert audit['status'] == 'PASS' and audit['records_sha256'] == completion['records_sha256']
    control_completion = json.loads((main / 'equal_information_joint_completion_v2.json').read_text())
    assert control_completion['main_records_sha256'] == completion['records_sha256']
    assert control_completion['reference_records_sha256'] == d.sha(development_control.DB)
    control_analysis = json.loads((main / 'equal_information_joint_analysis_v2.json').read_text())
    assert control_analysis['all_point_rules_pass'] and control_analysis['control_audits_pass']
    assert control_analysis['continue_to_fresh_confirmation']
    for name in ['localization_full_joint_integrity_v2.json', 'localization_full_joint_failure_audit_v2.json']:
        audit = json.loads((main / name).read_text())
        assert audit['status'] == 'PASS'
        assert audit['reference_records_sha256'] == control_completion['reference_records_sha256']
        assert audit['main_records_sha256'] == completion['records_sha256']
    return dict(development_main_records_sha256=completion['records_sha256'],
                development_reference_records_sha256=control_completion['reference_records_sha256'])


def run():
    p = check()
    entry = verify_development_gate()
    assert all(os.environ.get(k) == v for k, v in p['execution']['threads'].items())
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / 'run_state.json'
    state = dict(entry, confirmation_protocol_sha256=d.sha(P))
    if state_path.exists():
        assert json.loads(state_path.read_text()) == state
    else:
        assert not (OUT / 'records.sqlite3').exists()
        state_path.write_text(json.dumps(state, indent=2))
    with sqlite3.connect(OUT / 'records.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done = dict(db.execute('SELECT id,payload FROM records'))
        all_cases = list(cases(p))
        expected = {c['id']: c for c in all_cases}
        assert set(done).issubset(expected)
        for key, value in done.items():
            assert json.loads(value)['case'] == expected[key]
        remaining = [case for case in all_cases if case['id'] not in done]
        start = time.perf_counter()
        print('Confirmation8workers existing', len(done), 'remaining', len(remaining), flush=True)
        with ProcessPoolExecutor(max_workers=8) as pool:
            for i, row in enumerate(pool.map(execute, remaining, chunksize=1)):
                db.execute('INSERT INTO records VALUES(?,?)', (row['case']['id'], json.dumps(row)))
                db.commit()
                if (len(done) + i + 1) % 8 == 0:
                    print('Confirmation', len(done) + i + 1, '/3200 elapsed_s', round(time.perf_counter() - start, 1), flush=True)
    assert [r['case'] for r in read()] == list(cases(p))
    complete = dict(status='COMPLETE', cases=TOTAL, protocol_sha256=d.sha(P),
                    records_sha256=d.sha(OUT / 'records.sqlite3'), seconds_this_invocation=time.perf_counter() - start)
    (OUT / 'completion.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)
    analyze()


def analyze():
    p = check()
    rows = read()
    assert len(rows) == TOTAL and [r['case'] for r in rows] == list(cases(p))
    t = totals(rows, .01)
    point_rules = rules(t)
    point_pass = all(all(value.values()) for value in point_rules.values())
    uncertainty = primary_effects(rows, json.loads(DESIGN.read_text()))
    interval_pass = uncertainty['all4_positive_lower_bounds']
    paths = [OUT / name for name in ['integrity.json', 'parent_failure_audit.json',
                                     'failure_audit.json', 'uncertainty_audit.json']]
    audited = all(path.exists() and json.loads(path.read_text())['status'] == 'PASS' for path in paths)
    passed = point_pass and interval_pass
    result = dict(status='CORE_POINT_AND_UNCERTAINTY_PASS_ALL_AUDITS_AND_FULL_LR_PENDING' if passed else 'CONFIRMATION_CORE_JOINT_CRITERION_FAILED',
                  totals=t, preliminary_rule=point_rules, preliminary_point_rule_passed=point_pass,
                  primary_uncertainty=uncertainty, primary_uncertainty_rule_passed=interval_pass,
                  preliminary_rule_passed=passed, all_main_audits_pass=audited,
                  continue_to_full_LR_controls=passed and audited,
                  continue_to_submission=False,
                  descriptive_operating_points={str(a): totals(rows, a) for a in [.005, .01, .02]},
                  localization_warning_cases=[r['case']['id'] for r in rows if any(
                      not v['optimizer_success'] for v in r['localization_fits'].values())],
                  scope=p['scope'])
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    {'freeze': freeze, 'run': run, 'analyze': analyze}[sys.argv[1]]()
