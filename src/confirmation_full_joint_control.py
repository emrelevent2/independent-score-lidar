"""Output-isolated conditional full-law references for the fresh localization study.

Delegates each numerical worker to the already frozen V2 algorithm, passing
the NEW protocol and NEW row explicitly. Does not invoke the old run/read
functions or change any primary statistic, location or calibration draw.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sqlite3
import sys
import time

import null_bank_development as parent
import null_bank_equal_information_control_v2 as algorithm
import localization_fresh_confirmation as study

P = parent.ROOT / 'CONFIRMATION_FULL_JOINT_CONTROL_MANIFEST.json'
DB = study.OUT / 'equal_information_joint_reference_v2.sqlite3'
STATE = study.OUT / 'equal_information_joint_run_state_v2.json'


def freeze():
    if P.exists():
        raise FileExistsError('Conditional control manifest already frozen')
    study.check()
    assert not (study.OUT / 'analysis.json').exists(), 'Freeze before any fresh-study aggregate outcomes'
    assert not (study.OUT / 'records.sqlite3').exists(), 'Pin confirmation controls before the first confirmation query'
    assert not DB.exists(), 'No control efficacy data before manifest'
    meta = study.DESIGN
    assert json.loads(meta.read_text())['development_protocol_sha256'] == parent.sha(study.previous.P)
    record = dict(status='FROZEN_CONDITIONAL_CONTROL_RUNNER_BEFORE_FRESH_AGGREGATE_OUTCOMES',
                  study_protocol_sha256=parent.sha(study.P), confirmation_design_sha256=parent.sha(meta),
                  frozen_numerical_worker_addendum_sha256=parent.sha(algorithm.P),
                  sources={'confirmation_full_joint_control.py': parent.sha(__file__)},
                  execution='Onlyafterlocked3200confirmationpointANDall4uncertaintylowerboundsPASS, '
                            'mainintegrity,parentwarning,localizerwarningANDuncertaintyauditsPASS. '
                            'BOTHlegacyandstrongfull-LRreferences '
                            'onALL3200NEWqueries/ALL199commonpaths,8workers,threads1. '
                            'PassfreshrowandfreshprotocoltothefrozenV2worker. '
                            'No old-study run/read/export function is called.',
                  reporting='Preserveeveryfrozenmainrow. SeparateSQLiteinsideNEWstudyoutput. '
                            'Mainrecordshash/protocol/manifestpinnedinresumestatebeforewrites. '
                            'No extra export to old output paths.',
                  rule='Localizedprimaryatleastcorrectandnomorewrong/nullthanBOTHfull-LR '
                       'referencesatfrozenalpha.01. Indepcontrolintegrityandwarningaudits '
                       'mandatorybeforelargerfreshconfirmation. No subgroup or weakercontrol '
                       'substitution;no globalML,physical,uniformriskorQ1claim.')
    P.write_text(json.dumps(record, indent=2))
    print('Frozen output-isolated control manifest', parent.sha(P), flush=True)


def check():
    p = study.check()
    manifest = json.loads(P.read_text())
    assert manifest['study_protocol_sha256'] == parent.sha(study.P)
    assert manifest['confirmation_design_sha256'] == parent.sha(study.DESIGN)
    assert manifest['frozen_numerical_worker_addendum_sha256'] == parent.sha(algorithm.P)
    for name, digest in manifest['sources'].items():
        assert parent.sha(parent.ROOT / 'src' / name) == digest
    assert DB.resolve().parent == study.OUT.resolve()
    assert DB.resolve().parent != parent.OUT.resolve()
    return p


def execute(row):
    return algorithm.worker(row, check())


def read():
    with sqlite3.connect(f'file:{DB.as_posix()}?mode=ro', uri=True) as db:
        return [json.loads(r[0]) for r in db.execute('SELECT payload FROM records ORDER BY rowid')]


def run():
    p = check()
    study.verify_development_gate()
    a = json.loads((study.OUT / 'analysis.json').read_text())
    assert a['preliminary_rule_passed'], 'Conditional reference follows a passed fresh primary only'
    assert all(json.loads((study.OUT / name).read_text())['status'] == 'PASS' for name in
               ['integrity.json', 'parent_failure_audit.json', 'failure_audit.json', 'uncertainty_audit.json'])
    assert all(os.environ.get(k) == v for k, v in p['execution']['threads'].items())
    rows = study.read()
    assert len(rows) == 3200 and [r['case'] for r in rows] == list(study.cases(p))
    before = parent.sha(study.OUT / 'records.sqlite3')
    state = dict(study_protocol_sha256=parent.sha(study.P), main_records_sha256=before,
                 control_manifest_sha256=parent.sha(P))
    if STATE.exists():
        assert json.loads(STATE.read_text()) == state
    else:
        assert not DB.exists(), 'Existing reference DB needs its original pinned resume state'
        STATE.write_text(json.dumps(state, indent=2))
    with sqlite3.connect(DB) as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done = dict(db.execute('SELECT id,payload FROM records'))
        fresh = {r['case']['id']: r for r in rows}
        assert set(done).issubset(fresh)
        for key, value in done.items():
            assert json.loads(value)['case'] == fresh[key]['case']
        remaining = [r for r in rows if r['case']['id'] not in done]
        start = time.perf_counter()
        print('Fresh-localizationfull-LR8workers existing', len(done), 'remaining', len(remaining), flush=True)
        with ProcessPoolExecutor(max_workers=8) as pool:
            for i, result in enumerate(pool.map(execute, remaining, chunksize=1)):
                db.execute('INSERT INTO records VALUES(?,?)', (result['case']['id'], json.dumps(result)))
                db.commit()
                if (len(done) + i + 1) % 8 == 0:
                    print('Fresh-localizationfull-LR', len(done) + i + 1, '/3200 elapsed_s',
                          round(time.perf_counter() - start, 1), flush=True)
    controls = read()
    assert [r['case'] for r in controls] == [r['case'] for r in rows]
    assert before == parent.sha(study.OUT / 'records.sqlite3')
    complete = dict(status='COMPLETE', cases=3200, main_records_sha256=before,
                    reference_records_sha256=parent.sha(DB), study_protocol_sha256=parent.sha(study.P),
                    control_manifest_sha256=parent.sha(P), seconds_this_invocation=time.perf_counter() - start)
    (study.OUT / 'equal_information_joint_completion_v2.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)
    analyze()


def analyze():
    check()
    main, controls = study.read(), read()
    assert len(main) == len(controls) == 3200
    totals = {m: dict(correct=0, wrong=0, false=0, positive=0, null=0) for m in ['legacy', 'strong']}
    for row, ref in zip(main, controls):
        assert row['case'] == ref['case']
        augmented = dict(row, samples=dict(row['samples'], **ref['samples']))
        for method in totals:
            for k, value in parent.previous.old.endpoint(augmented, method, .01).items():
                totals[method][k] += value
    primary = study.totals(main, .01)[study.PRIMARY]
    rules = {m: dict(at_least_correct=primary['correct'] >= v['correct'],
                     no_more_wrong=primary['wrong'] <= v['wrong'], no_more_null=primary['false'] <= v['false'])
             for m, v in totals.items()}
    main_completion = json.loads((study.OUT / 'completion.json').read_text())
    reference_completion = json.loads((study.OUT / 'equal_information_joint_completion_v2.json').read_text())
    main_hash = parent.sha(study.OUT / 'records.sqlite3')
    reference_hash = parent.sha(DB)
    assert main_completion['records_sha256'] == reference_completion['main_records_sha256'] == main_hash
    assert reference_completion['reference_records_sha256'] == reference_hash
    paths = [study.OUT / n for n in ['confirmation_full_joint_integrity.json',
                                    'confirmation_full_joint_failure_audit.json']]
    audited = all(path.exists() and json.loads(path.read_text())['status'] == 'PASS'
                  and json.loads(path.read_text())['main_records_sha256'] == main_hash
                  and json.loads(path.read_text())['reference_records_sha256'] == reference_hash for path in paths)
    core = json.loads((study.OUT / 'analysis.json').read_text())
    main_paths = [study.OUT / n for n in ['integrity.json', 'parent_failure_audit.json',
                                         'failure_audit.json', 'uncertainty_audit.json']]
    main_audited = all(path.exists() and json.loads(path.read_text())['status'] == 'PASS'
                       and json.loads(path.read_text())['records_sha256'] == main_hash for path in main_paths)
    passed = all(all(v.values()) for v in rules.values())
    result = dict(status='FULL_LR_POINT_COMPARISON_REQUIRES_INDEPENDENT_AUDITS',
                  reference_totals=totals, primary_totals=primary, primary_rules=rules,
                  all_point_rules_pass=passed, control_audits_pass=audited,
                  main_audits_bound_to_current_records=main_audited,
                  confirmation_joint_evidence_verified=passed and audited and main_audited and core['preliminary_rule_passed'],
                  main_records_sha256=main_hash, reference_records_sha256=reference_hash,
                  scope='BOTHfrozenfull-LRreferencesonALLfresh3200/199paths. '
                        'NotglobalML/physical/uniformrisk/Q1certification. '
                        'Thisisthelockedlargerconfirmation;articleandartifactverificationstillrequired.')
    (study.OUT / 'equal_information_joint_analysis_v2.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    {'freeze': freeze, 'run': run, 'analyze': analyze}[sys.argv[1]]()
