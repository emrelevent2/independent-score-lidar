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
import score_peak_localization_development as study

P = parent.ROOT / 'LOCALIZATION_POSTFAILURE_FULL_JOINT_DIAGNOSIS_PROTOCOL.json'
OUT = parent.ROOT / 'results/localization_postfailure_reference_v1'
DB = OUT / 'equal_information_joint_reference_v2.sqlite3'
STATE = OUT / 'equal_information_joint_run_state_v2.json'


def freeze():
    if P.exists():
        raise FileExistsError('Conditional control manifest already frozen')
    study.check()
    assert json.loads((study.OUT / 'analysis.json').read_text())['preliminary_rule_passed'] is False, 'This is explicitly after the failed complete development, not a confirmation'
    complete = json.loads((study.OUT / 'completion.json').read_text())
    assert complete['cases'] == 640 and complete['records_sha256'] == parent.sha(study.OUT / 'records.sqlite3')
    assert all(json.loads((study.OUT / name).read_text())['status'] == 'PASS' for name in
               ['integrity.json', 'parent_failure_audit.json', 'failure_audit.json'])
    assert not DB.exists(), 'No control efficacy data before manifest'
    meta = parent.ROOT / 'SCORE_PEAK_LOCALIZATION_METADATA_ADDENDUM.json'
    assert json.loads(meta.read_text())['protocol_sha256'] == parent.sha(study.P)
    record = dict(status='FROZEN_AFTER_FAILED_LOCALIZATION_GATE_CLOSED_RESEARCH_DIAGNOSIS_ONLY', source_records_sha256=parent.sha(study.OUT / 'records.sqlite3'),
                  study_protocol_sha256=parent.sha(study.P), metadata_addendum_sha256=parent.sha(meta),
                  frozen_numerical_worker_addendum_sha256=parent.sha(algorithm.P),
                  sources={'localization_postfailure_full_joint_diagnosis.py': parent.sha(__file__)},
                  reason='Closedfreshdevelopmentyielded203/1/4vsraw185/2/5andmatched186/1/5, '
                         'butthelockedall-comparatorgateFAILEDduetonullcounts2--3insomecontrols. '
                         'Computebothfull-LRreferencesasfurtherresearchtoinformthenextproposal. '
                         'No reinterpretationofthefailedgateandnoautomaticauthorizationofconfirmation.',
                  execution='AFTERFAILEDmainruleandcompletedmainintegrity,parentwarning, '
                            'localizerwarningauditsPASS. BOTHlegacyandstrongfull-LRreferences '
                            'onALL640CLOSEDqueries/ALL199commonpaths,8workers,threads1. '
                            'PassfreshrowandfreshprotocoltothefrozenV2worker. '
                            'No old-study run/read/export function is called.',
                  reporting='Preserveeveryfrozenmainrow. SeparateSQLiteinsideNEWstudyoutput. '
                            'Mainrecordshash/protocol/manifestpinnedinresumestatebeforewrites. '
                            'No extra export to old output paths.',
                  rule='PointcontrastsaredescriptiveCLOSEDdiagnosticsonly. '
                       'Theearlier15-methodprimarygateFAILEDandisUNCHANGED. '
                       'No result here permits the locked3200confirmation or a superiority '
                       'claim. Both references and independent integrity/warning audits '
                       'coverALL640/199,notafavorable subset. No globalML,physicalorQ1claim.')
    P.write_text(json.dumps(record, indent=2))
    print('Frozen output-isolated control manifest', parent.sha(P), flush=True)


def check():
    p = study.check()
    manifest = json.loads(P.read_text())
    assert manifest['study_protocol_sha256'] == parent.sha(study.P)
    assert manifest['metadata_addendum_sha256'] == parent.sha(parent.ROOT / 'SCORE_PEAK_LOCALIZATION_METADATA_ADDENDUM.json')
    assert manifest['frozen_numerical_worker_addendum_sha256'] == parent.sha(algorithm.P)
    for name, digest in manifest['sources'].items():
        assert parent.sha(parent.ROOT / 'src' / name) == digest
    assert DB.resolve().parent == OUT.resolve()
    assert manifest['source_records_sha256'] == parent.sha(study.OUT / 'records.sqlite3')
    assert DB.resolve().parent != study.OUT.resolve()
    assert DB.resolve().parent != parent.OUT.resolve()
    return p


def execute(row):
    return algorithm.worker(row, check())


def read():
    with sqlite3.connect(f'file:{DB.as_posix()}?mode=ro', uri=True) as db:
        return [json.loads(r[0]) for r in db.execute('SELECT payload FROM records ORDER BY rowid')]


def run():
    p = check()
    a = json.loads((study.OUT / 'analysis.json').read_text())
    assert a['preliminary_rule_passed'] is False, 'This diagnosis cannot relabel the failed gate as passed'
    OUT.mkdir(parents=True, exist_ok=True)
    assert all(json.loads((study.OUT / name).read_text())['status'] == 'PASS' for name in
               ['integrity.json', 'parent_failure_audit.json', 'failure_audit.json'])
    assert all(os.environ.get(k) == v for k, v in p['execution']['threads'].items())
    rows = study.read()
    assert len(rows) == 640 and [r['case'] for r in rows] == list(study.cases(p))
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
                    print('Fresh-localizationfull-LR', len(done) + i + 1, '/640 elapsed_s',
                          round(time.perf_counter() - start, 1), flush=True)
    controls = read()
    assert [r['case'] for r in controls] == [r['case'] for r in rows]
    assert before == parent.sha(study.OUT / 'records.sqlite3')
    complete = dict(status='COMPLETE', cases=640, main_records_sha256=before,
                    reference_records_sha256=parent.sha(DB), study_protocol_sha256=parent.sha(study.P),
                    control_manifest_sha256=parent.sha(P), seconds_this_invocation=time.perf_counter() - start)
    (OUT / 'equal_information_joint_completion_v2.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)
    analyze()


def analyze():
    check()
    main, controls = study.read(), read()
    assert len(main) == len(controls) == 640
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
    paths = [OUT / n for n in ['postfailure_full_joint_integrity.json',
                                    'postfailure_full_joint_failure_audit.json']]
    audited = all(path.exists() and json.loads(path.read_text())['status'] == 'PASS' for path in paths)
    passed = all(all(v.values()) for v in rules.values())
    result = dict(status='POSTFAILURE_CLOSED_REFERENCE_DIAGNOSIS_NOT_CONFIRMATION',
                  reference_totals=totals, primary_totals=primary, primary_rules=rules,
                  all_point_rules_pass=passed, control_audits_pass=audited,
                  continue_to_fresh_confirmation=False, frozen_development_rule_failed=True, closed_diagnostic_only=True,
                  scope='BOTHfrozenfull-LRreferencesonALLCLOSED640/199paths. '
                        'POSTFAILUREresearchdiagnosis,notconfirmationorpermissionto '
                        'changefailedgates. NotglobalML/physical/uniformrisk/Q1certification.')
    (OUT / 'equal_information_joint_analysis_v2.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    {'freeze': freeze, 'run': run, 'analyze': analyze}[sys.argv[1]]()
