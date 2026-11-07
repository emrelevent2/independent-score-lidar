"""New frozen detect-then-localize development; old scientific records intact.

The score/rank test is exactly the frozen independent-bank algorithm. One
observed-record alternative fit changes only reported locations. Matched
censored-score controls receive the same localization opportunity. Because
the statistic is unchanged, localization is not fitted inside rank draws.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import os
import sqlite3
import sys
import time

import null_bank_development as d
import null_bank_equal_information_control_v2 as strong_control
from context_agreement_features import numeric_statistics
from outward_safety_lag import outward_safe_lag
from score_peak_localization import localize

P = d.ROOT / 'SCORE_PEAK_LOCALIZATION_DEVELOPMENT_PROTOCOL.json'
OUT = d.ROOT / 'results/score_peak_localization_development_v1'
PRIMARY = 'localized_full_bank'
METHODS = d.METHODS + ['localized_safe_score_common', 'localized_safe_score_original', PRIMARY]


def freeze():
    if P.exists():
        raise FileExistsError('Localization development already frozen')
    original = strong_control.check()
    sources = ['score_peak_localization.py', 'test_score_peak_localization.py',
               'score_peak_localization_development.py']
    p = dict(original, settings=dict(original['settings'], seed_base=27190463851))
    p.update(status='FROZEN_BEFORE_FRESH_LOCALIZATION_EFFICACY_COHORT',
             sources={name: d.sha(d.ROOT / 'src' / name) for name in sources},
             parent_protocol_sha256=d.sha(d.P),
             stronger_control_addendum_sha256=d.sha(strong_control.P),
             methods=METHODS, primary_method=PRIMARY,
             bootstrap_seed_offset_from_parent=300000000000,
             origin='ChosenafterCLOSEDpreviousbankdevelopment:206/3/2vs192/2/2. '
                    'Threeerrorsaresecondarydistances2.0177--2.2913bins; '
                    'strictfullalternativefitsmovethemwithintheUNCHANGED2bintolerance. '
                    'Thattruthselectedthree-casecheckisNOTefficacyconfirmation.',
             algorithm='Keepfrozenfull-bank/null-refitquartergridscore,pvalueand199rankdraws. '
                       'Oneobserved-record5parameterfull-likelihoodfitfromselectedscorepeak, '
                       'c=.02,tauwithin4.5bins,secondarywithin4.5bins,boundsandstoppingrules '
                       'fixedinsource;keepinitial/nestednullfeasiblepoints. Sortreturned '
                       'locationsbyfittedamplitude. Localizationneverchangesdeclaration. '
                       'No-hiddenregimescopythematchedcensoredlocalizerEXACTLY.',
             calibration='Parent12statisticsandEVERYadaptivecalibrationsteprecomputedon '
                         'new640queries/newcompletepaths. Localizationdependsonobserved '
                         'recordonlyanddoesnotenterthestatistic:itsrankdistributionis '
                         'identicaltotheparentbanktest,notanunrepeatedbootstrapselection. '
                         'All640observedlocalizationscomputedregardlessofrejection. '
                         'Censored-scorecontrolsgetownlikelihoodrefinementfromownscorepeak.',
             primary_rule='StrictmorecorrectANDstrictfewertotalerrorsvsBOTHoriginal/common '
                          'rawcensoredscores,oldempiricalfullscore,andunlocalizedbankprimary. '
                          'Nomorewrongornullseparately. Atleastcorrectandnomorewrong/null '
                          'vsALLremainingcomparatorsincludingBOTHMATCHEDlocalizedcensored '
                          'scores. Nochangedalpha/tolerance/subgrouporposthocablationwinner.',
             additional_control='Onpreliminarypass+ALLmain/localizerauditsPASS,runBOTH '
                                'frozenlegacyandV2equal-informationfull-LRreferences '
                                'onALL640freshrows/ALL199commonpathsusing '
                                'strong_control.worker(row,newprotocol). Primarymustbeatleast '
                                'correctandnomorewrong/nullvsboth;independentcontrol '
                                'integrity/warningauditsmandatorybeforefreshlargerconfirmation. '
                                'OldcontrolrunnerfixedoldOUTmustNOTbeinvokedonthiscohort.',
             scope='NewDEVELOPMENTnotconfirmationofclosedbankcohort. '
                   'Standarddetect-then-localize/MLingredients;novelty,Q1,physicalvalidity '
                   'andjointutilityunproved. Source/protocol/datafrozenaftercreation. '
                   'No externalpublicationorhardwareexecution.')
    P.write_text(json.dumps(p, indent=2))
    print('Frozen new localization study', d.sha(P), flush=True)


def check():
    strong_control.check()
    p = json.loads(P.read_text())
    assert p['parent_protocol_sha256'] == d.sha(d.P)
    assert p['stronger_control_addendum_sha256'] == d.sha(strong_control.P)
    for name, digest in p['sources'].items():
        assert d.sha(d.ROOT / 'src' / name) == digest, name
    return p


def cases(p):
    for case in d.cases(p):
        yield dict(case, bootstrap_seed=case['bootstrap_seed'] + p['bootstrap_seed_offset_from_parent'])


def append_localizations(row, p):
    case = row['case']
    g = p['regimes'][case['regime']]
    obs, _, _, _, _, digest = d.previous.old.base.data(case, p)
    assert digest == row['data_sha256']
    s, e = numeric_statistics(obs, g['cycles'], 64, outward_safe_lag(g['td'], g['te']))
    safe = localize(s, e, 1.5, row['safe_null'], row['samples']['safe_score_common']['delay'])
    if g['td'] >= g['te']:
        full = dict(safe)
    else:
        geo = d.kernel.prepare_record(obs, g['cycles'] * 64, 64, g['td'], g['te'])
        full = localize(s, e, 1.5, row['observed_full_null'], row['samples'][d.PRIMARY]['delay'], geo)
    samples = dict(row['samples'])
    for cal in ['common', 'original']:
        source = row['samples']['safe_score_' + cal]
        samples['localized_safe_score_' + cal] = dict(source, primary=safe['primary'], delay=safe['delay'])
    samples[PRIMARY] = dict(row['samples'][d.PRIMARY], primary=full['primary'], delay=full['delay'])
    for old, new in [('safe_score_common', 'localized_safe_score_common'),
                     ('safe_score_original', 'localized_safe_score_original'), (d.PRIMARY, PRIMARY)]:
        assert all(samples[old][k] == samples[new][k] for k in ['score', 'pvalue', 'bootstrap_scores'])
    if g['td'] >= g['te']:
        assert samples[PRIMARY] == samples['localized_safe_score_common'] == samples['localized_safe_score_original']
    return dict(row, samples=samples, localization_fits=dict(safe=safe, full=full))


def worker(case, p):
    start = time.perf_counter()
    row = append_localizations(d.worker(case, p), p)
    row['seconds'] = time.perf_counter() - start
    return row


def execute(case):
    return worker(case, check())


def read():
    with sqlite3.connect(OUT / 'records.sqlite3') as db:
        return [json.loads(v[0]) for v in db.execute('SELECT payload FROM records ORDER BY rowid')]


def totals(rows, alpha):
    result = {m: dict(correct=0, wrong=0, false=0, positive=0, null=0) for m in METHODS}
    for row in rows:
        for method in METHODS:
            for k, value in d.previous.old.endpoint(row, method, alpha).items():
                result[method][k] += value
    return result


def rules(t):
    x = t[PRIMARY]
    result = {}
    strict = ['safe_score_common', 'safe_score_original', 'full_opg_safe', d.PRIMARY]
    for method, value in t.items():
        if method == PRIMARY:
            continue
        rule = dict(no_more_wrong=x['wrong'] <= value['wrong'], no_more_null=x['false'] <= value['false'])
        if method in strict:
            rule.update(strict_more_correct=x['correct'] > value['correct'],
                        strict_fewer_errors=x['wrong'] + x['false'] < value['wrong'] + value['false'])
        else:
            rule['at_least_correct'] = x['correct'] >= value['correct']
        result[method] = rule
    return result


def analyze():
    p = check()
    rows = read()
    assert len(rows) == 640
    t = totals(rows, .01)
    r = rules(t)
    passed = all(all(v.values()) for v in r.values())
    result = dict(status='PRELIMINARY_PASS_CONTROLS_AUDITS_CONFIRMATION_PENDING' if passed else 'PRELIMINARY_JOINT_UTILITY_FAILED',
                  totals=t, preliminary_rule=r, preliminary_rule_passed=passed, continue_to_fresh_confirmation=False,
                  descriptive_operating_points={str(a): totals(rows, a) for a in [.005, .01, .02]},
                  localization_warning_cases=[row['case']['id'] for row in rows if any(
                      not f['optimizer_success'] for f in row['localization_fits'].values())],
                  scope=p['scope'])
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def run():
    p = check()
    assert all(os.environ.get(k) == v for k, v in p['execution']['threads'].items())
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUT / 'records.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        done = {r[0] for r in db.execute('SELECT id FROM records')}
        remaining = [c for c in cases(p) if c['id'] not in done]
        start = time.perf_counter()
        print('Fresh localization8workers existing', len(done), 'remaining', len(remaining), flush=True)
        with ProcessPoolExecutor(max_workers=8) as pool:
            for i, row in enumerate(pool.map(execute, remaining, chunksize=1)):
                db.execute('INSERT INTO records VALUES(?,?)', (row['case']['id'], json.dumps(row)))
                db.commit()
                if (len(done) + i + 1) % 8 == 0:
                    print('Fresh localization', len(done) + i + 1, '/640 elapsed_s', round(time.perf_counter() - start, 1), flush=True)
    assert [r['case'] for r in read()] == list(cases(p))
    complete = dict(status='COMPLETE', cases=640, protocol_sha256=d.sha(P),
                    records_sha256=d.sha(OUT / 'records.sqlite3'), seconds_this_invocation=time.perf_counter() - start)
    (OUT / 'completion.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)
    analyze()


def closed():
    check()
    original = d.check()
    before = d.sha(d.OUT / 'records.sqlite3')
    rows = [append_localizations(row, original) for row in d.read()]
    assert before == d.sha(d.OUT / 'records.sqlite3')
    result = dict(status='CLOSED_EXPLORATORY_LOCATION_DIAGNOSIS_NOT_CONFIRMATION',
                  totals=totals(rows, .01), source_records_sha256=before,
                  localization_warning_cases=[r['case']['id'] for r in rows if any(
                      not f['optimizer_success'] for f in r['localization_fits'].values())],
                  scope='All640closedrecords,includingmatchedlocalizedcensoredcontrols. '
                        'No changes to originals,teststatistics,pvalues,toleranceorphysics. '
                        'TheseareDEVELOPMENTdiagnosticsnotnewmeasuredgeneralization. '
                        'Frozenfreshstudyalgorithmunchangedregardlessoftheseoutcomes.')
    (d.OUT / 'closed_all_query_localization_diagnosis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    {'freeze': freeze, 'run': run, 'closed': closed, 'analyze': analyze}[sys.argv[1]]()
