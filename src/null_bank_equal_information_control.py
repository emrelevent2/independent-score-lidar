"""Pre-metrics additive stronger control;leaves the frozen12methodstudy intact."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,time,sqlite3,hashlib,sys,os
import numpy as np
import null_bank_development as d
import null_bank_full_joint_control as legacy_runner
from full_information_joint_reference import stronger_joint
from null_bank_score import refit_full_null,full_likelihood_ratio
from fast_two_return_fit import reports
from context_agreement_features import numeric_statistics
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null,pulse
from deadtime_path_bootstrap import refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times
P=d.ROOT/'NULL_BANK_EQUAL_INFORMATION_CONTROL_ADDENDUM.json'


def freeze():
    if P.exists():raise FileExistsError('Strongcontroladdendumalreadyfrozen')
    p=d.check();assert not (d.OUT/'analysis.json').exists(),'Addendum must precede main metric inspection';names=['full_information_joint_reference.py','test_full_information_joint_reference.py','null_bank_equal_information_control.py'];c=sqlite3.connect(d.OUT/'records.sqlite3');count=c.execute('SELECT count(*) FROM records').fetchone()[0];c.close();record=dict(status='FROZEN_ADDITIVE_STRONGER_CONTROL_BEFORE_ANY_MAIN_ENDPOINT_TOTALS',main_protocol_sha256=d.sha(d.P),sources={n:d.sha(d.ROOT/'src'/n) for n in names},main_committed_cases_at_addendum=count,primary_method_and_all12main_pipeline_sources_unchanged=True,reason='Legacyfulljointreferenceinitializesonlyfrom6censoredfitsandmaymissasecondaryvisibleinthefullinformation. StrongercomparisonmustgettheSAMEbank/fullphasegradientasprimary.',method='Keepfrozenlegacyfulljointsolutionasfeasiblebaseline;add6separatedquarterbinfull-bank-scorestarts,refinetop2fullscorestartsandbestnewinitialfulllikelihood(atmost3additional5parameterfits),keepmaxofallinitial/refined/legacy/nullsolutions. Repeatallselection/fullnullrefitsinsideeverycommonbootstrapdraw. No globalMLclaim.',execution='ConditionalonMAINpreliminarypassandmainauditsPASS,computeBOTHlegacyandstrongerreferenceforALL640queriesandALL199commonpaths. Samegenerators/seeds/recovery/contextdatasets/2bintolerance/alpha.01. No favorable-subgroupshortcut. Computeexpensivecontrolsisdeferredonlywhenprimaryhasalreadyfailed,notwaivedforpublication.',gate='PRIMARYmusthaveatleastcorrectandnomorewrongandnomorenullthanBOTHfulljointcontrols. Allcomparisonintegrityandnumerical-warningauditsMUSTpassbeforefreshconfirmation. Addendumcanonlystrengthenreadinessrequirements;itdoesnotpromoteanotherprimaryorchangealpha.',scope='Additionalpre-metricsrobustnesscomparator,not primaryefficacyresults orconfirmation. Genericexpectedinformation/score/MLingredientsestablished. Mainprotocolandscientificrecordsremainunchanged.');P.write_text(json.dumps(record,indent=2));print('Frozenstrongcontroladdendum',d.sha(P),flush=True)


def check():
    p=d.check();a=json.loads(P.read_text());assert a['main_protocol_sha256']==d.sha(d.P)
    for n,h in a['sources'].items():assert d.sha(d.ROOT/'src'/n)==h,n
    return p


def worker(row,p):
    start=time.perf_counter();case=row['case'];g=p['regimes'][case['regime']];obs,_,_,_,_,dh=d.previous.old.base.data(case,p);assert dh==row['data_sha256'];lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);pilot=fit_null(s,e,1.5);geo=d.kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);null=refit_full_null(s,1.5,pilot,geo);assert d.json_fit(null)==row['observed_full_null'];old=full_likelihood_ratio(s,1.5,null,row['observed_alternative_fits'],geo);strong=stronger_joint(s,1.5,null,row['observed_alternative_fits'],geo,d.bank(),case['regime'],old);theta=row['calibration_null']['theta'];rate=theta[0]*pulse(theta[1],64,1.5)[0]+theta[2]/64;scores={m:[] for m in ['legacy','strong']};warnings=[];legacy_warnings=[];h=hashlib.sha256();failed_full_nulls=[];unsupported=dict(primary=0.,secondary=0.,two_pulse_loglr=0.)
    for draw in range(199):
        bo,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);h.update(bo.tobytes());bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,theta,1.5,'duration');bg=d.kernel.prepare_record(bo,g['cycles']*64,64,g['td'],g['te']);nf=refit_full_null(bs,1.5,bf,bg);six=reports(bs,be,1.5,bf,unsupported);assert six['samples']['local_joint']['score']==row['samples']['safe_joint_common']['bootstrap_scores'][draw];lr=full_likelihood_ratio(bs,1.5,nf,six['fits'],bg);eq=stronger_joint(bs,1.5,nf,six['fits'],bg,d.bank(),case['regime'],lr);scores['legacy'].append(lr['score']);scores['strong'].append(eq['score']);failed=[i for i,v in enumerate(six['fits']) if not v['optimizer_success']];additional=[i for i,v in enumerate(eq['additional_refinements']) if not v['optimizer_success']]
        if not nf['optimizer_success']:failed_full_nulls.append(draw)
        if not bf['optimizer_success'] or not nf['optimizer_success'] or not lr['optimizer_success'] or failed:legacy_warnings.append(dict(draw=draw,safe_null_failure=not bf['optimizer_success'],full_null_failure=not nf['optimizer_success'],safe_alternative_failure_components=failed,full_joint_failure=not lr['optimizer_success']))
        if additional or not eq['optimizer_success']:warnings.append(dict(draw=draw,additional_failure_components=additional,strong_selected_failure=not eq['optimizer_success']))
    assert h.hexdigest()==row['bootstrap_path_sha256']['common'] and failed_full_nulls==[w['draw'] for w in row['bootstrap_warnings'] if w['calibration']=='common' and w['full_null_failure']];samples={}
    for m,v in [('legacy',old),('strong',strong)]:samples[m]=dict({k:v[k] for k in ['score','primary','delay']},pvalue=pvalue(v['score'],scores[m]),bootstrap_scores=scores[m])
    return dict(case=case,samples=samples,observed_full_joint=old,observed_stronger_joint=strong,legacy_bootstrap_warnings=legacy_warnings,stronger_bootstrap_warnings=warnings,bootstrap_path_sha256=h.hexdigest(),seconds=time.perf_counter()-start)


def execute(row):return worker(row,check())


def connect():
    c=sqlite3.connect(d.OUT/'equal_information_joint_reference.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(x[0]) for x in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();main=json.loads((d.OUT/'analysis.json').read_text());assert main['preliminary_rule_passed'];assert all(json.loads((d.OUT/n).read_text())['status']=='PASS' for n in ['integrity.json','failure_audit.json']);assert {n:os.environ.get(n) for n in p['execution']['threads']}==p['execution']['threads'];before=d.sha(d.OUT/'records.sqlite3');c=connect();done={x[0] for x in c.execute('SELECT id FROM records')};remaining=[row for row in d.read() if row['case']['id'] not in done];start=time.perf_counter();new=0;print('Equal-informationjoint8workers existing',len(done),'remaining',len(remaining),flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(execute,remaining,chunksize=1):
            c.execute('INSERT INTO records VALUES(?,?)',(row['case']['id'],json.dumps(row)));c.commit();new+=1
            if (len(done)+new)%8==0:print('Equal-informationjointcontrol',len(done)+new,'/640 elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();rows=read();assert [r['case'] for r in rows]==list(d.cases(p)) and before==d.sha(d.OUT/'records.sqlite3');lc=legacy_runner.connect()
    for row in rows:
        payload=dict(case=row['case'],sample=row['samples']['legacy'],observed_full_joint=row['observed_full_joint'],bootstrap_warnings=row['legacy_bootstrap_warnings'],bootstrap_path_sha256=row['bootstrap_path_sha256'],seconds=row['seconds']);old=lc.execute('SELECT payload FROM records WHERE id=?',(row['case']['id'],)).fetchone()
        if old:
            previous=json.loads(old[0]);assert {k:v for k,v in previous.items() if k!='seconds'}=={k:v for k,v in payload.items() if k!='seconds'}
        else:lc.execute('INSERT INTO records VALUES(?,?)',(row['case']['id'],json.dumps(payload)))
    lc.commit();lc.close();legacy_complete=dict(status='COMPLETE',cases=640,protocol_sha256=d.sha(d.P),reference_records_sha256=d.sha(d.OUT/'full_joint_reference.sqlite3'),main_records_sha256=before,timing_scope='Computedtogetherwithadditiveequal-informationcontrol;notisolatedlegacytime.');(d.OUT/'full_joint_completion.json').write_text(json.dumps(legacy_complete,indent=2));result=dict(status='COMPLETE',cases=640,main_records_sha256=before,reference_records_sha256=d.sha(d.OUT/'equal_information_joint_reference.sqlite3'),addendum_sha256=d.sha(P),seconds_this_invocation=time.perf_counter()-start);(d.OUT/'equal_information_joint_completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);analyze()


def analyze():
    p=check();main=d.read();rows=read();assert len(rows)==len(main)==640;totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in ['legacy','strong']}
    for query,ref in zip(main,rows):
        assert query['case']==ref['case'];augmented=dict(query,samples=dict(query['samples'],**ref['samples']))
        for m in totals:
            for k,v in d.previous.old.endpoint(augmented,m,.01).items():totals[m][k]+=v
    primary=d.totals(main,.01)[d.PRIMARY];rules={m:dict(at_least_correct=primary['correct']>=v['correct'],no_more_wrong=primary['wrong']<=v['wrong'],no_more_null=primary['false']<=v['false']) for m,v in totals.items()};audit_paths=[d.OUT/n for n in ['equal_information_joint_integrity.json','equal_information_joint_failure_audit.json']];audited=all(n.exists() and json.loads(n.read_text())['status']=='PASS' for n in audit_paths);result=dict(status='POINT_COMPARISON_REQUIRES_INDEPENDENT_CONTROL_AUDITS',reference_totals=totals,primary_totals=primary,primary_rules=rules,all_point_rules_pass=all(all(v.values()) for v in rules.values()),control_audits_pass=audited,continue_to_fresh_confirmation=all(all(v.values()) for v in rules.values()) and audited,scope='Additionalpre-metricsstrongercontrolusingSAMEfullgradient/bankstarts. Bothreferencesretained;no globalMLorQ1claim. Primarymethodandmainresultsunchanged. Freshlargerconfirmationandphysicalassessmentstillrequired.');(d.OUT/'equal_information_joint_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
