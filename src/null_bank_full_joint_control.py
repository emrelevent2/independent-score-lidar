"""Preplanned expensive full-law control,mandatory on a preliminary pass."""
import json,time,hashlib,sqlite3,os,sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import null_bank_development as d
from null_bank_score import refit_full_null,full_likelihood_ratio
from fast_two_return_fit import reports
from context_agreement_features import numeric_statistics
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null,pulse
from deadtime_path_bootstrap import refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times


def reference_worker(row,p):
    start=time.perf_counter();case=row['case'];g=p['regimes'][case['regime']];obs,_,_,_,_,dh=d.previous.old.base.data(case,p);assert dh==row['data_sha256'];lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);pilot=fit_null(s,e,1.5);geo=d.kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);null=refit_full_null(s,1.5,pilot,geo);assert d.json_fit(null)==row['observed_full_null'];observed=full_likelihood_ratio(s,1.5,null,row['observed_alternative_fits'],geo);theta=row['calibration_null']['theta'];rate=theta[0]*pulse(theta[1],64,1.5)[0]+theta[2]/64;scores=[];warnings=[];h=hashlib.sha256();full_null_failure_indices=[];unsupported=dict(primary=0.,secondary=0.,two_pulse_loglr=0.)
    for draw in range(p['settings']['bootstrap_replicates']):
        bo,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);h.update(bo.tobytes());bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,theta,1.5,'duration');bg=d.kernel.prepare_record(bo,g['cycles']*64,64,g['td'],g['te']);nf=refit_full_null(bs,1.5,bf,bg);six=reports(bs,be,1.5,bf,unsupported);assert six['samples']['local_joint']['score']==row['samples']['safe_joint_common']['bootstrap_scores'][draw];stat=full_likelihood_ratio(bs,1.5,nf,six['fits'],bg);scores.append(stat['score']);failed=[i for i,x in enumerate(six['fits']) if not x['optimizer_success']]
        if not nf['optimizer_success']:full_null_failure_indices.append(draw)
        if not bf['optimizer_success'] or not nf['optimizer_success'] or not stat['optimizer_success'] or failed:warnings.append(dict(draw=draw,safe_null_failure=not bf['optimizer_success'],full_null_failure=not nf['optimizer_success'],safe_alternative_failure_components=failed,full_joint_failure=not stat['optimizer_success']))
    assert h.hexdigest()==row['bootstrap_path_sha256']['common'];assert full_null_failure_indices==[w['draw'] for w in row['bootstrap_warnings'] if w['calibration']=='common' and w['full_null_failure']];sample={k:observed[k] for k in ['score','primary','delay']};sample.update(pvalue=pvalue(observed['score'],scores),bootstrap_scores=scores)
    return dict(case=case,sample=sample,observed_full_joint=observed,bootstrap_warnings=warnings,bootstrap_path_sha256=h.hexdigest(),seconds=time.perf_counter()-start)


def execute(row):return reference_worker(row,d.check())


def connect():
    c=sqlite3.connect(d.OUT/'full_joint_reference.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(x[0]) for x in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=d.check();a=json.loads((d.OUT/'analysis.json').read_text());assert a['preliminary_rule_passed'],'Preplanned expensivecontrolnotrequiredonfailedprimary';assert json.loads((d.OUT/'integrity.json').read_text())['status']=='PASS' and json.loads((d.OUT/'failure_audit.json').read_text())['status']=='PASS';assert {n:os.environ.get(n) for n in p['execution']['threads']}==p['execution']['threads'];before=d.sha(d.OUT/'records.sqlite3');c=connect();done={x[0] for x in c.execute('SELECT id FROM records')};remaining=[row for row in d.read() if row['case']['id'] not in done];start=time.perf_counter();new=0
    with ProcessPoolExecutor(max_workers=p['execution']['workers']) as pool:
        for row in pool.map(execute,remaining,chunksize=1):
            c.execute('INSERT INTO records VALUES(?,?)',(row['case']['id'],json.dumps(row)));c.commit();new+=1
            if (len(done)+new)%8==0:print('Preplanned fulljoint control',len(done)+new,'/640 elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();assert [x['case'] for x in read()]==list(d.cases(p)) and before==d.sha(d.OUT/'records.sqlite3');result=dict(status='COMPLETE',cases=640,protocol_sha256=d.sha(d.P),reference_records_sha256=d.sha(d.OUT/'full_joint_reference.sqlite3'),main_records_sha256=before,seconds_this_invocation=time.perf_counter()-start);(d.OUT/'full_joint_completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);analyze()


def analyze():
    p=d.check();rows=d.read();reference=read();assert len(reference)==len(rows)==640;t=dict(correct=0,wrong=0,false=0,positive=0,null=0)
    for row,ref in zip(rows,reference):
        assert row['case']==ref['case'];r=dict(row,samples=dict(row['samples'],full_joint_reference=ref['sample']))
        for k,n in d.previous.old.endpoint(r,'full_joint_reference',.01).items():t[k]+=n
    primary=d.totals(rows,.01)[d.PRIMARY];rule=dict(at_least_correct=primary['correct']>=t['correct'],no_more_wrong=primary['wrong']<=t['wrong'],no_more_null=primary['false']<=t['false']);audit_paths=[d.OUT/'full_joint_integrity.json',d.OUT/'full_joint_failure_audit.json'];audited=all(f.exists() and json.loads(f.read_text())['status']=='PASS' for f in audit_paths);result=dict(status='STRONG_CONTROL_POINT_COMPARISON_ONLY',reference_totals=t,primary_totals=primary,primary_rule=rule,point_rule_passed=all(rule.values()),all_strong_control_integrity_and_warning_audits_pass=audited,continue_to_fresh_confirmation=all(rule.values()) and audited,scope='Preplannedstrongerfulllawcomparisonatfrozenalpha.01. All640queries/all199commonpaths;not a global-ML/physical/Q1guarantee. Freshconfirmationalsorequiredifallgatespass.')
    (d.OUT/'full_joint_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':{'run':run,'analyze':analyze}[sys.argv[1]]()
