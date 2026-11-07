"""Fresh development:identical detection calls,different location estimators.

The five-parameter one-step estimate is an established-estimation adaptation.
No empirical advantage or nonregular-boundary efficiency theorem is assumed.
"""
from pathlib import Path
import sys,json,time,hashlib,sqlite3,os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import transported_joint_development as old
from context_agreement_features import numeric_statistics
from outward_safety_lag import outward_safe_lag
from joint_one_step_location import estimate
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'ONE_STEP_LOCATION_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/one_step_location_development_v1'
METHODS=old.METHODS+['transport_score_location','transport_one_step_location'];sha=old.sha


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol exists')
    previous=old.check();p={k:previous[k] for k in ['settings','regimes','families','environment']};p['settings']=dict(p['settings'],positive_per_family_regime=16,null_per_family_regime=32,seed_base=11809274613)
    names=list(previous['sources'])+['joint_one_step_location.py','test_joint_one_step_location.py','one_step_location_development.py','test_one_step_location_development.py']
    p.update(status='FROZEN_BEFORE_FRESH_ONE_STEP_LOCATION_DATA',sources={n:sha(ROOT/'src'/n) for n in names},methods=METHODS,previous_scientific_protocol_sha256=sha(old.P),primary_method='transport_one_step_location',execution=dict(process_workers=4,thread_environment={name:'1' for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},commit_order='inputcaseorder,singlewriter'),
             design='Fresh640querysets:256positive384null,16independent context paths each;four recovery/flux regimes and seven broad families retained. Primary contrasts/phase/background vary versus context;misleading,isolated,borrowed and abrupt-primary-jump cases remain. New data and bootstrap seeds,not the closed320labels. New DEVELOPMENT,not confirmation of either old candidate.',
             detection='EXACT frozen transported_joint_development.worker applied to new640cases and new seeds. All original4controls unchanged:local_score,local_joint,context_max,context_transport.199sharedfull event-path null replicates repeat all query-dependent null/alternative fits and weighted profile statistics. Approximate plug-in composite-null calibration,not e-values/finiteuniformguarantees. No source edit to old worker.',
             point_estimators='Under IDENTICAL context_transport detection score,pvalue,andbootstrap scores,compare iterated joint-fit location,raw local_score primary/secondary,and new query-only one-step location. One-step:query-null primary + quarter-bin projected-score secondary;profile3nonnegative amplitudes atfixedcentres;one damped5parameterFisher step with likelihood-increase backtracking. New estimator receives only querycounts/exposures,knownsigma,andquerynulltheta;no generatingtruth/donortruth. Both reported centers estimated;stronger fitted component labeled primary. Bounds/step rules/rcond frozen in source.',
             calibration_scope='Point estimation changes no detection event. It therefore does not require recomputing a different detection-score bootstrap;alloriginal bootstrap already repeated. No guarantee for location error under H1 follows. Synthetic positive truth evaluates location errors separately. Primary and secondary must BOTH bewithin2bins;no tolerance changes or posthoc threshold selection. Alpha=.01primary;otheralphasdescriptive only.',
             primary_rule='Atalpha.01 one-step point rule strictly more correct AND strictly fewer wrong locations than BOTH iterated-joint and rawscore point estimators at IDENTICAL transport calls;allfalsecounts must exactly match. Endtoend also strictly more correct AND strictly fewer totalerrors than stronglocal_joint,with no more wrong/null separately;atleast local_scorecorrect andno more local_scorewrong/null;Paretoimproveequalcontext_max(atleastcorrect,no morewrong/null,one strict). A pass requires freshlargerconfirmation and uncertainty assessment,not immediate manuscript promotion.',
             numerical_scope=previous['numerical'],physical_scope=previous['scope'],scope='Known idealized Gaussian/recovery model;new point-estimation comparison on broad fresh development. One-step Fisher estimation is established. No reducedbias/asymptoticefficiency claim at vanishing secondary amplitude,physicalvalidation,novelty,Q1 or submissionreadiness inferred.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());previous=old.check();assert previous['environment']==p['environment'];assert sha(old.P)==p['previous_scientific_protocol_sha256']
    for n,h in p['sources'].items():assert sha(ROOT/'src'/n)==h,n
    return p


def cases(p):
    for case in old.base.cases(p):case['bootstrap_seed']=79000000000+1000003*case['index'];yield case


def attach_locations(row,point):
    result=dict(row);result['samples']=dict(row['samples']);source=row['samples']['context_transport'];score=row['samples']['local_score']
    result['samples']['transport_score_location']=dict(source,primary=score['primary'],delay=score['delay'])
    result['samples']['transport_one_step_location']=dict(source,primary=point['primary'],delay=point['delay'])
    result['one_step_location']=point
    return result


def worker(case,p):
    row=old.worker(case,p);cfg=p['settings'];g=p['regimes'][case['regime']];obs,_,_,truth,params,dh=old.base.data(case,p);assert dh==row['data_sha256'] and truth==row['truth'];s,e=numeric_statistics(obs,g['cycles'],64,outward_safe_lag(g['td'],g['te']));start=time.perf_counter();point=estimate(s,e,cfg['sigma'],dict(theta=np.asarray(row['null_fit'])));seconds=time.perf_counter()-start;result=attach_locations(row,point);result['location_seconds']=seconds;result['seconds']=row['seconds']+seconds
    for m in ['transport_score_location','transport_one_step_location']:
        for key in ['score','pvalue','bootstrap_scores']:assert result['samples'][m][key]==row['samples']['context_transport'][key]
    return result


def execute(case):return worker(case,check())


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(z[0]) for z in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();assert {k:os.environ.get(k) for k in p['execution']['thread_environment']}==p['execution']['thread_environment'];connection=connect();done={z[0] for z in connection.execute('SELECT id FROM records')};remaining=[c for c in cases(p) if c['id'] not in done];start=time.perf_counter();new=0;print('One-step point development4processes;existing',len(done),'remaining',len(remaining),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(execute,remaining,chunksize=1):
            connection.execute('INSERT INTO records VALUES(?,?)',(row['case']['id'],json.dumps(row)));connection.commit();new+=1
            if (len(done)+new)%8==0:print('One-step location development',len(done)+new,'/640 elapsed_s',round(time.perf_counter()-start,1),flush=True)
    connection.close();rows=read();assert [r['case'] for r in rows]==list(cases(p));assert len(rows)==640;r=dict(status='COMPLETE',cases=640,preexisting_committed_cases=len(done),new_cases_this_invocation=new,protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),seconds_this_invocation=time.perf_counter()-start,timing_scope='4processes,1BLASthread each;not isolated serial/FPGA timing.');(OUT/'completion.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);analyze()


def endpoint(row,m,alpha=.01):return old.endpoint(row,m,alpha)


def totals(rows,alpha):
    t={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS}
    for row in rows:
        for m in METHODS:
            for k,n in endpoint(row,m,alpha).items():t[m][k]+=n
    return t


def rule(t):
    x=t['transport_one_step_location'];same={}
    for name in ['context_transport','transport_score_location']:
        b=t[name];same[name]=dict(strict_more_correct=x['correct']>b['correct'],strict_fewer_wrong=x['wrong']<b['wrong'],identical_null=x['false']==b['false'])
    l,s,b=[t[m] for m in ['local_joint','local_score','context_max']];end=dict(strict_more_correct_than_joint=x['correct']>l['correct'],strict_fewer_errors_than_joint=x['wrong']+x['false']<l['wrong']+l['false'],no_more_joint_wrong=x['wrong']<=l['wrong'],no_more_joint_null=x['false']<=l['false'],at_least_score_correct=x['correct']>=s['correct'],no_more_score_wrong=x['wrong']<=s['wrong'],no_more_score_null=x['false']<=s['false'],at_least_context_max_correct=x['correct']>=b['correct'],no_more_context_max_wrong=x['wrong']<=b['wrong'],no_more_context_max_null=x['false']<=b['false'],at_least_one_strict_context_max=x['correct']>b['correct'] or x['wrong']<b['wrong'] or x['false']<b['false'])
    return dict(identical_call_location=same,endtoend=end,pass_all=all(all(v.values()) for v in same.values()) and all(end.values()))


def errors(rows):
    result={}
    for m in METHODS:
        values=[];selected=[]
        for row in rows:
            if not row['case']['positive']:continue
            v=row['samples'][m];signed=(v['delay']-row['truth']['secondary']+32)%64-32;values.append(signed)
            if v['pvalue']<=.01:selected.append(signed)
        def summary(v):return dict(count=len(v),mean_signed_bins=float(np.mean(v)) if v else None,mae_bins=float(np.mean(np.abs(v))) if v else None,rmse_bins=float(np.sqrt(np.mean(np.square(v)))) if v else None)
        result[m]=dict(all_positive=summary(values),declared_positive=summary(selected))
    return result


def analyze():
    p=check();rows=read();assert len(rows)==640;t=totals(rows,.01);rules=rule(t);groups=[]
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:groups.append(dict(regime=g['name'],family=family,methods=totals([r for r in rows if r['case']['regime']==gi and r['case']['family']==family],.01)))
    r=dict(status='FRESH_ONE_STEP_LOCATION_DEVELOPMENT_ONLY',totals=t,primary_rule=rules,continue_to_fresh_confirmation=rules['pass_all'],per_regime_family=groups,descriptive_operating_points={str(alpha):totals(rows,alpha) for alpha in [.005,.01,.02]},secondary_errors=errors(rows),one_step_amplitude_fit_failures=sum(not z['one_step_location']['amplitude_optimizer_success'] for z in rows),one_step_no_accepted_step=sum(z['one_step_location']['backtracks']==21 for z in rows),bootstrap_null_fit_failures=sum(len(z['bootstrap_null_failure_indices']) for z in rows),bootstrap_alternative_fit_failures=sum(z['bootstrap_alternative_failures'] for z in rows),scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['per_regime_family','secondary_errors']},indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
