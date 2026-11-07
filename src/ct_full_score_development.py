"""Fresh full-observation score development on original continuous paths.

Shared full-null calibration AND original safe-null calibration controls.
No priority/finite-uniform-error/efficacy claim is built into this experiment.
"""
from pathlib import Path
import sys,json,time,sqlite3,os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import transported_joint_development as old
from ct_full_score import statistic as full_score,fit_full_null
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null,pulse
from deadtime_path_bootstrap import refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times
import ct_dual_fast as kernel
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import check_ct_dual_kernel as model
P=ROOT/'CT_FULL_SCORE_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/ct_full_score_development_v1';sha=old.sha
BASE={'local_score':'safe_score','local_joint':'safe_joint','context_max':'safe_context_max','context_transport':'safe_context_transport'}
METHODS=[f'{name}_{cal}' for cal in ['common','original'] for name in BASE.values()]+['full_score_common']


def freeze():
    if P.exists():raise FileExistsError('Frozen protocol exists')
    previous=old.check();model.check();p={k:previous[k] for k in ['settings','regimes','families','environment']};p['settings']=dict(p['settings'],positive_per_family_regime=16,null_per_family_regime=32,seed_base=14172809317);names=list(previous['sources'])+['ct_full_score.py','test_ct_full_score.py','ct_full_score_development.py']
    p.update(status='FROZEN_BEFORE_NEW_FULL_CT_SCORE_DATA',sources={name:sha(ROOT/'src'/name) for name in names},continuous_model_protocol_sha256=sha(model.P),previous_control_protocol_sha256=sha(old.P),methods=METHODS,primary_method='full_score_common',execution=dict(process_workers=4,thread_environment={n:'1' for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),
             design='Fresh640querysets,256positive384null;all4originalcontinuousfractionalrecoveryregimes,all7coherent/isolated/misleading/jump/borrowfamilies;16independentcontextrecords/query. Samephotonbudgets,contrast/phase/backgroundheterogeneity and2bintolerance. Newseeds;no closedcohort re-labeling. Preserveoriginalpiecewise64binrates,notafinerorclockreplacementphysics.',
             proposed='Querysafe3parameterMLEpilot. Fullcontinuousobservedloglikelihood derivative w.r.t64phaseintensities,withinitial,alltransition,andterminalterms. EmpiricalOPG=sum per-factor phase-score outerproducts. Project3querynullnuisancedirections,thenone-sidedmaxscore on256quarterbinsecondarytemplates,excludingwithin2.5sigmaofprimary. Reportquerypilotprimaryandfullscoremaxsecondary. NOdonortruth/generatingtruth input. OPGisempiricalinformation,not exactfiniteFisher. No location/evalueguarantee. IfTd>=Te,theproposedstatisticandlocation EXACTLY equal local safe projectedscore,as theobservationlawhasnohidden-eventinformationgain.',
             calibration='199completecontinuouslatentSPAD/electronicsfilteredpaths. Commoncalibrationgenerator=queryFULL3parameterobservednullMLE(3startsfromobservedcount/riskpeaks,notoracle). EVERYmethodrepeatssafequerynuisancerefitanditscompleteownstatistic/candidatefitinsideeachsharedpath. Fullscore evaluatedatthatreplicatesafepilotwithOPGprojection. All4safestatisticsalso getORIGINALcalibrationatquerySAFE-nullMLE,usingpairedrandomseedsandseparatelysimulatedpaths. InTd>=Te,nullfitsandpatharraysareidentical,and originalcalibration copiescommonscores exactly. Allcontrolsbenefitfromcommonfullcalibrationbutoldcalibrationcontrolsretainoriginalpipeline. Rankpvaluesplug-in/approximate forcompositenull;no finiteuniformguarantee.',
             controls='Originalquarterbinprojectedscore,6startjointfit,8startsame-contextmaximum,andweightedtransportprofile,allunder BOTHcommonfullnullandoriginalsafenullcalibration. Queryprimaryreported,neveroracleordefaultdonorprimary. Fullprimarymethodusesquerydataonly;contextcontrolsget16donorsasstrongerinformationcomparators.',
             primary_rule='Atalpha.01 full_score_common strictly morecorrect AND strictlyfewertotalerrors(wronglocation+nullfalse) than BOTHsafe_score_common andsafe_score_original,withnomorewrongandnomorenullseparately. Forother6joint/contextcontrols,atleastcorrectandnomorewrong/nullseparately. Oneprimarymethod/alpha;no posthoc operatingpoint or subgroup winner. A developmentpass stillrequiresfreshlargerconfirmation,uncertainty,noveltyandphysicalassessment.',
             numerical='ContinuousmodelkernelF0+F1+F2sourcesfrozen under CT_DUAL_KERNEL_VALIDATION_PROTOCOL. AlloriginalregimesTe<=3Td,no exclusions. Exact-dyadic safe durations withoutwardlag forcontrols/pilot. Fullfactor usesordinaryfloatinganalyticsegmentsandvalidatedmomentfallback,NOTintervalcertificate. Null/alternativeoptimizerwarningssavedbydrawforindependentreferenceaudit;no outcome-baseddropping.',
             scope='New DEVELOPMENT of a full-observation score,not confirmation of point-estimator variants or known-rateCDFchecks. KnownGaussianwidth/recovery,initialactive. Fullmodelscore/OPG/projection/bootstrapareestablishedingredients;no newgeneraltheorem/priority/Q1readinessassumed.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());oldp=old.check();model.check();assert p['environment']==oldp['environment'] and sha(old.P)==p['previous_control_protocol_sha256'] and sha(model.P)==p['continuous_model_protocol_sha256']
    for name,h in p['sources'].items():assert sha(ROOT/'src'/name)==h,name
    return p


def cases(p):
    for case in old.base.cases(p):case['bootstrap_seed']=97000000000+1000003*case['index'];yield case


def worker(case,p):
    start=time.perf_counter();cfg=p['settings'];g=p['regimes'][case['regime']];obs,ds,de,truth,params,dh=old.base.data(case,p);lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);context=donor_features(ds,de,1.5);pilot=fit_null(s,e,1.5);geometry=kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);full,diagnostic=full_score(s,e,1.5,pilot,geometry);observed=old.statistics(s,e,pilot,context,cfg);null=fit_full_null(s,e,1.5,pilot,geometry);rates={name:fit[0]*pulse(fit[1],64,1.5)[0]+fit[2]/64 for name,fit in [('common',null['theta']),('original',pilot['theta'])]};boots={m:[] for m in METHODS};hashes={};warnings=[];diagnostics=[]
    import hashlib
    for calibration in ['common','original']:
        if calibration=='original' and g['td']>=g['te']:
            assert np.array_equal(null['theta'],pilot['theta'])
            for name in BASE.values():boots[name+'_original']=list(boots[name+'_common'])
            hashes['original']=hashes['common'];continue
        pathhash=hashlib.sha256()
        for draw in range(cfg['bootstrap_replicates']):
            bo,_=simulate_two_dead_times(rates[calibration],g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);pathhash.update(bo.tobytes());bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,null['theta'] if calibration=='common' else pilot['theta'],1.5,'duration');r=old.statistics(bs,be,bf,context,cfg)
            for key,name in BASE.items():boots[name+'_'+calibration].append(r['samples'][key]['score'])
            if calibration=='common':
                bg=kernel.prepare_record(bo,g['cycles']*64,64,g['td'],g['te']);v,_=full_score(bs,be,1.5,bf,bg);boots['full_score_common'].append(v['score'])
                if g['td']>=g['te']:assert v==r['samples']['local_score']
            failed=[i for i,f in enumerate(r['fits']) if not f['optimizer_success']]
            if failed or not bf['optimizer_success']:warnings.append(dict(calibration=calibration,draw=draw,null_failure=not bf['optimizer_success'],alternative_failure_components=failed))
            if case['id'].endswith('_000') and draw in [0,198]:diagnostics.append(dict(calibration=calibration,draw=draw,null_theta=bf['theta'].tolist(),null_loglik=bf['loglik']))
        hashes[calibration]=pathhash.hexdigest()
    samples={}
    for calibration in ['common','original']:
        for key,name in BASE.items():
            stat=observed['samples'][key];m=name+'_'+calibration;samples[m]=dict(stat,pvalue=pvalue(stat['score'],boots[m]),bootstrap_scores=boots[m])
    samples['full_score_common']=dict(full,pvalue=pvalue(full['score'],boots['full_score_common']),bootstrap_scores=boots['full_score_common'])
    if g['td']>=g['te']:assert samples['full_score_common']==samples['safe_score_common']==samples['safe_score_original']
    return dict(case=case,truth=truth,generating_parameters=params,data_sha256=dh,observed_events=len(obs),scored_events=int(s.sum()),context={k:context[k] for k in ['primary','secondary','two_pulse_loglr']},safe_null=pilot['theta'].tolist(),calibration_null=dict(theta=null['theta'].tolist(),loglik=null['loglik'],optimizer_success=null['optimizer_success'],failures=null['failures'],mode=null['mode']),full_score_diagnostics=diagnostic,observed_alternative_fits=observed['fits'],samples=samples,bootstrap_path_sha256=hashes,bootstrap_warnings=warnings,refit_diagnostics=diagnostics,seconds=time.perf_counter()-start)


def execute(case):return worker(case,check())


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(r[0]) for r in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();assert {name:os.environ.get(name) for name in p['execution']['thread_environment']}==p['execution']['thread_environment'];connection=connect();done={r[0] for r in connection.execute('SELECT id FROM records')};remaining=[c for c in cases(p) if c['id'] not in done];start=time.perf_counter();new=0;print('Full CT score4processes;existing',len(done),'remaining',len(remaining),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(execute,remaining,chunksize=1):
            connection.execute('INSERT INTO records VALUES(?,?)',(row['case']['id'],json.dumps(row)));connection.commit();new+=1
            if (len(done)+new)%8==0:print('Full CT score development',len(done)+new,'/640 elapsed_s',round(time.perf_counter()-start,1),flush=True)
    connection.close();assert [r['case'] for r in read()]==list(cases(p));r=dict(status='COMPLETE',cases=640,new_cases_this_invocation=new,preexisting_committed_cases=len(done),protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),seconds_this_invocation=time.perf_counter()-start,timing_scope='4processes;not isolated/FPGA timing.');(OUT/'completion.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);analyze()


def endpoint(row,m,alpha=.01):return old.endpoint(row,m,alpha)


def totals(rows,alpha):
    t={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS}
    for row in rows:
        for m in METHODS:
            for key,n in endpoint(row,m,alpha).items():t[m][key]+=n
    return t


def primary_rule(t):
    x=t['full_score_common'];rules={}
    for name in ['safe_score_common','safe_score_original']:
        b=t[name];rules[name]=dict(strict_more_correct=x['correct']>b['correct'],strict_fewer_errors=x['wrong']+x['false']<b['wrong']+b['false'],no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'])
    for name in [m for m in METHODS if m not in ['full_score_common','safe_score_common','safe_score_original']]:
        b=t[name];rules[name]=dict(at_least_correct=x['correct']>=b['correct'],no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'])
    return rules


def analyze():
    p=check();rows=read();assert len(rows)==640;t=totals(rows,.01);rule=primary_rule(t);groups=[]
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:groups.append(dict(regime=g['name'],family=family,methods=totals([r for r in rows if r['case']['regime']==gi and r['case']['family']==family],.01)))
    r=dict(status='FRESH_FULL_CONTINUOUS_SCORE_DEVELOPMENT_ONLY',totals=t,primary_rule=rule,continue_to_fresh_confirmation=all(all(x.values()) for x in rule.values()),per_regime_family=groups,descriptive_operating_points={str(alpha):totals(rows,alpha) for alpha in [.005,.01,.02]},calibration_null_fit_failures=sum(not r['calibration_null']['optimizer_success'] for r in rows),bootstrap_null_warnings=sum(w['null_failure'] for r in rows for w in r['bootstrap_warnings']),bootstrap_alternative_warnings=sum(len(w['alternative_failure_components']) for r in rows for w in r['bootstrap_warnings']),scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='per_regime_family'},indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
