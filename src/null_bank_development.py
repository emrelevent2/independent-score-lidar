"""Fresh independent-bank/full-null-refit development with strong controls.

The expensive full-law joint-fit control is preplanned and mandatory if the
primary passes the preliminary utility gate. No confirmation is authorized
by an ablation win or an incomplete stronger-control comparison.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,sqlite3,os
import numpy as np
import ct_full_score_development as previous
import null_bank_score as candidate
from null_information_bank import NullInformationBank
from context_agreement_features import numeric_statistics,donor_features
from outward_safety_lag import outward_safe_lag
from deadtime_scan import fit_null,pulse
from deadtime_path_bootstrap import refit_from_start,pvalue
from safe_deadtime_exposure import simulate_two_dead_times
from fast_two_return_fit import reports
import ct_dual_fast as kernel
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import train_null_information_bank as training
P=ROOT/'NULL_BANK_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/null_bank_development_v1';sha=previous.sha
BASE=previous.BASE;METHODS=[name+'_'+cal for cal in ['common','original'] for name in BASE.values()]+candidate.METHODS;PRIMARY='full_bank_refit'


def json_fit(fit):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in fit.items()}


@lru_cache(maxsize=1)
def bank():return NullInformationBank.load(training.OUT/'bank.npz')


def bank_check():
    p=training.check();c=json.loads((training.OUT/'completion.json').read_text());v=json.loads((training.OUT/'validation.json').read_text());a=json.loads((training.OUT/'audit.json').read_text());va=json.loads((training.OUT/'validation_audit.json').read_text());assert v['status']=='NUMERICAL_WEIGHT_APPROXIMATION_PASS' and a['status']==va['status']=='PASS';assert c['bank_sha256']==v['bank_sha256']==a['bank_sha256']==va['bank_sha256']==sha(training.OUT/'bank.npz');return p


def freeze():
    if P.exists():raise FileExistsError('Frozen bank development protocol exists')
    old=previous.check();bp=bank_check();engineering=json.loads((ROOT/'results/null_bank_engineering_v1.json').read_text());assert engineering['status']=='ENGINEERING_ONLY_COMPLETE' and engineering['bank_sha256']==sha(training.OUT/'bank.npz') and engineering['score_source_sha256']==sha(ROOT/'src/null_bank_score.py');p={k:old[k] for k in ['settings','regimes','families','environment']};p['settings']=dict(p['settings'],seed_base=16304598271);names=list(old['sources'])+['null_information_bank.py','test_null_information_bank.py','null_bank_score.py','test_null_bank_score.py','null_bank_development.py','null_bank_full_joint_control.py']
    p.update(status='FROZEN_BEFORE_FRESH_NULL_BANK_EFFICACY_DATA',sources={n:sha(ROOT/'src'/n) for n in names},previous_study_protocol_sha256=sha(previous.P),bank_training_protocol_sha256=sha(training.P),bank_artifacts_sha256={name:sha(training.OUT/name) for name in ['bank.npz','completion.json','validation.json','audit.json','validation_audit.json']},engineering_sha256=sha(ROOT/'results/null_bank_engineering_v1.json'),methods=METHODS,primary_method=PRIMARY,execution=dict(workers=8,threads={n:'1' for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),
             design='Fresh640queries256positive384null;ALLoriginal4continuousfractionalrecoveryregimes and7heterogeneousquery/contextfamilies,16independentdonors,photon/acquisitionbudgets,contrastsand2bintolerance preserved. Seed16304598271;bootstrap120000000000. BanktrainedONLYonindependentknownnullsimulations andvalidatedBEFOREanynewquery. No closedlabel/cohort reuse.',
             primary='Full-null3parameterMLEfromONEcensored-pilotwarmstart,identicalalgorithmobservedandbootstrapped. Thenfull64phasegradient,independentfixedbanklookupatrefittedtheta,nuisanceprojection,one-sidedquarterbinsecondaryscan. Reportrefittedqueryprimaryandmaximizingsecondary. KnownTd>=Te EXACToldcensoredscorepipeline includingbootstraparrays. Bankisapproximatefixedscoreweight,not exactfiniteFisher;outsideamplitude/backgroundgridclampsflagged,neverdropqueries.',
             ablations='2x2:empiricalsame-queryOPGvsindependentbank,andcensoredpilotvsfull-nullrefit. full_opg_safe recreates originalfullscore mathematically usingvalidatedreversekernel. full_opg_refit/full_bank_safe areablationsONLY,noposthocpromotion. PRIMARYfull_bank_refit. All4repeatownscore/pilot/lookup/search ineverycommonpath.',
             controls='All4strongcensoredmethods(score,6startjoint,8startcontextmax,transportedprofile) under BOTHcommonfull-nullandoriginalcensored-nullcalibration. Preplanned additionalfull-lawjointreference:6safealternativestartsevaluatedunderfullfactor,bestrefinedbyone5parameterMLE,full-nullwarmrefitrepeated. Fullreferenceisnumerical/local,notglobal. ItiscomputedforALL640queryandALL199commonbootstrapdrawsIFANDONLYIFtheprimarypreliminarygatepasses. ThiscomputationaldeferraldoesNOTwaivecomparison:confirmationstayFalseuntilstrongercontrolandallintegrity/warningauditsfinish.',
             calibration='199completecontinuouslatentSPAD/electronicsfilteredpaths perbranch. Commongeneratorquery3startfullnullMLEfrompreviousfrozenmethod. Everydrawrefitscensorednull,all4safeadaptivefits andall4fullscorepilot/weight/searchvariants. Originalbranchusesquerycensorednull andsamepairedrandomseedsfor8safecontrols. InTd>=Te copy identicalcommon/originalscores exactly. Composite-nullplug-inrankpvaluesapproximate,notfiniteuniform/evalues. Primaryalpha.01;alpha.005/.02descriptiveonly,nowinnerselection.',
             preliminary_rule='Atalpha.01PRIMARYstrictmorecorrectANDstrictfewertotalerrorsvsBOTHsafe_score_common/originalANDfull_opg_safe,nomorewrong/nullseparately. Atleastcorrectandnomorewrong/nullvsALLother8comparators(including2ablations). Thus all11comparatorsconstrainprimary. No ablationpromotion. Ifpass,mandatorystrongfulljointreference mustalsohavePRIMARYatleastcorrect,nomorewrongandnomorenullbeforefreshconfirmation;no speed-onlysubstitution.',
             numerical='Allfrozencontinuousmodel/reverseparityandbanktraining/validationhasheschecked. Optimizerwarnings(savedfullnullandcalibrationnull,observedalternatives,eachbootstrapdraw)andbankclampsretainedbydrawforindependentaudit. EightprocesseswithsingleSQLitewriter/inputcasecommitorder,BLAS1. Resumeverifiedcommittedrows only;do not restartalivewriters.',
             scope='New DEVELOPMENT,not confirmationofclosed640orofknown-modelinformationratios. Knownwidth/recovery/initialstate,piecewise64binrates. Expectedinformationweighting,MLE,MonteCarlo,interpolationandbootstrapareestablished. Novelty/physicalscope/Q1andjointutilityunproved;preliminarypassisnotpapercompletion.')
    for g in bp['regimes']:
        source=p['regimes'][g['index']];assert all(source[n]==g[n] for n in ['cycles','td','te'])
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());previous.check();bank_check();assert p['previous_study_protocol_sha256']==sha(previous.P) and p['bank_training_protocol_sha256']==sha(training.P)
    for n,h in p['sources'].items():assert sha(ROOT/'src'/n)==h,n
    for n,h in p['bank_artifacts_sha256'].items():assert sha(training.OUT/n)==h,n
    assert p['engineering_sha256']==sha(ROOT/'results/null_bank_engineering_v1.json');return p


def cases(p):
    for case in previous.cases(p):case['bootstrap_seed']=120000000000+1000003*case['index'];yield case


def worker(case,p):
    import hashlib
    start=time.perf_counter();cfg=p['settings'];g=p['regimes'][case['regime']];obs,ds,de,truth,params,dh=previous.old.base.data(case,p);lag=outward_safe_lag(g['td'],g['te']);s,e=numeric_statistics(obs,g['cycles'],64,lag);context=donor_features(ds,de,1.5);pilot=fit_null(s,e,1.5);geometry=kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);full=candidate.statistics(s,e,1.5,pilot,geometry,bank(),case['regime']);observed=previous.old.statistics(s,e,pilot,context,cfg);calibration=previous.fit_full_null(s,e,1.5,pilot,geometry);rates={name:theta[0]*pulse(theta[1],64,1.5)[0]+theta[2]/64 for name,theta in [('common',calibration['theta']),('original',pilot['theta'])]};boots={m:[] for m in METHODS};hashes={};warnings=[];clamps=[];diagnostics=[]
    for cal in ['common','original']:
        if cal=='original' and g['td']>=g['te']:
            assert np.array_equal(calibration['theta'],pilot['theta'])
            for name in BASE.values():boots[name+'_original']=list(boots[name+'_common'])
            hashes['original']=hashes['common'];continue
        h=hashlib.sha256();warm=calibration['theta'] if cal=='common' else pilot['theta']
        for draw in range(cfg['bootstrap_replicates']):
            bo,_=simulate_two_dead_times(rates[cal],g['cycles'],g['td'],g['te'],case['bootstrap_seed']+1009*draw);h.update(bo.tobytes());bs,be=numeric_statistics(bo,g['cycles'],64,lag);bf=refit_from_start(bs,be,warm,1.5,'duration');r=previous.old.statistics(bs,be,bf,context,cfg);full_failure=False
            for key,name in BASE.items():boots[name+'_'+cal].append(r['samples'][key]['score'])
            if cal=='common':
                bg=kernel.prepare_record(bo,g['cycles']*64,64,g['td'],g['te']);v=candidate.statistics(bs,be,1.5,bf,bg,bank(),case['regime']);full_failure=not v['full_null']['optimizer_success']
                for method in candidate.METHODS:boots[method].append(v['samples'][method]['score'])
                for kind,flags in v['bank_flags'].items():
                    if flags is not None and (flags['amplitude_clamped'] or flags['background_clamped']):clamps.append(dict(draw=draw,pilot=kind,flags=flags))
                if g['td']>=g['te']:
                    for method in candidate.METHODS:assert v['samples'][method]==r['samples']['local_score']
            failed=[i for i,f in enumerate(r['fits']) if not f['optimizer_success']]
            if not bf['optimizer_success'] or full_failure or failed:warnings.append(dict(calibration=cal,draw=draw,safe_null_failure=not bf['optimizer_success'],full_null_failure=full_failure,alternative_failure_components=failed))
            if case['id'].endswith('_000') and draw in [0,198]:diagnostics.append(dict(calibration=cal,draw=draw,safe_null_theta=bf['theta'].tolist(),safe_loglik=bf['loglik'],full_null=json_fit(v['full_null']) if cal=='common' else None))
        hashes[cal]=h.hexdigest()
    samples={}
    for cal in ['common','original']:
        for key,name in BASE.items():
            m=name+'_'+cal;stat=observed['samples'][key];samples[m]=dict(stat,pvalue=pvalue(stat['score'],boots[m]),bootstrap_scores=boots[m])
    for m in candidate.METHODS:
        stat=full['samples'][m];samples[m]=dict(stat,pvalue=pvalue(stat['score'],boots[m]),bootstrap_scores=boots[m])
    if g['td']>=g['te']:
        for m in candidate.METHODS:assert samples[m]==samples['safe_score_common']==samples['safe_score_original']
    return dict(case=case,truth=truth,generating_parameters=params,data_sha256=dh,observed_events=len(obs),scored_events=int(s.sum()),context={k:context[k] for k in ['primary','secondary','two_pulse_loglr']},safe_null=json_fit(pilot),calibration_null=json_fit(calibration),observed_full_null=json_fit(full['full_null']),observed_bank_flags=full['bank_flags'],observed_alternative_fits=observed['fits'],samples=samples,bootstrap_path_sha256=hashes,bootstrap_warnings=warnings,bootstrap_bank_clamps=clamps,refit_diagnostics=diagnostics,seconds=time.perf_counter()-start)


def execute(case):return worker(case,check())


def connect():
    OUT.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(OUT/'records.sqlite3');c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT NOT NULL)');return c


def read():
    c=connect();rows=[json.loads(z[0]) for z in c.execute('SELECT payload FROM records ORDER BY rowid')];c.close();return rows


def run():
    p=check();assert {n:os.environ.get(n) for n in p['execution']['threads']}==p['execution']['threads'];c=connect();done={r[0] for r in c.execute('SELECT id FROM records')};remaining=[x for x in cases(p) if x['id'] not in done];start=time.perf_counter();new=0;print('Independent bank8workers existing',len(done),'remaining',len(remaining),flush=True)
    with ProcessPoolExecutor(max_workers=p['execution']['workers']) as pool:
        for row in pool.map(execute,remaining,chunksize=1):
            c.execute('INSERT INTO records VALUES(?,?)',(row['case']['id'],json.dumps(row)));c.commit();new+=1
            if (len(done)+new)%8==0:print('Null bank development',len(done)+new,'/640 elapsed_s',round(time.perf_counter()-start,1),flush=True)
    c.close();assert [r['case'] for r in read()]==list(cases(p));result=dict(status='COMPLETE',cases=640,protocol_sha256=sha(P),records_sha256=sha(OUT/'records.sqlite3'),new_cases_this_invocation=new,preexisting_committed_cases=len(done),seconds_this_invocation=time.perf_counter()-start);(OUT/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);analyze()


def totals(rows,alpha):
    result={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in METHODS}
    for row in rows:
        for m in METHODS:
            for k,n in previous.old.endpoint(row,m,alpha).items():result[m][k]+=n
    return result


def rule(t):
    x=t[PRIMARY];out={}
    for m in [m for m in METHODS if m!=PRIMARY]:
        b=t[m];out[m]=dict(no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'])
        if m in ['safe_score_common','safe_score_original','full_opg_safe']:out[m].update(strict_more_correct=x['correct']>b['correct'],strict_fewer_errors=x['wrong']+x['false']<b['wrong']+b['false'])
        else:out[m]['at_least_correct']=x['correct']>=b['correct']
    return out


def analyze():
    p=check();rows=read();assert len(rows)==640;t=totals(rows,.01);r=rule(t);passed=all(all(z.values()) for z in r.values());groups=[]
    for gi,g in enumerate(p['regimes']):
        for family in p['families']['positive']+p['families']['null']:groups.append(dict(regime=g['name'],family=family,methods=totals([row for row in rows if row['case']['regime']==gi and row['case']['family']==family],.01)))
    result=dict(status='PRELIMINARY_PASS_STRONG_FULL_LR_CONTROL_PENDING' if passed else 'PRELIMINARY_JOINT_UTILITY_FAILED',totals=t,preliminary_rule=r,preliminary_rule_passed=passed,continue_to_fresh_confirmation=False,stronger_full_joint_control='MANDATORY_PENDING' if passed else 'PREPLANNED_NOT_RUN_BECAUSE_PRIMARY_FAILED',per_regime_family=groups,descriptive_operating_points={str(a):totals(rows,a) for a in [.005,.01,.02]},observed_full_null_warnings=sum(not row['observed_full_null']['optimizer_success'] for row in rows),bootstrap_full_null_warnings=sum(w['full_null_failure'] for row in rows for w in row['bootstrap_warnings']),bootstrap_safe_null_warnings=sum(w['safe_null_failure'] for row in rows for w in row['bootstrap_warnings']),bootstrap_alternative_warnings=sum(len(w['alternative_failure_components']) for row in rows for w in row['bootstrap_warnings']),bootstrap_bank_clamp_events=sum(len(row['bootstrap_bank_clamps']) for row in rows),scope=p['scope']);(OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_regime_family'},indent=2),flush=True)


def benchmark():
    bp=bank_check();p=previous.check();p=dict(p,settings=dict(p['settings'],seed_base=2947816301));result=[]
    for gi in range(4):
        case=next(x for x in previous.cases(p) if x['regime']==gi);g=p['regimes'][gi];obs,ds,de,truth,params,dh=previous.old.base.data(case,p);s,e=numeric_statistics(obs,g['cycles'],64,outward_safe_lag(g['td'],g['te']));pilot=fit_null(s,e,1.5);geometry=kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);cal=previous.fit_full_null(s,e,1.5,pilot,geometry);rate=cal['theta'][0]*pulse(cal['theta'][1],64,1.5)[0]+cal['theta'][2]/64;timings=[]
        for draw in range(4):
            bo,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],3159871027+1000003*gi+1009*draw);bs,be=numeric_statistics(bo,g['cycles'],64,outward_safe_lag(g['td'],g['te']));bf=refit_from_start(bs,be,cal['theta'],1.5,'duration');bg=kernel.prepare_record(bo,g['cycles']*64,64,g['td'],g['te']);start=time.perf_counter();all_=candidate.statistics(bs,be,1.5,bf,bg,bank(),gi);all_seconds=time.perf_counter()-start;start=time.perf_counter();one,nf,flags=candidate.primary_statistic(bs,be,1.5,bf,bg,bank(),gi);primary_seconds=time.perf_counter()-start;assert one==all_['samples'][PRIMARY];start=time.perf_counter();nf=candidate.refit_full_null(bs,1.5,bf,bg);six=reports(bs,be,1.5,bf,dict(primary=0.,secondary=0.,two_pulse_loglr=0.));joint=candidate.full_likelihood_ratio(bs,1.5,nf,six['fits'],bg);joint_seconds=time.perf_counter()-start;timings.append(dict(all4_full_variants_seconds=all_seconds,primary_only_seconds=primary_seconds,full_joint_with6safe_starts_seconds=joint_seconds,full_null_evaluations=nf['evaluations'],full_joint_evaluations=joint['evaluations']))
        result.append(dict(regime=g['name'],timings=timings))
    out=dict(status='ENGINEERING_ONLY_COMPLETE',regimes=result,bank_sha256=sha(training.OUT/'bank.npz'),score_source_sha256=sha(ROOT/'src/null_bank_score.py'),scope='4knownphysicalengineeringqueryrecords plus4nullpaths/regime;no efficacyendpoint/pvalue. All4variants andseparateprimarypass exactparity. Timingexcludescommoncensoredpilot/exposure/geometryconstruction;fulljointincludesitsnullrefit+6safealternativefits+6fulllikevaluations+1fullrefinement. Not isolated/end-to-end/hardware claim;not new efficacydata.');(ROOT/'results/null_bank_engineering_v1.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze,'benchmark':benchmark}[sys.argv[1]]()
