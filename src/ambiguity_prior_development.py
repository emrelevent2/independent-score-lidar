"""Closed10kcalibration prior development;keep original location estimator."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time,sqlite3
from flint import arb
import causal_calibration_development as parent
from response_ambiguity_prior import reweight
from fixed_irf_control import fixed_components
from shoulder_alternative import numeric_and_sparse
from shoulder_envelope import shoulder_null_bound
from fixed_irf_control import fixed_null_bound
from certified_ratio_screen import sparse_numerator_upper
from envelope_evidence import outward_float

ROOT=parent.ROOT;P=ROOT/'AMBIGUITY_PRIOR_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/ambiguity_prior_development_v1';METHODS=['point_ci','ambiguity_ci','point','ambiguity_point']


def freeze():
    if P.exists():raise FileExistsError('Immutable protocol')
    old=parent.check();sources=dict(old['sources'])
    for n in ['response_ambiguity_prior.py','ambiguity_prior_development.py']:sources['src/'+n]=parent.base.original.sha(ROOT/'src'/n)
    p=dict(status='FROZEN_BEFORE_RESPONSE_PRIOR_OUTCOMES',sources=sources,cases=old['cases'],parent_sqlite_sha256=parent.base.original.sha(parent.OUT/'records.sqlite3'),
        mechanism='AllocateHALFglobaldelaymass to knownoptical-lobe location donorprimary+16 with donorwidth;otherhalf staysuniform. Spatialmass/rates/amplitudegrid unchanged. Exactrational kernel weights;raw query excluded from all prior features.',
        primary='ambiguityCI vs strongpointCI;strictlymorecorrect andnoincreasewrong/null counts. ReweightedpointIRFcontrol alwaysreported. SAME10kcalibration,1000query,4096boxes.',
        location='Hold original UNWEIGHTEDpointQ location estimate FIXED to isolate evidence power allocation;onlyexistence numerator changes.',
        crucial_limitation='CLOSED96positives use separation16or64;priorcenter16 derivesfromKNOWNinstrument response,butcoincideswithone designedsurfacegroup. Anygain here MUST be tested onnewcontinuoussecondarylags with etaandseparationindependentlyvaried;cannot claim generalization fromthis cohort.',
        attribution='Prior power allocation is standard;no newgeneralstatisticaltheorem. Q remainsproper and independentofrawquery,so originalfinite-calibration conditional-model bound is retained.',scope='Closeddevelopmentonly;noQ1/SOTA/independentaccuracyorhardwareclaim,noextraCaldata.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',parent.base.original.sha(P),flush=True)


def check():
    p=json.loads(P.read_text());parent.check();sha=parent.base.original.sha
    for n,d in p['sources'].items():assert sha(ROOT/n)==d,n
    assert sha(parent.OUT/'records.sqlite3')==p['parent_sqlite_sha256'];return p


def worker(old):
    start=time.perf_counter();c=old['case'];h,_,truth=parent.base.original.simulate_case(c);pred=old['full_calibration_predictor'];q=pred['proposal'];eta=pred['fit']['eta'];cap=old['interval_confidence']['eta_upper']
    model=reweight(fixed_components(q,eta),q);num=numeric_and_sparse(h[4,4],model);upper=sparse_numerator_upper(h[4,4],model,num['indices']);methods={};reports={m:old['reports'][m] for m in ['point_ci','point']}
    for m,base_m,point in [('ambiguity_ci','point_ci',False),('ambiguity_point','point',True)]:
        cv=arb(100).log() if point else (arb(1000)/9).log();cut=outward_float(cv);w=old['methods'][base_m]['refined_null_witness'];ratioUpper=outward_float(arb(upper)-arb(w['lower']));screen=ratioUpper<outward_float(cv,False)
        rep=dict(declared=False,log_ratio_lower=None,log_ratio_upper=ratioUpper,log_threshold_upper=cut,primary=q['primary'],secondary=pred['numerator']['location'],head=pred['numerator']['head']);bound=None
        if not screen:
            level=outward_float(arb(num['lower'])-arb(cut),False)
            bound=fixed_null_bound(h[4,4],eta,max_boxes=4096,decision_level=level) if point else shoulder_null_bound(h[4,4],cap=cap,max_boxes=4096,decision_level=level)
            score=outward_float(arb(num['lower'])-arb(bound['upper']),False);rep.update(declared=score>=cut,log_ratio_lower=score)
        methods[m]=dict(bound=bound,certified_screened_out=screen,report=rep);reports[m]=rep
    return dict(id=c['id'],case=c,truth=truth,raw_histogram_sha256=old['raw_histogram_sha256'],prior=model['ambiguity_prior'],new_numerator=num,new_numerator_upper=upper,original_location_held_fixed=True,methods=methods,reports=reports,elapsed_seconds=time.perf_counter()-start)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rs=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rs


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();ids={x[0] for x in db.execute('SELECT id FROM records')};old={r['id']:r for r in parent.read()};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(worker,old[c['id']]) for c in p['cases'] if c['id'] not in ids]):
            r=f.result();db.execute('INSERT INTO records VALUES(?,?)',(r['id'],json.dumps(r)));db.commit();ids.add(r['id'])
            if len(ids)%12==0:print('Ambiguity prior',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close();v=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-start,protocol_sha256=parent.base.original.sha(P),sqlite_sha256=parent.base.original.sha(OUT/'records.sqlite3'));(OUT/'completion.json').write_text(json.dumps(v,indent=2));print(v,flush=True)


def analyze():
    p=check();rs=read();assert len(rs)==len(p['cases']);totals={};groups={}
    for m in METHODS:
        groups[m]={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            ep=[parent.base.original.endpoint(r,m) for r in rs if r['case']['group']==g];groups[m][g]={k:sum(e[k] for e in ep) for k in ep[0]}
        totals[m]=dict(correct=sum(groups[m][g]['correct'] for g in ['near_two','far_two']),wrong=sum(groups[m][g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[m][g]['false_second_object'] for g in ['null_plain','null_lobe']))
    x,y=totals['ambiguity_ci'],totals['point_ci'];rule=dict(more_correct=x['correct']>y['correct'],no_more_wrong=x['wrong']<=y['wrong'],no_more_false=x['false']<=y['false'])
    v=dict(status='CLOSED_AMBIGUITY_PRIOR_COMPLETE',totals=totals,groups=groups,rule=rule,continue_to_continuous_lag_confirmation=all(rule.values()),protocol_sha256=parent.base.original.sha(P),scope=p['scope'],limitation=p['crucial_limitation']);(OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='groups'},indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run','analyze']);args=a.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
