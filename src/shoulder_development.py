"""Frozen stylized optical-lobe development; no real-sensor efficacy claim."""
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,ast,hashlib,json,sqlite3,time,platform
import numpy as np
import scipy
from flint import arb
from pilot import first_detection_probabilities
from guided_alternatives import guided_pool,fit_pool
from coverage_alternative import coverage_components
from shoulder_alternative import donor_proposal,components,numeric_and_sparse,pulse
from calibration_confidence import lobe_confidence
from shoulder_envelope import shoulder_null_bound
from envelope_evidence import continuous_null_bound,outward_float

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'SHOULDER_DEVELOPMENT_PROTOCOL.json'
OUT=ROOT/'results/shoulder_development_v1'
METHODS=['old_gaussian','new_gaussian','new_uncalibrated','cal_1000','cal_10000','cal_100000']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_graph():
    pending=[Path(__file__).name];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);path=ROOT/'src'/name
        for node in ast.walk(ast.parse(path.read_text())):
            names=([a.name for a in node.names] if isinstance(node,ast.Import) else
                   [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in names:
                local=module.split('.')[0]+'.py'
                if (ROOT/'src'/local).exists() and local not in seen:pending.append(local)
    return {f'src/{name}':sha(ROOT/'src'/name) for name in sorted(seen)}


def cases():
    rows=[]
    for g,group in enumerate(['null_plain','null_lobe','far_two','near_two']):
        for i in range(24):
            rows.append(dict(id=f'{group}_{i:02}',group=group,seed=2026112400+1000*g+i,
                primary=[100.,300.][i%2],background=[5.,20.][(i//2)%2],
                secondary=[8.,16.,32.][(i//4)%3] if group.endswith('two') else 0.,
                separation=16 if group=='near_two' else 64,
                eta=[.0625,.125][(i//4)%2] if group in ['null_lobe','far_two'] else 0.,
                geometry=['coherent','isolated'][(i//12)%2]))
    return rows


def freeze():
    if P.exists():raise FileExistsError('Protocol is immutable; add a new version')
    p=dict(status='FROZEN_BEFORE_ANY_DEVELOPMENT_ACCURACY',sources=source_graph(),cases=cases(),
        settings=dict(cycles=1000,bins=256,cap=.25,shift=16,calibration_cycles=[1000,10000,100000],
            primary_calibration_cycles=10000,calibration_failure_budget='1/1000',
            total_error_budget='1/100',calibrated_query_budget='9/1000',
            fwhm_range=[4.5,5.5],max_boxes=256,initial_cells=32,location_tolerance=2,
            sparse_global_components=16,rejection_screen=False),
        primary_comparison='cal_10000 versus old_gaussian, paired on identical queries/donors',
        continuation_rule=dict(all_required=True,
            more_correct_physical_two_surfaces='cal_10000 strictly exceeds old_gaussian over both two-surface groups',
            fewer_false_second_objects='cal_10000 strictly fewer declarations than old_gaussian over both physical-null groups',
            calibration_benefit='cal_10000 strictly more correct than new_uncalibrated over near_two'),
        secondary='Other calibration budgets and shared-numerator Gaussian/uncalibrated controls diagnostic; no primary switch.',
        model='Independent first-event cycles, repeatable optical lobe, fixed known shift, Gaussian main/side width, uniform background; common eta shared by independent calibration/query/donors.',
        exclusions='No correlated afterpulsing, calibration drift, FPGA speedup, actual sensor validity, imagewise/FDR guarantee or independent confirmation claimed.',
        design='96 new stylized development scenes, balanced designed groups, independent noise/calibration per case. Counts are not real-world prevalence.',
        numerical='NumPy-double multinomial simulation; all reported acceptances require Arb lower sparse proper-mixture numerator and continuous-cover null upper.',
        calibration_design='Known single-surface d in [31.5,32.5], w in [4.5,5.5]; main peak per-cycle height .01, uniform background total .005. Nested independent multinomial increments; do not minimum-aggregate bounds across budgets.',
        software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),len(p['sources']),'sources',flush=True)


def check_protocol():
    p=json.loads(P.read_text())
    for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
    assert p['cases']==cases()
    return p


def simulate_case(c):
    rng=np.random.default_rng(c['seed']);t=np.arange(256)
    d=float(rng.uniform(70.,90.));w=float(rng.uniform(4.5,5.5));s=d+c['separation']
    main=c['primary']*pulse(t,d,w,c['eta'])+c['background']/256
    second=c['secondary']*pulse(t,s,w,c['eta'])
    h=rng.multinomial(1000,first_detection_probabilities(main/1000),size=(9,9))[...,:-1]
    if c['secondary']>0:
        if c['geometry']=='coherent':
            h=rng.multinomial(1000,first_detection_probabilities((main+second)/1000),size=(9,9))[...,:-1]
        else:h[4,4]=rng.multinomial(1000,first_detection_probabilities((main+second)/1000))[:-1]
    cd=float(rng.uniform(31.5,32.5));cw=float(rng.uniform(4.5,5.5))
    g=np.exp(-4*np.log(2)*((t-cd)/cw)**2)
    side=np.exp(-4*np.log(2)*((t-(cd+16))/cw)**2)
    probs=first_detection_probabilities(.01*(g+c['eta']*side)+.005/256)
    cal=[];previous=0;total=np.zeros(256,dtype=np.int64)
    for n in [1000,10000,100000]:
        total+=rng.multinomial(n-previous,probs)[:-1];previous=n
        cal.append(total.copy())
    truth=dict(primary=d,secondary=s,width=w,calibration_delay=cd,calibration_width=cw)
    return h,cal,truth


def worker(c):
    start=time.perf_counter();h,cal,truth=simulate_case(c);query=h[4,4]
    pool,meta=guided_pool(h,4,4);old=fit_pool(pool);old.update(meta)
    new=donor_proposal(h)
    om=coverage_components(old);om['spatial']=np.arange(len(om['rates']))<4
    on=numeric_and_sparse(query,om);nn=numeric_and_sparse(query,components(new))
    confidence=[lobe_confidence(x,n) for x,n in zip(cal,[1000,10000,100000])]
    threshold=outward_float(arb(100).log());cal_threshold=outward_float((arb(1000)/9).log())
    gaussian=continuous_null_bound(query,fwhm_range=(4.5,5.5),max_boxes=256,
                                    initial_cells=32,normalized_envelope=False)
    caps=[.25]+[x['eta_upper'] for x in confidence];bounds={}
    # The stricter calibrated decision level is used even for a cached cap=M.
    # Every return retains a complete cover; only acceptance can stop early.
    decision_level=outward_float(arb(nn['lower'])-arb(cal_threshold),False)
    for cap in sorted(set(caps)):
        bounds[str(cap)]=shoulder_null_bound(query,cap=cap,decision_level=decision_level)
    reports={}
    for method in METHODS:
        numerator=on if method=='old_gaussian' else nn
        proposal=old if method=='old_gaussian' else new
        if method.endswith('gaussian'):bound=gaussian;cut=threshold;cap=0.
        else:
            cap=.25 if method=='new_uncalibrated' else confidence[[1000,10000,100000].index(int(method.split('_')[1]))]['eta_upper']
            bound=bounds[str(cap)];cut=threshold if method=='new_uncalibrated' else cal_threshold
        score=outward_float(arb(numerator['lower'])-arb(bound['upper']),False)
        reports[method]=dict(declared=score>=cut,log_e_lower=score,log_threshold_upper=cut,
            primary=proposal['primary'],secondary=numerator['location'],head=numerator['head'],
            numerator_lower=numerator['lower'],null_upper=bound['upper'],eta_cap=cap)
    return dict(id=c['id'],case=c,truth=truth,query=query.tolist(),
        raw_histogram_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
        calibration_histograms=[x.tolist() for x in cal],confidence=confidence,
        proposals=dict(old=old,new=new),numerators=dict(old=on,new=nn),
        gaussian_bound=gaussian,shoulder_bounds=bounds,reports=reports,
        elapsed_seconds=time.perf_counter()-start)


def connect():
    OUT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
    db.commit();return db


def run(workers):
    p=check_protocol();start=time.perf_counter();db=connect()
    ids={x[0] for x in db.execute('SELECT id FROM records')}
    pending=[c for c in p['cases'] if c['id'] not in ids]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs={ex.submit(worker,c):c['id'] for c in pending}
        for j in as_completed(jobs):
            row=j.result();db.execute('INSERT INTO records VALUES (?,?)',(row['id'],json.dumps(row)))
            db.commit();ids.add(row['id'])
            if len(ids)%12==0:print('Completed',len(ids),'of',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close()
    completion=dict(status='COMPLETE',cases=len(ids),new_records_this_invocation=len(pending),
        invocation_seconds=time.perf_counter()-start,protocol_sha256=sha(P),sqlite_sha256=sha(OUT/'records.sqlite3'))
    (OUT/'completion.json').write_text(json.dumps(completion,indent=2));print(completion,flush=True)


def endpoint(r,method):
    report=r['reports'][method];positive=r['case']['secondary']>0;declared=report['declared']
    correct=bool(positive and declared and abs(report['primary']-r['truth']['primary'])<=2 and
                 abs(report['secondary']-r['truth']['secondary'])<=2)
    return dict(correct=int(correct),wrong_positive_location=int(positive and declared and not correct),
                false_second_object=int(not positive and declared),declared=int(declared),n=1)


def analyze():
    p=check_protocol();db=connect();rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close()
    assert len(rows)==len(p['cases'])
    summary={}
    for method in METHODS:
        groups={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            endpoints=[endpoint(r,method) for r in rows if r['case']['group']==group]
            groups[group]={k:sum(x[k] for x in endpoints) for k in endpoints[0]}
        summary[method]=groups
    correct=lambda m:sum(summary[m][g]['correct'] for g in ['far_two','near_two'])
    false=lambda m:sum(summary[m][g]['false_second_object'] for g in ['null_plain','null_lobe'])
    criteria=dict(more_correct=correct('cal_10000')>correct('old_gaussian'),
        fewer_false_second_objects=false('cal_10000')<false('old_gaussian'),
        calibration_benefit=summary['cal_10000']['near_two']['correct']>summary['new_uncalibrated']['near_two']['correct'])
    paired={}
    for base in ['old_gaussian','new_gaussian','new_uncalibrated']:
        gains=losses=0
        for r in rows:
            a=endpoint(r,'cal_10000')['correct'];b=endpoint(r,base)['correct']
            gains+=a>b;losses+=a<b
        paired[base]=dict(correct_gains=gains,correct_losses=losses)
    calibration={str(n):dict(eta_upper_median=float(np.median([r['confidence'][j]['eta_upper'] for r in rows])),
        informative=sum(r['confidence'][j]['informative'] for r in rows),
        realized_undercoverage=sum(r['confidence'][j]['eta_upper']<r['case']['eta'] for r in rows),
        detected_photons_median=float(np.median([r['confidence'][j]['detected'] for r in rows])))
        for j,n in enumerate([1000,10000,100000])}
    result=dict(status='DEVELOPMENT_COMPLETE',methods=summary,criteria=criteria,
        continue_to_fresh_confirmation=all(criteria.values()),paired=paired,calibration=calibration,
        primary='cal_10000 frozen before outcomes; no best-budget substitution',
        limitations=p['exclusions'],protocol_sha256=sha(P))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','analyze']);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run(args.workers)
    else:analyze()
