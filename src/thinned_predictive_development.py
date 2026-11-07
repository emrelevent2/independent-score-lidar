"""Fixed quarter-query-training development comparison on closed real data."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import argparse,datetime,hashlib,json,math,sqlite3,time
import numpy as np
from thinned_predictive import split_query,predict,score,POLICIES
from real_poisson_methods_v2 import WorkingPulse,null_fit,local_report
from real_poisson_analysis_v2 import endpoints,threshold
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/real_poisson_development_v1';OLD=ROOT/'results/real_poisson_development_v2'
OUT=ROOT/'results/thinned_predictive_development_v1';P=ROOT/'THINNED_PREDICTIVE_DEVELOPMENT_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['thinned_predictive.py','thinned_predictive_development.py','real_poisson_methods_v2.py','real_poisson_analysis_v2.py','real_poisson_development_v2.py','likelihood_baseline.py']


def freeze():
    assert not P.exists() and not OUT.exists();m=json.loads((DATA/'manifest.json').read_text())
    old={f'gaussian_{i}_{q["id"]}.json':sha(OLD/f'gaussian_{i}_{q["id"]}.json') for i in range(3) for q in m['queries']}
    p=dict(status='DEVELOPMENT on closed Sequence7record; fixed before all thinned-predictive outputs.',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},manifest_sha256=sha(DATA/'manifest.json'),old_records_sha256=old,
        settings=dict(training_fraction=.25,pseudocount=2.,minimum_training_events=3,seed_base=2026112300,
            amplitudes_full_exposure=[1.,2.,4.,8.,16.,32.],retentions=[.01,.025,.05],alphas=[.005,.01],primary_alpha=.01,primary_retention=.025),
        primary='shrunken_nuisance',policies=POLICIES,controls=['heldout_local','full_local'],
        continuation='Primary shrunken_nuisance mean across3retentions of pair-rate minus known-one-unmatched-rate must exceed full_local; atprimary2.5%retention pairs must increase andknown-one-unmatched mustnotincrease. Otherwise retain negative result and do not consume new count frames for this unchanged method.',
        notes=['Each query is binomially thinned once withseed_base+100000*retention_index+pixel. Train+test equals original query exactly.',
            'Poisson thinning gives marginal train/test independence under that ideal law,not conditional independence given observed parent,andnot sensor validation.',
            'Neighbor donor counts are unchanged and exclude full raw query. Trained variants explicitly mark raw_center_excluded false; scored_counts_excluded true.',
            'Atleast3training events required,otherwise every predictor falls back to donor parameters scaled for heldout exposure.',
            'Amplitude grid and donor counts scaled by.75; donor support-based spatial mass unchanged across ablations.',
            'query_delay updates primary delay only;query_nuisance updates all primary nuisance parameters;shrunken_nuisance usesfixed2pseudocount means/width weighting.',
            'Fitted signal/background allocations are not observed labels;shrinkage is not called an exact conjugate posterior.',
            'Denominator is a numerical heldout single-return fit;all scores require calibration,no analytic sensor-control claim.',
            'Every policy/control uses own threshold on same399known-one calibration queries. Full local uses all query photons; heldout local uses75%.',
            'All earlier cohorts are closed development for this later variant. No empirical best-policy switch or grid search.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen thinned-predictive2697case development',flush=True)


def verify():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(DATA/'manifest.json')==p['manifest_sha256'];return p


@lru_cache(maxsize=1)
def frame(i):return np.load(DATA/f'thin_{i}.npz')['histograms']


@lru_cache(maxsize=1)
def pulse():return WorkingPulse('gaussian')


def worker(q,i,settings,old_digest):
    source=OLD/f'gaussian_{i}_{q["id"]}.json';assert sha(source)==old_digest;r=json.loads(source.read_text());original=frame(i)[q['row'],q['column']]
    seed=settings['seed_base']+100000*i+q['pixel'];train,test=split_query(original,settings['training_fraction'],seed)
    training_fit=null_fit(train,pulse()) if train.sum()>=settings['minimum_training_events'] else None
    null=null_fit(test,pulse());local=local_report(test,pulse(),null=null);reports={};proposals={}
    for policy in POLICIES:
        proposal=predict(r['proposal'],training_fit,settings['training_fraction'],policy,settings['pseudocount']);proposals[policy]=proposal
        reports[policy]=score(test,proposal,null['log_likelihood'],np.array(settings['amplitudes_full_exposure'])*(1-settings['training_fraction']))
    reports.update(heldout_local=local,full_local=r['reports']['local'])
    sparse=lambda h:[[int(j),int(h[j])] for j in np.flatnonzero(h)]
    return dict(query_id=q['id'],index=i,seed=seed,source_sha256=old_digest,query_parent_sha256=hashlib.sha256(np.asarray(original,dtype=np.int64).tobytes()).hexdigest(),
        bins=len(test),training=sparse(train),heldout=sparse(test),training_fit=training_fit,heldout_null=null,proposals=proposals,reports=reports)


def run(workers):
    p=verify();m=json.loads((DATA/'manifest.json').read_text());OUT.mkdir(exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,protocol TEXT,payload TEXT,sha256 TEXT)');t=time.perf_counter()
    for entry in m['inputs']:
        i=entry['index'];assert sha(DATA/f'thin_{i}.npz')==entry['sha256'];present={r[0] for r in db.execute('SELECT id FROM records')};todo=[q for q in m['queries'] if f'{i}_{q["id"]}' not in present]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(worker,q,i,p['settings'],p['old_records_sha256'][f'gaussian_{i}_{q["id"]}.json']) for q in todo]
            for j,job in enumerate(as_completed(jobs),1):
                r=job.result();raw=json.dumps(r,allow_nan=False);key=f'{i}_{r["query_id"]}'
                db.execute('INSERT INTO records VALUES(?,?,?,?)',(key,sha(P),raw,hashlib.sha256(raw.encode()).hexdigest()));db.commit()
                if j%200==0:print('Thinned predictive',i,j,'/',len(todo),round(time.perf_counter()-t,1),flush=True)
    count=db.execute('SELECT COUNT(*) FROM records').fetchone()[0];assert count==2697;db.close()
    (OUT/'completion.json').write_text(json.dumps(dict(cases=count,protocol_sha256=sha(P),elapsed_seconds=time.perf_counter()-t),indent=2))


def analyze():
    p=verify();m=json.loads((DATA/'manifest.json').read_text());db=sqlite3.connect(OUT/'records.sqlite3');rows={}
    for key,digest,raw,d in db.execute('SELECT * FROM records'):
        assert digest==sha(P) and hashlib.sha256(raw.encode()).hexdigest()==d;rows[key]=json.loads(raw)
    db.close();assert len(rows)==2697;summary={};utility={k:[] for k in POLICIES+p['controls']};calibration={}
    for i,prob in enumerate(p['settings']['retentions']):
        for method in utility:
            cal=[rows[f'{i}_{q["id"]}']['reports'][method]['score'] for q in m['queries'] if q['split']=='calibration']
            for alpha in p['settings']['alphas']:
                cut=threshold(cal,alpha);key=f'{i}_{alpha}_{method}';calibration[key]=cut
                groups={g:dict(n=0,**{k:0 for k in ['declared','pair','weak_window_agreement','accepted_unmatched','primary_agreement','duplicate_known_window']}) for g in ['evaluation_one','evaluation_two']}
                for q in m['queries']:
                    if q['split'] not in groups:continue
                    report=rows[f'{i}_{q["id"]}']['reports'][method];e=endpoints(q,report,report['score']>cut);g=groups[q['split']];g['n']+=1
                    for field,v in e.items():g[field]+=v
                summary[key]=dict(method=method,retention=prob,alpha=alpha,threshold=cut,groups=groups)
                if alpha==.01:
                    one=groups['evaluation_one'];two=groups['evaluation_two'];utility[method].append(two['pair']/two['n']-one['accepted_unmatched']/one['n'])
                    print(prob,method,'pairs',two['pair'],'unmatchedone',one['accepted_unmatched'],flush=True)
    u={k:float(np.mean(v)) for k,v in utility.items()};primary=summary['1_0.01_shrunken_nuisance']['groups'];control=summary['1_0.01_full_local']['groups']
    criteria=dict(mean_utility_gain=u['shrunken_nuisance']>u['full_local'],primary_pair_gain=primary['evaluation_two']['pair']>control['evaluation_two']['pair'],
        primary_unmatched_not_increased=primary['evaluation_one']['accepted_unmatched']<=control['evaluation_one']['accepted_unmatched'])
    result=dict(status='DEVELOPMENT on closed recorded data; no independent sensor efficacy.',protocol_sha256=sha(P),methods=summary,mean_utility=u,criteria=criteria,
        proceed_to_new_counts=all(criteria.values()),notes=['Primary policy frozen; ablations do not choose a replacement primary.','Real annotations are not exhaustive physical-negative truth.'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));(OUT/'calibration.json').write_text(json.dumps(calibration,indent=2));print('CONTINUATION',criteria,u,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','analyze']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    freeze() if a.action=='freeze' else run(a.workers) if a.action=='run' else analyze()
