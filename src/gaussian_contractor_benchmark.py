"""Frozen closed18case serial whole-query decision benchmark,with strong control."""
from pathlib import Path
import argparse,ast,json,hashlib,sqlite3,time,platform
import numpy as np
from flint import arb
import coverage_sparse_benchmark as source
from guided_alternatives import guided_pool,fit_pool
from coverage_alternative import coverage_components,numeric_scores,selected_indices,sparse_numerator_lower,location_report
from adaptive_evidence import mixture_interval,feasible_local_null
from envelope_evidence import continuous_null_bound,gaussian_envelope,outward_float
from gaussian_decision_bound import decision_bound
from query_likelihood_contractor import gaussian_mass_upper

ROOT=source.ROOT;P=ROOT/'GAUSSIAN_CONTRACTOR_BENCHMARK_PROTOCOL.json';OUT=ROOT/'results/gaussian_contractor_benchmark_v1'
METHODS=['original','ordinary','contracted']


def graph():
    todo=[Path(__file__).name];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for n in ast.walk(ast.parse((ROOT/'src'/name).read_text())):
            names=([a.name for a in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) and n.module else [])
            for module in names:
                local=module.split('.')[0]+'.py'
                if (ROOT/'src'/local).exists() and local not in seen:todo.append(local)
    return {f'src/{n}':source.sha(ROOT/'src'/n) for n in sorted(seen)}


def freeze():
    if P.exists():raise FileExistsError('Immutable benchmark')
    old=source.verify()
    p=dict(status='FROZEN_BEFORE_CONTRACTOR_PIPELINE_TIMINGS_OR_DECISIONS',sources=graph(),cases=old['cases'],records_sha256=old['records_sha256'],parent_benchmark_protocol_sha256=source.sha(source.P),
        settings=dict(cycles=1000,fwhm_range=[4.5,5.5],initial_cells=32,max_cover_cells=256,repeats=3,math_threads=1,alpha='1/100'),
        methods=dict(original='Frozen v5fullnull solver,gap.01,max256cells',ordinary='Strongcontrol:sameoldcellbound,parentcaps,checkedfrontwitness anddecision-directedstop/fixed-numeratorrejection',contracted='Same2Ddelay/widthcover;Gaussian-windowmass/zero-binbackgroundlikelihoodcontractors,boundedconcave2parameterprofile;decision-level floor fordiscardedunboundeddomains'),
        design='SAME18CLOSEDqueries fixed by earlier sparsebenchmarkacrosslevels/families,selections notbasedonnewoutcomes. Recompute donorproposal/rates/numericselection/sparseArbnumerator from fullhistogram,hold allidentical acrossmethods.',
        timing='ONEserialPythonprocess,singlemaththread. ThreeLatinrotationspercase,eachmethodoccupieseachpositiononce. Gaussianenvelope/globalmasscachesclearedBEFOREeachpipeline. Timed in-memory pipeline includesdonorfit,propermixture,oldscreenpolicy,feasiblewitness,sparsecertification andnull backend;excludesdatasetI/O,JSONwriting,wholeimagework. OtherPCactivitynotcontrolled;notisolatedmachine,FPGA,energyorframerateclaim.',
        numerical='Same256covercellceiling,NOTsameprofilecount. Contractedhasextraoutside1Dprofiles,allincludedintiming. ReturnednulluppermustcoverentireGaussianfamily evenwhenquery-dependentcontractorsdiscarddomains.',
        continuation_rule='Allordinaryacceptedcasesretained;noincreasewrong/nullcounts;andEITHERstrictcorrectgain ORmedianpairedpercaseordinary/contractedwholepipelineruntime>=1.2. Descriptiveengineeringgate,NOTindependentefficacyorstatisticalsignificance.',
        limitations='Closed18cases,notrepresentativeimageorindependentaccuracy;identicaloriginalratehash/numerator/olddecisionsmustbereproduced. No new opticalcalibrationdata orlocationheuristics.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',source.sha(P),len(p['sources']),'sources',flush=True)


def check():
    p=json.loads(P.read_text());source.verify()
    for n,d in p['sources'].items():assert source.sha(ROOT/n)==d,n
    for c in p['cases']:
        assert source.sha(source.DATA/(c['id']+'.npz'))==c['sha256']
        assert source.sha(source.OLD/(c['id']+'.json'))==p['records_sha256'][c['id']]
    return p


def clear_caches():
    gaussian_envelope.cache_clear();gaussian_mass_upper.cache_clear()


def pipeline(h,method):
    clear_caches();start=time.perf_counter();pool,info=guided_pool(h,8,8);q=dict(**fit_pool(pool),**info);query=h[8,8]
    model=coverage_components(q);_,weighted,full=numeric_scores(query,model);location=location_report(model,weighted)
    old=mixture_interval(query,model['rates'][:4]);witness=feasible_local_null(query)
    cv=arb(100).log();cut=outward_float(cv);lowcut=outward_float(cv,False)
    candidate=(outward_float(arb(old['upper'])-arb(witness['lower']))>=lowcut or full-witness['lower']>=cut-1.)
    sparse=None;bound=None;score=None;declared=False;backend_seconds=0.
    if candidate:
        sparse=sparse_numerator_lower(query,model,selected_indices(weighted,16))
        level=outward_float(arb(sparse['lower'])-arb(cut),False);t=time.perf_counter()
        if method=='original':bound=continuous_null_bound(query,fwhm_range=(4.5,5.5),initial_cells=32,max_boxes=256,normalized_envelope=False)
        else:bound=decision_bound(query,level,method,initial_cells=32,max_boxes=256,witness=witness)
        backend_seconds=time.perf_counter()-t
        if bound['upper'] is not None:score=outward_float(arb(sparse['lower'])-arb(bound['upper']),False);declared=bool(score>=cut)
    elapsed=time.perf_counter()-start
    return dict(method=method,seconds=elapsed,backend_seconds=backend_seconds,proposal=q,rate_matrix_sha256=hashlib.sha256(model['rates'].tobytes()).hexdigest(),sparse_numerator=sparse,bound=bound,screened=not candidate,witness=witness,
        report=dict(declared=declared,primary=q['primary'],delay=location['delay'],head=location['head']),log_e_lower=score,threshold_upper=cut)


def read():
    db=sqlite3.connect(OUT/'records.sqlite3');rows=[json.loads(x[0]) for x in db.execute('SELECT data FROM records ORDER BY id')];db.close();return rows


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'records.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,data TEXT NOT NULL)');db.commit();ids={r[0] for r in db.execute('SELECT id FROM records')};started=time.perf_counter()
    for index,c in enumerate(p['cases']):
        if c['id'] in ids:continue
        h=np.load(source.DATA/(c['id']+'.npz'))['histogram'];old=json.loads((source.OLD/(c['id']+'.json')).read_text());samples=[]
        for repeat in range(3):
            rotation=(index+repeat)%3;order=METHODS[rotation:]+METHODS[:rotation]
            for method in order:
                sample=pipeline(h,method);sample['repeat']=repeat;sample['order_position']=order.index(method)
                assert sample['proposal']==old['proposal'] and sample['rate_matrix_sha256']==old['evidence']['rate_matrix_sha256']
                assert sample['screened']==old['evidence']['screened'] and sample['sparse_numerator']==old['evidence']['sparse_numerator']
                if method=='original':assert sample['bound']==old['evidence']['bound'] and sample['report']==old['reports']['coverage_cert']
                samples.append(sample)
        r=dict(id=c['id'],case=c,samples=samples);db.execute('INSERT INTO records VALUES(?,?)',(c['id'],json.dumps(r,allow_nan=False)));db.commit();ids.add(c['id']);print('Gaussian whole-query benchmark',len(ids),'/',len(p['cases']),flush=True)
    assert len(ids)==len(p['cases']);db.close();result=dict(status='COMPLETE',records=len(ids),seconds=time.perf_counter()-started,protocol_sha256=source.sha(P),sqlite_sha256=source.sha(OUT/'records.sqlite3'),python=platform.python_version(),numpy=np.__version__);(OUT/'completion.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def analyze():
    p=check();rows=read();assert len(rows)==len(p['cases']);percase=[];totals={m:dict(correct=0,wrong=0,false=0,declared=0) for m in METHODS}
    for row in rows:
        methods={};truth=row['case']['metadata']
        for m in METHODS:
            ss=[s for s in row['samples'] if s['method']==m];assert len(ss)==3
            canonical=[{k:v for k,v in s.items() if k not in ['seconds','backend_seconds','repeat','order_position']} for s in ss];assert all(v==canonical[0] for v in canonical)
            rep=ss[0]['report'];positive=truth['positive'];correct=bool(positive and rep['declared'] and abs(rep['primary']-truth['primary_delay'])<=2 and abs(rep['delay']-truth['secondary_delay'])<=2)
            ep=dict(correct=int(correct),wrong=int(positive and rep['declared'] and not correct),false=int(not positive and rep['declared']),declared=int(rep['declared']))
            for k in ep:totals[m][k]+=ep[k]
            methods[m]=dict(median_seconds=float(np.median([s['seconds'] for s in ss])),median_backend_seconds=float(np.median([s['backend_seconds'] for s in ss])),endpoint=ep,boxes=0 if ss[0]['bound'] is None else ss[0]['bound']['boxes_evaluated'])
        percase.append(dict(id=row['id'],methods=methods,ordinary_over_contracted=methods['ordinary']['median_seconds']/methods['contracted']['median_seconds']))
    ratios=[x['ordinary_over_contracted'] for x in percase];retained=all(not x['methods']['ordinary']['endpoint']['declared'] or x['methods']['contracted']['endpoint']['declared'] for x in percase)
    a,b=totals['contracted'],totals['ordinary'];rule=dict(all_ordinary_accepts_retained=retained,no_more_wrong=a['wrong']<=b['wrong'],no_more_false=a['false']<=b['false'],correct_gain_or_runtime=bool(a['correct']>b['correct'] or np.median(ratios)>=1.2))
    v=dict(status='CLOSED_SERIAL_PIPELINE_BENCHMARK_COMPLETE',totals=totals,percase=percase,median_paired_ordinary_over_contracted=float(np.median(ratios)),paired_ratio_range=[min(ratios),max(ratios)],sum_case_median_seconds={m:sum(x['methods'][m]['median_seconds'] for x in percase) for m in METHODS},rule=rule,continue_to_confirmation=all(rule.values()),protocol_sha256=source.sha(P),scope=p['limitations'],timing_scope=p['timing']);(OUT/'analysis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='percase'},indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run','analyze']);args=a.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run()
    else:analyze()
