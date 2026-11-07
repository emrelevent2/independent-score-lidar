"""Numerical-work/retained-likelihood audit on fixed closed study queries."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from fractions import Fraction
import argparse,datetime,hashlib,json,time
import numpy as np
from flint import arb
from coverage_confirmation import verify as method_verify
from coverage_alternative import coverage_components,numeric_scores,selected_indices,sparse_numerator_lower
from envelope_evidence import certified_log_likelihood,outward_float
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/coverage_confirmation_v1';OLD=ROOT/'results/coverage_confirmation_v1'
OUT=ROOT/'results/coverage_sparse_benchmark_v1';P=ROOT/'COVERAGE_SPARSE_BENCHMARK_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    assert not P.exists() and not OUT.exists();method_verify();m=json.loads((DATA/'manifest.json').read_text());cs=[]
    for family in ['coherent','singleton']:
        for level in [3.,6.,10.]:
            for sf in [0,1]:cs.append(next(c for c in m['cases'] if c.get('support_family')==family and c['secondary']==level and c['separation_family']==sf))
    for family in ['filament','crossing_step']:cs.extend([c for c in m['cases'] if c.get('support_family')==family][:2])
    cs.extend([c for c in m['cases'] if c['kind']=='null' and c['split']=='evaluation'][:2]);assert len(cs)==18
    p=dict(status='Frozen before full-component Arb sums/timings on18closed confirmatory queries.',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        own_source_sha256=sha(Path(__file__)),method_protocol_sha256=sha(ROOT/'COVERAGE_CONFIRMATORY_PROTOCOL.json'),cases=cs,
        records_sha256={c['id']:sha(OLD/(c['id']+'.json')) for c in cs},settings=dict(workers=2,global_top=16,cycles=1000),
        notes=['Descriptive numerical-work audit,not independent efficacy or a representative image-latency benchmark.',
            'Full and sparse lower-bound calculations timed once per case,alternating order; outward upper bounds untimed.',
            'Two concurrent single-thread tasks share CPU. No isolated latency,frame throughput,energy or FPGA claim.',
            'All exact original mixture weights retained; sparse/full log probability gap enclosed with outward arithmetic.',
            'Cases fixed across photon/separation strata and donor-support families; no favorable-query selection by output.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen18case sparse/full benchmark',flush=True)


def verify():
    p=json.loads(P.read_text());assert sha(Path(__file__))==p['own_source_sha256'];method_verify()
    assert sha(ROOT/'COVERAGE_CONFIRMATORY_PROTOCOL.json')==p['method_protocol_sha256'];return p


def upper_sum(values,weights,ids):
    terms=[arb(values[i])+(arb(weights[i].numerator)/weights[i].denominator).log() for i in ids]
    shift=max(float(v) for v in terms)
    return outward_float(arb(shift)+sum(((v-arb(shift)).exp() for v in terms),arb(0)).log(),True)


def worker(c,digest,order):
    assert sha(OLD/(c['id']+'.json'))==digest and sha(DATA/(c['id']+'.npz'))==c['sha256']
    old=json.loads((OLD/(c['id']+'.json')).read_text());h=np.load(DATA/(c['id']+'.npz'))['histogram'][8,8]
    m=coverage_components(old['proposal']);_,weighted,numeric=numeric_scores(h,m)
    ids=selected_indices(weighted,16).tolist();all_ids=list(range(len(m['weights'])));values={};seconds={}
    for key,indices in ([('sparse',ids),('full',all_ids)] if order%2==0 else [('full',all_ids),('sparse',ids)]):
        start=time.perf_counter();values[key]=sparse_numerator_lower(h,m,indices);seconds[key]=time.perf_counter()-start
    upper={i:certified_log_likelihood(h,m['rates'][i],1000,'first',upper=True) for i in all_ids}
    us=upper_sum(upper,m['weights'],ids);uf=upper_sum(upper,m['weights'],all_ids)
    low=max(0.,outward_float(arb(values['full']['lower'])-arb(us),False));high=max(0.,outward_float(arb(uf)-arb(values['sparse']['lower']),True))
    assert low<=high and values['sparse']['lower']<=uf
    assert values['full']['lower']-1e-8<=numeric<=uf+1e-8
    bounded=old['evidence']['bound'] is not None;decisions=None
    if bounded:
        u=old['evidence']['bound']['upper'];th=old['evidence']['threshold_upper']
        decisions={key:outward_float(arb(values[key]['lower'])-arb(u),False)>=th for key in values}
        assert decisions['sparse']==old['reports']['coverage_cert']['declared']
    return dict(id=c['id'],components_full=len(all_ids),components_sparse=len(ids),seconds=seconds,
        speed_ratio_full_over_sparse=seconds['full']/seconds['sparse'],log_probability_gap_interval=[low,high],
        full_log_interval=[values['full']['lower'],uf],sparse_log_interval=[values['sparse']['lower'],us],
        full_numeric_log_mixture=numeric,shared_bound_decisions=decisions,source_record_sha256=digest)


def run():
    p=verify();OUT.mkdir(exist_ok=True);assert not (OUT/'analysis.json').exists();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,c,p['records_sha256'][c['id']],i) for i,c in enumerate(p['cases'])]
        for i,job in enumerate(as_completed(jobs),1):records.append(job.result());print('Sparse/full benchmark',i,'/18',flush=True)
    records.sort(key=lambda r:r['id']);ratios=[r['speed_ratio_full_over_sparse'] for r in records];gaps=[r['log_probability_gap_interval'][1] for r in records]
    result=dict(status='Completed frozen descriptive numerical-work audit.',protocol_sha256=sha(P),n=18,records=records,
        full_components_range=[min(r['components_full'] for r in records),max(r['components_full'] for r in records)],sparse_components=20,
        median_speed_ratio=float(np.median(ratios)),speed_ratio_range=[min(ratios),max(ratios)],
        log_gap_upper_quantiles=np.quantile(gaps,[0,.5,1]).tolist(),
        scope='Original-weight sparse/full Arb sums checked on18closed observations; concurrent CPU timings,not independent efficacy or isolated/image/hardware performance.')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);a=ap.parse_args();freeze() if a.action=='freeze' else run()
