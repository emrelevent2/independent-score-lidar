"""Replay decision computations;retain original observed latency measurements."""
import json,time
import numpy as np
from flint import arb
import gaussian_contractor_benchmark_v2 as s
from envelope_evidence import exact_gaussian,certified_log_likelihood


def canonical(sample):
    # SQLite records are JSON: normalize tuple/list representation before
    # comparing fresh in-memory results. No scientific value is altered.
    return json.loads(json.dumps({k:v for k,v in sample.items() if k not in ['seconds','backend_seconds','repeat','order_position']}))


def main():
    start=time.perf_counter();p=s.check();rows=s.read();sha=s.source.sha
    hashes={n:sha(s.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']};a=json.loads((s.OUT/'analysis.json').read_text());totals={m:dict(correct=0,wrong=0,false=0,declared=0) for m in s.METHODS};percase=[];replays=points=0
    for row in rows:
        h=np.load(s.source.DATA/(row['id']+'.npz'))['histogram'];t=row['case']['metadata'];m={}
        g=exact_gaussian(256,t['primary_delay'],t['fwhm'],True);rates=[(arb(t['primary_incident'])*x+arb(t['background_incident'])/256)/1000 for x in g];ll=certified_log_likelihood(h[8,8],rates,1000,'first',upper=True)
        for method in s.METHODS:
            samples=[v for v in row['samples'] if v['method']==method];assert len(samples)==3 and sorted(v['order_position'] for v in samples)==[0,1,2]
            assert all(canonical(v)==canonical(samples[0]) for v in samples)
            actual=s.pipeline(h,method);assert canonical(actual)==canonical(samples[0]);replays+=1
            if actual['bound'] is not None and actual['bound']['upper'] is not None:
                assert actual['bound']['upper']>=ll and actual['bound']['upper']>=actual['witness']['lower'];points+=1
            r=actual['report'];correct=bool(t['positive'] and r['declared'] and abs(r['primary']-t['primary_delay'])<=2 and abs(r['delay']-t['secondary_delay'])<=2)
            ep=dict(correct=int(correct),wrong=int(t['positive'] and r['declared'] and not correct),false=int(not t['positive'] and r['declared']),declared=int(r['declared']))
            for k in ep:totals[method][k]+=ep[k]
            m[method]=dict(median_seconds=float(np.median([v['seconds'] for v in samples])),median_backend_seconds=float(np.median([v['backend_seconds'] for v in samples])),endpoint=ep,boxes=0 if actual['bound'] is None else actual['bound']['boxes_evaluated'])
        percase.append(dict(id=row['id'],methods=m,ordinary_over_contracted=m['ordinary']['median_seconds']/m['contracted']['median_seconds']))
        print('Verified whole-query benchmark',len(percase),'/',len(rows),flush=True)
    assert totals==a['totals'] and percase==a['percase'];ratios=[r['ordinary_over_contracted'] for r in percase]
    assert a['median_paired_ordinary_over_contracted']==float(np.median(ratios)) and a['paired_ratio_range']==[min(ratios),max(ratios)]
    retained=all(not r['methods']['ordinary']['endpoint']['declared'] or r['methods']['contracted']['endpoint']['declared'] for r in percase);x,y=totals['contracted'],totals['ordinary']
    rule=dict(all_ordinary_accepts_retained=retained,no_more_wrong=x['wrong']<=y['wrong'],no_more_false=x['false']<=y['false'],correct_gain_or_runtime=bool(x['correct']>y['correct'] or np.median(ratios)>=1.2));assert rule==a['rule'] and a['continue_to_confirmation']==all(rule.values())
    assert all(sha(s.OUT/n)==d for n,d in hashes.items())
    v=dict(status='PASS',queries=len(rows),entire_pipeline_replays=replays,primary_only_feasible_null_points_checked=points,timing_records_recounted=9*len(rows),latencies_not_expected_bit_identical=True,outputs_sha256=hashes,seconds=time.perf_counter()-start,scope='All18queries/all3pipelines replayed;properQ/hash/oldoutput/certifiedbounds/decisions verified. Originaltimings retainedandindependentlyrecounted,not replacedbyverificationtimings. Negative continuation gate retained.')
    (s.OUT/'integrity.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2),flush=True)


if __name__=='__main__':main()
