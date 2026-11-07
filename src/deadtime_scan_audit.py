"""Audit all development paths/rank decisions;replay fixed full pipelines."""
import json,time,hashlib
import numpy as np
import deadtime_scan_development as d


def main():
    start=time.perf_counter();p=d.check();rows=d.read();expected=list(d.cases(p));a=json.loads((d.OUT/'analysis.json').read_text());completion=json.loads((d.OUT/'completion.json').read_text())
    hashes={n:d.sha(d.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']}
    assert hashes['records.sqlite3']==completion['records_sha256'] and completion['protocol_sha256']==d.sha(d.P)
    assert [r['case'] for r in rows]==expected and len(rows)==1308
    # Deterministic prespecified diagnostic sample: first and last held-out
    # null/positive of each regime,never chosen based on observed errors.
    chosen={f'{gi}_{group}_{j:03}' for gi in range(4) for group in ['test_null','test_positive'] for j in [0,63]}
    replayed=0;events_checked=0
    for row in rows:
        case=row['case'];events,s,e,op,truth=d.data(case,p);h=hashlib.sha256()
        for v in [events,s,e,op]:h.update(v.tobytes())
        assert h.hexdigest()==row['data_sha256'] and truth==row['truth'] and len(events)==row['events'];events_checked+=len(events)
        assert sorted(v['method'] for v in row['samples'])==sorted(d.METHODS)
        if case['id'] in chosen:
            for method in d.METHODS:
                actual=d.pipeline(s,e,op,case,p,method);saved=next(v for v in row['samples'] if v['method']==method)
                assert {k:v for k,v in actual.items() if k!='seconds'}=={k:v for k,v in saved.items() if k!='seconds'}
                replayed+=1
    # Independent recount,using the strict threshold equivalent to rank p<=.01.
    totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS};med={m:[] for m in d.METHODS};decisions=0
    for gi in range(4):
        for method in d.METHODS:
            get=lambda r:next(v for v in r['samples'] if v['method']==method)
            cal=np.array([get(r)['score'] for r in rows if r['case']['regime']==gi and r['case']['group']=='calibration_null']);assert len(cal)==199;threshold=np.sort(cal)[-2]
            per=dict(correct=0,wrong=0,false=0,positive=0,null=0)
            for r in rows:
                if r['case']['regime']!=gi or r['case']['group']=='calibration_null':continue
                v=get(r);declared=v['score']>threshold;pv=(1+np.count_nonzero(cal>=v['score']))/200;assert declared==(pv<=.01);decisions+=1;pos=r['case']['group']=='test_positive'
                distance=lambda x,y:min(abs((x-y)%64),abs((y-x)%64))
                correct=pos and declared and distance(v['primary'],r['truth']['primary'])<=2 and distance(v['delay'],r['truth']['secondary'])<=2
                ep=dict(correct=int(correct),wrong=int(pos and declared and not correct),false=int(not pos and declared),positive=int(pos),null=int(not pos))
                for k,z in ep.items():per[k]+=z;totals[method][k]+=z
                med[method].append(v['seconds'])
            saved=a['per_regime'][gi]['methods'][method]
            assert per=={k:saved[k] for k in per} and saved['calibration_second_largest_threshold']==threshold
    med={m:float(np.median(v)) for m,v in med.items()};assert med==a['median_method_pipeline_seconds'] and totals==a['totals']
    x,b,g=[totals[m] for m in ['duration_efficient','rounded_ded','duration_glrt']]
    rule=dict(more_correct_than_rounded=x['correct']>b['correct'],no_more_wrong_than_rounded=x['wrong']<=b['wrong'],no_more_null_than_rounded=x['false']<=b['false'],retains_ninety_percent_glrt=x['correct']>=.9*g['correct'],no_more_wrong_than_glrt=x['wrong']<=g['wrong'],no_more_null_than_glrt=x['false']<=g['false'],twice_faster_than_glrt=med['duration_glrt']>=2*med['duration_efficient'])
    assert rule==a['rule'] and a['continue_to_fresh_confirmation']==all(rule.values()) and not all(rule.values())
    assert all(d.sha(d.OUT/n)==z for n,z in hashes.items())
    v=dict(status='PASS',all_generated_paths_regenerated=len(rows),raw_detections_replayed=events_checked,all_rank_decisions_recounted=decisions,fixed_full_pipeline_replays=replayed,fixed_cases=len(chosen),original_timings_retained=True,original_failed_continuation_rule_retained=True,outputs_sha256=hashes,seconds=time.perf_counter()-start,scope='All1308data/predictors/thresholds/endpoints checked;16fixedheldout cases x5fullpipelines replayed,not all6540methodfits. MonteCarlocalibration limitedto specifiedsimulatorregime;no physicaloruniformcompositenullguarantee.')
    (d.OUT/'integrity.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2),flush=True)


if __name__=='__main__':main()
