"""Full frozen-development replay and separate endpoint recount."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
from flint import arb
import background_lobe_development as s
from envelope_evidence import exact_gaussian,certified_log_likelihood


def verify(pair,row):
    r=s.worker(pair);r.pop('elapsed_seconds');saved=dict(row);saved.pop('elapsed_seconds');assert r==saved
    witnesses=0;c=row['case'];t=row['truth']
    if c['secondary']==0:
        g=exact_gaussian(256,t['primary'],t['width'],False)
        side=exact_gaussian(272,arb(t['primary'])+16,t['width'],False)[:256]
        amp=arb(c['primary'])/sum(g,arb(0))
        rates=[(amp*(x+arb(c['eta'])*y)+arb(c['background'])/256)/1000 for x,y in zip(g,side)]
        ll=certified_log_likelihood(row['query'],rates,1000,'first',upper=True)
        for n in s.BUDGETS:
            key=f'background_ci_{n}'
            if row['confidence'][key]['eta_upper']>=c['eta']:
                assert ll<=row['bounds'][key]['upper'];witnesses+=1
    return witnesses


def main():
    start=time.perf_counter();p=s.check_protocol();sha=s.original.sha
    rows=s.read_records(s.OUT);parents={r['id']:r for r in s.read_records(s.parent.OUT)}
    originals={r['id']:r for r in s.read_records(s.original.OUT)}
    assert {r['id']:r['case'] for r in rows}=={c['id']:c for c in p['cases']}
    snapshot={name:sha(s.OUT/name) for name in ['records.sqlite3','completion.json','analysis.json']}
    done=witnesses=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(verify,(originals[r['id']],parents[r['id']]),r) for r in rows]):
            witnesses+=f.result();done+=1
            if done%24==0:print('Audited background-aware',done,'of',len(rows),flush=True)
    a=json.loads((s.OUT/'analysis.json').read_text());totals={}
    def good(r,m):
        v=r['reports'][m];t=r['truth']
        return bool(r['case']['secondary']>0 and v['declared'] and abs(v['primary']-t['primary'])<=2 and abs(v['secondary']-t['secondary'])<=2)
    for m in s.METHODS:
        groups={}
        for group in ['null_plain','null_lobe','far_two','near_two']:
            rs=[r for r in rows if r['case']['group']==group]
            v=dict(correct=sum(good(r,m) for r in rs),
                wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][m]['declared'] and not good(r,m) for r in rs),
                false_second_object=sum(r['case']['secondary']==0 and r['reports'][m]['declared'] for r in rs),
                declared=sum(r['reports'][m]['declared'] for r in rs),n=len(rs))
            assert v==a['groups'][m][group];groups[group]=v
        totals[m]=dict(correct=sum(groups[g]['correct'] for g in ['near_two','far_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
    assert totals==a['totals']
    for n in s.BUDGETS:
        key=f'background_ci_{n}';diff=[r['confidence'][key]['eta_upper']-parents[r['id']]['confidence'][f'cal_{n}']['eta_upper'] for r in rows]
        c=dict(tighter=sum(x<0 for x in diff),looser=sum(x>0 for x in diff),same=sum(x==0 for x in diff),
            median_upper=float(np.median([r['confidence'][key]['eta_upper'] for r in rows])),
            realized_undercoverage=sum(r['confidence'][key]['eta_upper']<r['case']['eta'] for r in rows),
            positive_background_lower=sum(r['confidence'][key].get('background_per_bin_lower',0)>0 for r in rows))
        assert c==a['calibration'][str(n)]
        for control in [f'old_ci_{n}',f'point_{n}']:
            assert a['paired'][f'{key}_vs_{control}']==dict(gains=sum(good(r,key) and not good(r,control) for r in rows),losses=sum(good(r,control) and not good(r,key) for r in rows))
    pri=totals['background_ci_100000'];old=totals['old_ci_100000']
    criteria=dict(more_correct=pri['correct']>old['correct'],no_more_wrong=pri['wrong']<=old['wrong'],no_more_false=pri['false']<=old['false'])
    assert criteria==a['primary_rule'] and all(criteria.values())==a['continue_to_new_confirmation']
    assert all(sha(s.OUT/name)==digest for name,digest in snapshot.items())
    result=dict(status='PASS',protocol_sha256=sha(s.P),records_fully_refit=done,applicable_exact_physical_null_witness_checks=witnesses,
        group_endpoint_recounts=44,output_sha256=snapshot,seconds=time.perf_counter()-start,
        scope='Every new data/predictor/calibration endpoint/sparse numerator/full continuous bound/decision replayed; grouped and paired endpoints and calibration statistics independently recounted. Closed development only.')
    (s.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
