"""Closed location experiment: complete worker replay and independent recount."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
import continuous_location_development as s
from ordered_audit import correct


def replay(pair):
    old,row=pair;actual=s.worker(old);expected=dict(row);actual.pop('elapsed_seconds');expected.pop('elapsed_seconds');assert actual==expected,row['id'];return row['id']


def main():
    start=time.perf_counter();p=s.check();rs=s.read();old={r['id']:r for r in s.parent.read()};sha=s.parent.base.original.sha
    assert {r['id']:r['case'] for r in rs}=={c['id']:c for c in p['cases']}
    hashes={n:sha(s.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']};a=json.loads((s.OUT/'analysis.json').read_text());totals={};errs={}
    for m in s.METHODS:
        groups={}
        for g in ['null_plain','null_lobe','far_two','near_two']:
            rows=[r for r in rs if r['case']['group']==g]
            v=dict(correct=sum(correct(r,m) for r in rows),wrong_positive_location=sum(r['case']['secondary']>0 and r['reports'][m]['declared'] and not correct(r,m) for r in rows),false_second_object=sum(r['case']['secondary']==0 and r['reports'][m]['declared'] for r in rows),declared=sum(r['reports'][m]['declared'] for r in rows),n=len(rows));assert v==a['groups'][m][g];groups[g]=v
        totals[m]=dict(correct=sum(groups[g]['correct'] for g in ['near_two','far_two']),wrong=sum(groups[g]['wrong_positive_location'] for g in ['near_two','far_two']),false=sum(groups[g]['false_second_object'] for g in ['null_plain','null_lobe']))
        error=np.array([r['reports'][m]['secondary']-r['truth']['secondary'] for r in rs if r['case']['secondary']>0 and r['reports'][m]['declared']]);errs[m]=dict(n=len(error),mae=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))))
    assert totals==a['totals'] and errs==a['declared_positive_errors']
    for row in rs:
        for b,key in [('point','point'),('ci','point_ci')]:
            assert row['reports'][b+'_old']==old[row['id']]['reports'][key]
            for e in ['window','profile']:
                before=dict(row['reports'][b+'_old']);after=dict(row['reports'][b+'_'+e]);before.pop('secondary');after.pop('secondary');assert before==after
        assert row['window']['working_window_mass']>=row['window']['old_working_window_mass']-1e-12
    eligible=[e for e in ['window','profile'] if totals['point_'+e]['correct']>totals['point_old']['correct'] and totals['point_'+e]['wrong']<totals['point_old']['wrong'] and totals['ci_'+e]['wrong']<=totals['ci_old']['wrong']]
    assert eligible==a['eligible'] and a['selected']==(max(eligible,key=lambda e:totals['point_'+e]['correct']) if eligible else None)
    done=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(replay,(old[r['id']],r)) for r in rs]):done.append(f.result())
    assert all(sha(s.OUT/n)==v for n,v in hashes.items())
    v=dict(status='PASS',records_fully_replayed=len(done),independent_groups_recounted=24,existence_reports_identical=True,selection_reproduced=True,error_metrics_reproduced=True,output_sha256=hashes,seconds=time.perf_counter()-start,scope='ALL768locationworkers fully replayed;existing detection likelihoods/covers unchanged and inherited from parent audited study. Negative development result retained.')
    (s.OUT/'integrity.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2),flush=True)


if __name__=='__main__':main()
