"""All path/decision checks and fixed full bootstrap replays,with strong-fit checks."""
import json,time,hashlib
import numpy as np
from scipy.stats import beta
import deadtime_bootstrap_development as d
from deadtime_path_bootstrap import bootstrap_paths,fixed_fit_scores,pvalue
from deadtime_scan import fit_null


def strip_time(row):
    v=json.loads(json.dumps(row))
    for s in v['samples'].values():
        s.pop('seconds',None);s.pop('stage_seconds',None)
    return v


def main():
    start=time.perf_counter();p=d.check();cfg=p['settings'];rows=d.read();expected=list(d.cases(p));assert [r['case'] for r in rows]==expected and len(rows)==384
    a=json.loads((d.OUT/'analysis.json').read_text());c=json.loads((d.OUT/'completion.json').read_text());hashes={n:d.sha(d.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']};assert c['records_sha256']==hashes['records.sqlite3'] and c['protocol_sha256']==d.sha(d.P)
    chosen={f'{gi}_{group}_000' for gi in range(4) for group in ['null','positive']};full=0;paths=drawfits=failures=0;maxgap=-np.inf;warnings=[]
    totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS};per=[{m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS} for gi in range(4)]
    for n,row in enumerate(rows):
        case=row['case'];g=p['regimes'][case['regime']];events,s,e,op,truth,dh=d.data(case,p);assert dh==row['data_sha256'] and truth==row['truth'] and len(events)==row['events']
        for mode in ['duration','ded']:
            ss,ee,parts=bootstrap_paths(np.array(row['fitted_nulls'][mode]),cfg['sigma'],g['cycles'],g['dead'],cfg['bins'],cfg['bootstrap_replicates'],case['bootstrap_seed'],mode);h=hashlib.sha256()
            for v in parts:h.update(v)
            assert h.hexdigest()==row['bootstrap_path_hashes'][mode];paths+=len(ss)
            if mode=='duration':
                scores,_=fixed_fit_scores(ss,ee,row['fitted_nulls'][mode],cfg['sigma'],mode,subdivision=cfg['subdivision']);np.testing.assert_array_equal(scores,row['samples']['duration_projected']['bootstrap_scores'])
            for diagnostic in row['refit_diagnostics']:
                if diagnostic['mode']!=mode:continue
                j=diagnostic['draw'];strong=fit_null(ss[j],ee[j],cfg['sigma'],mode);gap=strong['loglik']-diagnostic['loglik'];maxgap=max(maxgap,gap);drawfits+=1
                if gap>1e-5:warnings.append(dict(id=case['id'],mode=mode,draw=j,strong_minus_warm_loglik=gap))
        if case['id'] in chosen:
            assert strip_time(d.worker(case,p))==strip_time(row);full+=1
        for m in d.METHODS:
            v=row['samples'][m];boot=np.asarray(v['bootstrap_scores']);assert len(boot)==199 and np.all(np.isfinite(boot)) and v['pvalue']==pvalue(v['score'],boot)
            decisions=bool(v['score']>np.sort(boot)[-2]);assert decisions==(v['pvalue']<=.01)
            pos=case['group']=='positive';distance=lambda x,y:min(abs((x-y)%64),abs((y-x)%64))
            correct=pos and decisions and distance(v['primary'],truth['primary'])<=2 and distance(v['delay'],truth['secondary'])<=2
            ep=dict(correct=int(correct),wrong=int(pos and decisions and not correct),false=int(not pos and decisions),positive=int(pos),null=int(not pos))
            for k,z in ep.items():totals[m][k]+=z;per[case['regime']][m][k]+=z
        if (n+1)%64==0:print('Verified unknown-nuisance paths',n+1,'/',len(rows),flush=True)
    assert totals==a['totals']
    for gi,v in enumerate(per):assert v==a['per_regime'][gi]['methods']
    med={m:float(np.median([r['samples'][m]['seconds'] for r in rows])) for m in d.METHODS};assert med==a['median_composed_processing_seconds']
    x,b,z=[totals[m] for m in ['duration_projected','rounded_refit','duration_refit']]
    rule=dict(strict_more_correct=x['correct']>b['correct'],strict_fewer_total_errors=x['wrong']+x['false']<b['wrong']+b['false'],no_more_wrong=x['wrong']<=b['wrong'],no_more_null=x['false']<=b['false'],retains_ninetyfive_percent_refit=x['correct']>=.95*z['correct'],no_more_wrong_than_refit=x['wrong']<=z['wrong'],no_more_null_than_refit=x['false']<=z['false'])
    assert rule==a['rule'] and a['continue_to_fresh_efficacy_confirmation']==all(rule.values())
    intervals=[]
    for gi,v in enumerate(per):
        methods={}
        for m,r in v.items():
            f=r['false'];n=r['null'];lo=0 if f==0 else float(beta.ppf(.025,f,n-f+1));hi=1 if f==n else float(beta.ppf(.975,f+1,n-f));methods[m]=dict(false=f,null=n,clopper_pearson_95=[lo,hi])
        intervals.append(dict(regime=p['regimes'][gi]['name'],methods=methods))
    assert all(d.sha(d.OUT/n)==h for n,h in hashes.items())
    v=dict(status='PASS',observed_paths_regenerated=len(rows),bootstrap_paths_regenerated=paths,all_projected_bootstrap_scores_replayed=True,all_method_decisions_recounted=len(rows)*len(d.METHODS),fixed_full_workers_replayed=full,source_style_three_start_null_fits=drawfits,maximum_strong_minus_warm_loglik=maxgap,refit_control_warnings=warnings,original_times_and_results_retained=True,outputs_sha256=hashes,false_alarm_intervals_per_regime=intervals,seconds=time.perf_counter()-start,scope='All384raw paths/all152832bootstrap paths and all1536decisions verified;all projected scores replayed;8fixedfullworkers,notallfullnonlinearbootstrapfits. 64selectedbootstrapfits compared with source-style three-start controls;report any discrepancy. Binomial intervals per regime are descriptive,not proof of uniform1%control.')
    (d.OUT/'integrity.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='false_alarm_intervals_per_regime'},indent=2),flush=True)


if __name__=='__main__':main()
