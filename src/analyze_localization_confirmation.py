"""Analysis only: no fitted method or threshold may depend on these outcomes."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from scipy.stats import beta, binomtest
from localization_confirmation import verify, DATA, OUT, P

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def binomial_ci(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def stratified_indices(cases,n,rng):
    keys=[(c['secondary'],c['separation_family']) for c in cases]
    return np.hstack([rng.choice(ids,size=(n,len(ids)),replace=True)
        for key in sorted(set(keys))
        for ids in [np.flatnonzero(np.array([k==key for k in keys]))]])


def endpoint(c,r,declared):
    truth=c['metadata'];d=r['delay'];p=r.get('primary');positive=truth['positive']
    correct=bool(declared and positive and abs(d-truth['secondary_delay'])<=2.)
    any_channel=bool(declared and positive and min(abs(d-truth['secondary_delay']),
        abs(p-truth['secondary_delay']) if p is not None else np.inf)<=2.)
    pair=bool(declared and positive and p is not None and min(
        max(abs(p-truth['primary_delay']),abs(d-truth['secondary_delay'])),
        max(abs(d-truth['primary_delay']),abs(p-truth['secondary_delay'])))<=2.)
    return dict(declared=bool(declared),correct=correct,
        wrong_depth=bool(declared and positive and not correct),
        any_channel=any_channel,unordered_pair=pair)


def summary(rows,indices=None):
    out={'n':len(rows)}
    for metric in rows[0]:
        a=np.array([r[metric] for r in rows],float);k=int(a.sum())
        ci=binomial_ci(k,len(rows)) if indices is None else np.quantile(a[indices].mean(-1),[.025,.975]).tolist()
        out[metric]=dict(count=k,rate=float(a.mean()),ci95=ci,
            interval='exact binomial' if indices is None else 'fixed-stratum acquisition bootstrap')
        if indices is not None and k==0:
            # A zero bootstrap interval is degenerate. Independently distributed
            # Bernoulli risks need not coincide: AM-GM gives
            # P(all zero)=prod(1-p_i)<=(1-mean(p_i))**n.
            out[metric]['zero_event_average_risk_upper95']=float(1-.05**(1/len(rows)))
            out[metric]['boundary_note']='Bootstrap interval degenerates at zero; the additional one-sided bound uses independent Bernoulli trials and does not assert zero true risk.'
    return out


def main():
    p=verify();manifest=json.loads((DATA/'manifest.json').read_text());cases=manifest['cases']
    assert manifest['protocol_sha256']==sha(P)
    assert json.loads((OUT/'completion.json').read_text())['cases']==5199
    thresholds=json.loads((OUT/'thresholds.json').read_text());assert thresholds['protocol_sha256']==sha(P)
    rows={}
    for c in cases:
        r=json.loads((OUT/(c['id']+'.json')).read_text());assert r['protocol_sha256']==sha(P)
        rows[c['id']]=r['reports']
        assert r['reports']['spatial_report']['score']==r['reports']['complete_guided']['score']
    cal=[c for c in cases if c['split']=='calibration']
    null=[c for c in cases if c['split']=='evaluation' and c['kind']=='null']
    positive=[c for c in cases if c['split']=='evaluation' and c['kind']=='positive']
    stress=[c for c in cases if c['split']=='stress']
    rng=np.random.default_rng(20261108)
    boot_positive=stratified_indices(positive,5000,rng)
    boot_stress=stratified_indices(stress,5000,rng)
    boot_null=rng.integers(0,len(null),size=(5000,len(null)))
    out=dict(status='Fresh independent confirmation completed',protocol_sha256=sha(P),
        analysis_sha256=sha(Path(__file__)),primary='spatial_report_cal0.005',
        methods={},comparisons={},notes=[
            'Main positives have a fixed six-stratum design. Do not call their pooled intervals exact binomial.',
            'Fixed-calibration intervals condition on observed calibration acquisitions.',
            'Two-stage bootstrap additionally resamples calibration and is an uncertainty audit, not threshold reselection.',
            'Total error counts depend on the designed2000null/900positive composition; no population precision claim.',
            'Singleton stress results are separate and were not used to tune any method.'])
    configurations=[(name,a) for name in rows[cal[0]['id']] for a in [.005,.01]]+[('guided_certified',None)]
    table=[];records={}
    for name,alpha in configurations:
        label=name if alpha is None else f'{name}_cal{alpha}'
        th=None if alpha is None else thresholds['methods'][name][str(alpha)]['threshold']
        groups={};records[label]={}
        for group,seq,boot in [('null',null,None),('positive',positive,boot_positive),('singleton_stress',stress,boot_stress)]:
            rr=[]
            for c in seq:
                r=rows[c['id']][name]
                declared=r['declared'] if alpha is None else r['score']>th
                rr.append(endpoint(c,r,declared))
            records[label][group]=rr;groups[group]=summary(rr,boot)
        strata={}
        for level in [3.,6.,10.]:
            for family in [0,1]:
                ids=[i for i,c in enumerate(positive) if c['secondary']==level and c['separation_family']==family]
                strata[f'{level}_{family}']=summary([records[label]['positive'][i] for i in ids])
        out['methods'][label]=dict(method=name,alpha=alpha,threshold=th,summaries=groups,strata=strata,
            total_errors=groups['null']['declared']['count']+groups['positive']['wrong_depth']['count'])
        table.append(dict(method=label,null_false=groups['null']['declared']['count'],null_n=len(null),
            correct=groups['positive']['correct']['count'],wrong_depth=groups['positive']['wrong_depth']['count'],positive_n=len(positive),
            total_errors=out['methods'][label]['total_errors'],
            singleton_correct=groups['singleton_stress']['correct']['count'],singleton_wrong=groups['singleton_stress']['wrong_depth']['count'],singleton_n=len(stress)))
    primary='spatial_report_cal0.005'
    for label in out['methods']:
        comparison={}
        for group,metric,boot in [('positive','correct',boot_positive),('positive','wrong_depth',boot_positive),
                                  ('null','declared',boot_null),('singleton_stress','correct',boot_stress)]:
            a=np.array([r[metric] for r in records[primary][group]],int)
            b=np.array([r[metric] for r in records[label][group]],int);d=a-b
            won=int((d>0).sum());lost=int((d<0).sum())
            comparison[f'{group}_{metric}']=dict(count_difference=int(d.sum()),rate_difference=float(d.mean()),
                ci95=np.quantile(d[boot].mean(-1),[.025,.975]).tolist(),primary_only=won,control_only=lost,
                paired_exact_p=float(binomtest(won,won+lost,.5).pvalue) if won+lost else 1.,conditional_on_calibration=True)
        out['comparisons'][label]=comparison
    # Calibration uncertainty using common bootstrap samples for paired arms.
    b=1000;ci=rng.integers(0,len(cal),size=(b,len(cal)))
    pi=stratified_indices(positive,b,rng);ni=rng.integers(0,len(null),size=(b,len(null)))
    resampled={};audit={}
    for name in rows[cal[0]['id']]:
        cs=np.array([rows[c['id']][name]['score'] for c in cal]);rank=int(np.ceil((len(cal)+1)*(1-.005)))
        th=np.maximum(0.,np.partition(cs[ci],rank-1,axis=1)[:,rank-1])
        ps=np.array([rows[c['id']][name]['score'] for c in positive]);ns=np.array([rows[c['id']][name]['score'] for c in null])
        loc=np.array([abs(rows[c['id']][name]['delay']-c['metadata']['secondary_delay'])<=2 for c in positive])
        good=((ps[pi]>th[:,None])&loc[pi]).mean(-1)
        wrong=((ps[pi]>th[:,None])&~loc[pi]).mean(-1)
        false=(ns[ni]>th[:,None]).mean(-1)
        resampled[name]=(good,wrong,false)
        audit[name]=dict(correct_ci95=np.quantile(good,[.025,.975]).tolist(),
            wrong_depth_ci95=np.quantile(wrong,[.025,.975]).tolist(),null_false_ci95=np.quantile(false,[.025,.975]).tolist())
    out['two_stage_calibration_bootstrap']=dict(alpha=.005,replicates=b,methods=audit,comparisons={})
    for name,values in resampled.items():
        out['two_stage_calibration_bootstrap']['comparisons'][name]={k:np.quantile(a-z,[.025,.975]).tolist()
            for k,a,z in zip(['correct_difference','wrong_depth_difference','null_false_difference'],resampled['spatial_report'],values)}
    for a in [.005,.01]:
        # Exact null-tail tolerance interval from the continuous order-statistic law;
        # tied/floored scores can only make the tail conservative.
        r=int(np.ceil((len(cal)+1)*(1-a)))
        out.setdefault('calibration_tolerance',{})[str(a)]=dict(rank=r,n=len(cal),
            continuous_score_null_tail_ci95=[float(beta.ppf(.025,len(cal)+1-r,r)),float(beta.ppf(.975,len(cal)+1-r,r))],
            note='Across repeated calibration samples under exchangeability; not a conditional per-pixel guarantee. Ties/flooring conservative.')
    (OUT/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    (OUT/'endpoint_records.json').write_text(json.dumps(records))
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    for r in table: print(r)
    print('Primary paired comparisons',json.dumps(out['comparisons']['complete_guided_cal0.005'],indent=2))


if __name__=='__main__':main()
