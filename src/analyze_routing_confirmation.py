"""Frozen-policy routing evaluation; outcomes never change the chosen rule."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.stats import beta
from routing_confirmation import verify,ROOT,DATA,OUT,P
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def ci(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def boot_indices(cases,n,rng):
    keys=[(c['secondary'],c['separation_family']) for c in cases]
    return np.hstack([rng.choice(ids,size=(n,len(ids)),replace=True)
        for key in sorted(set(keys)) for ids in [np.flatnonzero([k==key for k in keys])]])


def endpoint(c,r,declared):
    m=c['metadata'];s=r['delay'];p=r['primary'];positive=m['positive']
    correct=bool(declared and positive and abs(s-m['secondary_delay'])<=2.)
    any_channel=bool(declared and positive and min(abs(p-m['secondary_delay']),abs(s-m['secondary_delay']))<=2.)
    pair=bool(declared and positive and min(max(abs(p-m['primary_delay']),abs(s-m['secondary_delay'])),
        max(abs(s-m['primary_delay']),abs(p-m['secondary_delay'])))<=2.)
    return dict(declared=bool(declared),correct=correct,wrong_depth=bool(declared and positive and not correct),any_channel=any_channel,pair=pair)


def summarize(records,boot=None):
    n=len(records);out=dict(n=n)
    for key in records[0]:
        values=np.array([r[key] for r in records],float);k=int(values.sum())
        interval=ci(k,n) if boot is None else np.quantile(values[boot].mean(-1),[.025,.975]).tolist()
        out[key]=dict(count=k,rate=k/n,ci95=interval,interval='exact binomial' if boot is None else 'six-stratum acquisition bootstrap')
        if k==0 and boot is not None:out[key]['zero_event_average_risk_upper95']=float(1-.05**(1/n))
    return out


def main():
    p=verify();digest=sha(P);manifest=json.loads((DATA/'manifest.json').read_text());cases=manifest['cases']
    assert len(cases)==9099 and manifest['protocol_sha256']==digest
    assert json.loads((OUT/'completion.json').read_text())['cases']==9099
    thresholds=json.loads((OUT/'thresholds.json').read_text());assert thresholds['protocol_sha256']==digest
    rows={}
    for c in cases:
        r=json.loads((OUT/(c['id']+'.json')).read_text());assert r['id']==c['id'] and r['protocol_sha256']==digest
        rows[c['id']]=r['reports']
    cal=[c for c in cases if c['split']=='calibration']
    groups={'null':[c for c in cases if c['split']=='evaluation' and c['kind']=='null']}
    groups.update({f:[c for c in cases if c['kind']=='positive' and c['support_family']==f] for f in ['coherent','singleton','filament','crossing_step']})
    rng=np.random.default_rng(20261109)
    boot={g:None if g=='null' else boot_indices(seq,5000,rng) for g,seq in groups.items()}
    boot_null=rng.integers(0,len(groups['null']),size=(5000,len(groups['null'])))
    result=dict(status='Independent routing confirmation completed',protocol_sha256=digest,analysis_sha256=sha(Path(__file__)),
        primary='routed_cal0.005',methods={},paired_comparisons={},notes=[
            'Primary routing policy and gate were frozen before all9099inputs.',
            'False-null and wrong-positive-location counts are separate.',
            'Null calibration targets are marginal across exchangeable acquisitions, not model-conditional guarantees.',
            'Two new geometry families are stress results; no primary variant was selected from their outcomes.',
            'Descriptive noninferiority margins and bootstrap intervals do not establish general sensor robustness.'])
    records={};table=[]
    methods=list(rows[cal[0]['id']])
    for method in methods:
        for alpha in [.005,.01]:
            label=f'{method}_cal{alpha}';threshold=thresholds['methods'][method][str(alpha)]['threshold'];summaries={};records[label]={}
            for g,seq in groups.items():
                rr=[endpoint(c,rows[c['id']][method],rows[c['id']][method]['score']>threshold) for c in seq]
                records[label][g]=rr;summaries[g]=summarize(rr,boot[g])
            strata={}
            for g,seq in groups.items():
                if g=='null':continue
                strata[g]={}
                for level in [3.,6.,10.]:
                    for sf in [0,1]:
                        ids=[i for i,c in enumerate(seq) if c['secondary']==level and c['separation_family']==sf]
                        strata[g][f'{level}_{sf}']=summarize([records[label][g][i] for i in ids])
            result['methods'][label]=dict(method=method,alpha=alpha,threshold=threshold,summaries=summaries,strata=strata)
            row=dict(method=label,null_false=summaries['null']['declared']['count'])
            for g in ['coherent','singleton','filament','crossing_step']:
                row[g+'_correct']=summaries[g]['correct']['count'];row[g+'_wrong']=summaries[g]['wrong_depth']['count']
            table.append(row)
    primary='routed_cal0.005'
    for label in records:
        comparisons={}
        for g in groups:
            for key in (['declared'] if g=='null' else ['correct','wrong_depth','any_channel','pair']):
                a=np.array([r[key] for r in records[primary][g]],float);b=np.array([r[key] for r in records[label][g]],float);d=a-b
                samples=d[boot_null if g=='null' else boot[g]].mean(-1)
                comparisons[g+'_'+key]=dict(count_difference=int(d.sum()),rate_difference=float(d.mean()),
                    ci95=np.quantile(samples,[.025,.975]).tolist(),lower_one_sided95=float(np.quantile(samples,.05)),
                    primary_only=int((d>0).sum()),control_only=int((d<0).sum()),conditional_on_calibration=True)
        result['paired_comparisons'][label]=comparisons
    # Include calibration uncertainty without changing any recorded decisions.
    reps=1000;ci_idx=rng.integers(0,len(cal),size=(reps,len(cal)))
    eval_idx={g:rng.integers(0,len(seq),size=(reps,len(seq))) if g=='null' else boot_indices(seq,reps,rng)
        for g,seq in groups.items() if g in ['null','coherent','singleton']}
    resampled={};audit={}
    for method in methods:
        cs=np.array([rows[c['id']][method]['score'] for c in cal]);rank=int(np.ceil((len(cal)+1)*.995))
        ts=np.maximum(0.,np.partition(cs[ci_idx],rank-1,axis=1)[:,rank-1]);resampled[method]={}
        audit[method]={}
        for g,index in eval_idx.items():
            seq=groups[g];score=np.array([rows[c['id']][method]['score'] for c in seq]);declared=score[index]>ts[:,None]
            if g=='null':values={'declared':declared.mean(-1)}
            else:
                localized=np.array([abs(rows[c['id']][method]['delay']-c['metadata']['secondary_delay'])<=2. for c in seq])
                values={'correct':(declared&localized[index]).mean(-1),'wrong_depth':(declared&~localized[index]).mean(-1)}
            for k,v in values.items():
                resampled[method][g+'_'+k]=v;audit[method][g+'_'+k]=np.quantile(v,[.025,.975]).tolist()
    comparisons={m:{k:dict(ci95=np.quantile(resampled['routed'][k]-v,[.025,.975]).tolist(),
        lower_one_sided95=float(np.quantile(resampled['routed'][k]-v,.05))) for k,v in values.items()}
        for m,values in resampled.items()}
    result['two_stage_bootstrap']=dict(replicates=reps,alpha=.005,methods=audit,paired=comparisons)
    result['prespecified_descriptive_criteria']=dict(
        coherent_gain_vs_local=comparisons['local_fit']['coherent_correct']['lower_one_sided95']>0,
        coherent_not_more_than2pp_below_spatial=comparisons['spatial_report']['coherent_correct']['lower_one_sided95']>-.02,
        singleton_not_more_than5pp_below_local=comparisons['local_fit']['singleton_correct']['lower_one_sided95']>-.05,
        note='Prespecified descriptive bootstrap criteria; no claim of exact finite-sample noninferiority coverage.')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False));(OUT/'endpoint_records.json').write_text(json.dumps(records))
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    for row in table:print(row)
    print('Criteria',result['prespecified_descriptive_criteria'])


if __name__=='__main__':main()
