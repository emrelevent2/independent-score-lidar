"""Final analysis of separately frozen methods, thresholds and paired queries.

Positive outcomes use a fixed six-stratum design: intervals/bootstrap retain
photon-level x separation strata. Null rates have exact binomial intervals.
The role-invariant endpoints are additional, declared before accuracy inspection.
"""
from pathlib import Path
import json,hashlib,math,csv
import numpy as np
from scipy.io import loadmat
from scipy.stats import beta
from likelihood_baseline import fitted_flux_report
from continuous_alternative import alternative_at
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'guided_confirmation_v1';OUT=ROOT/'results'/'guided_confirmation_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def exact_interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def value(report):return -np.inf if report.get('score') is None else float(report['score'])


def caspi_reports(expected):
    cache=OUT/'caspi_reports.json'
    if cache.exists():
        saved=json.loads(cache.read_text());assert saved['main_protocol_sha256']==sha(ROOT/'GUIDED_CONFIRMATORY_PROTOCOL.json')
        assert set(saved['reports'])==expected;return saved['reports']
    reports={}
    for file in sorted(OUT.glob('caspi_output_*.mat')):
        d=loadmat(file,squeeze_me=True);ids=np.asarray(d['case_ids'],dtype=object).reshape(-1)
        flux=np.asarray(d['query_flux']);times=np.asarray(d['runtimes']).reshape(-1)
        if flux.ndim==1:flux=flux[None,:]
        assert flux.shape==(len(ids),256) and np.all(np.isfinite(flux))
        for i,raw in enumerate(ids):
            identifier=str(np.asarray(raw).reshape(-1)[0]);assert identifier not in reports
            report=fitted_flux_report(flux[i]);report['seconds']=float(times[i])
            if not np.isfinite(report['score']):report['score']=None
            reports[identifier]=report
    if set(reports)!=expected:raise RuntimeError(f'CASPI pending: {len(reports)}/{len(expected)} outputs')
    cache.write_text(json.dumps(dict(reports=reports,main_protocol_sha256=sha(ROOT/'GUIDED_CONFIRMATORY_PROTOCOL.json')),indent=2,allow_nan=False))
    return reports


def bootstrap_indices(cases,replicates,rng):
    # Preserve the designed photon x separation composition in every resample.
    keys=[(c['metadata']['secondary_incident'],c['separation_family']) for c in cases]
    groups={key:np.array([i for i,k in enumerate(keys) if k==key]) for key in sorted(set(keys))}
    return np.hstack([rng.choice(ids,size=(replicates,len(ids)),replace=True) for ids in groups.values()])


def stats(records,cases,bootstrap=None):
    n=len(records);positive=bool(cases[0]['metadata']['positive'])
    out=dict(n=n)
    for key in ['declared','localized','wrong_depth','weak_any_channel','unordered_pair']:
        values=np.array([r[key] for r in records],float);k=int(values.sum())
        if positive and bootstrap is not None:
            interval=np.quantile(values[bootstrap].mean(-1),[.025,.975]).tolist();kind='stratified acquisition bootstrap'
        else:interval=exact_interval(k,n);kind='exact binomial'
        out[key]=dict(count=k,rate=k/n,ci95=interval,interval_method=kind)
    return out


def main():
    manifest=json.loads((DATA/'manifest.json').read_text());cases=manifest['cases'];p=json.loads((ROOT/'GUIDED_CONFIRMATORY_PROTOCOL.json').read_text())
    assert len(cases)==4899 and manifest['protocol_sha256']==sha(ROOT/'GUIDED_CONFIRMATORY_PROTOCOL.json')
    assert json.loads((OUT/'completion.json').read_text())['cases']==4899
    assert json.loads((OUT/'complete_guided'/'completion.json').read_text())['cases']==4899
    assert json.loads((OUT/'pointcloud'/'completion.json').read_text())['cases']==579
    # Enforce every frozen method and comparator source before inspecting scores.
    for protocol,field in [('GUIDED_CONFIRMATORY_PROTOCOL.json','source_sha256'),('COMPLETE_GUIDED_COMPARATOR_PROTOCOL.json','source_sha256'),('POINTCLOUD_COMPARATOR_PROTOCOL.json','own_source_sha256')]:
        proto=json.loads((ROOT/protocol).read_text())
        for n,digest in proto[field].items():assert sha(ROOT/'src'/n)==digest,n
    expected={c['id'] for c in cases if c['caspi']};caspi=caspi_reports(expected)
    reports={}
    for c in cases:
        r=json.loads((OUT/(c['id']+'.json')).read_text());assert r['protocol_sha256']==manifest['protocol_sha256']
        reports[c['id']]=r['reports']
        extra=json.loads((OUT/'complete_guided'/(c['id']+'.json')).read_text())
        reports[c['id']]['complete_guided_likelihood']=extra['report']
        if c['caspi']:
            reports[c['id']]['caspi']=caspi[c['id']]
            reports[c['id']].update(json.loads((OUT/'pointcloud'/(c['id']+'.json')).read_text())['reports'])
    calibrated=sorted({m for rr in reports.values() for m,r in rr.items() if 'score' in r and 'declared' not in r})
    analytic=sorted({m for rr in reports.values() for m,r in rr.items() if 'declared' in r})
    primary_cache={};recompute_count=0
    def primary(case,method,report):
        nonlocal recompute_count
        if 'primary' in report:return report['primary']
        rr=reports[case['id']]
        if method in ['guided_certified','guided_full256','guided_fast']:return rr['guided_fast']['proposal']['primary']
        if method.startswith('oracle_'):return case['metadata']['primary_delay']
        if method.startswith('uniform9') or method.startswith('local3'):
            size=9 if method.startswith('uniform9') else 3;key=(case['id'],size)
            if key not in primary_cache:
                h=np.load(DATA/(case['id']+'.npz'))['histogram'];_,q=alternative_at(h,8,8,size=size)
                assert abs(q['secondary']-report['delay'])<1e-7
                primary_cache[key]=q['primary'];recompute_count+=1
                if recompute_count%1000==0:print('Verified donor-coordinate reconstructions',recompute_count,flush=True)
            return primary_cache[key]
        return None
    result=dict(status='Complete final comparison',protocol_sha256=manifest['protocol_sha256'],analysis_sha256=sha(Path(__file__)),
        primary_metric='Declared secondary within2bins; role-invariant metrics added in pre-inspection addendum.',
        positive_interval_note='Fixed six-stratum design: stratified bootstrap, not pooled exact binomial.',
        full={},paired={},paired_differences={},notes=['Calibrated modes have marginal exchangeability guarantees; certified modes have model-conditional guarantees.',
            'Paired CASPI/neural controls all recalibrated on same199 null observations.',
            'Pretrained neural reference is a transfer pipeline with additional local reporting; not general published-method accuracy.',
            'No final test data have tuned method settings.'])
    table=[];rng=np.random.default_rng(20261107)
    for cohort in ['full','paired']:
        available=[c for c in cases if cohort=='full' or c['caspi']]
        # Full-study results cannot silently include smaller-subset methods.
        methods=[m for m in calibrated if all(m in reports[c['id']] for c in available)]+analytic
        evaluation=[c for c in available if c['split']=='evaluation'];null=[c for c in evaluation if not c['metadata']['positive']];positive=[c for c in evaluation if c['metadata']['positive']]
        boot=bootstrap_indices(positive,5000,rng)
        configurations=[(m,None) for m in methods if m in analytic]+[(m,a) for m in methods if m in calibrated for a in [.005,.01]]
        for method,alpha in configurations:
            label=method if alpha is None else method+f'_cal{alpha}'
            cal=[c for c in available if c['split']=='calibration' and method in reports[c['id']]]
            if alpha is None:threshold=None;rank=None
            else:
                rank=math.ceil((len(cal)+1)*(1-alpha));threshold=max(0.,float(np.sort([value(reports[c['id']][method]) for c in cal])[rank-1]))
            records=[]
            for c in evaluation:
                report=reports[c['id']][method];truth=c['metadata'];declared=bool(report['declared']) if alpha is None else value(report)>threshold
                if method=='guided_fast':declared=declared and report['proposal']['valid']
                delay=report.get('delay');d0=primary(c,method,report)
                localized=bool(declared and truth['positive'] and delay is not None and abs(delay-truth['secondary_delay'])<=2.)
                weak=pair=False
                if declared and truth['positive'] and delay is not None:
                    locs=[delay] if d0 is None else [delay,d0]
                    weak=any(abs(x-truth['secondary_delay'])<=2. for x in locs)
                    if d0 is not None:
                        pair=min(max(abs(d0-truth['primary_delay']),abs(delay-truth['secondary_delay'])),
                                 max(abs(delay-truth['primary_delay']),abs(d0-truth['secondary_delay'])))<=2.
                records.append(dict(id=c['id'],positive=truth['positive'],declared=bool(declared),localized=localized,
                    wrong_depth=bool(declared and truth['positive'] and not localized),weak_any_channel=bool(weak),unordered_pair=bool(pair),
                    geometry=truth['geometry'],level=truth['secondary_incident'],separation_family=c['separation_family']))
            lookup={r['id']:r for r in records};nr=[lookup[c['id']] for c in null];pr=[lookup[c['id']] for c in positive]
            summary=dict(null=stats(nr,null),positive=stats(pr,positive,boot),
                strata={f'{level}_{family}':None for level in [3.,6.,10.] for family in [0,1]})
            for key in summary['strata']:
                level,family=key.split('_');level=float(level);family=int(family)
                ids=[i for i,c in enumerate(positive) if c['metadata']['secondary_incident']==level and c['separation_family']==family]
                summary['strata'][key]=stats([pr[i] for i in ids],[positive[i] for i in ids])
            result[cohort][label]=dict(method=method,alpha=alpha,threshold=threshold,calibration_n=len(cal) if alpha else 0,
                calibration_rank=rank,oracle=method.startswith('oracle_'),summaries=summary,records=records)
            table.append(dict(cohort=cohort,method=label,null_n=len(null),null_false=summary['null']['declared']['count'],
                positive_n=len(positive),localized=summary['positive']['localized']['count'],wrong=summary['positive']['wrong_depth']['count'],
                weak_any=summary['positive']['weak_any_channel']['count'],pair_localized=summary['positive']['unordered_pair']['count']))
        # Shared acquisition bootstrap, preserving positive strata, conditional
        # on the observed calibration sample. A separate two-stage audit can
        # resample calibration uncertainty without altering any decisions.
        anchor=result[cohort]['guided_fast_cal0.005']['records'];a={r['id']:r for r in anchor}
        null_boot=rng.integers(0,len(null),size=(5000,len(null)))
        result['paired_differences'][cohort]={}
        for label,entry in result[cohort].items():
            if entry['oracle']:continue
            b={r['id']:r for r in entry['records']};d={}
            for group,key,indexes in [(null,'declared',null_boot),(positive,'localized',boot),(positive,'weak_any_channel',boot),(positive,'unordered_pair',boot)]:
                values=np.array([float(a[c['id']][key])-float(b[c['id']][key]) for c in group])
                d['false' if key=='declared' else key]=dict(n=len(values),difference=float(values.mean()),
                    ci95=np.quantile(values[indexes].mean(-1),[.025,.975]).tolist(),conditional_on_calibration=True)
            result['paired_differences'][cohort][label]=d
    result['recomputed_primary_fits']=recompute_count
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print('Final comparison complete; full study below (paired subset is separate)')
    for r in table:
        if r['cohort']=='full':print(r)


if __name__=='__main__':main()
