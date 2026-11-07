"""Pre-specified calibration ranks and query-level outcome definitions."""
from pathlib import Path
import hashlib,json,csv,math
import numpy as np
from scipy.io import loadmat
from scipy.stats import beta
from likelihood_baseline import fitted_flux_report

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'confirmation_v1';RESULTS=ROOT/'results'/'confirmation_v1'


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def value(report):
    return -np.inf if report.get('score') is None else float(report['score'])


def outcome(report,truth,threshold,proposed=False):
    declared=bool(report['declared']) if proposed else value(report)>threshold
    delay=report.get('delay')
    localized=bool(declared and truth['positive'] and delay is not None and abs(delay-truth['secondary_delay'])<=2.)
    return dict(declared=declared,localized=localized,
                wrong_depth=bool(declared and truth['positive'] and not localized))


def group_summary(records):
    n=len(records)
    if not n:return None
    keys=['declared','localized','wrong_depth']
    return dict(n=n,**{k:dict(count=sum(r[k] for r in records),rate=sum(r[k] for r in records)/n,
                             ci95=interval(sum(r[k] for r in records),n)) for k in keys})


def caspi_reports():
    path=RESULTS/'caspi_reports.json'
    if path.exists():return json.loads(path.read_text())['reports']
    reports={}
    for file in sorted(RESULTS.glob('caspi_output_*.mat')):
        d=loadmat(file,squeeze_me=True);ids=np.asarray(d['case_ids'],dtype=object).reshape(-1)
        flux=np.asarray(d['query_flux']);times=np.asarray(d['runtimes']).reshape(-1)
        if flux.ndim==1:flux=flux[None,:]
        for i,identifier in enumerate(ids):
            name=str(np.asarray(identifier).reshape(-1)[0])
            if name in reports:raise RuntimeError('Duplicate CASPI case')
            r=fitted_flux_report(flux[i]);r['seconds']=float(times[i])
            if not np.isfinite(r['score']):r['score']=None
            reports[name]=r
    if len(reports)==579:
        path.write_text(json.dumps(dict(reports=reports,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2,allow_nan=False))
    return reports


def main():
    completion=json.loads((RESULTS/'completion.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text());cases=manifest['cases']
    if completion['cases']!=1449 or len(cases)!=1449:raise RuntimeError('Incomplete CPU confirmation')
    reports={r['id']:json.loads((RESULTS/(r['id']+'.json')).read_text())['reports'] for r in cases}
    rt=json.loads((RESULTS/'rt3d_reports.json').read_text())
    for identifier,rr in rt['reports'].items():
        # The 100-iteration runs logged background-update errors in every
        # bucket. Exclude that entire planned configuration, including cases
        # without an individually attributed warning; never select successes.
        reports[identifier].update({m:v for m,v in rr.items() if m!='rt3d_100'})
    # The additional authors-default executable run terminated with exit 0 but
    # logged illegal CUDA memory accesses and failed FFTs. Its clouds are not
    # valid algorithm outputs: exclude them, rather than score them as misses.
    excluded=[]
    for folder in sorted((ROOT/'tools').glob('rt3d_*v1_*')):
        log=folder/'run.log'
        if not log.exists():continue
        lines=log.read_text(errors='replace').splitlines()
        bad=[line for line in lines if any(s in line.lower() for s in
             ['gpuassert','illegal memory','cufftexecr2c fails','cufftexecc2r fails','error bkg update'])]
        if bad:
            if 'authors_default' not in folder.name and 'v1_100_' not in folder.name:
                raise RuntimeError('Invalid primary RT3D output: '+folder.name)
            excluded.append(dict(run=folder.name,error_lines=len(bad),first_error=bad[0],
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
    if (RESULTS/'rt3d_author_default_reports.json').exists() and not any('authors_default' in r['run'] for r in excluded):
        raise RuntimeError('Additional reference requires explicit validity audit')
    caspi=caspi_reports()
    if len(caspi)==579:
        for identifier,rr in caspi.items():reports[identifier]['caspi']=rr
    methods=sorted(set(m for rr in reports.values() for m in rr))
    # Proposed is absent from calibration by design; all other controls use
    # their complete, pre-specified null calibration subset.
    result=dict(status='complete CPU and RT3D; CASPI complete' if len(caspi)==579 else 'CASPI pending',
                caspi_cases_available=len(caspi),freeze_sha256=manifest['freeze_sha256'],
                analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                methods={},per_case={},paired_differences={},excluded_runs=excluded,notes=[
                    'Primary threshold/calibration rules frozen before test.',
                    'Additional RT3D authors-default runs excluded: logged CUDA/FFT failures despite exit 0; not scored as zero recall.',
                    'All RT3D 100-iteration batches excluded: explicit background-update errors; planned attempted comparison failed.',
                    'Rates are not claimed exactly matched across methods.',
                    'No calibration data are used to set the proposed threshold.'])
    table=[]
    for method in methods:
        if method=='caspi' and len(caspi)!=579:continue
        cal=[reports[r['id']][method] for r in cases if r['split']=='calibration' and method in reports[r['id']]]
        if method=='proposed':threshold=None;rank=None
        else:
            rank=math.ceil((len(cal)+1)*.99)
            if rank>len(cal):raise RuntimeError('Insufficient calibration size')
            threshold=float(np.sort([value(r) for r in cal])[rank-1])
        records=[]
        for case in cases:
            if case['split']!='evaluation' or method not in reports[case['id']]:continue
            rr=reports[case['id']][method];meta=case['metadata']
            record=dict(id=case['id'],positive=meta['positive'],context=meta['context'],geometry=meta['geometry'],
                photon_level=meta['secondary_incident'],caspi_subset=case['caspi'],
                **outcome(rr,meta,threshold,method=='proposed'))
            records.append(record)
        null=[r for r in records if not r['positive']];positive=[r for r in records if r['positive']]
        summaries=dict(null=group_summary(null),positive=group_summary(positive),
            null_contexts={c:group_summary([r for r in null if r['context']==c]) for c in ['globally_single','hole','edge']},
            photon_levels={str(n):group_summary([r for r in positive if r['photon_level']==n]) for n in [3.,6.,10.]},
            positive_geometries={str(g):group_summary([r for r in positive if r['geometry']==g]) for g in range(5)},
            paired_caspi_subset=dict(null=group_summary([r for r in null if r['caspi_subset']]),
                positive=group_summary([r for r in positive if r['caspi_subset']])))
        result['methods'][method]=dict(calibration_n=len(cal),calibration_rank=rank,
            threshold=threshold if threshold is None or np.isfinite(threshold) else None,
            threshold_negative_infinity=bool(threshold is not None and threshold==-np.inf),summaries=summaries)
        result['per_case'][method]=records
        table.append(dict(method=method,calibration_n=len(cal),null_n=len(null),null_false=summaries['null']['declared']['count'],
            null_rate=summaries['null']['declared']['rate'],positive_n=len(positive),
            localized=summaries['positive']['localized']['count'],recall=summaries['positive']['localized']['rate'],
            wrong_depth=summaries['positive']['wrong_depth']['count']))
    # Nominal paired scene-bootstrap intervals, not multiplicity-adjusted tests.
    rng=np.random.default_rng(20261106)
    proposed={r['id']:r for r in result['per_case']['proposed']}
    for method,records in result['per_case'].items():
        if method=='proposed':continue
        shared=[r for r in records if r['id'] in proposed]
        differences={}
        for positive,key in [(False,'declared'),(True,'localized')]:
            subset=[r for r in shared if r['positive']==positive]
            delta=np.asarray([float(proposed[r['id']][key])-float(r[key]) for r in subset])
            boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(-1)
            differences['recall' if positive else 'false_declarations']=dict(n=len(delta),
                difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
        result['paired_differences'][method]=differences
    (RESULTS/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (RESULTS/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(result['status'])
    for r in table:print(r)


if __name__=='__main__':main()
