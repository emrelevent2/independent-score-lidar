"""Recompute new bank scores and current external-method comparisons."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,hashlib,math,sqlite3
from fractions import Fraction
import numpy as np
from scipy.special import logsumexp
from real_primary_bank_development import ROOT,DATA,OLD,OUT,P,sha,verify,frame
from real_coverage_mixture import components
from real_primary_bank import bank_report
from real_poisson_methods_v2 import WorkingPulse


def bank_worker(q,i):
    path=OUT/f'{i}_{q["id"]}.json';r=json.loads(path.read_text());assert r['protocol_sha256']==sha(P)
    original=OLD/f'gaussian_{i}_{q["id"]}.json';old=json.loads(original.read_text());saved=r['coverage'];h=frame(i);query=h[q['row'],q['column']]
    logs=[];candidates=[];n=0;bank=saved['donor_proposals']
    for j,qz in enumerate(bank):
        assert qz['raw_center_excluded'];m=components(qz,len(query));assert sum(m['weights'],Fraction(0))==1
        rates=m['means'];z=np.log(rates)@query-rates.sum(1)+np.log([float(w) for w in m['weights']])-np.log(len(bank))
        logs.append(float(logsumexp(z)));n+=len(z)
        candidates.append((float(logsumexp(z[:4])),qz['primary'],qz['secondary'],'spatial',j))
        g=logsumexp(z[4:].reshape(len(m['global_delays']),len(m['amplitudes'])),axis=1);k=int(np.argmax(g));near=abs(m['global_delays']-m['global_delays'][k])<=2
        delay=float(np.exp(g[near]-logsumexp(g[near]))@m['global_delays'][near]);candidates.append((float(g[k]),qz['primary'],delay,'global',j))
    best=max(candidates,key=lambda c:c[0]);full=float(logsumexp(logs))
    assert saved['log_mixture']==full and saved['score']==full-old['null']['log_likelihood']
    assert (saved['primary'],saved['delay'],saved['head'],saved['selected_bank'])==best[1:]
    assert saved['components']==n and saved['bank_size']==len(bank)
    return path.name,sha(path),sha(original)


def audit_bank(workers):
    p=verify();m=json.loads((DATA/'manifest.json').read_text());hashes={};regenerated=[]
    assert json.loads((OUT/'completion.json').read_text())['cases']==2697
    for entry in m['inputs']:
        i=entry['index'];assert sha(DATA/f'thin_{i}.npz')==entry['sha256']
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs={pool.submit(bank_worker,q,i):q for q in m['queries']}
            for job in as_completed(jobs):
                name,digest,original=job.result();q=jobs[job]
                assert original==p['old_record_sha256'][f'gaussian_{i}_{q["id"]}.json'];hashes[name]=digest
        h=np.load(DATA/f'thin_{i}.npz')['histograms'];pulse=WorkingPulse('gaussian')
        for split in ['calibration','evaluation_one','evaluation_two']:
            q=next(q for q in m['queries'] if q['split']==split);old=json.loads((OLD/f'gaussian_{i}_{q["id"]}.json').read_text())
            expected=bank_report(h,pulse,q['row'],q['column'],old['null']['log_likelihood']);assert expected==json.loads((OUT/f'{i}_{q["id"]}.json').read_text())['coverage']
            regenerated.append(f'{i}_{q["id"]}')
        print('Audited primary bank',len(hashes),flush=True)
    (OUT/'result_sha256.json').write_text(json.dumps(dict(protocol_sha256=sha(P),records=hashes),indent=2))
    result=dict(status='PASS',cases=len(hashes),all_saved_bank_mixture_scores_and_reports_recomputed=True,
        donor_bank_regenerations=regenerated,scope='Frozen sources,original records,input hashes,unit prior mass,every new report and9complete donor-bank regenerations; not independent efficacy or physical calibration.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


def external_worker(c,settings,digest):
    from coverage_external_comparison import worker
    return worker(c,settings,digest)


def audit_external(workers):
    import coverage_external_comparison as ext
    p=ext.verify();db=sqlite3.connect(ext.OUT/'records.sqlite3');saved={i:json.loads(raw) for i,raw in db.execute('SELECT id,payload FROM records')};db.close()
    assert len(saved)==579;done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(external_worker,c,p['settings'],p['proposal_record_sha256'][c['id']]) for c in p['cases']]
        for job in as_completed(jobs):
            r=job.result();assert r==saved[r['id']];done+=1
    # Independently recount all six fixed published/local controls, including
    # their common-cohort calibration. Do not rely on the old summary alone.
    cs=p['cases'];caspi=json.loads((ext.OLD/'caspi_reports.json').read_text())['reports'];reports={}
    for c in cs:
        r=json.loads((ext.OLD/(c['id']+'.json')).read_text())['reports'];network=json.loads((ext.OLD/'pointcloud'/(c['id']+'.json')).read_text())['reports']
        reports[c['id']]=dict(guided_fast=r['guided_fast'],local_likelihood=r['local_likelihood'],
            complete_guided_likelihood=json.loads((ext.OLD/'complete_guided'/(c['id']+'.json')).read_text())['report'],caspi=caspi[c['id']],**network)
    analysis=json.loads((ext.OUT/'analysis.json').read_text());recounts={}
    for method in ['guided_fast','local_likelihood','complete_guided_likelihood','caspi','unrolled_raw','unrolled_coates']:
        cal=sorted(reports[c['id']][method]['score'] for c in cs if c['split']=='calibration');cut=max(0.,cal[math.ceil(200*.995)-1])
        result=dict(correct=0,wrong_depth=0,null_false=0)
        for c in cs:
            if c['split']=='calibration':continue
            r=reports[c['id']][method];d=r['score']>cut
            if method=='guided_fast':d=d and r['proposal']['valid']
            if c['metadata']['positive']:
                good=abs(r['delay']-c['metadata']['secondary_delay'])<=2;result['correct']+=bool(d and good);result['wrong_depth']+=bool(d and not good)
            else:result['null_false']+=bool(d)
        old=analysis['methods'][method+'_cal0.005'];assert old['threshold']==cut
        for field,v in result.items():assert old[field]==v
        recounts[method]=result
    result=dict(status='PASS',new_coverage_records_fully_recomputed=done,all_six_controls_recounted=recounts,
        protocol_sha256=sha(ext.P),records_sqlite_sha256=sha(ext.OUT/'records.sqlite3'),
        scope='Current coverage inputs/proposals/certificates and fixed-output control cutoffs/endpoints; no new network prediction or in-distribution retraining.')
    (ext.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['bank','external']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    audit_bank(a.workers) if a.action=='bank' else audit_external(a.workers)
