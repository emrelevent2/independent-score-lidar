"""Post hoc development on CLOSED guided_confirmation_v1; NOT new confirmation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,hashlib,datetime,time,math
import numpy as np
from profile_proposals import profile_at_proposals
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'guided_confirmation_v1';BASE=ROOT/'results'/'guided_confirmation_v1'
OUT=ROOT/'results'/'profile_development_v1';P=ROOT/'PROFILE_DEVELOPMENT_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES=['profile_development.py','profile_proposals.py','likelihood_baseline.py','adaptive_evidence.py']
def freeze():
    if P.exists():raise RuntimeError('Existing snapshot retained')
    p=dict(status='POST HOC development using closed first guided confirmation; fresh confirmation mandatory',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
        variants=['fixed_secondary','fixed_both','refined_presence_donor_report'],alpha=[.005,.01],
        notes=['No new hand-tuned spatial bandwidth or shrinkage coefficient. Test two structural choices: fix secondary only or both donor centers.',
            'Third variant retains complete guided likelihood decisions exactly and changes only reported secondary to the fixed donor coordinate.',
            'Every current result is development, including reuse of calibration/test data. No claim of new independent confirmation.'])
    P.write_text(json.dumps(p,indent=2))
def worker(item,digest):
    path=OUT/(item['id']+'.json')
    if path.exists():
        r=json.loads(path.read_text());assert r['protocol_sha256']==digest;return r
    original=json.loads((BASE/(item['id']+'.json')).read_text());q=original['reports']['guided_fast']['proposal']
    h=np.load(DATA/(item['id']+'.npz'))['histogram'];reports={}
    witness=original['reports'].get('guided_certified',{}).get('evidence',{}).get('local_null_witness')
    for mode in ['fixed_secondary','fixed_both']:reports[mode]=profile_at_proposals(h[8,8],q['primary'],q['secondary'],mode,null_witness=witness)
    control=json.loads((BASE/'complete_guided'/(item['id']+'.json')).read_text())['report']
    reports['refined_presence_donor_report']=dict(control,delay=q['secondary'])
    out=dict(id=item['id'],protocol_sha256=digest,reports=reports);path.write_text(json.dumps(out,indent=2,allow_nan=False));return out
def analyze():
    cases=json.loads((DATA/'manifest.json').read_text())['cases'];rows={c['id']:json.loads((OUT/(c['id']+'.json')).read_text()) for c in cases}
    cal=[c for c in cases if c['split']=='calibration'];evaluation=[c for c in cases if c['split']=='evaluation'];out={}
    for mode in ['fixed_secondary','fixed_both','refined_presence_donor_report']:
        for alpha in [.005,.01]:
            rank=math.ceil((len(cal)+1)*(1-alpha));threshold=max(0.,float(np.sort([rows[c['id']]['reports'][mode]['score'] for c in cal])[rank-1]))
            tp=wrong=fp=0
            for c in evaluation:
                r=rows[c['id']]['reports'][mode];positive=c['metadata']['positive'];declared=r['score']>threshold
                localized=declared and positive and abs(r['delay']-c['metadata']['secondary_delay'])<=2.
                tp+=localized;wrong+=declared and positive and not localized;fp+=declared and not positive
            out[f'{mode}_{alpha}']=dict(correct=int(tp),wrong_depth=int(wrong),false_null=int(fp),threshold=threshold,positive_n=900,null_n=2000)
    (OUT/'analysis.json').write_text(json.dumps(dict(status='POST HOC DEVELOPMENT; not final confirmation',methods=out),indent=2))
    print(json.dumps(out,indent=2),flush=True)
def run(workers):
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    cases=json.loads((DATA/'manifest.json').read_text())['cases'];OUT.mkdir(exist_ok=True);start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,c,sha(P)) for c in cases]
        for i,j in enumerate(as_completed(jobs),1):
            j.result()
            if i%1000==0 or i==len(jobs):print('Profile development',i,'/',len(jobs),'seconds',round(time.perf_counter()-start,1),flush=True)
    analyze()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','analyze']);parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    freeze() if args.action=='freeze' else run(args.workers) if args.action=='run' else analyze()
