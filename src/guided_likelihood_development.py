from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,math,argparse
import numpy as np
from guided_likelihood import guided_likelihood_report
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'guided_likelihood_development_v1'
DATA=ROOT/'data'/'improvement_development_v1';AD=ROOT/'results'/'adaptive_development_v1';BASE=ROOT/'results'/'improvement_development_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def worker(item):
    path=OUT/(item['id']+'.json')
    if path.exists():return json.loads(path.read_text())
    a=json.loads((AD/(item['id']+'.json')).read_text());h=np.load(DATA/(item['id']+'.npz'))['histogram']
    r=guided_likelihood_report(h[8,8],a['parameters']['secondary'],null_witness=a['evidence']['local_null_witness'])
    out=dict(**item,metadata=a['metadata'],report=r);path.write_text(json.dumps(out,indent=2,allow_nan=False));return out
def main(workers):
    p=json.loads((ROOT/'IMPROVEMENT_DEVELOPMENT_PROTOCOL_v1.json').read_text());OUT.mkdir(exist_ok=True)
    (OUT/'snapshot.json').write_text(json.dumps(dict(status='Additional strong development control; not independent confirmation',
        source_sha256={n:sha(ROOT/'src'/n) for n in ['guided_likelihood_development.py','guided_likelihood.py','adaptive_evidence.py','likelihood_baseline.py']},
        calibration_alpha=[.005,.01],secondary_refinement_bins=2.,starts=[-1.,0.,1.]),indent=2))
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(worker,p['cases']))
    cal=[r['report']['score'] for r in rows if r['split']=='calibration'];ev=[r for r in rows if r['split']=='evaluation'];summary={}
    for alpha in [.005,.01]:
        threshold=float(np.sort(cal)[math.ceil((len(cal)+1)*(1-alpha))-1]);fp=tp=wrong=0
        for r in ev:
            declared=r['report']['score']>threshold;positive=r['metadata']['positive']
            localized=declared and positive and abs(r['report']['delay']-r['metadata']['secondary_delay'])<=2.
            fp+=declared and not positive;tp+=localized;wrong+=declared and positive and not localized
        summary[str(alpha)]=dict(threshold=threshold,false=int(fp),null_n=100,localized=int(tp),positive_n=150,wrong=int(wrong))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);main(parser.parse_args().workers)
