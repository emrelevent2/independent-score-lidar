"""Additional declared development grid; only CLOSED data are used."""
from pathlib import Path
import argparse,datetime,hashlib,json,math
import numpy as np
from support_routing import route_report
from disagreement_routing import disagreement_report
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results'/'localization_confirmation_v1'
DATA=ROOT/'data'/'localization_confirmation_v1';OUT=ROOT/'results'/'disagreement_development_v1'
P=ROOT/'DISAGREEMENT_DEVELOPMENT_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing development snapshot retained')
    p=dict(status='POST HOC DEVELOPMENT on closed localization confirmation; no independent new-method evidence.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in ['disagreement_development.py','disagreement_routing.py','support_routing.py']},
        normalizers=json.loads((ROOT/'SUPPORT_DEVELOPMENT_PROTOCOL.json').read_text())['normalizers'],
        gate=3.5,ratios=[None,1.25,1.5,2.],separation_for_override=4.,alpha=[.005,.01],
        selection='Same equal-main/singleton correct-minus-wrong utility at0.005; null<=25/2000; simpler baseline wins exact tie.',
        notes=['Strongly separated local candidate can override a confident but misplaced donor proposal.',
            'No fresh inputs exist. All variants including failures will remain in the development record.'])
    P.write_text(json.dumps(p,indent=2))


def run():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    cases=json.loads((DATA/'manifest.json').read_text())['cases'];rows={c['id']:json.loads((BASE/(c['id']+'.json')).read_text()) for c in cases}
    cal=[c for c in cases if c['split']=='calibration'];evaluation=[c for c in cases if c['split']!='calibration'];out={};best=None
    for ratio in p['ratios']:
        name='route35' if ratio is None else f'disagreement_{ratio}'
        reports={}
        for c in cases:
            r=rows[c['id']];args=(r['proposal'],r['reports']['complete_guided'],r['reports']['local_likelihood'],p['normalizers'])
            reports[c['id']]=route_report(*args,'route',3.5) if ratio is None else disagreement_report(*args,ratio=ratio,gate=3.5)
        for alpha in p['alpha']:
            rank=math.ceil((len(cal)+1)*(1-alpha));threshold=max(0.,float(np.sort([reports[c['id']]['score'] for c in cal])[rank-1]))
            stats={g:dict(n=0,correct=0,wrong=0,declared=0,overrides=0) for g in ['null','main','singleton']}
            for c in evaluation:
                group='null' if c['kind']=='null' else 'singleton' if c['split']=='stress' else 'main';s=stats[group];r=reports[c['id']]
                declared=r['score']>threshold;correct=bool(declared and c['kind']=='positive' and abs(r['delay']-c['metadata']['secondary_delay'])<=2.)
                s['n']+=1;s['declared']+=declared;s['correct']+=correct;s['wrong']+=declared and c['kind']=='positive' and not correct;s['overrides']+=r.get('disagreement_override',False)
            m,s,n=stats['main'],stats['singleton'],stats['null'];utility=.5*(m['correct']/m['n']+s['correct']/s['n'])-.5*(m['wrong']/m['n']+s['wrong']/s['n'])
            label=f'{name}_cal{alpha}';out[label]=dict(ratio=ratio,alpha=alpha,threshold=threshold,utility=utility,stats=stats)
            if alpha==.005 and n['declared']<=25 and (best is None or utility>out[best]['utility']):best=label
            print(label,stats,'utility',utility,flush=True)
    OUT.mkdir(exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(dict(status=p['status'],protocol_sha256=sha(P),selected=best,methods=out),indent=2))
    print('Selected on DEVELOPMENT',best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
