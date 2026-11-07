"""Explicit post-hoc development on CLOSED localization confirmation data."""
from pathlib import Path
import argparse,datetime,hashlib,json,math
import numpy as np
from support_routing import donor_support,route_report
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results'/'localization_confirmation_v1'
DATA=ROOT/'data'/'localization_confirmation_v1';OUT=ROOT/'results'/'support_development_v1'
P=ROOT/'SUPPORT_DEVELOPMENT_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing development snapshot retained')
    th=json.loads((BASE/'thresholds.json').read_text())['methods']
    normalizers={k:th[name]['0.005']['threshold'] for k,name in [('spatial','complete_guided'),('local','local_likelihood')]}
    assert min(normalizers.values())>0
    variants=[dict(name=p,policy=p,gate=2.5) for p in ['spatial','local','naive_union']]
    variants +=[dict(name=f'{p}_{g}',policy=p,gate=g) for p in ['route','support_union','aligned_union'] for g in [1.5,2.5,3.5,4.5]]
    protocol=dict(status='POST HOC DEVELOPMENT: closed localization cohort, including singleton stress. No independent performance claim.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in ['support_development.py','support_routing.py']},
        source_protocol_sha256=sha(ROOT/'LOCALIZATION_CONFIRMATORY_PROTOCOL.json'),
        normalizers=normalizers,variants=variants,alpha=[.005,.01],
        selection_utility='At0.005:0.5*(main recall+singleton recall)-0.5*(main wrong-rate+singleton wrong-rate), with null count <=25/2000. Prefer smaller gate on exact tie.',
        notes=['Support is a donor-only variance-inspired proxy, not a statistical confidence bound.',
            'Simple score normalization uses CLOSED development null thresholds. It is fixed preprocessing, not an assertion that head scores are p-values.',
            'Joint policy is recalibrated on the development nulls for this screen; a fresh independent calibration is mandatory for any final test.',
            'Main and singleton are equally weighted in selection; never conceal singleton failure with the larger main denominator.',
            'No existing frozen source or result is modified. All current test outcomes are development for a future hybrid.'])
    P.write_text(json.dumps(protocol,indent=2));print('Frozen',len(variants),'development variants',flush=True)


def run():
    p=json.loads(P.read_text());digest=sha(P)
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    cases=json.loads((DATA/'manifest.json').read_text())['cases'];rows={}
    for i,c in enumerate(cases):
        r=json.loads((BASE/(c['id']+'.json')).read_text());rows[c['id']]=r
        if (i+1)%1000==0:print('Read',i+1,flush=True)
    cal=[c for c in cases if c['split']=='calibration'];evaluation=[c for c in cases if c['split']!='calibration']
    out={};best=None;support_distribution={}
    for group,seq in [('calibration',cal),('null',[c for c in evaluation if c['kind']=='null']),
                      ('main',[c for c in evaluation if c['split']=='evaluation' and c['kind']=='positive']),
                      ('singleton',[c for c in evaluation if c['split']=='stress'])]:
        values=[donor_support(rows[c['id']]['proposal']) for c in seq]
        support_distribution[group]=dict(n=len(values),quantiles=np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist())
    for variant in p['variants']:
        rr={c['id']:route_report(rows[c['id']]['proposal'],rows[c['id']]['reports']['complete_guided'],
            rows[c['id']]['reports']['local_likelihood'],p['normalizers'],variant['policy'],variant['gate']) for c in cases}
        for alpha in p['alpha']:
            rank=math.ceil((len(cal)+1)*(1-alpha));th=max(0.,float(np.sort([rr[c['id']]['score'] for c in cal])[rank-1]))
            stats={g:dict(n=0,correct=0,wrong=0,declared=0,spatial_head=0) for g in ['null','main','singleton']}
            for c in evaluation:
                group='null' if c['kind']=='null' else 'singleton' if c['split']=='stress' else 'main';s=stats[group]
                r=rr[c['id']];declared=r['score']>th
                correct=bool(declared and c['kind']=='positive' and abs(r['delay']-c['metadata']['secondary_delay'])<=2.)
                s['n']+=1;s['declared']+=declared;s['correct']+=correct
                s['wrong']+=declared and c['kind']=='positive' and not correct;s['spatial_head']+=r['head']=='spatial'
            m,s,n=stats['main'],stats['singleton'],stats['null']
            utility=.5*(m['correct']/m['n']+s['correct']/s['n'])-.5*(m['wrong']/m['n']+s['wrong']/s['n'])
            label=f'{variant["name"]}_cal{alpha}';out[label]=dict(variant=variant,alpha=alpha,threshold=th,stats=stats,utility=utility)
            if alpha==.005 and n['declared']<=25 and (best is None or utility>out[best]['utility']):best=label
            print(label,'main',m['correct'],m['wrong'],'singleton',s['correct'],s['wrong'],'null',n['declared'],'utility',round(utility,4),flush=True)
    OUT.mkdir(exist_ok=True)
    result=dict(status=p['status'],protocol_sha256=digest,methods=out,selected=best,support_distribution=support_distribution)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print('Selected on DEVELOPMENT',best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
