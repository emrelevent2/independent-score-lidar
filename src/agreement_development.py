"""Closed real development selection; held-out histograms remain unread."""
from pathlib import Path
import argparse,datetime,hashlib,json,itertools
import numpy as np
from agreement_policy import agreement_report
from real_poisson_analysis_v2 import endpoints,threshold
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/real_poisson_development_v2'
DATA=ROOT/'data/real_poisson_development_v1';P=ROOT/'AGREEMENT_DEVELOPMENT_PROTOCOL.json'
OUT=ROOT/'results/agreement_development_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing development rule retained')
    assert not (ROOT/'REAL_POISSON_TEST_PROTOCOL.json').exists()
    settings=[dict(distance=d,bonus=b,blend=w) for d,b,w in itertools.product([4.,8.,16.],[0.,.5,1.,2.],[0.,.5,1.])]
    p=dict(status='Additional DEVELOPMENT after negative support-only routing; all test histograms still unread.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in ['agreement_development.py','agreement_policy.py','real_poisson_analysis_v2.py']},
        prior_analysis_sha256=sha(D/'analysis.json'),prior_calibration_sha256=sha(D/'calibration.json'),
        manifest_sha256=sha(DATA/'manifest.json'),settings=settings,kinds=['gaussian','empirical'],primary_alpha=.01,
        selection='Maximize sum of recovered pairs across3attenuations, constrained to accepted-unmatched count on known-one queries <= selected local baseline at EACH attenuation. Include unchanged local fallback. Tie: fewer unmatched, smaller bonus, smaller distance, smaller blend, Gaussian.',
        notes=['A finite supervised development search; its gains are not independent confirmation.',
            'Role-invariant local/donor pair association; boost only if BOTH local returns agree with donors and donor support>=3.5.',
            'Report nudging changes both local positions using the matched donor pair. Local score remains the base.',
            'Every candidate is recalibrated on the same399known-one development calibration pixels.',
            'No physical false-positive or e-value claim. Test outcome cannot select candidates.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen',len(settings)*2,'development candidates plus local fallback',flush=True)


def run():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(D/'analysis.json')==p['prior_analysis_sha256'] and sha(D/'calibration.json')==p['prior_calibration_sha256']
    assert sha(DATA/'manifest.json')==p['manifest_sha256']
    a=json.loads((D/'analysis.json').read_text());cal=json.loads((D/'calibration.json').read_text());m=json.loads((DATA/'manifest.json').read_text())
    queries=m['queries'];cache={}
    for kind in p['kinds']:
        for i in range(3):cache[kind,i]={q['id']:json.loads((D/f'{kind}_{i}_{q["id"]}.json').read_text()) for q in queries}
    local_kind=a['selected_kinds']['local'];bounds=[a['methods'][f'{local_kind}_{i}_local_0.01']['groups']['evaluation_one']['accepted_unmatched'] for i in range(3)]
    fallback=dict(id='local_fallback',kind=local_kind,setting=None,groups=[a['methods'][f'{local_kind}_{i}_local_0.01']['groups'] for i in range(3)])
    fallback.update(total_pairs=sum(g['evaluation_two']['pair'] for g in fallback['groups']),total_unmatched=sum(bounds),eligible=True)
    candidates=[fallback]
    for kind in p['kinds']:
        for j,setting in enumerate(p['settings']):
            groups=[];cuts={}
            for i in range(3):
                norms=cal[f'{kind}_{i}']['normalizers'];records=cache[kind,i]
                reports={q['id']:agreement_report(records[q['id']],norms,setting) for q in queries}
                cut=threshold([reports[q['id']]['score'] for q in queries if q['split']=='calibration'],.01);cuts[str(i)]=cut
                group={s:dict(n=0,pair=0,accepted_unmatched=0,declared=0,weak_window_agreement=0) for s in ['evaluation_one','evaluation_two']}
                for q in queries:
                    if q['split'] not in group:continue
                    r=reports[q['id']];e=endpoints(q,r,r['score']>cut);g=group[q['split']];g['n']+=1
                    for key in g:
                        if key!='n':g[key]+=e[key]
                groups.append(group)
            unmatched=[g['evaluation_one']['accepted_unmatched'] for g in groups]
            result=dict(id=f'{kind}_{j:02d}',kind=kind,setting=setting,thresholds=cuts,groups=groups,
                total_pairs=sum(g['evaluation_two']['pair'] for g in groups),total_unmatched=sum(unmatched),eligible=all(v<=b for v,b in zip(unmatched,bounds)))
            candidates.append(result)
    def key(c):
        s=c['setting'] or dict(bonus=0,distance=0,blend=0)
        return (-c['total_pairs'],c['total_unmatched'],s['bonus'],s['distance'],s['blend'],c['kind']!='gaussian')
    selected=min([c for c in candidates if c['eligible']],key=key)
    OUT.mkdir(exist_ok=True);out=dict(status='DEVELOPMENT ONLY',protocol_sha256=sha(P),unmatched_bounds=bounds,candidates=candidates,selected=selected)
    (OUT/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(bounds=bounds,baseline=fallback,selected=selected),indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
