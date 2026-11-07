"""Finite real DEVELOPMENT selection; unchanged local fallback mandatory."""
from pathlib import Path
import argparse,datetime,hashlib,json
from real_poisson_analysis_v2 import route,endpoints,threshold
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/anchored_development_v1';BASE=ROOT/'results/real_poisson_development_v2'
DATA=ROOT/'data/real_poisson_development_v1';P=ROOT/'ANCHORED_MODEL_SELECTION_RULE.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=['guided','spatial','routed','naive_union','disagreement']


def freeze():
    if P.exists():raise RuntimeError('Existing rule retained')
    assert not (ROOT/'REAL_POISSON_TEST_PROTOCOL.json').exists()
    p=dict(status='Finite development choice fixed before aggregated anchored accuracy.',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in ['anchored_analysis.py','real_poisson_analysis_v2.py']},
        anchored_protocol_sha256=sha(ROOT/'ANCHORED_DEVELOPMENT_PROTOCOL.json'),base_analysis_sha256=sha(BASE/'analysis.json'),
        methods=METHODS,depth_sigmas=[4.,8.,16.],primary_alpha=.01,
        selection='Maximize recovered two-interval pairs summed across3attenuations, with known-one accepted-unmatched <= Gaussian local at EACH attenuation. Unchanged local fallback included. Tie: fewer unmatched across all evaluation pixels, smaller sigma, methods in listed order; local wins equal performance.',
        notes=['Development selection, not independent efficacy confirmation. No physical-negative labels.',
            'All methods retain primary/secondary association-free interval endpoints and identical normalization/calibration queries.',
            'Best anchored policy will be frozen before any test histogram acquisition; all unsuccessful policies retained.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen15anchored candidates plus local fallback',flush=True)


def run():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(BASE/'analysis.json')==p['base_analysis_sha256']
    assert json.loads((D/'completion.json').read_text())['cases']==8091
    qs=json.loads((DATA/'manifest.json').read_text())['queries'];a=json.loads((BASE/'analysis.json').read_text());basecal=json.loads((BASE/'calibration.json').read_text())
    bounds=[a['methods'][f'gaussian_{i}_local_0.01']['groups']['evaluation_one']['accepted_unmatched'] for i in range(3)]
    groups=[a['methods'][f'gaussian_{i}_local_0.01']['groups'] for i in range(3)]
    fallback=dict(id='local_fallback',kind='gaussian',depth_sigma=None,method='local',groups=groups,eligible=True,
        total_pairs=sum(g['evaluation_two']['pair'] for g in groups),total_unmatched=sum(g[s]['accepted_unmatched'] for g in groups for s in ['evaluation_one','evaluation_two']))
    candidates=[fallback];calibration={};summary={}
    for sigma in p['depth_sigmas']:
        for i in range(3):
            records={q['id']:json.loads((D/f'sigma{int(sigma)}_{i}_{q["id"]}.json').read_text()) for q in qs}
            assert all(r['protocol_sha256']==p['anchored_protocol_sha256'] for r in records.values())
            norms={n:threshold([records[q['id']]['reports'][n]['score'] for q in qs if q['split']=='normalization'],.01,1e-6) for n in ['local','guided']}
            key=f'sigma{int(sigma)}_{i}';calibration[key]=dict(normalizers=norms,methods={})
            for method in METHODS:
                reports={q['id']:route(records[q['id']],norms,method) for q in qs}
                for alpha in [.005,.01]:
                    cut=threshold([reports[q['id']]['score'] for q in qs if q['split']=='calibration'],alpha)
                    gs={s:dict(n=0,declared=0,pair=0,weak_window_agreement=0,accepted_unmatched=0,primary_agreement=0,duplicate_known_window=0) for s in ['evaluation_one','evaluation_two']}
                    for q in qs:
                        if q['split'] not in gs:continue
                        r=reports[q['id']];e=endpoints(q,r,r['score']>cut);g=gs[q['split']];g['n']+=1
                        for f,v in e.items():g[f]+=v
                    label=f'{key}_{method}_{alpha}';summary[label]=dict(depth_sigma=sigma,method=method,frame_index=i,alpha=alpha,groups=gs)
                    calibration[key]['methods'].setdefault(method,{})[str(alpha)]=cut
        for method in METHODS:
            gs=[summary[f'sigma{int(sigma)}_{i}_{method}_0.01']['groups'] for i in range(3)]
            c=dict(id=f'sigma{int(sigma)}_{method}',kind='gaussian',depth_sigma=sigma,method=method,groups=gs,
                total_pairs=sum(g['evaluation_two']['pair'] for g in gs),
                total_unmatched=sum(g[s]['accepted_unmatched'] for g in gs for s in ['evaluation_one','evaluation_two']),
                eligible=all(g['evaluation_one']['accepted_unmatched']<=b for g,b in zip(gs,bounds)))
            candidates.append(c)
    def order(c):return (-c['total_pairs'],c['total_unmatched'],c['depth_sigma'] or 0.,METHODS.index(c['method']) if c['method'] in METHODS else -1)
    selected=min([c for c in candidates if c['eligible']],key=order)
    result=dict(status='DEVELOPMENT ONLY',rule_sha256=sha(P),unmatched_bounds=bounds,candidates=candidates,selected=selected,all_outcomes=summary)
    (D/'analysis.json').write_text(json.dumps(result,indent=2));(D/'calibration.json').write_text(json.dumps(calibration,indent=2))
    for c in candidates:print(c['id'],'pairs',c['total_pairs'],'unmatched',c['total_unmatched'],'eligible',c['eligible'],flush=True)
    print('SELECTED',selected['id'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
