"""Reclassify the completed routing cohort as DEVELOPMENT for a new rule."""
from pathlib import Path
import argparse,datetime,hashlib,json,itertools
from guarded_localization import guarded_report
from analyze_routing_confirmation import endpoint
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/routing_confirmation_v1'
DATA=ROOT/'data/routing_confirmation_v1';P=ROOT/'GUARDED_LOCALIZATION_DEVELOPMENT_PROTOCOL.json';OUT=ROOT/'results/guarded_development_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing guarded-development rule retained')
    settings=[dict(distance=d,fraction=f,blend=b) for d,f,b in itertools.product([2.,4.,8.,16.],[0.,.5,1.,2.],[.5,1.])]
    p=dict(status='CLOSED previous routing test is DEVELOPMENT for this new pure localization rule. It cannot independently confirm it.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=settings,
        source_sha256={n:sha(ROOT/'src'/n) for n in ['guarded_development.py','guarded_localization.py','support_routing.py','analyze_routing_confirmation.py']},
        base_analysis_sha256=sha(D/'analysis.json'),manifest_sha256=sha(DATA/'manifest.json'),primary_alpha=.005,
        selection='Keep EXACT local score and threshold. Maximize coherent+singleton correct, constrained to no reduction of correct and no increase of wrong depth in EACH of coherent,singleton,filament,crossing_step. Include unchanged local fallback. Tie: fewer changed reports, smaller distance, larger fraction, smaller blend.',
        notes=['Every candidate has the identical null declarations by algebraic score identity, not a calibration claim.',
            'All previous stress geometries become development for this new rule. Fresh confirmation is mandatory.',
            'Real-data working-count extension remains a separate frozen comparison; this simulator-specific count noise proxy is not silently applied there.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen64guarded candidates plus unchanged local',flush=True)


def run():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    assert sha(D/'analysis.json')==p['base_analysis_sha256'] and sha(DATA/'manifest.json')==p['manifest_sha256']
    m=json.loads((DATA/'manifest.json').read_text());cases=[c for c in m['cases'] if c['kind']=='positive'];rows={c['id']:json.loads((D/(c['id']+'.json')).read_text()) for c in cases}
    cut=json.loads((D/'thresholds.json').read_text())['methods']['local_fit']['0.005']['threshold']
    families=['coherent','singleton','filament','crossing_step'];baseline={g:dict(n=0,correct=0,wrong_depth=0) for g in families}
    for c in cases:
        r=rows[c['id']]['reports']['local_fit'];e=endpoint(c,r,r['score']>cut);g=baseline[c['support_family']];g['n']+=1
        for f in ['correct','wrong_depth']:g[f]+=e[f]
    candidates=[dict(id='local_fallback',setting=None,groups=baseline,eligible=True,changed=0,total_correct=sum(baseline[g]['correct'] for g in ['coherent','singleton']))]
    for i,setting in enumerate(p['settings']):
        groups={g:dict(n=0,correct=0,wrong_depth=0) for g in families};changed=0
        for c in cases:
            r=rows[c['id']];report=guarded_report(r['proposal'],r['reports']['guided_fit'],r['reports']['local_fit'],setting)
            assert report['score']==r['reports']['local_fit']['score'];e=endpoint(c,report,report['score']>cut);g=groups[c['support_family']];g['n']+=1
            for f in ['correct','wrong_depth']:g[f]+=e[f]
            changed+=report['location_adjusted'] and e['declared']
        eligible=all(groups[g]['correct']>=baseline[g]['correct'] and groups[g]['wrong_depth']<=baseline[g]['wrong_depth'] for g in families)
        candidates.append(dict(id=f'guarded_{i:02d}',setting=setting,groups=groups,eligible=eligible,changed=changed,total_correct=sum(groups[g]['correct'] for g in ['coherent','singleton'])))
    def key(c):
        s=c['setting'] or dict(distance=0,fraction=0,blend=0)
        return (-c['total_correct'],c['changed'],s['distance'],-s['fraction'],s['blend'])
    selected=min([c for c in candidates if c['eligible']],key=key);OUT.mkdir(exist_ok=True)
    (OUT/'analysis.json').write_text(json.dumps(dict(status='DEVELOPMENT ONLY',protocol_sha256=sha(P),baseline=baseline,candidates=candidates,selected=selected),indent=2))
    print(json.dumps(dict(baseline=baseline,selected=selected),indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
