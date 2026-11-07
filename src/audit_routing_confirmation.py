"""Source, observation, score-routing and fresh-calibration integrity audit."""
from collections import Counter
import hashlib,json
import numpy as np
from routing_confirmation import verify,ROOT,DATA,OUT,P
from routing_scenes import make_routing_case
from support_routing import route_report
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=verify();digest=sha(P);manifest=json.loads((DATA/'manifest.json').read_text());cases=manifest['cases']
    assert manifest['protocol_sha256']==digest and len(cases)==9099
    for field in ['scene_seed','noise_seed']:
        values={c[field] for c in cases};assert len(values)==9099
        for folder in ['confirmation_v1','guided_confirmation_v1','localization_confirmation_v1']:
            old=json.loads((ROOT/'data'/folder/'manifest.json').read_text())['cases']
            assert not values.intersection(c[field] for c in old),(field,folder)
    rows={};heads=Counter();families=Counter();invalid=0;first_eval=float('inf')
    for c in cases:
        f=DATA/(c['id']+'.npz');assert sha(f)==c['sha256'],c['id']
        result=OUT/(c['id']+'.json');r=json.loads(result.read_text())
        assert r['protocol_sha256']==digest and r['id']==c['id'];q=r['proposal'];assert q['raw_center_excluded']
        assert r['reports']['guided_fit']['score']==r['reports']['spatial_report']['score']
        expected=route_report(q,r['reports']['guided_fit'],r['reports']['local_fit'],p['settings']['normalizers'],'route',3.5)
        assert expected==r['reports']['routed'],c['id']
        rows[c['id']]=r['reports'];heads[(c['support_family'],expected['head'])]+=1
        families[(c['split'],c['support_family'])]+=1;invalid+=not q['valid']
        for name,report in r['reports'].items():
            assert np.all(np.isfinite([report['score'],report['delay'],report['primary']])),(c['id'],name)
        if c['split']!='calibration':first_eval=min(first_eval,result.stat().st_mtime)
    cal=[c for c in cases if c['split']=='calibration'];thresholds=json.loads((OUT/'thresholds.json').read_text())
    assert thresholds['protocol_sha256']==digest
    for method in thresholds['methods']:
        scores=np.sort([rows[c['id']][method]['score'] for c in cal])
        for alpha in [.005,.01]:
            rank=int(np.ceil((len(cal)+1)*(1-alpha)));entry=thresholds['methods'][method][str(alpha)]
            assert entry['n']==3999 and entry['rank']==rank and entry['threshold']==max(0.,float(scores[rank-1]))
    regenerated=['cal_0000','null_2999','coherent_0000','singleton_0599','filament_0000','crossing_step_0299']
    for identifier in regenerated:
        c=next(c for c in cases if c['id']==identifier);h,m,_=make_routing_case(c)
        assert np.array_equal(h,np.load(DATA/(identifier+'.npz'))['histogram']) and m==c['metadata']
    a=json.loads((OUT/'analysis.json').read_text());assert a['protocol_sha256']==digest and a['analysis_sha256']==sha(ROOT/'src'/'analyze_routing_confirmation.py')
    result=dict(status='PASS',protocol_sha256=digest,input_hashes_checked=len(cases),
        exact_routing_reports_checked=len(cases),frozen_source_count=len(p['source_sha256']),
        head_counts={str(k):v for k,v in heads.items()},family_counts={str(k):v for k,v in families.items()},
        invalid_proposals=int(invalid),regenerated=regenerated,
        threshold_saved_before_first_evaluation=(OUT/'thresholds.json').stat().st_mtime<=first_eval,
        all_shared_calibration_order_statistics_verified=True,
        scope='Implementation/data integrity; not a physical sensor or population-generalization guarantee.')
    (OUT/'integrity_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
