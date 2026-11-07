"""Integrity and paired-decision audit, independent of numerical superiority."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from localization_confirmation import ROOT,DATA,OUT,P,verify,observation
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    verify();digest=sha(P);manifest=json.loads((DATA/'manifest.json').read_text());cases=manifest['cases']
    assert manifest['protocol_sha256']==digest and len(cases)==5199
    assert len(set(c['id'] for c in cases))==5199
    for field in ['scene_seed','noise_seed']:
        new={c[field] for c in cases};assert len(new)==5199
        for old in ['confirmation_v1','guided_confirmation_v1']:
            old_manifest=json.loads((ROOT/'data'/old/'manifest.json').read_text())
            assert not new.intersection(c[field] for c in old_manifest['cases']),(field,old)
    counts=Counter();decisions=Counter();invalid=Counter();boxes=[]
    for c in cases:
        assert sha(DATA/(c['id']+'.npz'))==c['sha256'],c['id']
        r=json.loads((OUT/(c['id']+'.json')).read_text());assert r['id']==c['id'] and r['protocol_sha256']==digest
        assert r['proposal']['raw_center_excluded'] is True
        reports=r['reports'];assert reports['spatial_report']['score']==reports['complete_guided']['score']
        assert reports['spatial_report']['delay']==r['proposal']['secondary']
        for name,report in reports.items():
            for field in ['score','delay','primary']:
                if field in report: assert np.isfinite(report[field]),(c['id'],name,field)
        counts[c['split']]+=1
        invalid[c['split']]+=not r['proposal']['valid']
        if 'guided_certified' in reports:
            e=reports['guided_certified']['evidence'];decisions[e['decision']]+=1;boxes.append(e['boxes_evaluated'])
            if reports['guided_certified']['declared']:
                assert e['log_e_lower']>=e['log_threshold_upper']
    # Representative deterministic regeneration includes all split/generator paths.
    ids=['cal_0000','null_1999','positive_0000','positive_0899','singleton_0000','singleton_0299']
    for identifier in ids:
        c=next(c for c in cases if c['id']==identifier);h,m=observation(c)
        assert np.array_equal(h,np.load(DATA/(identifier+'.npz'))['histogram'])
        assert m==c['metadata']
    th=json.loads((OUT/'thresholds.json').read_text());assert th['protocol_sha256']==digest
    assert th['methods']['spatial_report']==th['methods']['complete_guided']
    result=dict(status='PASS',protocol_sha256=digest,case_hashes_checked=len(cases),
        split_counts=dict(counts),invalid_proposals=dict(invalid),
        same_presence_scores_and_thresholds=True,regenerated_inputs=ids,
        certified_decisions=dict(decisions),box_quantiles=np.quantile(boxes,[0,.5,.9,.99,1]).tolist(),
        source='Frozen sources verified; each acquisition hash verified; all reported scores finite.',
        limitation='Integrity audit does not establish sensor calibration or general scene accuracy.')
    (OUT/'integrity_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
