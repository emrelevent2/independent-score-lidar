"""Independent execution-integrity and nuisance-mismatch development audit."""
import json,math
from fractions import Fraction
import numpy as np
from real_coverage_development import verify,ROOT,DATA,OLD,OUT,P,sha
from real_coverage_mixture import components,report
from real_poisson_analysis_v2 import interval_match


def main():
    p=verify();m=json.loads((DATA/'manifest.json').read_text())
    assert json.loads((OUT/'completion.json').read_text())['cases']==2697
    hashes={};diagnostics={};done=0
    for entry in m['inputs']:
        i=entry['index'];assert sha(DATA/f'thin_{i}.npz')==entry['sha256']
        h=np.load(DATA/f'thin_{i}.npz')['histograms'];groups={g:dict(n=0,donor_primary_matches_any=0,local_primary_matches_any=0,donor_primary_median_abs_nearest_window_center=[]) for g in ['evaluation_one','evaluation_two']}
        for q in m['queries']:
            old_name=f'gaussian_{i}_{q["id"]}.json';source=OLD/old_name
            assert sha(source)==p['old_record_sha256'][old_name];old=json.loads(source.read_text())
            path=OUT/f'{i}_{q["id"]}.json';r=json.loads(path.read_text())
            assert r['query_id']==q['id'] and r['index']==i and r['protocol_sha256']==sha(P)
            qz=old['proposal'];assert qz['raw_center_excluded'];model=components(qz,672)
            assert sum(model['weights'],Fraction(0))==1
            expected=report(h[q['row'],q['column']],qz,old['null']['log_likelihood'])
            assert expected==r['coverage'];hashes[path.name]=sha(path);done+=1
            if q['split'] in groups:
                g=groups[q['split']];g['n']+=1
                g['donor_primary_matches_any']+=any(interval_match(qz['primary'],w) for w in q['windows'])
                g['local_primary_matches_any']+=any(interval_match(old['reports']['local']['primary'],w) for w in q['windows'])
                g['donor_primary_median_abs_nearest_window_center'].append(min(abs(qz['primary']-(a+b)/2) for a,b in q['windows']))
        for g in groups.values():
            g['donor_primary_median_abs_nearest_window_center']=float(np.median(g['donor_primary_median_abs_nearest_window_center']))
        diagnostics[str(i)]=groups
        print('Audited real coverage',done,flush=True)
    assert done==2697
    (OUT/'result_sha256.json').write_text(json.dumps(dict(protocol_sha256=sha(P),records=hashes),indent=2))
    result=dict(status='PASS',protocol_sha256=sha(P),cases=done,all_new_reports_recomputed=True,
        original_frozen_sources_and_inputs_checked=True,exact_prior_mass_checked=True,
        diagnostics=diagnostics,diagnostic_scope='Post-analysis descriptive nuisance mismatch on closed development record, not an independently tested repair.',
        scope='Execution integrity and proper Poisson-mixture construction; does not establish actual detector law, exhaustive annotation truth, independent efficacy or sensor calibration.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
