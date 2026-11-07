"""Supplement the frozen audit with old-mixture checks and result hashes.

This is a verifier, not a new inference rule. It does not change outcomes.
"""
import json
import numpy as np
from flint import arb
from coverage_confirmation import verify, DATA, OUT, P, sha
from coverage_alternative import coverage_components, numeric_scores, location_report
from adaptive_evidence import mixture_interval, feasible_local_null
from envelope_evidence import outward_float, continuous_null_bound


def main():
    p=verify(); m=json.loads((DATA/'manifest.json').read_text())
    assert json.loads((OUT/'completion.json').read_text())['cases']==6099
    hashes={c['id']:sha(OUT/(c['id']+'.json')) for c in m['cases']}
    manifest_path=OUT/'result_sha256.json'
    snapshot=dict(protocol_sha256=sha(P),records=hashes)
    if manifest_path.exists(): assert json.loads(manifest_path.read_text())==snapshot
    else: manifest_path.write_text(json.dumps(snapshot,indent=2))
    threshold=(arb(1)/arb(str(p['settings']['alpha']))).log()
    th_up=outward_float(threshold,True); th_lo=outward_float(threshold,False)
    bounded=accepted=0; regenerated_bounds={}
    for c in m['cases']:
        r=json.loads((OUT/(c['id']+'.json')).read_text()); e=r['evidence']
        assert sha(DATA/(c['id']+'.npz'))==c['sha256']
        h=np.load(DATA/(c['id']+'.npz'))['histogram'][8,8]
        models=coverage_components(r['proposal'],cycles=1000,amplitudes=p['settings']['secondary_amplitudes'])
        old=mixture_interval(h,models['rates'][:4],1000)
        witness=feasible_local_null(h,1000,tuple(p['settings']['fwhm_range']))
        assert old==e['old_mixture_interval'] and witness==e['local_null_witness']
        _,weighted,full=numeric_scores(h,models,1000)
        assert full==e['numeric_full_log_mixture']
        report=location_report(models,weighted)
        assert all(r['reports']['coverage_cert'][k]==v for k,v in report.items())
        old_upper=outward_float(arb(old['upper'])-arb(witness['lower']),True)
        candidate=old_upper>=th_lo or full-witness['lower']>=th_up-p['settings']['numerical_screen_margin']
        assert e['screened']==(not candidate) and e['threshold_upper']==th_up
        if candidate:
            bounded+=1
            delta=outward_float(arb(old['lower'])-arb(e['bound']['upper']),False)
            assert e['old_log_e_lower']==delta
            assert r['reports']['spatial_cert']['declared']==(delta>=th_up)
            accepted+=r['reports']['spatial_cert']['declared']
            group='calibration' if c['split']=='calibration' else 'null' if c['kind']=='null' else c['support_family']
            if group not in regenerated_bounds:
                b=continuous_null_bound(h,1000,tuple(p['settings']['fwhm_range']),initial_cells=32,max_boxes=256,normalized_envelope=False)
                assert b==e['bound'];regenerated_bounds[group]=c['id']
        else:
            assert e['bound'] is None and not r['reports']['spatial_cert']['declared']
        assert sha(OUT/(c['id']+'.json'))==hashes[c['id']]
    result=dict(status='PASS',protocol_sha256=sha(P),cases=len(hashes),
        old_mixtures_and_witnesses_recomputed=len(hashes),full_numeric_scores_and_location_reports_recomputed=len(hashes),
        bounded_old_acceptance_inequalities_checked=bounded,spatial_accepted=accepted,
        continuous_null_bounds_regenerated=regenerated_bounds,
        result_hash_manifest_sha256=sha(manifest_path),
        scope='Complements frozen integrity.json. Hash snapshot created after completion; not a prospective timestamp or physical validation.')
    (OUT/'supplemental_integrity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
