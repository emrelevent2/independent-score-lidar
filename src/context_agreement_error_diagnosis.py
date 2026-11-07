"""Post-development diagnostic; never changes fitted locations or decisions.

Truth-dependent location ceilings are descriptive oracle diagnostics, not a
proposed classifier, hyperparameter-selection rule, or achievable guarantee.
"""
import json
import context_agreement_development as d


def diagnose(rows,method):
    c=dict(positive=0,correct=0,declared_wrong_primary_only=0,declared_wrong_secondary_only=0,declared_both_locations_wrong=0,missed_with_both_locations_correct=0,missed_with_location_error=0,null=0,false=0,truth_dependent_fixed_location_ceiling=0)
    for r in rows:
        z=r['samples'][method];declared=z['pvalue']<=.01
        if not r['case']['positive']:
            c['null']+=1;c['false']+=int(declared);continue
        c['positive']+=1
        distance=lambda a,b:abs((a-b+32)%64-32)
        primary=distance(z['primary'],r['truth']['primary'])<=2;secondary=distance(z['delay'],r['truth']['secondary'])<=2
        c['truth_dependent_fixed_location_ceiling']+=int(primary and secondary)
        if not declared:c['missed_with_both_locations_correct' if primary and secondary else 'missed_with_location_error']+=1
        elif primary and secondary:c['correct']+=1
        elif secondary:c['declared_wrong_primary_only']+=1
        elif primary:c['declared_wrong_secondary_only']+=1
        else:c['declared_both_locations_wrong']+=1
    assert sum(c[k] for k in ['correct','declared_wrong_primary_only','declared_wrong_secondary_only','declared_both_locations_wrong','missed_with_both_locations_correct','missed_with_location_error'])==c['positive']
    return c


def main():
    d.check();rows=d.read();assert len(rows)==640;assert json.loads((d.OUT/'integrity.json').read_text())['status']=='PASS'
    result=dict(status='POST_DEVELOPMENT_DIAGNOSTIC_ONLY',methods={m:diagnose(rows,m) for m in d.METHODS},per_regime_family=[],scope='Truth-dependent error provenance on a closed development cohort;not new efficacy evidence,not a fitted method,and not a basis for reusing these labels to claim improvement. Same fixed local maximum is reported by local/full/pilot methods;their fixed-location oracle ceilings must match. A threshold-only change cannot exceed this observed ceiling on these saved estimates.')
    for gi,g in enumerate(d.check()['regimes']):
        for family in d.check()['families']['positive']+d.check()['families']['null']:
            subset=[r for r in rows if r['case']['regime']==gi and r['case']['family']==family];result['per_regime_family'].append(dict(regime=g['name'],family=family,methods={m:diagnose(subset,m) for m in d.METHODS}))
    assert len({result['methods'][m]['truth_dependent_fixed_location_ceiling'] for m in ['local','full_supported','pilot_supported']})==1
    (d.OUT/'error_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_regime_family'},indent=2))


if __name__=='__main__':main()
