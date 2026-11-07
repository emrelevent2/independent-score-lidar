"""Post-study arithmetic correction;retain all original scientific records."""
from pathlib import Path
import sys,json,time
import numpy as np
import safe_donor_development as d
from safe_donor_alternative import build_alternative
from certified_safe_components import exact_safe_statistics,mixture_log_lower
from certified_safe_null import declare_from_fixed_mixture
from outward_safety_lag import outward_safe_lag
from safe_deadtime_exposure import safe_lag
P=d.ROOT/'SAFE_LAG_ARITHMETIC_CORRECTION_PROTOCOL.json';OUT=d.ROOT/'results/safe_lag_arithmetic_correction_v1'


def freeze():
    if P.exists():raise FileExistsError('Protocol already frozen')
    names=['src/safe_lag_correction_replay.py','src/outward_safety_lag.py','src/test_outward_safety_lag.py'];v=dict(status='FROZEN_BEFORE_FULL_ARITHMETIC_CORRECTION_REPLAY',sources={n:d.sha(d.ROOT/n) for n in names},base_protocol_sha256=d.sha(d.P),inputs={str(p.relative_to(d.ROOT)):d.sha(p) for p in [d.OUT/'records.sqlite3',d.OUT/'analysis.json']},reason='Original safe_lag uses nearest IEEE sum. For exact supplied Td<Te,that can be below exact Td+Te. Bound routine correctly certifies its supplied lag,but bridge from recovery parameters to guaranteed-safe lag needs outward rounding.',change='Recompute only observed certified decisions using exact recovery sum rounded UP. Original donor predictors unchanged:they are arbitrary fixed positive intensities for the proof,so their training-risk approximation does not affect conditional validity. Original approximate bootstrap controls and all original outputs retained. No efficacy,threshold,prior or model-family tuning.',scope='Post-study arithmetic repair on CLOSED data,not independent confirmation. Exact recorded dyadic paths and exact supplied recovery parameters,not physical timestamp/recovery uncertainty.')
    P.write_text(json.dumps(v,indent=2));print('Frozen arithmetic correction',d.sha(P),flush=True)


def run():
    p=d.check();v=json.loads(P.read_text());assert d.sha(d.P)==v['base_protocol_sha256']
    for n,h in {**v['sources'],**v['inputs']}.items():assert d.sha(d.ROOT/n)==h,n
    OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'completion.json').exists();start=time.perf_counter();cfg=p['settings'];rows=[];changed=0;different_counts=0;affected=0;totals=dict(correct=0,wrong=0,false=0,positive=0,null=0)
    for i,row in enumerate(d.read()):
        case=row['case'];g=p['regimes'][case['regime']];obs,ds,de,truth,dh=d.data(case,p);assert dh==row['data_sha256'];alternative=build_alternative(ds,de,cfg['sigma']);assert alternative['sha256']==row['alternative_sha256'];lag=outward_safe_lag(g['td'],g['te']);oldlag=safe_lag(g['td'],g['te']);affected+=int(lag!=oldlag)
        stats=exact_safe_statistics(obs,g['cycles'],cfg['bins'],lag);oldstats=exact_safe_statistics(obs,g['cycles'],cfg['bins'],oldlag);different_counts+=int(not np.array_equal(stats['counts'],oldstats['counts']));num=mixture_log_lower(stats,alternative['rates'],alternative['weights']);result=declare_from_fixed_mixture(stats,num,sigma_range=(cfg['sigma'],cfg['sigma']),initial_cells=cfg['cover_initial_cells'],max_boxes=cfg['cover_max_boxes'],fit=dict(theta=np.array(row['null_fit'])))
        original=row['samples']['donor_certified'];changed+=int(result['declared']!=original['declared']);copy=dict(row);copy['samples']=dict(row['samples'],donor_certified=dict(declared=result['declared'],primary=original['primary'],delay=original['delay']))
        for k,z in d.endpoint(copy,'donor_certified',p).items():totals[k]+=z
        rows.append(dict(id=case['id'],old_lag=oldlag,outward_lag=lag,certificate=result))
        if (i+1)%64==0:print('Outward recovery correction',i+1,'/',448,flush=True)
    (OUT/'records.json').write_text(json.dumps(rows,indent=2));completion=dict(status='COMPLETE',cases=len(rows),recovery_sum_changed_cases=affected,scored_count_changed_cases=different_counts,decision_changes=changed,totals=totals,protocol_sha256=d.sha(P),records_sha256=d.sha(OUT/'records.json'),seconds=time.perf_counter()-start,scope=v['scope']);(OUT/'completion.json').write_text(json.dumps(completion,indent=2));assert all(d.sha(d.ROOT/n)==h for n,h in v['inputs'].items());print(json.dumps(completion,indent=2),flush=True)


if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
