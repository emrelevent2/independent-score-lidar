"""Frozen coverage efficacy,proper-mass audit and calibrated controls."""
import argparse,csv,json,math
from fractions import Fraction
import numpy as np
from flint import arb
from coverage_confirmation import verify,ROOT,DATA,OUT,P,sha,EMPIRICAL
from coverage_alternative import coverage_components,numeric_scores,sparse_numerator_lower,location_report
from envelope_evidence import outward_float,continuous_null_bound
from guided_alternatives import guided_pool,fit_pool
from routing_scenes import make_routing_case
from analyze_routing_confirmation import endpoint,boot_indices,summarize


def analyze():
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P)
    assert json.loads((OUT/'completion.json').read_text())['cases']==len(m['cases'])==6099
    cases=m['cases'];rows={c['id']:json.loads((OUT/(c['id']+'.json')).read_text()) for c in cases}
    assert all(r['protocol_sha256']==sha(P) for r in rows.values());cuts=json.loads((OUT/'thresholds.json').read_text())['methods']
    groups=dict(null=[c for c in cases if c['kind']=='null' and c['split']=='evaluation'])
    groups.update({g:[c for c in cases if c['kind']=='positive' and c['support_family']==g] for g in ['coherent','singleton','filament','crossing_step']})
    rng=np.random.default_rng(20261119);boots={g:rng.integers(len(cs),size=(5000,len(cs))) if g=='null' else boot_indices(cs,5000,rng) for g,cs in groups.items()}
    outputs={};methods={};table=[]
    configurations=[(method,.01,True) for method in ['spatial_cert','coverage_cert']]+[(method,alpha,False) for method in EMPIRICAL for alpha in p['empirical_alphas']]
    for method,alpha,analytic in configurations:
        label=f'{method}_{alpha}';outputs[label]={};summary={}
        for g,cs in groups.items():
            es=[]
            for c in cs:
                r=rows[c['id']]['reports'][method];declared=r['declared'] if analytic else r['score']>cuts[method][str(alpha)]
                es.append(endpoint(c,r,declared))
            outputs[label][g]=es;summary[g]=summarize(es,None if g=='null' else boots[g])
        methods[label]=dict(method=method,alpha=alpha,analytic=analytic,groups=summary)
        row=dict(method=method,alpha=alpha,analytic=analytic,null_false=summary['null']['declared']['count'])
        for g in ['coherent','singleton','filament','crossing_step']:
            for field in ['correct','wrong_depth','pair']:row[g+'_'+field]=summary[g][field]['count']
        table.append(row)
    comparisons={}
    for control in methods:
        if methods[control]['analytic']:
            primary='coverage_cert_0.01'
        else:primary=f'coverage_fast_{methods[control]["alpha"]}'
        comparisons[control]={}
        for g,cs in groups.items():
            for field in ['declared'] if g=='null' else ['correct','wrong_depth','pair','any_channel']:
                a=np.array([r[field] for r in outputs[primary][g]],int);b=np.array([r[field] for r in outputs[control][g]],int);d=a-b
                samples=d[boots[g]].mean(1)
                comparisons[control][g+'_'+field]=dict(count_difference=int(d.sum()),rate_difference=float(d.mean()),
                    ci95=np.quantile(samples,[.025,.975]).tolist(),lower_one_sided95=float(np.quantile(samples,.05)),
                    gain=int((d>0).sum()),loss=int((d<0).sum()),conditional_on_calibration=not methods[control]['analytic'])
    # Independent calibration resampling for the secondary empirical comparisons.
    cal=[c for c in cases if c['split']=='calibration'];reps=1000;calidx=rng.integers(len(cal),size=(reps,len(cal)))
    evalidx={g:(rng.integers(len(cs),size=(reps,len(cs))) if g=='null' else boot_indices(cs,reps,rng)) for g,cs in groups.items()}
    values={}
    for method in EMPIRICAL:
        cs=np.array([rows[c['id']]['reports'][method]['score'] for c in cal]);rank=math.ceil((len(cal)+1)*.995)
        thresholds=np.maximum(0.,np.partition(cs[calidx],rank-1,axis=1)[:,rank-1]);values[method]={}
        for g,seq in groups.items():
            scores=np.array([rows[c['id']]['reports'][method]['score'] for c in seq]);index=evalidx[g];declared=scores[index]>thresholds[:,None]
            if g=='null':values[method][g+'_declared']=declared.mean(1)
            else:
                good=np.array([abs(rows[c['id']]['reports'][method]['delay']-c['metadata']['secondary_delay'])<=2 for c in seq])
                values[method][g+'_correct']=(declared&good[index]).mean(1)
                values[method][g+'_wrong_depth']=(declared&~good[index]).mean(1)
    two_stage={method:{k:dict(ci95=np.quantile(values['coverage_fast'][k]-v,[.025,.975]).tolist(),
        lower_one_sided95=float(np.quantile(values['coverage_fast'][k]-v,.05))) for k,v in fields.items()} for method,fields in values.items()}
    primary=comparisons['spatial_cert_0.01'];criteria=dict(
        isolated_gain_lower95_positive=primary['singleton_correct']['lower_one_sided95']>0,
        coherent_cost_above_prespecified_minus3pp=primary['coherent_correct']['lower_one_sided95']>-.03,
        interpretation='Prespecified descriptive paired-bootstrap criteria; not an exact finite-sample noninferiority theorem.')
    result=dict(status='Fresh prospective proper-coverage confirmation completed.',protocol_sha256=sha(P),primary='coverage_cert_0.01',methods=methods,
        paired_comparisons=comparisons,two_stage_empirical=dict(replicates=reps,alpha=.005,paired=two_stage),criteria=criteria,
        notes=['Analytic declarations require certified sparse numerator with original weights and the shared complete continuous-null upper.',
            'Fast scores are empirical and do not inherit analytic control. All observed null and wrong-positive counts remain separate.',
            'Every geometry family is within the development design; no out-of-distribution or physical sensor-accuracy claim.'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));(OUT/'endpoint_records.json').write_text(json.dumps(outputs))
    with (OUT/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    for row in table:print(row,flush=True)
    print('PRIMARY CRITERIA',criteria,flush=True)


def audit():
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P) and len(m['cases'])==6099
    cuts=json.loads((OUT/'thresholds.json').read_text());assert cuts['protocol_sha256']==sha(P)
    cal={k:[] for k in EMPIRICAL};regens={};accepted=bounded=0
    for c in m['cases']:
        assert sha(DATA/(c['id']+'.npz'))==c['sha256'];r=json.loads((OUT/(c['id']+'.json')).read_text())
        assert r['id']==c['id'] and r['protocol_sha256']==sha(P);e=r['evidence'];q=r['proposal'];assert q['raw_center_excluded']
        h=np.load(DATA/(c['id']+'.npz'))['histogram'][8,8]
        if e['bound'] is not None:
            bounded+=1;models=coverage_components(q,cycles=1000,amplitudes=p['settings']['secondary_amplitudes'])
            assert sum(models['weights'],Fraction(0))==1 and str(models['spatial_mass'])==e['spatial_mass']
            assert __import__('hashlib').sha256(models['rates'].tobytes()).hexdigest()==e['rate_matrix_sha256']
            sparse=sparse_numerator_lower(h,models,e['sparse_numerator']['indices'],1000)
            assert sparse==e['sparse_numerator'] and sparse['original_weights_retained']
            delta=outward_float(arb(sparse['lower'])-arb(e['bound']['upper']),False)
            assert delta==e['coverage_log_e_lower']
            assert r['reports']['coverage_cert']['declared']==(delta>=e['threshold_upper'])
            if r['reports']['coverage_cert']['declared']:accepted+=1
        else:
            assert e['screened'] and not r['reports']['coverage_cert']['declared'] and not r['reports']['spatial_cert']['declared']
        if c['split']=='calibration':
            for method in EMPIRICAL:cal[method].append(r['reports'][method]['score'])
        group='calibration' if c['split']=='calibration' else 'null' if c['kind']=='null' else c['support_family']
        regens.setdefault(group,c)
    for method,values in cal.items():
        values=sorted(values)
        for alpha in p['empirical_alphas']:assert cuts['methods'][method][str(alpha)]==max(0.,values[math.ceil((len(values)+1)*(1-alpha))-1])
    regenerated=[]
    for g,c in regens.items():
        h,meta,_=make_routing_case(c);assert np.array_equal(h,np.load(DATA/(c['id']+'.npz'))['histogram']) and meta==c['metadata']
        r=json.loads((OUT/(c['id']+'.json')).read_text());pool,info=guided_pool(h,8,8);q=dict(**fit_pool(pool),**info);assert q==r['proposal']
        regenerated.append(c['id'])
    completion=json.loads((OUT/'completion.json').read_text());assert completion['cases']==6099
    result=dict(status='PASS',protocol_sha256=sha(P),cases=6099,source_files=len(p['source_sha256']),
        bounded_numerators_fully_recomputed=bounded,coverage_accepted=accepted,regenerated_observations_and_donor_proposals=regenerated,
        scope='All source/input/result hashes and identities,all computed sparse numerators/exact prior masses/acceptance inequalities,all calibration cutoffs,and6exact observation/proposal regenerations. Continuous-null upper algorithm independently covered by mathematical/software checks; no physical calibration claim.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['analyze','audit']);args=ap.parse_args()
    analyze() if args.action=='analyze' else audit()
