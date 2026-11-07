"""Prespecified paired real-interval agreement, never physical-negative truth."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from guarded_real_confirmation import verify,ROOT,DATA,OUT,P,sha,query_settings
from guarded_real_policy import guarded_real_report
from real_poisson_analysis_v2 import route,endpoints
from real_multiframe_test import select_queries
METHODS=['local','guarded','routed','guided','spatial']


def get_report(record,normalizers,method):
    return guarded_real_report(record,normalizers) if method=='guarded' else route(record,normalizers,method)


def joint_block_difference(frames,field,replicates=5000,seed=20261117,block_size=16):
    """Conditional fixed-frame Exp(1) block bootstrap, shared within sequence.

    Positive weights retain sparse windows without conditioning away zero-
    count resamples. This approximate interval is not scene-level inference.
    """
    rng=np.random.default_rng(seed);weights={};values=[];observed=[];gain=loss=0
    blocks_count=(192//block_size)*(256//block_size)
    for f in frames:
        qs=f['queries'];left=f['left'];right=f['right'];sequence=f['sequence']
        if sequence not in weights:weights[sequence]=rng.exponential(1.,size=(replicates,blocks_count))
        blocks=np.array([(q['row']//block_size)*(256//block_size)+q['column']//block_size for q in qs])
        delta=np.array([int(a[field])-int(b[field]) for a,b in zip(left,right)])
        n=np.bincount(blocks,minlength=blocks_count);sums=np.bincount(blocks,weights=delta,minlength=blocks_count)
        draws=weights[sequence];values.append((draws@sums)/(draws@n));observed.append(float(delta.mean()))
        gain+=int((delta>0).sum());loss+=int((delta<0).sum())
    samples=np.mean(np.stack(values),axis=0)
    return dict(equal_frame_mean_difference=float(np.mean(observed)),interval_95=np.quantile(samples,[.025,.975]).tolist(),
        frame_differences=observed,frames=len(frames),sequences=len(weights),gain=gain,loss=loss,
        observed_zero_discordance=gain+loss==0,replicates=replicates,
        interpretation='Approximate conditional-frame spatial block-weighting interval; same sequence shares block weights. No independent-scene,physical-truth or calibration-uncertainty coverage claim. A zero-discordance interval is empirical degeneracy,not zero population risk.')


def analyze():
    p=verify();manifest=json.loads((DATA/'manifest.json').read_text());assert manifest['protocol_sha256']==sha(P)
    completion=json.loads((OUT/'completion.json').read_text());assert completion['cases']==completion['expected']
    summary={};outputs={};rows=[];comparisons={};primary_frames=[]
    for entry in manifest['frames']:
        name=entry['name'];fm=json.loads((DATA/name/'manifest.json').read_text());queries=fm['queries']
        for i,probability in enumerate(p['settings']['probabilities']):
            records={q['id']:json.loads((OUT/name/f'{i}_{q["id"]}.json').read_text()) for q in queries}
            assert all(r['protocol_sha256']==sha(P) for r in records.values());cal=p['calibration'][f'gaussian_{i}']
            for alpha in p['settings']['alphas']:
                for method in METHODS:
                    cut=cal['methods']['local' if method=='guarded' else method][str(alpha)]
                    groups={g:dict(n=0,declared=0,pair=0,weak_window_agreement=0,accepted_unmatched=0,primary_agreement=0,
                        duplicate_known_window=0,declared_incomplete_pair=0,location_adjusted=0,actual_location_changed=0) for g in ['known_one','two_interval']}
                    es=[]
                    for q in queries:
                        r=records[q['id']];report=get_report(r,cal['normalizers'],method);local=route(r,cal['normalizers'],'local')
                        if method=='guarded':assert report['score']==local['score'] and report['primary']==local['primary']
                        declared=report['score']>cut;e=endpoints(q,report,declared)
                        e['declared_incomplete_pair']=bool(q['split']=='two_interval' and declared and not e['pair'])
                        e['location_adjusted']=bool(declared and report.get('location_adjusted',False))
                        e['actual_location_changed']=bool(declared and report['delay']!=local['delay'])
                        g=groups[q['split']];g['n']+=1
                        for key,value in e.items():g[key]+=value
                        es.append(dict(query_id=q['id'],split=q['split'],row=q['row'],column=q['column'],primary=report['primary'],
                            delay=report['delay'],score=report['score'],threshold=cut,head=report['head'],**e))
                    key=f'{name}_{i}_{alpha}_{method}';summary[key]=dict(frame=name,sequence=fm['sequence'],probability=probability,alpha=alpha,method=method,groups=groups);outputs[key]=es
                    for split,group in groups.items():rows.append(dict(frame=name,sequence=fm['sequence'],probability=probability,alpha=alpha,method=method,split=split,**group))
                local=outputs[f'{name}_{i}_{alpha}_local'];guard=outputs[f'{name}_{i}_{alpha}_guarded']
                assert [x['declared'] for x in local]==[x['declared'] for x in guard]
                for split,field in [('two_interval','pair'),('two_interval','accepted_unmatched'),('known_one','accepted_unmatched')]:
                    ids=[j for j,q in enumerate(queries) if q['split']==split]
                    f=dict(sequence=fm['sequence'],queries=[queries[j] for j in ids],left=[guard[j] for j in ids],right=[local[j] for j in ids])
                    comparisons[f'{name}_{i}_{alpha}_{split}_{field}']=joint_block_difference([f],field,
                        p['settings']['bootstrap_replicates'],p['settings']['bootstrap_seed'],p['settings']['spatial_block_size'])
                    if probability==p['settings']['primary_retention'] and alpha==p['settings']['primary_alpha'] and split=='two_interval' and field=='pair':primary_frames.append(f)
            gs=[summary[f'{name}_{i}_0.01_{method}']['groups'] for method in ['local','guarded','routed']]
            print(name,probability,'pairs',[g['two_interval']['pair'] for g in gs],
                'known-one unmatched',[g['known_one']['accepted_unmatched'] for g in gs],flush=True)
    primary=joint_block_difference(primary_frames,'pair',p['settings']['bootstrap_replicates'],p['settings']['bootstrap_seed'],p['settings']['spatial_block_size'])
    result=dict(status='Prospective final-guard real interval-agreement evaluation; not physical detection truth.',protocol_sha256=sha(P),
        primary=dict(alpha=p['settings']['primary_alpha'],retention=p['settings']['primary_retention'],comparison='guarded-minus-local',**primary),
        methods=summary,paired_comparisons=comparisons,
        notes=p['notes']+['Same-sequence new captures are not independently new scenes. All fixed frames/retentions/policies retained.',
            'Known-one unmatched is not a physical false-positive count; score/decision identity holds exactly independently of labels.'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));(OUT/'endpoint_records.json').write_text(json.dumps(outputs))
    with (OUT/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('PRIMARY',json.dumps(result['primary']),flush=True)


def audit():
    p=verify();manifest=json.loads((DATA/'manifest.json').read_text());assert manifest['protocol_sha256']==sha(P)
    cases=0;regens=[];old=json.loads((ROOT/'REAL_POISSON_TEST_PROTOCOL.json').read_text())['frames']
    old_members={f['member'] for f in old}
    for fi,(item,entry) in enumerate(zip(p['frames'],manifest['frames'])):
        assert item['member'] not in old_members and item['sequence']!=7
        directory=DATA/entry['name'];assert sha(directory/'manifest.json')==entry['manifest_sha256'];fm=json.loads((directory/'manifest.json').read_text())
        assert sha(directory/'frame_histograms.npz')==fm['frame_sha256'];h=np.load(directory/'frame_histograms.npz')['histograms']
        assert sha(directory/Path(fm['download']['member']).name)==fm['download']['sha256']
        qs=select_queries(np.load(ROOT/item['label_file_relative'],allow_pickle=False)['labels'],query_settings(p,item),fi)
        assert json.loads((directory/'query_coordinates.json').read_text())==qs and sha(directory/'query_coordinates.json')==fm['coordinate_sha256']
        assert len({q['pixel'] for q in qs})==len(qs)==len(fm['queries'])
        for q,stored in zip(qs,fm['queries']):
            assert all(stored[k]==v for k,v in q.items())
            if len(stored['windows'])==2:assert stored['weak_window']==int(np.argmin([h[stored['row'],stored['column'],a:b+1].sum() for a,b in stored['windows']]))
        for i,inp in enumerate(fm['inputs']):
            assert sha(directory/f'thin_{i}.npz')==inp['sha256']
            if i==1:
                child=np.random.default_rng(inp['seed']).binomial(h,inp['probability']).astype(np.int32)
                assert np.array_equal(child,np.load(directory/f'thin_{i}.npz')['histograms']);regens.append(entry['name']+'_1')
            cal=p['calibration'][f'gaussian_{i}']
            for q in fm['queries']:
                r=json.loads((OUT/entry['name']/f'{i}_{q["id"]}.json').read_text());assert r['protocol_sha256']==sha(P) and r['query_id']==q['id']
                assert r['attenuation_index']==i and r['frame_name']==entry['name']
                local=route(r,cal['normalizers'],'local');guard=guarded_real_report(r,cal['normalizers'])
                assert local['score']==guard['score'] and local['primary']==guard['primary']
                for alpha in p['settings']['alphas']:
                    cut=cal['methods']['local'][str(alpha)];assert (local['score']>cut)==(guard['score']>cut)
                cases+=1
            assert len(list((OUT/entry['name']).glob(f'{i}_*.json')))==len(qs)
    completion=json.loads((OUT/'completion.json').read_text());assert cases==completion['cases']==completion['expected']
    result=dict(status='PASS',protocol_sha256=sha(P),frames=len(manifest['frames']),cases=cases,source_files=len(p['source_sha256']),
        regenerated_primary_retention_tensors=regens,local_guarded_score_and_decision_identity=True,
        scope='All frozen sources,labels,downloaded-member and count-tensor hashes; exact query selection; all fitted-record identities; primary thinning regenerated perframe. No proof of label completeness,scene independence or physical calibration.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['analyze','audit']);args=ap.parse_args()
    analyze() if args.action=='analyze' else audit()
