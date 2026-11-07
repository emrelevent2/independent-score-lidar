"""Predefined held-out annotation endpoints and paired block resampling."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from real_poisson_analysis_v2 import route,endpoints
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'REAL_POISSON_TEST_PROTOCOL.json'
DATA=ROOT/'data/real_multiframe_test_v1';OUT=ROOT/'results/real_multiframe_test_v1'
METHODS=['local','guided','spatial','routed','naive_union','disagreement','selected_anchored']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def report(record,p,method,i,alpha):
    if method=='selected_anchored':
        selected=p['anchored_selected']
        if selected['depth_sigma'] is None:method='local'
        else:
            key=f'sigma{int(selected["depth_sigma"])}_{i}';cal=p['anchored_calibration'][key]
            r=route(record['anchored'],cal['normalizers'],selected['method'])
            return r,cal['methods'][selected['method']][str(alpha)]
    cal=p['development_calibration'][f'{p["selected_kinds"][method]}_{i}']
    r=route(record['base'],cal['normalizers'],method)
    return r,cal['methods'][method][str(alpha)]


def block_difference(queries,left,right,field,block_size=16,reps=2000,seed=2026111400):
    """Paired approximate spatial bootstrap; conditional on this single frame."""
    blocks=np.array([(q['row']//block_size)*(256//block_size)+q['column']//block_size for q in queries])
    unique,index=np.unique(blocks,return_inverse=True);n=np.bincount(index)
    delta=np.array([int(l[field])-int(r[field]) for l,r in zip(left,right)])
    sums=np.bincount(index,weights=delta);rng=np.random.default_rng(seed)
    draws=rng.integers(len(unique),size=(reps,len(unique)))
    values=sums[draws].sum(1)/n[draws].sum(1)
    return dict(difference=float(delta.mean()),interval_95=np.quantile(values,[.025,.975]).tolist(),
        sampled_blocks=len(unique),n=len(queries),replicates=reps,
        interpretation='Approximate paired spatial-block bootstrap conditional on this frame; overlapping neighborhoods and calibration uncertainty are not independently identified.')


def analyze():
    from real_multiframe_test import verify
    p=verify();m=json.loads((DATA/'manifest.json').read_text());completion=json.loads((OUT/'completion.json').read_text())
    assert completion['protocol_sha256']==sha(P) and completion['cases']==completion['expected']
    summary={};outputs={};comparisons={};rows=[]
    for f in m['frames']:
        name=f['name'];fm=json.loads((DATA/name/'manifest.json').read_text());qs=fm['queries']
        for i,prob in enumerate(p['settings']['probabilities']):
            cache={}
            for kind in set(p['selected_kinds'].values())|{p['anchored_selected']['kind']}:
                cache[kind]={q['id']:json.loads((OUT/name/f'{kind}_{i}_{q["id"]}.json').read_text()) for q in qs}
                assert all(r['protocol_sha256']==sha(P) for r in cache[kind].values())
            for alpha in p['settings']['alphas']:
                for method in METHODS:
                    kind=p['anchored_selected']['kind'] if method=='selected_anchored' else p['selected_kinds'][method]
                    gs={s:dict(n=0,declared=0,pair=0,weak_window_agreement=0,accepted_unmatched=0,primary_agreement=0,duplicate_known_window=0,declared_incomplete_pair=0) for s in ['known_one','two_interval']}
                    records=[]
                    for q in qs:
                        r,cut=report(cache[kind][q['id']],p,method,i,alpha);e=endpoints(q,r,r['score']>cut)
                        e['declared_incomplete_pair']=bool(q['split']=='two_interval' and e['declared'] and not e['pair'])
                        g=gs[q['split']];g['n']+=1
                        for k,v in e.items():g[k]+=v
                        records.append(dict(query_id=q['id'],split=q['split'],row=q['row'],column=q['column'],primary=r['primary'],delay=r['delay'],score=r['score'],threshold=cut,head=r['head'],**e))
                    key=f'{name}_{i}_{alpha}_{method}';summary[key]=dict(frame=name,probability=prob,alpha=alpha,method=method,groups=gs);outputs[key]=records
                    for split,g in gs.items():rows.append(dict(frame=name,probability=prob,alpha=alpha,method=method,split=split,**g))
                if alpha==p['settings']['primary_alpha']:
                    local=outputs[f'{name}_{i}_{alpha}_local']
                    for method in ['selected_anchored','routed','spatial']:
                        records=outputs[f'{name}_{i}_{alpha}_{method}']
                        for split,field in [('two_interval','pair'),('two_interval','accepted_unmatched'),('known_one','accepted_unmatched')]:
                            ids=[j for j,q in enumerate(qs) if q['split']==split]
                            result=block_difference([qs[j] for j in ids],[records[j] for j in ids],[local[j] for j in ids],field,
                                p['settings']['spatial_block_size'],p['settings']['bootstrap_replicates'])
                            comparisons[f'{name}_{i}_{method}_{split}_{field}']=result
            print(name,'attenuation',prob,'primary alpha',p['settings']['primary_alpha'],flush=True)
            for method in METHODS:
                gs=summary[f'{name}_{i}_{p["settings"]["primary_alpha"]}_{method}']['groups']
                print(method,'pair',gs['two_interval']['pair'],'/500','unmatched',gs['two_interval']['accepted_unmatched'],'/500 +',gs['known_one']['accepted_unmatched'],'/1000',flush=True)
    result=dict(status='HELD-OUT real annotation agreement; not physical detection truth.',protocol_sha256=sha(P),methods=summary,
        comparisons=comparisons,primary_frame='Sequence17_first',validation_frame='Sequence3_middle',
        selected_anchored=p['anchored_selected']['id'],notes=p['notes'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));(OUT/'endpoint_records.json').write_text(json.dumps(outputs))
    with (OUT/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit():
    from real_multiframe_test import verify,select_queries
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P)
    count=0;regenerated=[]
    kinds=set(p['selected_kinds'].values())|{p['anchored_selected']['kind']}
    for fi,(item,entry) in enumerate(zip(p['frames'],m['frames'])):
        directory=DATA/entry['name'];assert sha(directory/'manifest.json')==entry['manifest_sha256'];fm=json.loads((directory/'manifest.json').read_text())
        assert fm['protocol_sha256']==sha(P) and item['sequence']!=7
        h=np.load(directory/'frame_histograms.npz')['histograms'];assert sha(directory/'frame_histograms.npz')==fm['frame_sha256']
        assert sha(directory/Path(fm['download']['member']).name)==fm['download']['sha256']
        qs=select_queries(np.load(ROOT/item['label_file_relative'],allow_pickle=False)['labels'],p['settings'],fi)
        saved=json.loads((directory/'query_coordinates.json').read_text());assert saved==qs
        assert sha(directory/'query_coordinates.json')==fm['coordinate_sha256']
        assert len(set(q['pixel'] for q in qs))==1500 and len(qs)==len(fm['queries'])
        for old,new in zip(qs,fm['queries']):
            assert all(new[k]==v for k,v in old.items())
            if len(new['windows'])==2:assert new['weak_window']==int(np.argmin([h[new['row'],new['column'],a:b+1].sum() for a,b in new['windows']]))
        for i,input in enumerate(fm['inputs']):
            path=directory/f'thin_{i}.npz';assert sha(path)==input['sha256']
            if i==0:
                regen=np.random.default_rng(input['seed']).binomial(h,input['probability']).astype(np.int32)
                assert np.array_equal(regen,np.load(path)['histograms']);regenerated.append(entry['name']+'_0')
            for kind in kinds:
                for q in fm['queries']:
                    record=json.loads((OUT/entry['name']/f'{kind}_{i}_{q["id"]}.json').read_text())
                    assert record['protocol_sha256']==sha(P) and record['query_id']==q['id']
                    assert record['kind']==kind and record['attenuation_index']==i;count+=1
        assert len(list((OUT/entry['name']).glob('*.json')))==1500*3*len(kinds)
    completion=json.loads((OUT/'completion.json').read_text());assert count==completion['cases']==completion['expected']
    result=dict(status='PASS',protocol_sha256=sha(P),cases=count,frames=len(m['frames']),regenerated_thinnings=regenerated,
        scope='Source/input hashes, whole downloaded members, all prepared count files, all query records, exact annotation selection, weak-role assignment and two deterministic attenuation regenerations. No claim of annotation completeness or physical calibration.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['analyze','audit']);args=ap.parse_args()
    analyze() if args.action=='analyze' else audit()
