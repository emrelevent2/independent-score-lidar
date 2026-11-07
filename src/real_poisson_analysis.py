"""Real development annotation agreement; never physical false-positive truth."""
from pathlib import Path
import argparse,datetime,hashlib,json,math
import numpy as np
from real_poisson_development import ROOT,DATA,OUT,P,verify
RULE=ROOT/'REAL_MODEL_SELECTION_RULE.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze_rule():
    if RULE.exists():raise RuntimeError('Existing selection rule retained')
    p=dict(status='Selection rule fixed before aggregate development accuracy inspection.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),development_protocol_sha256=sha(P),
        primary_alpha=.01,kinds=['gaussian','empirical'],
        utility='Mean across the3fixed attenuation probabilities: two-interval pair recovery minus accepted-unmatched rate on known-one-interval queries.',
        selection='Choose kind independently for routed,local,spatial,guided,naive_union,disagreement;Gaussian wins exact tie. Keep all development outcomes.',
        notes=['Baseline local method receives the same two-kind selection opportunity as the proposed router.',
            'These are development agreement metrics, not physical true/false-positive rates.',
            'Untouched Sequence17 test is not accessed. The complete selected test settings must be frozen later.'])
    RULE.write_text(json.dumps(p,indent=2))


def route(record,normalizers,method):
    q=record['proposal'];r=record['reports'];ss=r['guided']['score']/normalizers['guided'];ls=r['local']['score']/normalizers['local']
    spatial=dict(score=ss,primary=r['guided']['primary'],delay=q['secondary'],head='spatial')
    local=dict(score=ls,primary=r['local']['primary'],delay=r['local']['delay'],head='local')
    if method=='guided':return dict(spatial,delay=r['guided']['delay'])
    if method=='spatial':return spatial
    if method=='local':return local
    if method=='naive_union':return spatial if ss>=ls else local
    chosen=spatial if q['support_proxy']>=3.5 else local
    if method=='disagreement' and abs(local['delay']-q['secondary'])>4 and ls>1.5*ss:chosen=local
    if method not in ['routed','disagreement']:raise ValueError('Unknown method')
    return chosen


def interval_match(delay,window,tolerance=2.):return window[0]-tolerance<=delay<=window[1]+tolerance


def endpoints(query,report,declared):
    windows=query['windows'];p,s=report['primary'],report['delay'];locations=[p,s] if declared else [p]
    matched=[any(interval_match(d,w) for d in locations) for w in windows]
    primary_agreement=any(interval_match(p,w) for w in windows)
    unmatched=bool(declared and any(not any(interval_match(d,w) for w in windows) for d in locations))
    pair=False
    if len(windows)==2 and declared:
        pair=(interval_match(p,windows[0]) and interval_match(s,windows[1])) or (interval_match(p,windows[1]) and interval_match(s,windows[0]))
    weak=bool(matched[query.get('weak_window',0)])
    return dict(declared=bool(declared),pair=bool(pair),weak_window_agreement=weak,
        accepted_unmatched=unmatched,primary_agreement=bool(primary_agreement),
        duplicate_known_window=bool(declared and len(windows)==1 and not unmatched))


def threshold(values,alpha,floor=0.):
    rank=math.ceil((len(values)+1)*(1-alpha))
    if rank>len(values):return float('inf')
    return max(floor,float(np.sort(values)[rank-1]))


def main():
    p=verify();assert json.loads((OUT/'completion.json').read_text())['cases']==5394
    rule=json.loads(RULE.read_text());assert rule['development_protocol_sha256']==sha(P)
    m=json.loads((DATA/'manifest.json').read_text());queries=m['queries'];methods=['local','guided','spatial','routed','naive_union','disagreement'];summary={};calibration={};all_records={}
    normal=[q for q in queries if q['split']=='normalization'];cal=[q for q in queries if q['split']=='calibration']
    evaluation=[q for q in queries if q['split'].startswith('evaluation')]
    for kind in p['settings']['kinds']:
        for index,probability in enumerate(p['settings']['probabilities']):
            records={}
            for q in queries:
                r=json.loads((OUT/f'{kind}_{index}_{q["id"]}.json').read_text());assert r['protocol_sha256']==sha(P)
                records[q['id']]=r
            norms={name:threshold([records[q['id']]['reports'][name]['score'] for q in normal],.01,1e-6) for name in ['local','guided']}
            key=f'{kind}_{index}';calibration[key]=dict(kind=kind,probability=probability,normalizers=norms,methods={})
            for method in methods:
                scores=[route(records[q['id']],norms,method)['score'] for q in cal]
                for alpha in [.005,.01]:
                    cut=threshold(scores,alpha);label=f'{kind}_{index}_{method}_{alpha}'
                    groups={g:dict(n=0,declared=0,pair=0,weak_window_agreement=0,accepted_unmatched=0,primary_agreement=0,duplicate_known_window=0) for g in ['evaluation_one','evaluation_two']}
                    outputs=[]
                    for q in evaluation:
                        r=route(records[q['id']],norms,method);e=endpoints(q,r,r['score']>cut);s=groups[q['split']];s['n']+=1
                        for field,value in e.items():s[field]+=value
                        outputs.append(dict(query_id=q['id'],row=q['row'],column=q['column'],split=q['split'],**e))
                    summary[label]=dict(kind=kind,probability=probability,method=method,alpha=alpha,threshold=cut,groups=groups)
                    all_records[label]=outputs
                    if alpha==.01:print(kind,probability,method,'pair',groups['evaluation_two']['pair'],'/200','unmatched_one',groups['evaluation_one']['accepted_unmatched'],'/100',flush=True)
                    calibration[key]['methods'].setdefault(method,{})[str(alpha)]=cut
    selected={};utilities={}
    for method in methods:
        utilities[method]={}
        for kind in p['settings']['kinds']:
            values=[]
            for index in range(3):
                s=summary[f'{kind}_{index}_{method}_0.01']['groups']
                values.append(s['evaluation_two']['pair']/s['evaluation_two']['n']-s['evaluation_one']['accepted_unmatched']/s['evaluation_one']['n'])
            utilities[method][kind]=float(np.mean(values))
        selected[method]=max(p['settings']['kinds'],key=lambda k:utilities[method][k])
    result=dict(status='DEVELOPMENT annotation agreement only',protocol_sha256=sha(P),selection_rule_sha256=sha(RULE),
        methods=summary,selected_kinds=selected,selection_utilities=utilities,
        notes=['Unknown/unlabeled bins never become true negatives. Accepted-unmatched is not a physical false-positive count.',
            'One thinned full frame per exposure; nearby query windows overlap. No iid-pixel confidence intervals are asserted.',
            'Working Poisson scores on attenuated recorded counts do not establish sensor calibration.'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False));(OUT/'calibration.json').write_text(json.dumps(calibration,indent=2))
    (OUT/'endpoint_records.json').write_text(json.dumps(all_records));print('Selected kinds on DEVELOPMENT',selected,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze_rule','analyze']);args=ap.parse_args()
    freeze_rule() if args.action=='freeze_rule' else main()
