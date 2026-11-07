"""Frozen paired evaluation, including calibration uncertainty and integrity."""
import argparse,csv,hashlib,json,math
import numpy as np
from scipy.stats import binomtest
from guarded_confirmation import verify,ROOT,DATA,OUT,P
from guarded_localization import guarded_report
from routing_scenes import make_routing_case
from analyze_routing_confirmation import endpoint,boot_indices,summarize
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def analyze():
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(P)
    assert json.loads((OUT/'completion.json').read_text())['cases']==len(m['cases'])==6099
    cases=m['cases'];rows={c['id']:json.loads((OUT/(c['id']+'.json')).read_text()) for c in cases}
    cuts=json.loads((OUT/'thresholds.json').read_text());assert cuts['protocol_sha256']==sha(P)
    groups=dict(null=[c for c in cases if c['kind']=='null' and c['split']=='evaluation'])
    groups.update({g:[c for c in cases if c['kind']=='positive' and c['support_family']==g] for g in ['coherent','singleton','filament','crossing_step']})
    rng=np.random.default_rng(20261115);boots={g:None if g=='null' else boot_indices(cs,5000,rng) for g,cs in groups.items()}
    methods=list(rows[cases[0]['id']]['reports']);outputs={};summaries={};tables=[];paired={}
    for alpha in p['alphas']:
        for method in methods:
            key=f'{method}_{alpha}';outputs[key]={};summary={}
            for g,cs in groups.items():
                es=[endpoint(c,rows[c['id']]['reports'][method],rows[c['id']]['reports'][method]['score']>cuts['methods'][method][str(alpha)]) for c in cs]
                outputs[key][g]=es;summary[g]=summarize(es,boots[g])
            summaries[key]=dict(method=method,alpha=alpha,threshold=cuts['methods'][method][str(alpha)],groups=summary)
            line=dict(method=method,alpha=alpha,null_false=summary['null']['declared']['count'])
            for g in ['coherent','singleton','filament','crossing_step']:
                for f in ['correct','wrong_depth','pair']:line[g+'_'+f]=summary[g][f]['count']
            tables.append(line)
        for g,cs in groups.items():
            for field in ['declared'] if g=='null' else ['correct','wrong_depth','pair','any_channel']:
                left=np.array([r[field] for r in outputs[f'guarded_{alpha}'][g]],int)
                right=np.array([r[field] for r in outputs[f'local_{alpha}'][g]],int);d=left-right
                index=rng.integers(len(cs),size=(5000,len(cs))) if g=='null' else boots[g]
                gain=int((d>0).sum());loss=int((d<0).sum())
                paired[f'{alpha}_{g}_{field}']=dict(difference_count=int(d.sum()),difference_rate=float(d.mean()),
                    ci95=np.quantile(d[index].mean(1),[.025,.975]).tolist(),gain=gain,loss=loss,
                    paired_discordance_two_sided_p=float(binomtest(gain,gain+loss).pvalue) if gain+loss else 1.,
                    conditional_on_calibration=True)
        assert [r['declared'] for r in outputs[f'guarded_{alpha}']['null']]==[r['declared'] for r in outputs[f'local_{alpha}']['null']]
    cal=[c for c in cases if c['split']=='calibration'];scores=np.array([rows[c['id']]['reports']['local']['score'] for c in cal])
    reps=1000;draw=rng.integers(len(cal),size=(reps,len(cal)));rank=math.ceil((len(cal)+1)*(1-p['primary_alpha']))
    thresholds=np.maximum(0.,np.partition(scores[draw],rank-1,axis=1)[:,rank-1]);two_stage={}
    for g,cs in groups.items():
        if g=='null':continue
        idx=boot_indices(cs,reps,rng);score=np.array([rows[c['id']]['reports']['local']['score'] for c in cs])
        local=np.array([abs(rows[c['id']]['reports']['local']['delay']-c['metadata']['secondary_delay'])<=2 for c in cs],int)
        guarded=np.array([abs(rows[c['id']]['reports']['guarded']['delay']-c['metadata']['secondary_delay'])<=2 for c in cs],int)
        declarations=score[idx]>thresholds[:,None];delta=(declarations*(guarded-local)[idx]).mean(1)
        two_stage[g]=dict(ci95=np.quantile(delta,[.025,.975]).tolist(),lower_one_sided95=float(np.quantile(delta,.05)),
            interpretation='Paired six-stratum test plus independent null-calibration bootstrap, descriptive intervals.')
    result=dict(status='Fresh independent guarded-localization confirmation',protocol_sha256=sha(P),primary='guarded_0.005',
        methods=summaries,paired_comparisons=paired,two_stage_bootstrap=dict(replicates=reps,groups=two_stage),
        notes=['Null decision equality follows score identity; it does not mean the population false-positive rate is smaller.',
            'The empirical localization rule is separate from certified donor-only e-value construction.',
            'Previously used stress geometries were development here; no novel-geometry generalization claim.',
            'All comparisons to previous routing are on these same fresh inputs; counts from older cohorts are not pooled.'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));(OUT/'endpoint_records.json').write_text(json.dumps(outputs))
    with (OUT/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    for line in tables:print(line,flush=True)
    print('Two-stage',two_stage,flush=True)


def audit():
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert len(m['cases'])==6099 and m['protocol_sha256']==sha(P)
    cal={};regenerated=[];identity=0;chosen={}
    for c in m['cases']:
        path=DATA/(c['id']+'.npz');assert sha(path)==c['sha256'];r=json.loads((OUT/(c['id']+'.json')).read_text())
        assert r['protocol_sha256']==sha(P) and r['id']==c['id'];rs=r['reports']
        assert rs['guarded']['score']==rs['local']['score'];identity+=1
        assert rs['guarded']==guarded_report(r['proposal'],rs['guided'],rs['local'],p['settings'])
        if c['split']=='calibration':
            for n,report in rs.items():cal.setdefault(n,[]).append(report['score'])
        group='calibration' if c['split']=='calibration' else ('null' if c['kind']=='null' else c['support_family'])
        chosen.setdefault(group,c)
    cuts=json.loads((OUT/'thresholds.json').read_text())
    for n,values in cal.items():
        values=sorted(values)
        for alpha in p['alphas']:assert cuts['methods'][n][str(alpha)]==max(0.,values[math.ceil((len(values)+1)*(1-alpha))-1])
    assert cuts['methods']['local']==cuts['methods']['guarded']
    for g,c in chosen.items():
        h,meta,_=make_routing_case(c);assert np.array_equal(h,np.load(DATA/(c['id']+'.npz'))['histogram']) and meta==c['metadata'];regenerated.append(c['id'])
    result=dict(status='PASS',cases=identity,source_files=len(p['source_sha256']),protocol_sha256=sha(P),regenerated_inputs=regenerated,
        local_score_identity_cases=identity,identical_calibration_thresholds=True,
        scope='All frozen sources, input hashes and record IDs; exact guarded reconstruction; all calibration thresholds;6deterministically regenerated acquisitions.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['analyze','audit']);args=ap.parse_args()
    analyze() if args.action=='analyze' else audit()
