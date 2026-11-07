"""Post-development omnibus/likelihood-location hypothesis,not confirmation."""
from pathlib import Path
import sys,json
import numpy as np
import safe_donor_development as d
import safe_spatial_control_addendum as extra
from safe_donor_alternative import build_alternative
from certified_safe_components import exact_safe_statistics
from safe_deadtime_exposure import safe_lag
from paired_rank_fusion import paired_rank_pvalue
P=d.ROOT/'CLOSED_OMNIBUS_LOCATION_DIAGNOSTIC_PROTOCOL.json';OUT=d.ROOT/'results/closed_omnibus_location_diagnostic_v1'


def freeze():
    if P.exists():raise FileExistsError('Frozen protocol exists')
    v=dict(status='FROZEN_ON_ALREADY_CLOSED_DEVELOPMENT_BEFORE_OMNIBUS_EVALUATION',sources={str(Path(__file__).relative_to(d.ROOT)):d.sha(Path(__file__)),'src/paired_rank_fusion.py':d.sha(d.ROOT/'src/paired_rank_fusion.py')},inputs={str(p.relative_to(d.ROOT)):d.sha(p) for p in [d.OUT/'records.sqlite3',extra.OUT/'records.sqlite3',d.OUT/'analysis.json',extra.OUT/'analysis.json']},rules='Joint symmetric minP rank of local,spatial-prior and fixed-mixture likelihood statistics. Smallest component p selects location,local then spatial on ties. Compare original MAP location versus unweighted maximum component likelihood when mixture is selected. Original numerator weights/presence statistics unchanged;no ground truth in either selector.',scope='New CLOSED development hypotheses after failed methods;not prospective efficacy or novelty proof. Standard joint resampling/omnibus and marginal-test/profile-localization ideas. Any promotion needs fresh independently frozen data.')
    P.write_text(json.dumps(v,indent=2));print('Frozen closed omnibus diagnostic',d.sha(P))


def run():
    p=d.check();extra.check();v=json.loads(P.read_text())
    for n,h in {**v['sources'],**v['inputs']}.items():assert d.sha(d.ROOT/n)==h,n
    base=d.read();controls=extra.read();totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in ['omnibus_map','omnibus_profile_location','mixture_profile_location']};out=[]
    for row,control in zip(base,controls):
        case=row['case'];g=p['regimes'][case['regime']];obs,ds,de,truth,dh=d.data(case,p);assert dh==row['data_sha256'];alt=build_alternative(ds,de,p['settings']['sigma']);assert alt['sha256']==row['alternative_sha256'];s=exact_safe_statistics(obs,g['cycles'],p['settings']['bins'],safe_lag(g['td'],g['te']));e=np.array([float(z.mid()) for z in s['exposure']]);ll=s['counts']@np.log(alt['rates']).T-e@alt['rates'].T;ml=int(np.argmax(ll));mlloc=alt['locations'][ml];local=row['samples']['local_bootstrap'];spatial=control['sample'];mixture=row['samples']['donor_bootstrap'];parts=[local,spatial,mixture];b=np.column_stack([z['bootstrap_scores'] for z in parts]);fusion=paired_rank_pvalue([z['score'] for z in parts],b);chosen=parts[fusion['selected']];sample={}
        for method in totals:
            pv=mixture['pvalue'] if method=='mixture_profile_location' else fusion['pvalue'];loc=mlloc if method=='mixture_profile_location' or (method=='omnibus_profile_location' and fusion['selected']==2) else [chosen['primary'],chosen['delay']];declare=pv<=.01;distance=lambda x,y:abs((x-y+32)%64-32);correct=case['positive'] and declare and distance(loc[0],truth['primary'])<=2 and distance(loc[1],truth['secondary'])<=2;ep=dict(correct=int(correct),wrong=int(case['positive'] and declare and not correct),false=int(not case['positive'] and declare),positive=int(case['positive']),null=int(not case['positive']))
            for k,z in ep.items():totals[method][k]+=z
            sample[method]=dict(pvalue=pv,primary=float(loc[0]),delay=float(loc[1]))
        out.append(dict(id=case['id'],fusion=fusion,maximum_likelihood_component=ml,samples=sample))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'records.json').write_text(json.dumps(out,indent=2));r=dict(status='CLOSED_DIAGNOSTIC_ONLY',totals=totals,protocol_sha256=d.sha(P),scope=v['scope']);(OUT/'analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))


if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
