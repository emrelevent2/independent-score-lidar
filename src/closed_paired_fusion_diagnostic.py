"""Post-development diagnostic only;never called independent confirmation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import scipy
import safe_donor_development as d
import safe_spatial_control_addendum as extra
from paired_rank_fusion import paired_rank_pvalue
P=d.ROOT/'CLOSED_PAIRED_FUSION_DIAGNOSTIC_PROTOCOL.json';OUT=d.ROOT/'results/closed_paired_fusion_diagnostic_v1'


def freeze():
    if P.exists():raise FileExistsError('Protocol exists')
    assert (d.OUT/'analysis.json').exists() and (extra.OUT/'analysis.json').exists()
    names=['src/closed_paired_fusion_diagnostic.py','src/paired_rank_fusion.py','src/test_paired_rank_fusion.py'];v=dict(status='FROZEN_AFTER_BASE_RESULTS_CLOSED_BEFORE_FUSION_RECOUNT',sources={n:d.sha(d.ROOT/n) for n in names},inputs={str(p.relative_to(d.ROOT)):d.sha(p) for p in [d.OUT/'records.sqlite3',extra.OUT/'records.sqlite3',d.OUT/'analysis.json',extra.OUT/'analysis.json']},methods=['paired_min_rank','paired_min_rank_with_agreement'],selection='Symmetric paired ranks of local and equal-context prior-score statistics,including observed row. Select smaller component p,local on ties. Second candidate additionally requires reported secondary delays agree within one pulseFWHM=2sqrt(2log2)*sigma. This only removes declarations,no extra chance to reject. No labels in either rule.',scope='CLOSED development diagnostic following failed base methods. Any apparent gain is hypothesis generation,not efficacy confirmation. Generic minP/rankfusion and agreement gating are established. Fitted-null bootstrap pvalues remain approximate;certified guarantee does not transfer.')
    P.write_text(json.dumps(v,indent=2));print('Frozen closed diagnostic',d.sha(P))


def run():
    p=d.check();extra.check();v=json.loads(P.read_text())
    for n,h in {**v['sources'],**v['inputs']}.items():assert d.sha(d.ROOT/n)==h,n
    base=d.read();controls=extra.read();out=[];totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in v['methods']}
    for row,control in zip(base,controls):
        assert row['case']==control['case'];local=row['samples']['local_bootstrap'];spatial=control['sample'];boots=np.column_stack([local['bootstrap_scores'],spatial['bootstrap_scores']]);r=paired_rank_pvalue([local['score'],spatial['score']],boots);chosen=[local,spatial][r['selected']];distance=abs((local['delay']-spatial['delay']+32)%64-32);agrees=distance<=2*np.sqrt(2*np.log(2))*p['settings']['sigma'];samples={}
        for method in v['methods']:
            declared=r['pvalue']<=.01 and (method=='paired_min_rank' or agrees);samples[method]=dict(declared=bool(declared),primary=chosen['primary'],delay=chosen['delay'],pvalue=r['pvalue'])
            pos=row['case']['positive'];dist=lambda x,y:abs((x-y+32)%64-32);correct=pos and declared and dist(chosen['primary'],row['truth']['primary'])<=2 and dist(chosen['delay'],row['truth']['secondary'])<=2;ep=dict(correct=int(correct),wrong=int(pos and declared and not correct),false=int(not pos and declared),positive=int(pos),null=int(not pos))
            for k,z in ep.items():totals[method][k]+=z
        out.append(dict(id=row['case']['id'],fusion=r,agreement=bool(agrees),samples=samples))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'records.json').write_text(json.dumps(out,indent=2));result=dict(status='CLOSED_DIAGNOSTIC_ONLY',totals=totals,protocol_sha256=d.sha(P),scope=v['scope']);(OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
