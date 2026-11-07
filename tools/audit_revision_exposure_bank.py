import json,sqlite3
import numpy as np
from scipy.linalg import qr
import revision_exposure_bank as ex
from ct_full_score import templates

def main():
    ex.check();base=ex.ab.study.read()
    with sqlite3.connect(f'file:{(ex.OUT/"records.sqlite3").as_posix()}?mode=ro',uri=True) as db:rows=[json.loads(v[0]) for v in db.execute('SELECT payload FROM records ORDER BY rowid')]
    assert len(rows)==1280;largest=0.;checks=identities=0
    for branch in ['original','common']:
        rr=[r for r in rows if r['branch']==branch];assert len(rr)==640
        for a,b in zip(base,rr):
            assert a['case']==b['case'] and a['data_sha256']==b['data_sha256'];z=b['sample']
            assert len(z['bootstrap_scores'])==199 and z['pvalue']==(1+sum(v>=z['score'] for v in z['bootstrap_scores']))/200
            if a['case']['regime']<2:
                assert z==a['samples']['localized_safe_score_'+branch];identities+=1;continue
            assert b['bootstrap_path_sha256']==a['bootstrap_path_sha256'][branch]
            if not a['case']['id'].endswith('_000'):continue
            p=ex.ab.study.check();g=p['regimes'][a['case']['regime']]
            obs,_,_,_,_,_=ex.ab.study.d.previous.old.base.data(a['case'],p)
            s,e=ex.ab.numeric_statistics(obs,g['cycles'],64,ex.ab.outward_safe_lag(g['td'],g['te']))
            theta=a['safe_null']['theta'];rate,j=ex.ab.rates_and_design(theta,64,1.5);h,_=ex.bank().lookup(theta,a['case']['regime'])
            roots=np.sqrt(np.diag(h));q,_=qr(roots[:,None]*j,mode='economic');grid,t=templates(64,1.5)
            res=t*roots-(t*roots)@q@q.T;raw=res@((s/rate-e)/roots);v=np.sum(res**2,axis=1)
            good=(np.abs((grid-theta[1]+32)%64-32)>=3.75)&(v>1e-8);score=np.max(np.where(good,np.maximum(raw,0)**2/np.maximum(v,1e-8),-np.inf))
            gap=abs(score-z['score']);largest=max(largest,gap);assert gap<1e-8;checks+=1
    result=dict(status='PASS',rows=1280,rank_checks=1280,no_hidden_exact_identities=identities,independent_QR_score_checks=checks,max_QR_score_gap=float(largest),records_sha256=ex.ab.sha(ex.OUT/'records.sqlite3'),validation_sha256=ex.ab.sha(ex.OUT/'validation.json'),protocol_sha256=ex.ab.sha(ex.P),auditor_sha256=ex.ab.sha(__file__))
    (ex.OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
