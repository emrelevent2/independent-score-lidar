"""Independent algebra/record checks for the two censored-bank ablations."""
import sys,json,sqlite3,time
from pathlib import Path
import numpy as np
import revision_censored_bank as ab
import revision_censored_bank_common as common
from scipy.linalg import qr
from ct_full_score import templates
from deadtime_path_bootstrap import fixed_fit_scores

def read(path):
    with sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True) as db:return [json.loads(v[0]) for v in db.execute('SELECT payload FROM records ORDER BY rowid')]

def main():
    start=time.perf_counter();ab.check();common.check();base=ab.study.read();a=read(ab.OUT/'records.sqlite3');b=read(common.OUT/'records.sqlite3')
    assert len(base)==len(a)==len(b)==640
    identities=rankchecks=0;largest=0.
    for row,aa,bb in zip(base,a,b):
        assert row['case']==aa['case']==bb['case'] and row['data_sha256']==aa['data_sha256']==bb['data_sha256']
        for r in [aa,bb]:
            z=r['sample'];assert len(z['bootstrap_scores'])==199 and np.all(np.isfinite(z['bootstrap_scores']))
            assert z['pvalue']==(1+sum(v>=z['score'] for v in z['bootstrap_scores']))/200;rankchecks+=1
        if row['case']['regime']<2:
            assert aa['sample']==bb['sample']==row['samples']['localized_safe_score_original'];identities+=1;continue
        assert aa['bootstrap_path_sha256']==row['bootstrap_path_sha256']['original']
        assert bb['bootstrap_path_sha256']==row['bootstrap_path_sha256']['common']
        assert aa['sample']['score']==bb['sample']['score'] and aa['localization']==bb['localization']
        if row['case']['id'].endswith('_000'):
            g=ab.study.check()['regimes'][row['case']['regime']]
            obs,_,_,_,_,_=ab.study.d.previous.old.base.data(row['case'],ab.study.check())
            s,e=ab.numeric_statistics(obs,g['cycles'],64,ab.outward_safe_lag(g['td'],g['te']))
            theta=row['safe_null']['theta'];rate,j=ab.rates_and_design(theta,64,1.5);h,_=ab.bank().lookup(theta,row['case']['regime']);w=np.diag(h)
            grid,t=templates(64,1.5);roots=np.sqrt(w)
            q,_=qr(roots[:,None]*j,mode='economic');residual=t*roots-(t*roots)@q@q.T
            u=s/rate-e;raw=residual@(u/roots);variance=np.sum(residual**2,axis=1)
            allowed=(np.abs((grid-theta[1]+32)%64-32)>=3.75)&(variance>1e-8)
            scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(variance,1e-8),-np.inf)
            score=float(np.max(scores));gap=abs(score-aa['sample']['score']);largest=max(largest,gap);assert gap<1e-8
    # Independent special-case check: replacing bank weights by observed weights
    # reproduces the original FFT statistic using explicit template projection.
    rng=np.random.default_rng(510739)
    for i in range(20):
        theta=np.array([rng.uniform(.2,3),rng.uniform(0,64),rng.uniform(.1,2)])
        rate,j=ab.rates_and_design(theta,64,1.5);e=rng.uniform(100,500,64);s=rng.poisson(rate*e)
        grid,t=templates(64,1.5);w=e/rate;roots=np.sqrt(w);q,_=qr(roots[:,None]*j,mode='economic')
        residual=t*roots-(t*roots)@q@q.T;raw=residual@((s/rate-e)/roots);v=np.sum(residual**2,axis=1)
        allowed=(np.abs((grid-theta[1]+32)%64-32)>=3.75)&(v>1e-8)
        scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(v,1e-8),-np.inf)
        old,_=fixed_fit_scores(s,e,theta,1.5);assert abs(float(old[0])-float(np.max(scores)))<1e-8
    bp=ab.oldbank.check(); errors=[];paths=0
    for case in ab.oldbank.validation_cases(bp):
        g=bp['regimes'][case['ri']];rate,j=ab.rates_and_design(case['theta'],64,1.5);summed=np.zeros(64)
        import hashlib
        digest=hashlib.sha256()
        for draw in range(bp['validation_records_per_profile']):
            obs,_=ab.simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],case['seed']+1009*draw);digest.update(obs.tobytes())
            _,exposure=ab.numeric_statistics(obs,g['cycles'],64,ab.outward_safe_lag(g['td'],g['te']));summed+=exposure/rate;paths+=1
        with np.load(ab.oldbank.OUT/'validation'/(str(case['index'])+'.npz'),allow_pickle=False) as saved:
            assert digest.hexdigest()==json.loads(str(saved['metadata']))['observed_arrays_sha256']
        h=np.diag(summed/bp['validation_records_per_profile']);hb,_=ab.bank().lookup(case['theta'],g['index']);grid,t=templates(64,1.5)
        def variance(matrix):
            roots=np.sqrt(np.diag(matrix));q,_=qr(roots[:,None]*j,mode='economic');r=t*roots-(t*roots)@q@q.T;return np.sum(r*r,axis=1)
        v,vb=variance(h),variance(hb);allowed=np.abs((grid-case['theta'][1]+32)%64-32)>=3.75
        assert np.all(v[allowed]>0) and np.all(vb[allowed]>0);errors.extend(np.abs(np.log(vb[allowed]/v[allowed])).tolist())
    validation=dict(heldout_records=paths,profiles=32,directions=len(errors),absolute_log_variance_ratio_quantiles=dict(zip(['median','p95','max'],np.quantile(errors,[.5,.95,1]).tolist())),interpretation='Held-out descriptive approximation check. No newly invented acceptance gate.')
    result=dict(status='PASS',query_pairs=640,rank_checks=rankchecks,no_hidden_exact_identities=identities,training_cell_path_hash_checks=json.loads((ab.OUT/'training.json').read_text())['path_hash_checks'],independent_QR_projected_scores=14,max_score_gap=largest,FFT_original_equivalence_fixtures=20,heldout_validation=validation,source_sha256=ab.sha(__file__),original_bank_records_sha256=ab.sha(ab.OUT/'records.sqlite3'),common_bank_records_sha256=ab.sha(common.OUT/'records.sqlite3'),seconds=time.perf_counter()-start)
    (ab.ROOT/'results/revision_censored_bank_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
