"""Replay every known-rate two-dead-time diagnostic path and recount summaries."""
import json,time,math
import numpy as np
import safe_exposure_model_check as d
from safe_deadtime_exposure import simulate_two_dead_times,safe_lag,safe_statistics


def main():
    start=time.perf_counter();p=d.check();rows=json.loads((d.OUT/'records.json').read_text());a=json.loads((d.OUT/'analysis.json').read_text());c=json.loads((d.OUT/'completion.json').read_text());hashes={n:d.sha(d.OUT/n) for n in ['records.json','analysis.json','completion.json']};assert c['records_sha256']==hashes['records.json'] and c['protocol_sha256']==d.sha(d.P)
    cfg=p['settings'];expected=[(gi,j) for gi in range(4) for j in range(cfg['replicates_per_regime'])];assert [(r['regime'],r['index']) for r in rows]==expected
    for row in rows:assert d.case_data(row['regime'],row['index'],p)==row
    # Direct intersection of every active interval with individual phase bins,
    # independent of the endpoint-histogram implementation,in fixed8paths.
    differences=[]
    for gi,g in enumerate(p['regimes']):
        for j in [0,cfg['replicates_per_regime']-1]:
            obs,latent=simulate_two_dead_times(np.full(cfg['bins'],g['rate']),cfg['cycles'],g['td'],g['te'],cfg['seed_base']+1000003*gi+1009*j)
            s,e,info=safe_statistics(obs,cfg['cycles'],cfg['bins'],safe_lag(g['td'],g['te']));direct=np.zeros(cfg['bins'])
            for lo,hi in zip(info['open_starts'],info['open_ends']):
                k=np.arange(int(np.floor(lo)),int(np.ceil(hi)));v=np.minimum(k+1,hi)-np.maximum(k,lo);np.add.at(direct,k%cfg['bins'],v)
            np.testing.assert_allclose(e,direct,rtol=0,atol=1e-10);differences.append(float(np.abs(e-direct).max()))
    for gi,g in enumerate(p['regimes']):
        group=[r for r in rows if r['regime']==gi];saved=a['per_regime'][gi];assert saved['paths']==len(group)
        for method in ['safe','naive']:
            values=[math.exp(v[f'{method}_log_ratio']) for v in group];n=len(values);mean=math.fsum(values)/n;var=math.fsum((x-mean)**2 for x in values)/(n-1);rate=sum(v[f'{method}_count'] for v in group)/math.fsum(v[f'{method}_time'] for v in group)
            np.testing.assert_allclose(mean,saved['methods'][method]['mean_point_likelihood_ratio'],rtol=1e-14)
            np.testing.assert_allclose(math.sqrt(var/n),saved['methods'][method]['ratio_mean_monte_carlo_se'],rtol=1e-14)
            np.testing.assert_allclose(rate,saved['methods'][method]['pooled_rate_estimate'],rtol=1e-14)
    assert all(d.sha(d.OUT/n)==h for n,h in hashes.items())
    result=dict(status='PASS',all_paths_replayed=len(rows),all_latent_safe_state_checks_replayed=True,direct_phase_interval_checks=len(differences),maximum_direct_exposure_difference=max(differences),all_rate_ratio_summaries_recounted=True,outputs_sha256=hashes,seconds=time.perf_counter()-start,scope='Arithmetic and simulated two-detector state verification only. Known-point martingale ratios,not secondary-surface sensitivity,uniform composite-null performance,novelty or physical validation.')
    (d.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
