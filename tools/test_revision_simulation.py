"""Meaningful physical-reduction and implementation-parity tests."""
import numpy as np
from scipy.stats import ks_2samp
import revision_sensitivity as rev

def main():
    rate=np.full(64,.025);old=[];new=[]
    for i in range(40):
        a,_=rev.simulate_two_dead_times(rate,500,31.7,80.,914000+1009*i)
        b,latent=rev.with_afterpulses(rate,500,31.7,80.,491000+1009*i,0.)
        old.extend(np.diff(a));new.extend(np.diff(b));assert np.all(np.diff(latent)>=31.7-1e-9)
    ks=ks_2samp(old,new);assert ks.statistic<.035
    for i in range(20):
        obs,latent=rev.with_afterpulses(rate,500,31.7,80.,730000+1009*i,.05)
        again,again_latent=rev.with_afterpulses(rate,500,31.7,80.,730000+1009*i,.05)
        assert np.array_equal(obs,again) and np.array_equal(latent,again_latent)
        assert np.all(np.diff(latent)>=31.7-1e-9) and np.all(np.diff(obs)>=80-1e-9)
        # Independent electronics filter must reproduce every recorded event.
        filtered=[];next_active=0.
        for t in latent:
            if t>=next_active:filtered.append(t);next_active=t+80.
        assert np.array_equal(obs,np.array(filtered))
    p=rev.check();assert len(list(rev.cases(p,'power')))==480 and len(list(rev.cases(p,'robust')))==864
    ids=[c['index'] for k in ['power','robust'] for c in rev.cases(p,k)];assert len(set(ids))==len(ids)
    calibration_seeds=[p['bootstrap_seed_base']+1000003*i+1009*j for i in ids for j in range(199)]
    query_seeds=[p['seed_base']+1000003*(i+1) for i in ids]
    parameter_seeds=[p['seed_base']+1009*i for i in ids]
    assert len(set(calibration_seeds))==len(calibration_seeds)
    assert len(set(query_seeds+parameter_seeds))==len(query_seeds+parameter_seeds)
    assert set(query_seeds+parameter_seeds).isdisjoint(calibration_seeds)
    # Compare the optimized primary-only evaluator against the original full
    # four-way evaluator on independently generated nominal acquisitions.
    import null_bank_score
    largest=0.
    for c in list(rev.cases(p,'power'))[:2]:
        obs,_,_=rev.acquisition(c,p);g=p['regime'];s,e=rev.numeric_statistics(obs,g['cycles'],64,rev.outward_safe_lag(g['td'],g['te']))
        fit=rev.fit_null(s,e,1.5);geo=rev.kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te'])
        a,af,_=rev.primary_statistic(s,e,1.5,fit,geo,rev.d.bank(),2)
        b=null_bank_score.statistics(s,e,1.5,fit,geo,rev.d.bank(),2)
        assert a==b['samples']['full_bank_refit'];largest=max(largest,abs(af['loglik']-b['full_null']['loglik']))
    result=dict(status='PASS',zero_afterpulse_reduction=dict(old_gaps=len(old),new_gaps=len(new),KS_statistic=float(ks.statistic),limit=.035),afterpulse_physical_replay_cases=20,all_query_and_calibration_seeds_distinct=True,primary_evaluator_parity_cases=2,max_null_loglik_gap=largest,protocol_sha256=rev.sha(rev.P),test_source_sha256=rev.sha(__file__))
    (rev.ROOT/'results/revision_simulation_tests.json').write_text(rev.json.dumps(result,indent=2));print(rev.json.dumps(result,indent=2))

if __name__=='__main__':main()
