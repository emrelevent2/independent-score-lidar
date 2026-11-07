"""Recount measured coverage and independently check saved fitted predictions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from measurement_recovery_model import raw_pairs,GapLikelihood,Pulse
import ct_dual_fast as kernel
OUT=ROOT/'results/measured_recovery_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a=json.loads((OUT/'analysis.json').read_text());p=json.loads((ROOT/'MEASURED_RECOVERY_PROTOCOL_20261107.json').read_text())
    assert a['protocol_sha256']==sha(ROOT/'MEASURED_RECOVERY_PROTOCOL_20261107.json')
    for n,d in p['sources'].items():assert sha(ROOT/'tools'/n)==d
    for n,d in p['inputs'].items():assert sha(ROOT/n)==d
    summary=[]
    for row in a['rows']:
        phase,gaps,period,meta=raw_pairs(ROOT/'data/highflux_raw/2019_01_28'/row['file']);start,stop=row['evaluation_pair_range']
        z=np.load(OUT/(Path(row['file']).stem+'_heldout.npz'));assert np.array_equal(z['phase'],phase[start:stop]) and np.array_equal(z['gaps'],gaps[start:stop])
        assert meta==row['metadata'] and len(gaps)==row['metadata']['pairs']
        par=row['parameters'];assert row['random_zero_density_all']==np.count_nonzero(gaps<=par['detector_ns'])==0
        assert row['nominal80_zero_density_all']==np.count_nonzero(gaps<80)
        assert row['hard_zero_density_all']==np.count_nonzero(gaps<par['hard_recovery_ns'])
        assert all(r['success'] for r in row['starts']+row['hard_starts'])
        assert row['best']==min(row['starts'],key=lambda x:x['negative_mean_logdensity'])
        pulse=Pulse(ROOT/'data/highflux_raw/2019_01_28/laser_calib_100.mat',period,p['bins'])
        assert np.array_equal(pulse.rate(np.array(row['best']['x'])),z['rate'])
        selected=np.arange(0,len(z['gaps']),251)
        replay=GapLikelihood(z['phase'][selected],z['gaps'][selected],period,50.,p['integration_step_ns']).logdensity(z['rate'],par['recovery_location_ns'],par['recovery_scale_ns'])
        assert np.array_equal(replay,z['logdensity'][selected])
        assert np.isfinite(z['logdensity']).sum()==row['evaluation']['random_finite']==20000
        assert np.isfinite(z['hard_logdensity']).sum()==row['evaluation']['hard_finite']
        assert float(z['logdensity'].mean())==row['evaluation']['random_mean_logdensity']
        # Average the separate fixed-recovery kernel, not the implemented
        # recovery-mixture expression, on deterministic observed transitions.
        fixtures=[]
        order=np.argsort(z['gaps']);indices=[int(order[j]) for j in [0,10,100,1000,10000,19999]]
        for i in indices:
            ph=float(z['phase'][i]);gap=float(z['gaps'][i]);mu=par['recovery_location_ns'];sd=par['recovery_scale_ns'];k=p['bins']
            def integrand(te):
                geom=kernel.prepare_rows([ph*k/period],[gap*k/period],k,50*k/period,te*k/period)
                value=kernel.evaluate_rows(z['rate']*period/k,np.zeros((k,1)),geom)[0][0]
                return float(np.exp(value)*k/period)*truncnorm.pdf(te,(50-mu)/sd,(100-mu)/sd,loc=mu,scale=sd)
            upper=min(gap,100.);points=[v for v in [mu-6*sd,mu,mu+6*sd] if 50<v<upper]
            density=quad(integrand,50,upper,points=points,epsabs=1e-12,epsrel=2e-7,limit=180)[0]
            assert density>0
            difference=float(abs(np.log(density)-z['logdensity'][i]));assert difference<.015
            fixtures.append(dict(evaluation_index=i,independent_logdensity=float(np.log(density)),saved_logdensity=float(z['logdensity'][i]),abs_log_difference=difference))
        refinement=np.abs(z['logdensity']-z['refined_logdensity'])
        assert float(refinement.max())==row['evaluation']['refinement_max_abs_logdensity']
        summary.append(dict(file=row['file'],raw_gaps_recounted=len(gaps),heldout_records_checked=stop-start,deterministic_density_replays=len(selected),independent_mixture_quadratures=fixtures,refinement_max_abs_logdensity=float(refinement.max()),refinement_mean_abs_logdensity=float(refinement.mean())))
    result=dict(status='PASS',analysis_sha256=sha(OUT/'analysis.json'),source_sha256=sha(__file__),rows=summary)
    (OUT/'results_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
