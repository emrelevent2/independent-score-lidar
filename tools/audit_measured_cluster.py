"""Replay new held-out mixture results and check component normalization."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.stats import ks_2samp
from measured_cluster_extension import ScaledLikelihood,burst_logdensity
from measurement_recovery_model import raw_pairs
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/measured_cluster_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=json.loads((ROOT/'MEASURED_CLUSTER_PROTOCOL_20261107.json').read_text());a=json.loads((OUT/'analysis.json').read_text());base=json.loads((ROOT/'results/measured_recovery_v1/analysis.json').read_text())
    assert sha(ROOT/'tools/measured_cluster_extension.py')==p['source_sha256'] and sha(ROOT/'MEASURED_CLUSTER_PROTOCOL_20261107.json')==a['protocol_sha256']
    rows=[]
    for row,b in zip(a['rows'],base['rows']):
        file=OUT/(Path(row['file']).stem+'.npz');assert sha(file)==row['data_sha256'];z=np.load(file)
        phase,gaps,period,meta=raw_pairs(ROOT/'data/highflux_raw/2019_01_28'/row['file']);lo,hi=row['evaluation_pair_range']
        assert np.array_equal(z['phase'],phase[lo:hi]) and np.array_equal(z['gaps'],gaps[lo:hi])
        assert lo>=p['calibration_pair_range'][1] and lo>=40200
        assert row['best']==min(row['starts'],key=lambda v:v['negative_mean_logdensity']) and all(v['success'] for v in row['starts'])
        par=b['parameters'];indices=np.arange(0,hi-lo,67);x=np.array(row['best']['x'])
        evaluator=ScaledLikelihood(phase[lo:hi][indices],gaps[lo:hi][indices],period,z['rate'],par['recovery_location_ns'],par['recovery_scale_ns'])
        assert np.array_equal(evaluator.mixture(x),z['logdensity'][indices])
        assert np.array_equal(evaluator.physical(1.),z['baseline_logdensity'][indices])
        gain=float(np.mean(z['logdensity']-z['baseline_logdensity']));assert gain==row['mean_log_score_gain']
        assert float(ks_2samp(z['gaps'],z['predicted']).statistic)==row['cdf_distance']
        assert float(ks_2samp(z['gaps'],z['baseline_predicted']).statistic)==row['baseline_cdf_distance']
        assert np.all(np.isfinite(z['logdensity'])) and row['all_finite'] and len(z['gaps'])==20000
        assert row['fixed_cutoff_zero_densities']==sum(z['gaps']<par['hard_recovery_ns'])
        tau=row['parameters']['cluster_exponential_mean_ns'];mu=par['recovery_location_ns'];sd=par['recovery_scale_ns']
        mass=quad(lambda s:float(np.exp(burst_logdensity(s,mu,sd,50.,tau))),50,100+tau*40,points=[mu,100.],epsabs=2e-8,limit=100)[0]
        assert abs(mass-1)<1e-7
        fine=ScaledLikelihood(phase[lo:hi][indices],gaps[lo:hi][indices],period,z['rate'],mu,sd,step=.0625)
        # Quantify integration sensitivity of the *contrast*, not just raw
        # densities; this matters when the fitted extra component is negligible.
        contrast=fine.mixture(x)-fine.physical(1.)
        coarse=z['logdensity'][indices]-z['baseline_logdensity'][indices]
        delta=float(np.mean(contrast-coarse))
        assert abs(delta)<.002
        rows.append(dict(file=row['file'],replayed= len(indices),heldout_gap_count=20000,mean_log_score_gain=gain,burst_component_total_mass=mass,mesh_refinement_mean_change_in_contrast=delta))
    result=dict(status='PASS',analysis_sha256=sha(OUT/'analysis.json'),protocol_sha256=sha(ROOT/'MEASURED_CLUSTER_PROTOCOL_20261107.json'),source_sha256=sha(__file__),rows=rows)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
