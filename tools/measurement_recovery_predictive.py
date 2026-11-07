"""Direct physical simulation of conditional next-gap predictions after calibration."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import ks_2samp
from measurement_recovery_model import raw_pairs
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/measured_recovery_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def predict(rate,phases,period,d,mu,sd,seed,n=100000):
    rng=np.random.default_rng(seed);phase=rng.choice(phases,n,replace=True)
    lower=ndtr((d-mu)/sd);upper=ndtr((2*d-mu)/sd)
    recovery=mu+sd*ndtri(lower+(upper-lower)*rng.uniform(size=n))
    width=period/len(rate);prefix=np.r_[0.,np.cumsum(rate)*width];mass=prefix[-1]
    def arrival(start):
        cycle=np.floor(start/period);local=start-cycle*period;index=np.minimum((local/width).astype(int),len(rate)-1)
        z=cycle*mass+prefix[index]+rate[index]*(local-index*width)+rng.exponential(size=len(start))
        c=np.floor(z/mass);rem=z-c*mass;j=np.minimum(np.searchsorted(prefix,rem,side='right')-1,len(rate)-1)
        return c*period+j*width+(rem-prefix[j])/rate[j]
    first=arrival(phase+d);hidden=first-phase<recovery
    first[hidden]=arrival(first[hidden]+d)
    gap=first-phase
    assert np.all(gap>=recovery-1e-9) and np.all(gap>=d-1e-9)
    return gap


def main():
    analysis=json.loads((OUT/'analysis.json').read_text());rows=[]
    for i,row in enumerate(analysis['rows']):
        data=np.load(OUT/(Path(row['file']).stem+'_heldout.npz'));p=row['parameters']
        predicted=predict(data['rate'],data['phase'],row['period_ns'],p['detector_ns'],p['recovery_location_ns'],p['recovery_scale_ns'],8607199401+i)
        target=OUT/(Path(row['file']).stem+'_predictive.npz');np.savez_compressed(target,gaps=predicted)
        rows.append(dict(file=row['file'],conditional_predictions=len(predicted),evaluation_gaps=len(data['gaps']),descriptive_marginal_cdf_distance=float(ks_2samp(data['gaps'],predicted).statistic),predictions_sha256=sha(target)))
    result=dict(status='COMPLETE',analysis_sha256=sha(OUT/'analysis.json'),source_sha256=sha(__file__),rows=rows,interpretation='Predictions condition on empirical preceding phases. CDF distances are descriptive model checks, not a two-surface accuracy measure or a calibrated goodness-of-fit p-value.')
    (OUT/'predictive.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
