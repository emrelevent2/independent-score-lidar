"""Independent quadrature, normalization and boundary checks for recovery mixture."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from measurement_recovery_model import GapLikelihood
import ct_dual_fast as kernel
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    period=100.;td=50.;k=64;grid=np.arange(k)
    rates=[np.full(k,.01),.009+.02*np.exp(-.5*((grid-12)/2.)**2)]
    comparisons=[]
    for rate in rates:
        for phase,gap in [(3.1,76.),(18.4,80.),(3.1,100.),(18.4,125.),(3.1,180.)]:
            g=GapLikelihood(np.array([phase]),np.array([gap]),period,td,.03125)
            actual=float(np.exp(g.logdensity(rate,80.,1.))[0])
            def density(te):
                geom=kernel.prepare_rows([phase*k/period],[gap*k/period],k,td*k/period,te*k/period)
                value=kernel.evaluate_rows(rate*period/k,np.zeros((k,1)),geom)[0][0]
                return float(np.exp(value)*k/period)*truncnorm.pdf(te,td-80,2*td-80,loc=80,scale=1)
            expected=quad(density,td,min(gap,2*td),points=[80.] if gap>80 else None,epsabs=1e-11,limit=120)[0]
            comparisons.append(dict(phase=phase,gap=gap,actual=actual,independent=expected,abs_difference=abs(actual-expected),relative_difference=abs(actual-expected)/expected))
    assert max(c['relative_difference'] for c in comparisons)<.006
    # With constant rate the mixture must integrate to one, including the tail.
    norm=quad(lambda s:np.exp(GapLikelihood(np.array([3.1]),np.array([s]),period,td,.125).logdensity(rates[0],80.,1.))[0],td,3000,points=[76,80,84,100,150],epsabs=2e-7,limit=150)[0]
    assert abs(norm-1)<2e-6
    result=dict(status='PASS',mixture_formula_comparisons=comparisons,constant_rate_total_mass=norm,model_source_sha256=sha(ROOT/'tools/measurement_recovery_model.py'),audit_source_sha256=sha(__file__),note='Independent reference averages the existing fixed-recovery path-density kernel over electronics recovery. Period/bin Jacobian applied explicitly.')
    out=ROOT/'results/measured_recovery_v1';out.mkdir(exist_ok=True);(out/'formula_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='mixture_formula_comparisons'},indent=2))


if __name__=='__main__':main()
