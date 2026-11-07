"""Numerical/physical checks for the random-recovery extension, not efficacy."""
from pathlib import Path
import sys, time, json
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import calibrated_recovery as r
import ct_dual_fast as hard


def main():
    template = r.Template(); theta = np.array([2.4, 15.2, .8])
    rate, design = template.rate(theta)
    checks = []
    for td, mu, sd, jitter in [(31.7, 80., 1.2, 0.), (32., 51.3, .95, 0.), (31.7, 80., 1.2, .5)]:
        observed = r.simulate(rate, 200, td, mu, sd, 712458, jitter)
        start = time.perf_counter()
        geometry = r.prepare(observed, 64, td, mu, sd, jitter)
        ll, gradient = r.evaluate(rate, design, geometry)
        epsilon = 1e-5; errors = []
        for j in range(3):
            plus = theta.copy(); minus = theta.copy(); plus[j] += epsilon; minus[j] -= epsilon
            numeric = (r.evaluate(*template.rate(plus), geometry)[0]-r.evaluate(*template.rate(minus), geometry)[0])/(2*epsilon)
            errors.append(float(abs(numeric-gradient[j])/max(1., abs(numeric))))
        assert max(errors)<2e-6, errors
        full = r.evaluate(rate, np.eye(64), geometry)[1]
        assert np.max(abs(full @ design-gradient))<1e-8
        if not jitter:
            # An independent integral of fixed-recovery likelihood densities.
            # Split at the gap so the hard-support discontinuity is explicit.
            normz = norm.cdf((3*td-mu)/sd)-norm.cdf((td-mu)/sd)
            for index in [0, min(3, len(observed)-2)]:
                phase = observed[index] % 64; gap = observed[index+1]-observed[index]
                def integrand(te):
                    geo = hard.prepare_rows([phase], [gap], 64, td, te)
                    value = hard.evaluate_rows(rate, design, geo)[0][0]
                    return np.exp(value)*norm.pdf((te-mu)/sd)/sd/normz
                expected = quad(integrand, max(td, mu-9*sd), min(gap, 3*td, mu+9*sd), epsabs=1e-11, epsrel=1e-7, limit=100)[0]
                pair = r.prepare(observed[index:index+2], 64, td, mu, sd, recovery_order=24)
                actual = np.exp(r.evaluate(rate, design, pair)[0])
                assert abs(actual-expected)/expected<3e-4, (actual, expected)
        fit_start = time.perf_counter()
        fit = r.fit_null(observed, template, geometry)
        fit_seconds = time.perf_counter()-fit_start
        checks.append(dict(td=td, mu=mu, sd=sd, jitter=jitter, events=len(observed),
            derivative_errors=errors, fit=fit, fit_seconds=fit_seconds,
            total_seconds=time.perf_counter()-start))
        print(json.dumps(checks[-1]), flush=True)
    out = ROOT/'results/calibrated_recovery_v1'; out.mkdir(exist_ok=True)
    (out/'kernel_checks.json').write_text(json.dumps(dict(status='PASS', checks=checks), indent=2))


if __name__=='__main__':main()
