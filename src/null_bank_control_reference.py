"""Independent quadrature refinement for read-only control sensitivity checks."""
import numpy as np
from scipy.optimize import minimize
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from full_information_joint_reference_v2 import reported_locations


def refine_component(fit, centre, null, counts, geo, label):
    """Two stricter starts, independent GL32 objective, same parameter box."""
    k = len(counts)
    a, t, b = null['theta']
    scale = max(1., sum(counts))
    bounds = [(0, 20), (t - 4.5, t + 4.5), (1e-6, 30),
              (0, 20), (centre - 4.5, centre + 4.5)]
    initial = np.array(fit['theta'], float)
    initial[1] += k * round((t - initial[1]) / k)
    initial[4] += k * round((centre - initial[4]) / k)
    initial = np.clip(initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
    candidates = []

    def objective(theta):
        rate, design = rates_and_design(theta, k, 1.5)
        value, gradient = reference.full_factor(rate, design, geo, 32)
        return -value / scale, -gradient / scale

    for x in [initial, np.array([a, t, b, .02, centre])]:
        result = minimize(objective, x, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(ftol=1e-13, gtol=1e-9, maxiter=1000, maxls=60))
        theta = result.x.copy()
        primary, delay = reported_locations(theta, k)
        candidates.append(dict(theta=theta.tolist(), loglik=-float(result.fun) * scale,
                               primary=primary, delay=delay,
                               optimizer_success=bool(result.success)))
    best = max([fit] + candidates, key=lambda v: v['loglik'])
    return best, dict(label=label, improvement=best['loglik'] - fit['loglik'],
                      reference_optimizer_success=best['optimizer_success'])
