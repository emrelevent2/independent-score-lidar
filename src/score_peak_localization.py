"""Detection-independent continuous two-return localization after a score scan.

The null score and its calibration are untouched. Only reported locations
are refined using an explicitly fixed one-start likelihood optimization.
This is established detect-then-estimate methodology, not a new principle.
"""
import numpy as np
from scipy.optimize import minimize
import ct_dual_fast as model
from clock_pulse_likelihood import rates_and_design
from full_information_joint_reference_v2 import reported_locations


def localize(counts, exposure, sigma, null_fit, centre, geometry=None):
    k = len(counts)
    a, t, b = np.asarray(null_fit['theta'], float)
    centre = float(centre)
    scale = max(1., sum(counts))
    initial = np.array([a, t, b, .02, centre])
    nested = np.array([a, t, b, 0., centre])

    def objective(theta):
        rate, j = rates_and_design(theta, k, sigma)
        if geometry is None:
            value = counts @ np.log(rate) - exposure @ rate
            gradient = j.T @ (counts / rate - exposure)
        else:
            value, gradient = model.full_factor(rate, j, geometry)
        return -float(value) / scale, -gradient / scale

    bounds = [(0, 20), (t - 3 * sigma, t + 3 * sigma), (1e-6, 30),
              (0, 20), (centre - 3 * sigma, centre + 3 * sigma)]
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                      options=dict(ftol=1e-11, gtol=1e-7, maxiter=160, maxls=30))
    candidates = [(initial, -objective(initial)[0] * scale),
                  (nested, -objective(nested)[0] * scale),
                  (result.x, -float(result.fun) * scale)]
    theta, ll = max(candidates, key=lambda value: value[1])
    theta = theta.copy()
    theta[1] %= k
    theta[4] %= k
    primary, delay = reported_locations(theta, k)
    return dict(theta=theta.tolist(), loglik=float(ll), primary=primary, delay=delay,
                optimizer_success=bool(result.success), evaluations=int(result.nfev),
                start=centre, mode='FULL_LOCALIZATION' if geometry is not None else 'SAFE_LOCALIZATION')
