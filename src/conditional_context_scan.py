"""Prototype context-weighted maximum on a complete secondary score field.

Classical prior weighting/weighted scanning, not a new general test theorem.
Independent donor summaries are held fixed, but the transported centre and
the exclusion set depend on the query null fit and must be recomputed in
EVERY complete-path calibration draw. This is not an e-value or a posterior.
No efficacy study has selected or validated this prototype.
"""
import numpy as np
from deadtime_scan import pulse
from deadtime_path_bootstrap import fine_kernels, fine_correlations


def censored_field(counts, exposure, theta, sigma):
    """Expose exactly the quarter-grid field used by the existing safe scan.

    The FFT kernel, nuisance inverse cutoff and exclusion rule are identical
    to fixed_fit_scores(duration). Kept separate from frozen scientific code.
    """
    s, e = np.asarray(counts, float), np.asarray(exposure, float)
    theta = np.asarray(theta, float)
    if s.ndim != 1 or e.shape != s.shape or theta.shape != (3,):
        raise ValueError('One count/exposure vector and three null parameters required')
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(e)) or np.any(s < 0) or np.any(e < 0):
        raise ValueError('Finite nonnegative counts and exposures required')
    if sigma <= 0 or not np.isfinite(sigma) or not np.all(np.isfinite(theta)):
        raise ValueError('Finite parameters and positive width required')
    k = len(s)
    a, t, b = theta
    f, d = pulse(t, k, sigma)
    rate = np.maximum(a * f + b / k, 1e-14)
    u, w = s / rate - e, e / rate
    j = np.column_stack([f, a*d, np.ones(k)/k])
    ff, ff2 = fine_kernels(k, sigma, 4)
    raw = fine_correlations(u, ff)[0]
    variance = fine_correlations(w, ff2)[0]
    cross = np.stack([fine_correlations(w*j[:,i], ff)[0] for i in range(3)], axis=1)
    inverse = np.linalg.pinv(j.T @ (w[:,None] * j), rcond=1e-12)
    raw = raw - cross @ inverse @ (j.T @ u)
    variance = variance - np.einsum('mi,ij,mj->m', cross, inverse, cross)
    grid = np.arange(4*k)/4
    allowed = (np.abs((grid-t+k/2)%k-k/2) >= 2.5*sigma) & (variance > 1e-8)
    scores = np.where(allowed, np.maximum(raw,0)**2/np.maximum(variance,1e-8), -np.inf)
    return dict(grid=grid, scores=scores, primary=float(t), variance=variance)


def _circular_gaussian(grid, centre, bins, width):
    distance = np.abs((grid-centre+bins/2)%bins-bins/2)
    # Stable normalization even if the centre lies in an excluded region.
    log_mass = -.5*(distance/width)**2
    mass = np.exp(log_mass - np.max(log_mass))
    return mass / mass.sum()


def weighted_scan(field, context, sigma, bins):
    """Fixed 1/2 global +1/4 absolute +1/4 transported prior weighting.

    Donor two-pulse score must exceed20, the existing context-support cutoff.
    Otherwise the original max and centre are returned exactly. Supported
    weights are normalized on ALL admissible quarter-grid centres. The
    statistic is max(score +2log(weight/uniform)); it may be negative.
    The reported centre initializes the same observed-record localization
    opportunity for candidate and matched controls in any future study.
    """
    grid, scores = np.asarray(field['grid'],float), np.asarray(field['scores'],float)
    if grid.ndim != 1 or scores.shape != grid.shape or not len(grid):
        raise ValueError('Equal nonempty one-dimensional grid and score fields required')
    if bins <= 0 or sigma <= 0 or not np.isfinite(bins+sigma):
        raise ValueError('Finite positive period and width required')
    if not np.all(np.isfinite(grid)) or np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError('Finite grid and finite or negative-infinite scores required')
    values = [float(context[name]) for name in ['primary','secondary','two_pulse_loglr']]
    primary = float(field['primary'])
    if not np.all(np.isfinite(values + [primary])):
        raise ValueError('Finite context and query primary required')
    allowed = np.isfinite(scores)
    if not np.any(allowed):
        raise ValueError('At least one admissible score required')
    n = int(allowed.sum())
    supported = values[2] > 20.
    weights = np.zeros_like(scores)
    weights[allowed] = 1./n
    weighted = scores.copy()
    transported = (primary + values[1] - values[0]) % bins
    if supported:
        absolute = _circular_gaussian(grid[allowed], values[1], bins, sigma)
        relative = _circular_gaussian(grid[allowed], transported, bins, sigma)
        weights[allowed] = .5/n + .25*absolute + .25*relative
        weighted[allowed] = scores[allowed] + 2*np.log(weights[allowed]*n)
    best = int(np.argmax(weighted))
    return dict(score=float(weighted[best]), delay=float(grid[best]), primary=primary,
                supported=bool(supported), transported_centre=float(transported),
                weights=weights, scores=weighted, grid=grid.copy(), allowed=allowed)
