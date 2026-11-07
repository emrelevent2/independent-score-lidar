"""Random-recovery transition score and a jitter-integrated pair criterion.

Random recovery is integrated inside each density, before taking logarithms.
The physical transition expansion is exact for R<=3*Td, up to quadrature.
With timestamp noise the adjacent-pair criterion averages both endpoint errors;
it is a working conditional criterion, not an exact noisy-record likelihood.
Calibration simulations retain the complete noisy path and shared endpoints.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import minimize
import ct_dual_fast as base
from deadtime_path_bootstrap import njit, simulate as latent_simulate
from deadtime_scan import pulse as gaussian_pulse


@lru_cache(maxsize=12)
def legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x+1)/2, w/2


@lru_cache(maxsize=8)
def hermite(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    return x*np.sqrt(2), w/np.sqrt(np.pi)


class Template:
    def __init__(self, bins=64, sigma=1.5, mass=None, raw_step=None):
        self.bins, self.sigma = bins, sigma
        self.mass = mass
        if mass is not None:
            mass = np.maximum(np.asarray(mass, float)-np.min(mass), 0)
            self.mass = mass/mass.sum()
            self.step = float(raw_step)
            centre = np.dot(np.arange(len(mass))+.5, self.mass)*self.step
            self.edges = np.arange(len(mass)+1)*self.step-centre
            self.cdf = np.r_[0., np.cumsum(self.mass)]

    def pulse(self, tau):
        if self.mass is None:
            return gaussian_pulse(tau, self.bins, self.sigma)
        x = np.arange(self.bins+1)-tau % self.bins
        f = np.zeros(self.bins); derivative = np.zeros(self.bins)
        for offset in (-2, -1, 0, 1, 2):
            xx = x+offset*self.bins
            f += np.diff(np.interp(xx, self.edges, self.cdf, left=0., right=1.))
            index = np.floor((xx-self.edges[0])/self.step).astype(int)
            density = np.zeros_like(xx)
            ok = (index >= 0) & (index < len(self.mass))
            density[ok] = self.mass[index[ok]]/self.step
            derivative += density[:-1]-density[1:]
        return f, derivative

    def rate(self, theta):
        a, t, b = theta[:3]; f, df = self.pulse(t)
        rate = a*f+b/self.bins
        design = [f, a*df, np.ones(self.bins)/self.bins]
        if len(theta) == 5:
            c, d = theta[3:]; h, dh = self.pulse(d)
            rate = rate+c*h; design.extend([h, c*dh])
        return rate, np.column_stack(design)

    def bank(self, subdivision=4):
        grid = np.arange(self.bins*subdivision)/subdivision
        return grid, np.array([self.pulse(t)[0] for t in grid])


def recovery_nodes(gaps, td, mu, sd, order):
    z, w = legendre(order)
    lower = ndtr((td-mu)/sd); upper = ndtr((3*td-mu)/sd)
    if upper-lower <= 1e-14:
        raise ValueError('Recovery normal has negligible mass in [Td,3Td]')
    top = ndtr((np.minimum(gaps, 3*td)-mu)/sd)
    mass = np.maximum(0., top-lower)/(upper-lower)
    probability = lower+(top-lower)[:, None]*z[None, :]
    values = mu+sd*ndtri(np.clip(probability, 1e-300, 1-1e-16))
    return np.clip(values, td, 3*td), mass[:, None]*w[None, :]


def endpoint_quadrature(phase, sigma, order=1):
    """Split Gaussian integration at every piecewise-constant event-rate edge."""
    z, w = legendre(order)
    phases = np.asarray(phase)
    segments = int(np.ceil(14*sigma))+2
    errors = np.zeros((segments*order, len(phases)))
    weights = np.zeros_like(errors)
    for i, p in enumerate(phases):
        lo, hi = -7*sigma, 7*sigma
        breaks = [lo, hi]
        for integer in range(math.floor(p-hi), math.ceil(p-lo)+1):
            edge = p-integer
            if lo < edge < hi: breaks.append(edge)
        breaks = sorted(breaks)
        for j, (left, right) in enumerate(zip(breaks[:-1], breaks[1:])):
            xx = left+(right-left)*z
            errors[j*order:(j+1)*order, i] = xx
            weights[j*order:(j+1)*order, i] = (right-left)*w*np.exp(-.5*(xx/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        weights[:,i] /= weights[:,i].sum()
    return errors, weights


def prepare(observed, bins, td, mu, sd, jitter=0., recovery_order=8, jitter_order=3, endpoint_order=3):
    observed = np.asarray(observed, float)
    if len(observed)<2 or np.any(np.diff(observed)<=0):
        raise ValueError('At least two chronological timestamps required')
    phase = observed[:-1] % bins; gaps = np.diff(observed)
    if jitter:
        errors, weights = hermite(jitter_order)
        ends, end_weights = endpoint_quadrature(observed[1:] % bins, jitter, endpoint_order)
        shifts = [(jitter*a, b, wa*wb)
                  for a, wa in zip(errors, weights) for b, wb in zip(ends,end_weights)]
    else:
        shifts = [(0., 0., 1.)]
    phases = np.concatenate([(phase-a) % bins for a, b, w in shifts])
    lengths = np.concatenate([gaps+a-b for a, b, w in shifts])
    rw, ww = recovery_nodes(lengths, td, mu, sd, recovery_order)
    component_weights = np.array([np.broadcast_to(s[2],len(gaps)) for s in shifts])
    ok = (lengths >= td) & (component_weights.reshape(-1)>0)
    safe_lengths = np.maximum(lengths, td)
    maximum = min(3*td, max(td+1e-8, float(rw[ww>0].max()) if np.any(ww>0) else mu))
    geometry = base.prepare_rows(phases, safe_lengths, bins, td, maximum)
    geometry.update(recoveries=rw, recovery_weights=ww,
                    component_weights=component_weights,
                    usable=ok, pairs=len(gaps), components=len(shifts))
    return geometry


@njit(cache=True)
def _soft_rows(rate, design, prefix, gp, phases, gaps, events, all_edges,
               eo, co, indices, fw, bw, comp, td, rec, weights, usable, nodes, qw):
    n = len(phases); dim = design.shape[1]
    values = np.full(n, -np.inf); scores = np.zeros((n, dim))
    for r in range(n):
        if not usable[r]:
            continue
        start, end = co[r], co[r+1]
        edges = all_edges[eo[r]:eo[r+1]]
        p = base._profile(rate, design, prefix, gp, edges, indices[start:end],
                          fw[start:end], bw[start:end], td, nodes, qw)
        factor = 0.; gradient = np.zeros(dim)
        for j in range(rec.shape[1]):
            if weights[r, j] <= 0:
                continue
            v, g = base._F(min(rec[r, j], gaps[r]-td), edges, p, nodes, qw, td)
            factor += weights[r, j]*v
            gradient += weights[r, j]*g
        if factor <= 0:
            continue
        idx = events[r]
        values[r] = math.log(rate[idx])-base._iv(rate, prefix, comp[r])+math.log(factor)
        scores[r] = design[idx]/rate[idx]+gradient/factor
        for j in range(dim):
            scores[r, j] -= base._ivg(design, gp, comp[r], j)
    return values, scores


def evaluate(rate, design, geometry, return_pieces=False):
    rate = np.asarray(rate, float); design = np.asarray(design, float); g = geometry
    prefix = np.r_[0., np.cumsum(rate)]
    gp = np.vstack([np.zeros(design.shape[1]), np.cumsum(design, axis=0)])
    nodes, weights = base.quadrature(8)
    ll, grad = _soft_rows(rate, design, prefix, gp, g['phases'], g['gaps'],
        g['event_bins'], g['edges'], g['edge_offsets'], g['cell_offsets'],
        g['indices'], g['forward'], g['backward'], g['compensator'], g['td'],
        g['recoveries'], g['recovery_weights'], g['usable'], nodes, weights)
    n = g['pairs']; m = g['components']
    ll = ll.reshape(m, n)
    gradient = grad.reshape(m, n, design.shape[1])
    maximum = np.max(ll, axis=0)
    if not np.all(np.isfinite(maximum)):
        return (-np.inf, np.zeros(design.shape[1]))
    mass = np.exp(ll-maximum)*g['component_weights']
    total = mass.sum(axis=0)
    pieces = maximum+np.log(total)
    derivatives = np.einsum('mn,mnj->nj', mass/total, gradient)
    result = float(pieces.sum()), derivatives.sum(axis=0)
    return result+(pieces, derivatives) if return_pieces else result


def fit_null(observed, template, geometry, starts=1):
    k = template.bins
    counts = np.bincount(np.floor(np.asarray(observed) % k).astype(int), minlength=k)
    _, candidates = template.bank(1)
    ranking = candidates @ counts
    centres = []
    for index in np.argsort(ranking)[::-1]:
        if all(abs((index-v+k/2) % k-k/2) > max(3., 3*template.sigma) for v in centres):
            centres.append(float(index))
        if len(centres) == starts:
            break
    scale = max(1., geometry['pairs'])
    flux = max(.05, k/max(np.diff(observed).mean()-geometry['td'], 1.))
    fits = []
    for centre in centres:
        initial = np.array([min(10., .8*flux), centre, min(10., .2*flux)])
        def objective(x):
            rate, design = template.rate(x)
            ll, grad = evaluate(rate, design, geometry)
            return -ll/scale, -grad/scale
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
            bounds=[(1e-5, 20.), (centre-4., centre+4.), (1e-5, 30.)],
            options=dict(ftol=1e-10, gtol=1e-6, maxiter=80, maxls=25))
        fits.append(dict(theta=result.x.tolist(), loglik=-float(result.fun)*scale,
                         success=bool(result.success), evaluations=int(result.nfev)))
    return max(fits, key=lambda r:r['loglik'])


def scan(template, theta, geometry, information):
    rate, design = template.rate(theta)
    ll, gradient = evaluate(rate, np.eye(template.bins), geometry)
    grid, candidates = template.bank()
    h = information; cross = candidates @ h @ design
    inverse = np.linalg.pinv(design.T @ h @ design, rcond=1e-10)
    raw = candidates @ gradient-cross @ inverse @ (design.T @ gradient)
    variance = np.einsum('mi,ij,mj->m', candidates, h, candidates)-np.einsum('mi,ij,mj->m', cross, inverse, cross)
    distance = np.abs((grid-theta[1]+template.bins/2) % template.bins-template.bins/2)
    allowed = (distance>=max(3., 2.5*template.sigma)) & (variance>1e-9)
    scores = np.where(allowed, np.maximum(raw, 0.)**2/np.maximum(variance, 1e-9), -np.inf)
    best = int(np.argmax(scores))
    return dict(score=float(scores[best]), primary=float(theta[1] % template.bins),
                delay=float(grid[best]), loglik=float(ll))


def localize(template, theta, centre, geometry):
    a, t, b = theta; scale = max(1., geometry['pairs'])
    start = np.array([a, t, b, .02, centre])
    def objective(x):
        rate, design = template.rate(x)
        ll, grad = evaluate(rate, design, geometry)
        return -ll/scale, -grad/scale
    result = minimize(objective, start, jac=True, method='L-BFGS-B',
        bounds=[(1e-5, 20.), (t-4., t+4.), (1e-5, 30.), (0., 20.), (centre-4., centre+4.)],
        options=dict(ftol=1e-10, gtol=1e-6, maxiter=100, maxls=25))
    choices = [start, np.asarray(result.x), np.array([a, t, b, 0., centre])]
    x = min(choices, key=lambda x:objective(x)[0])
    p, d = x[1] % template.bins, x[4] % template.bins
    if x[3] > x[0]: p, d = d, p
    return dict(primary=float(p), delay=float(d), theta=x.tolist(), success=bool(result.success))


@njit(cache=True)
def _readout(latent, recoveries):
    out = np.empty(len(latent)); n = 0; available = 0.
    for t in latent:
        if t >= available:
            out[n] = t; available = t+recoveries[n]; n += 1
    return out[:n]


def simulate(rate, cycles, td, mu, sd, seed, jitter=0., random_recovery=True, afterpulse=0.):
    if afterpulse:
        raise NotImplementedError('Afterpulse paths require an explicit branching simulator')
    latent = latent_simulate(rate, cycles, td, seed)
    rng = np.random.default_rng(seed+104729)
    if random_recovery and sd > 0:
        lo, hi = ndtr((td-mu)/sd), ndtr((3*td-mu)/sd)
        recoveries = mu+sd*ndtri(lo+(hi-lo)*rng.uniform(size=len(latent)))
    else:
        recoveries = np.full(len(latent), mu)
    observed = _readout(latent, recoveries)
    if jitter:
        observed = np.sort(observed+rng.normal(0., jitter, len(observed)))
    return observed
