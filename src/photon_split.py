"""Split independent laser cycles, rather than treating bins as independent.

The observation model records at most one event per cycle. Adding the no-event
category and sampling multivariate hypergeometric counts is equivalent to a
uniform random partition of the original cycles. Marginal independence of the
two halves holds under the independent-cycle multinomial model, not conditional
on a fixed observed histogram. This distinction matters for validation claims.
"""
import numpy as np


def split_cycles(h,cycles,seed,left_cycles=None):
    h=np.asarray(h)
    if not np.all(h==np.floor(h)) or np.any(h<0) or np.any(h.sum(-1)>cycles):
        raise ValueError('Expected valid integer first-detection histograms')
    if left_cycles is None: left_cycles=cycles//2
    if not 0<left_cycles<cycles: raise ValueError('Empty partition')
    rng=np.random.default_rng(seed)
    left=np.empty_like(h,dtype=np.int64)
    for index in np.ndindex(h.shape[:-1]):
        histogram=h[index].astype(np.int64)
        categories=np.r_[histogram,cycles-int(histogram.sum())]
        counts=rng.multivariate_hypergeometric(categories,left_cycles)
        left[index]=counts[:-1]
    return left,h.astype(np.int64)-left,left_cycles,cycles-left_cycles
