"""Active durations for a nonparalyzable, continuously timestamped detector.

Units are phase-bin widths. For a piecewise-constant periodic incident rate,
the path log likelihood is sum S*log(rate)-E*rate. E is a random duration;
this does NOT imply independent Poisson counts conditional on E.
Recorded timestamp quantization/jitter and uncertain recovery remain physical
model limitations. The likelihood and exposure principle are established.
"""
from __future__ import annotations
import numpy as np


def periodic_measure(starts, ends, bins):
    """Lebesgue measure of intervals in each periodic unit-width phase bin."""
    starts=np.asarray(starts,dtype=float);ends=np.asarray(ends,dtype=float)
    if starts.shape!=ends.shape or np.any(ends<starts):
        raise ValueError('Invalid intervals')
    def ramps(x):
        p=np.remainder(x,bins);j=np.floor(p).astype(np.int64)
        n=np.bincount(j,minlength=bins)
        return len(x)-np.cumsum(n)+np.bincount(j,weights=p-j,minlength=bins)
    cycles=np.sum(np.floor(ends/bins)-np.floor(starts/bins))
    return cycles+ramps(ends)-ramps(starts)


def sufficient_statistics(events, cycles, bins, dead_bins, previous=None):
    """Count/exposure of a whole-period block, given local continuous times.

    events in [0,cycles*bins); previous is the latest pre-block detection,
    when known. Absence of previous means the detector is active at t=0.
    A detection causes [t,t+dead_bins) blindness; unobserved arrivals do not
    extend it. The final event's truncated dead interval is included.
    """
    if int(cycles)!=cycles or cycles<=0 or int(bins)!=bins or bins<=0:
        raise ValueError('Positive integer cycles and bins required')
    if not np.isfinite(dead_bins) or dead_bins<0:
        raise ValueError('Nonnegative finite dead time required')
    bins=int(bins);events=np.asarray(events,dtype=float).reshape(-1)
    end=float(cycles*bins)
    if np.any(~np.isfinite(events)) or np.any(events<0) or np.any(events>=end):
        raise ValueError('Events must be inside the half-open block')
    history=events
    if previous is not None:
        if not np.isfinite(previous) or previous>=0:raise ValueError('Previous must precede block')
        history=np.r_[float(previous),events]
    if np.any(np.diff(history)<dead_bins-1e-7):
        raise ValueError('Observed hit inside reconstructed dead time')
    lo=np.maximum(history,0.);hi=np.minimum(history+dead_bins,end)
    keep=hi>lo;lo=lo[keep];hi=hi[keep]
    inactive=periodic_measure(lo,hi,bins)
    exposure=float(cycles)-inactive
    if np.any(exposure < -1e-7) or np.any(exposure>cycles+1e-7):
        raise ArithmeticError('Exposure out of range')
    exposure=np.clip(exposure,0,float(cycles))
    counts=np.bincount(np.floor(np.remainder(events,bins)).astype(np.int64),minlength=bins)
    if not np.isclose(exposure.sum(),end-np.sum(hi-lo),rtol=1e-12,atol=2e-7):
        raise ArithmeticError('Active-time conservation failure')
    return counts,exposure


def block_statistics(frames, phase_ns, period_ns, dead_ns, start, cycles, bins):
    """Keep raw timestamps including phase>=period (carry into next cycle).

    Absolute floats locate the block only. Local times are reconstructed with
    integer frame differences before multiplication to avoid cancellation.
    Input must be a chronologically ordered complete record including the
    last pre-block detection. No events are dropped at rounded period edges.
    """
    frames=np.asarray(frames,dtype=np.int64).reshape(-1)
    phase_ns=np.asarray(phase_ns,dtype=float).reshape(-1)
    if frames.shape!=phase_ns.shape or period_ns<=0:raise ValueError('Invalid raw record')
    periods=frames+phase_ns/period_ns
    if np.any(np.diff(periods)<0):raise ValueError('Raw events are not ordered')
    a,b=np.searchsorted(periods,[start,start+cycles],side='left')
    width=period_ns/bins
    local=(frames[a:b]-int(start))*bins+phase_ns[a:b]/width
    previous=None if a==0 else float((int(frames[a-1])-int(start))*bins+phase_ns[a-1]/width)
    counts,exposure=sufficient_statistics(local,cycles,bins,dead_ns/width,previous)
    meta=dict(events=int(b-a),previous_available=bool(a),previous_time_bins=previous,
              active_bins=float(exposure.sum()),dead_bins=float(cycles*bins-exposure.sum()),
              phase_bin_width_ns=float(width),first_record_index=int(a),last_record_index_exclusive=int(b),
              out_of_period_timestamps_preserved=int(np.sum(phase_ns[a:b]>=period_ns)))
    return counts,exposure,meta


def path_loglik(counts,exposure,rates):
    counts=np.asarray(counts);exposure=np.asarray(exposure);rates=np.asarray(rates)
    if counts.shape!=exposure.shape or counts.shape!=rates.shape or np.any(rates<0):
        raise ValueError('Invalid sufficient statistics or intensity')
    if np.any((rates==0)&(counts>0)):return -np.inf
    positive=counts>0
    return float(np.dot(counts[positive],np.log(rates[positive]))-np.dot(exposure,rates))


def simulate_events(rates,cycles,dead_bins,seed):
    """Exact integrated-hazard inversion for piecewise-constant periodic rate.

    No resetting at cycle edges. Initial state is active. Dead time may be
    fractional and span several periods. Output continuous recorded times.
    """
    rates=np.asarray(rates,dtype=float).reshape(-1);bins=len(rates)
    if np.any(~np.isfinite(rates)) or np.any(rates<=0) or dead_bins<0:
        raise ValueError('Strictly positive finite rates and nonnegative dead time required')
    prefix=np.r_[0.,np.cumsum(rates)];mass=prefix[-1]
    rng=np.random.default_rng(seed);available=0.;end=cycles*bins;out=[]
    while available<end:
        cycle=np.floor(available/bins);p=available-cycle*bins;j=min(int(p),bins-1)
        hazard=cycle*mass+prefix[j]+rates[j]*(p-j)+rng.exponential()
        cyc=np.floor(hazard/mass);v=hazard-cyc*mass
        j=min(int(np.searchsorted(prefix,v,side='right')-1),bins-1)
        t=cyc*bins+j+(v-prefix[j])/rates[j]
        if t>=end:break
        out.append(t);available=t+dead_bins
    return np.asarray(out)
