"""Complete the terminal no-record interval in measured-path counterfactuals.

The earlier study is retained unchanged. This addendum changes only latent
completion after the last source record, conditioning it on no further record.
Source blocks, injection strengths, estimators and thresholds stay fixed.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import semisynthetic_recovery_study as original
import calibrated_recovery_study as study
from measurement_recovery_model import instantaneous
OUT=ROOT/'results/semisynthetic_recovery_v2'
P=ROOT/'SEMISYNTHETIC_TERMINAL_PROTOCOL_20261107.json'
BASE_COMPLETE=original.complete
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check():
    p=json.loads(P.read_text())
    assert sha(__file__)==p['terminal_source_sha256']
    assert sha(ROOT/'tools/semisynthetic_recovery_study.py')==p['source_sha256']
    assert sha(study.P)==p['parent_protocol_sha256']
    return p,study.check()


def complete(observed,rate,cfg,sensor,seed):
    incident,marks,diagnostic=BASE_COMPLETE(observed,rate,cfg,sensor,seed)
    rng=np.random.default_rng(seed+32452843)
    last=float(observed[-1]);end=cfg['cycles']*64.;length=end-last
    d=cfg['td'];mu=sensor['mu'];sd=sensor['sd'];lo=ndtr((d-mu)/sd);hi=ndtr((2*d-mu)/sd)
    def recovery(n):return mu+sd*ndtri(np.clip(lo+(hi-lo)*rng.uniform(size=n),1e-300,1-1e-16))
    for attempt in range(1000000):
        anchor_recovery=recovery(1)[0]
        candidates=last+length*rng.uniform(size=rng.poisson(rate.max()*length))
        candidates=candidates[rng.uniform(size=len(candidates))<instantaneous(rate,candidates,64)/rate.max()]
        candidates=np.sort(candidates);tail_marks=recovery(len(candidates))
        tail_events=np.r_[last,candidates];trial_marks=np.r_[anchor_recovery,tail_marks]
        if len(original.filter_incident(tail_events,trial_marks,d))==1:break
    else:raise RuntimeError('Terminal conditional sampler exhausted')
    prefix=incident<=last
    new_events=np.r_[incident[prefix],candidates];new_marks=np.r_[marks[prefix],tail_marks]
    new_marks[np.flatnonzero(new_events==last)[0]]=anchor_recovery
    assert np.array_equal(original.filter_incident(new_events,new_marks,d),observed)
    diagnostic.update(terminal_completion=True,terminal_rejection_attempts=attempt+1,
        terminal_incident_photons=len(candidates),incident_photons=len(new_events),
        completed_path_sha256=hashlib.sha256(np.c_[new_events,new_marks].tobytes()).hexdigest())
    return new_events,new_marks,diagnostic


def freeze():
    assert not P.exists()
    p=json.loads((ROOT/'SEMISYNTHETIC_RECOVERY_PROTOCOL_20261107.json').read_text())
    p.update(status='FIXED_BEFORE_TERMINAL_COMPLETE_COUNTERFACTUAL_RESULTS',
        terminal_source_sha256=sha(__file__),earlier_protocol_sha256=sha(ROOT/'SEMISYNTHETIC_RECOVERY_PROTOCOL_20261107.json'),
        amendment='Review of latent boundary handling identified that the earlier completion chose no unrecorded avalanche after the last source event. Here the entire terminal incident path and recovery mark are sampled by rejection conditional on no additional record before acquisition end. The earlier192 outcomes are retained. Blocks, ratios, target seeds and inference are unchanged. This is a physical boundary correction, not outcome-based tuning.')
    P.write_text(json.dumps(p,indent=2));print('Terminal completion addendum frozen',sha(P),flush=True)


# These assignments are also executed in spawn-created worker interpreters.
original.OUT=OUT;original.P=P;original.check=check;original.complete=complete


if __name__=='__main__':
    action=sys.argv[1]
    if action=='freeze':freeze()
    elif action=='run':original.run()
    elif action=='analyze':original.analyze()
