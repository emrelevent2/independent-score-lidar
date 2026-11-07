"""Descriptive donor-anchor coverage on CLOSED development data only."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from real_coverage_development import ROOT,DATA,OLD,OUT,sha
from real_poisson_methods_v2 import WorkingPulse,_matched,_peaks
from real_poisson_analysis_v2 import interval_match


def anchors(h,pulse,row,col):
    y,x=np.mgrid[-4:5,-4:5];keep=(x!=0)|(y!=0);donors=h[row-4:row+5,col-4:col+5][keep]
    weights=np.exp(-.5*((x[keep]**2+y[keep]**2)/1.5**2));weights/=weights.sum()
    primary=np.argmax(_matched(donors,pulse),axis=-1)
    votes=np.bincount(primary,weights=weights,minlength=h.shape[-1]);smooth=gaussian_filter1d(votes,2.,mode='constant')
    pool=weights@donors;pooled_match=_matched(pool,pulse)
    result={}
    for name,v in [('primary_votes',smooth),('pooled_counts',pooled_match)]:
        selected=[]
        for d in _peaks(v,n=len(v)):
            if all(abs(d-s)>=8 for s in selected):selected.append(d)
            if len(selected)==3:break
        result[name]=selected
    return result


def main():
    m=json.loads((DATA/'manifest.json').read_text());h=np.load(DATA/'thin_1.npz')['histograms'];pulse=WorkingPulse('gaussian')
    queries=[q for q in m['queries'] if q['split']=='evaluation_two'];summary={k:0 for k in ['donor_anchor','local_anchor','three_vote_anchors','three_pooled_anchors']};records=[]
    for q in queries:
        old=json.loads((OLD/f'gaussian_1_{q["id"]}.json').read_text());a=anchors(h,pulse,q['row'],q['column'])
        hit=lambda d:any(interval_match(d,w) for w in q['windows'])
        # Match the known window selected by the local primary where possible;
        # this avoids declaring two candidate peaks in the same window a repair.
        local=old['reports']['local']['primary'];target=[w for w in q['windows'] if interval_match(local,w)]
        target_hit=lambda ds:bool(target and any(interval_match(d,w) for d in ds for w in target))
        e=dict(donor_anchor=target_hit([old['proposal']['primary']]),local_anchor=bool(target),
            three_vote_anchors=target_hit(a['primary_votes']),three_pooled_anchors=target_hit(a['pooled_counts']))
        for k,v in e.items():summary[k]+=v
        records.append(dict(query_id=q['id'],anchors=a,windows=q['windows'],target_window=target,**e))
    result=dict(status='Post-analysis descriptive diagnostic on closed development; not a new method or efficacy result.',
        retention=.025,n=len(queries),matching_local_primary_window=summary,records=records,
        parameters=dict(spatial_sigma=1.5,vote_smoothing_sigma_bins=2.,max_anchors=3,minimum_separation_bins=8),
        notes=['All candidate anchors use only raw query-excluded donors. Local primary and annotation windows evaluate coverage, never generate anchors.',
            'Coverage of a reference window is not detection,correct pair recovery,calibration or physical truth. No new reporter has been selected.'])
    (OUT/'primary_mode_diagnostic.json').write_text(json.dumps(result,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
