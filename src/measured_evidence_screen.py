"""Exploratory fixed ROI, measured counts and measured IRF; no truth claims."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from read_rt3d import read_rbin
from poisson_evidence import poisson_neighbor_alternative,irf_templates,poisson_null_bound,poisson_log_likelihood

ROOT=Path(__file__).resolve().parents[1]


def summarize(result):
    records=result['records'];n=len(records)
    return dict(tested_pixels=n,
        declared_per_pixel_alpha01=int(sum(r['log_e']>=float(np.log(100)) for r in records)),
        declared_bonferroni_image_alpha05=int(sum(r['log_e']>=float(np.log(n/.05)) for r in records)),
        numerical_gap_max=max(r['global_gap'] for r in records),
        fallback_count=sum(r['fallback_count'] for r in records),
        median_log_e=float(np.median([r['log_e'] for r in records])))


def main():
    path=ROOT/'data'/'rt3d_measured'/'4_camouflage_3_2ms.rbin'
    d=read_rbin(path)
    h=d['histograms'][0,64:88,27:51,:]
    alt,p,s=poisson_neighbor_alternative(h,d['irf'])
    templates=irf_templates(d['irf'],h.shape[-1])
    result=dict(status='exploratory measured-data evidence; no ground truth',
                roi_zero_based_exclusive={'rows':[64,88],'columns':[27,51]},
                scored_roi_indices=[5,19],alpha_per_pixel=.01,
                observation_model='independent Poisson; known measured IRF; uniform background',
                source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()
                    for n in ['measured_evidence_screen.py','poisson_evidence.py','read_rt3d.py']},
                caveats=['No supplied ground-truth return map.',
                         'IRF, Poisson and uniform-background assumptions are not experimentally certified.',
                         'Integer-delay null; fractional shifts remain a model limitation.',
                         'ROI is development data, not an untouched real-data benchmark.',
                         'Per-pixel threshold is not a family-wise image guarantee.'],records=[])
    start=time.perf_counter()
    for y in range(5,19):
        for col in range(5,19):
            null=poisson_null_bound(h[y,col],templates)
            log_e=poisson_log_likelihood(h[y,col],alt[y,col])-null['upper']
            result['records'].append(dict(row=y+64,column=col+27,log_e=log_e,
                primary_proposal=int(p[y,col]),secondary_proposal=int(s[y,col]),**null))
        result['elapsed_seconds']=time.perf_counter()-start
        (ROOT/'results'/'measured_likelihood_evidence.json').write_text(json.dumps(result,indent=2))
        print('Completed ROI row',y,'pixels',len(result['records']),flush=True)
    result['summary']=summarize(result)
    (ROOT/'results'/'measured_likelihood_evidence.json').write_text(json.dumps(result,indent=2))
    print(result['summary'],flush=True)


if __name__=='__main__':main()
