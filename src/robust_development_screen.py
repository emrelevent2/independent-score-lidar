"""Development check of continuous-delay and uncertain-width evidence."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.stats import beta
from pilot import simulate
from envelope_evidence import independent_alternative,robust_evidence
from likelihood_evidence import evaluate_evidence

ROOT=Path(__file__).resolve().parents[1]


def conditions():
    mask=np.ones((12,12),bool);mask[6,6]=False
    return {
        'fractional_null':dict(secondary=0.,offset=.37,pulse_fwhm=5.),
        'narrow_null':dict(secondary=0.,offset=.37,pulse_fwhm=4.5),
        'wide_null':dict(secondary=0.,offset=.37,pulse_fwhm=5.5),
        'bright_fractional_null':dict(secondary=0.,primary=600.,offset=.37,pulse_fwhm=5.),
        'neighbor_only_null':dict(secondary=10.,secondary_mask=mask,offset=.37,pulse_fwhm=5.2),
        'secondary_3':dict(secondary=3.,offset=.37,pulse_fwhm=5.2),
        'secondary_6':dict(secondary=6.,offset=.37,pulse_fwhm=5.2),
        'secondary_10':dict(secondary=10.,offset=.37,pulse_fwhm=5.2)}


def main():
    out=dict(status='development only; 24 acquisitions per condition',
        seeds=[15000,15023],paired_seeds=True,source_sha256={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()
           for n in ['robust_development_screen.py','envelope_evidence.py','likelihood_evidence.py','pilot.py','crossfit.py','screen.py']},
        settings={'initial_cells':32,'max_boxes':256,'gap_tolerance':.01,'width_range':[4.5,5.5]},details={})
    start=time.perf_counter()
    for name,cfg in conditions().items():
        records=[]
        for seed in range(15000,15024):
            h,_,_=simulate(seed,size=12,**cfg)
            alt,p,s,v=independent_alternative(h,size=9)
            outputs={}
            t=time.perf_counter()
            old=evaluate_evidence(h[6,6],alt[6,6])
            outputs['integer_known']={'log_e':old['log_e'],'declared':bool(old['log_e']>=np.log(100)),
                                      'seconds':time.perf_counter()-t}
            for method,width in [('continuous_known',(5.,5.)),('continuous_uncertain',(4.5,5.5))]:
                t=time.perf_counter();r=robust_evidence(h[6,6],alt[6,6],fwhm_range=width)
                outputs[method]=dict(**r,seconds=time.perf_counter()-t)
            records.append(dict(seed=seed,primary=int(p[6,6]),secondary=int(s[6,6]),valid=bool(v[6,6]),methods=outputs))
            if len(records)%8==0:
                out['details'][name]={'in_progress':True,'records':records}
                (ROOT/'results'/'robust_development_screen.json').write_text(json.dumps(out,indent=2))
                print(name,len(records),'of 24',flush=True)
        summary={}
        for m in outputs:
            k=sum(r['methods'][m]['declared'] and r['valid'] for r in records)
            summary[m]={'declared':k,'n':len(records),'rate':k/len(records),
                'ci95':[float(beta.ppf(.025,k,len(records)-k+1)) if k else 0.,float(beta.ppf(.975,k+1,len(records)-k)) if k<len(records) else 1.],
                'seconds_mean':float(np.mean([r['methods'][m]['seconds'] for r in records]))}
            if m!='integer_known':
                summary[m]['max_remaining_gap']=max(r['methods'][m]['gap'] for r in records)
                summary[m]['conservative_fallbacks']=sum(r['methods'][m]['conservative_fallbacks'] for r in records)
        out['details'][name]={'in_progress':False,'records':records,'summary':summary}
        out['elapsed_seconds']=time.perf_counter()-start
        (ROOT/'results'/'robust_development_screen.json').write_text(json.dumps(out,indent=2))
        print(name,summary,flush=True)


if __name__=='__main__':main()
