"""Descriptive sequential timing, after concurrent experiments have stopped."""
from pathlib import Path
import json,time,os,platform
import numpy as np
from continuous_alternative import alternative_at,amplitude_components
from envelope_evidence import robust_mixture_evidence
from likelihood_baseline import local_likelihood_report
ROOT=Path(__file__).resolve().parents[1]
ids=[f'null_{i:04d}' for i in range(5)]+[f'positive_{i:04d}' for i in [0,1,150,300,301]]
records=[]
def run(name,h):
 t=time.perf_counter()
 if name=='proposed':
  _,p=alternative_at(h,8,8);robust_mixture_evidence(h[8,8],amplitude_components(p),normalized_envelope=False)
 else:local_likelihood_report(h[8,8])
 return time.perf_counter()-t
if __name__=='__main__':
 warm=np.load(ROOT/'data'/'confirmation_v1'/'null_0005.npz')['histogram']
 for m in ['proposed','local_likelihood']:run(m,warm)
 for i,identifier in enumerate(ids):
  h=np.load(ROOT/'data'/'confirmation_v1'/(identifier+'.npz'))['histogram']
  methods=['proposed','local_likelihood'] if i%2==0 else ['local_likelihood','proposed']
  records.append(dict(id=identifier,**{m:run(m,h) for m in methods}))
 out=dict(records=records,note='One warmup per method; ten fixed inputs; sequential single-thread CPU wall time; not a full-image throughput benchmark.',
  environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
  platform=platform.platform(),summary={m:dict(median=float(np.median([r[m] for r in records])),
  q25=float(np.quantile([r[m] for r in records],.25)),q75=float(np.quantile([r[m] for r in records],.75))) for m in ['proposed','local_likelihood']})
 (ROOT/'results'/'confirmation_v1'/'isolated_timing.json').write_text(json.dumps(out,indent=2));print(out['summary'])
