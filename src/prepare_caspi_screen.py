from pathlib import Path
import json
import numpy as np
from scipy.io import savemat
from pilot import simulate
from screen import CONDITIONS

root=Path(__file__).resolve().parents[1]
for condition,kw in CONDITIONS.items():
    observations=[]; manifest=[]
    for split,first,count in [('calibration',3000,20),('screen',4000,30)]:
        for seed in range(first,first+count):
            for null in ([True] if split=='calibration' else [True,False]):
                h,_,_=simulate(seed,**({**kw,'secondary':0.} if null else kw))
                observations.append(h.astype(np.uint16))
                manifest.append({'split':split,'seed':seed,'null':null})
    savemat(root/'data'/f'screen_{condition}.mat',{'observations':np.stack(observations,axis=-1)},do_compression=True)
    (root/'data'/f'screen_{condition}.json').write_text(json.dumps({'condition':condition,'config':kw,'samples':manifest},indent=2))
    print(condition,len(observations),flush=True)
