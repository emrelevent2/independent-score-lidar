from pathlib import Path
import json
import argparse
import numpy as np
from scipy.io import loadmat,savemat
from photon_split import split_cycles

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--condition',default='nominal')
args=parser.parse_args(); condition=args.condition
source=root/'data'/f'screen_{condition}.mat'
manifest=json.loads((root/'data'/f'screen_{condition}.json').read_text())
observations=loadmat(source)['observations']
left=[]; right=[]
for i,sample in enumerate(manifest['samples']):
    seed=100000+sample['seed']*2+int(sample['null'])
    a,b,na,nb=split_cycles(observations[...,i],1000,seed)
    left.append(a.astype(np.uint16)); right.append(b.astype(np.uint16))
savemat(root/'data'/f'split_{condition}.mat',{'left':np.stack(left,axis=-1),
 'right':np.stack(right,axis=-1),'left_cycles':na,'right_cycles':nb},do_compression=True)
manifest['split_seed_rule']='100000 + acquisition_seed*2 + int(null)'
manifest['cycles']=[na,nb]
(root/'data'/f'split_{condition}.json').write_text(json.dumps(manifest,indent=2))
print(condition,observations.shape,na,nb)
