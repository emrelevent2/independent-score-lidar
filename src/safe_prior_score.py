"""Classical weighted maximum matched score,using the SAME context locations.

This is a strong context-informed comparator,not a proper-mixture e-value.
Full event-path refitted bootstrap provides approximate plug-in calibration.
"""
import numpy as np
from deadtime_scan import pulse


def candidate_bank(alternative,sigma):
    k=alternative['rates'].shape[1];spatial=alternative['locations'][0];global_locations=[];seen=set()
    for loc,kind in zip(alternative['locations'],alternative['kinds']):
        if kind!='coverage' or float(loc[1]) in seen:continue
        seen.add(float(loc[1]));global_locations.append(loc)
    locations=np.asarray([spatial,*global_locations]);weights=np.r_[.75,np.full(len(global_locations),.25/len(global_locations))]
    templates=np.array([pulse(v[1],k,sigma)[0] for v in locations])
    return locations,weights,templates


def prior_score(s,e,theta,sigma,bank):
    locations,weights,g=bank;s=np.asarray(s,dtype=float);e=np.asarray(e,dtype=float);k=len(s);a,t,b=theta;f,d=pulse(t,k,sigma);lam=np.maximum(a*f+b/k,1e-14);u=s/lam-e;w=e/lam;j=np.column_stack([f,a*d,np.ones(k)/k]);inverse=np.linalg.pinv(j.T@(w[:,None]*j),rcond=1e-12)
    cross=g@(w[:,None]*j);raw=g@u-cross@inverse@(j.T@u);information=(g*g)@w-np.einsum('mi,ij,mj->m',cross,inverse,cross);distance=np.abs((locations[:,1]-t+k/2)%k-k/2);allowed=(distance>=2.5*sigma)&(information>1e-8)
    scores=np.where(allowed,np.maximum(raw,0)**2/np.maximum(information,1e-8)+2*np.log(weights),-np.inf);best=int(np.argmax(scores))
    return dict(score=float(scores[best]),primary=float(locations[best,0]),delay=float(locations[best,1]),index=best)
