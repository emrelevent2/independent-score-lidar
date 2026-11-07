"""Reverse sensitivities of the frozen positive continuous path expansion.

Same F0+F1+F2 law and floating moments as ct_dual_fast;no changed physical
family. Reverse-mode differentiation is established. Its phase-rate score
avoids propagating64dense sensitivities through every integration segment.
Initial and terminal factors are retained;terminal uses the frozen forward
implementation. This new version does not replace any frozen experiment.
"""
import math
import numpy as np
from deadtime_path_bootstrap import njit
import ct_dual_fast as forward

prepare_record=forward.prepare_record
prepare_rows=forward.prepare_rows


@njit(cache=True)
def _add_integral(d,v,direct,difference):
    lo=int(d[1]);hi=int(d[2]);difference[lo]+=v;difference[hi]-=v
    direct[hi]+=v*d[4];direct[lo]-=v*d[3]
    return v*d[0]


@njit(cache=True)
def _one(rate,prefix,edges,indices,fw,bw,comp,event,upper,td,te):
    k=len(rate);n=len(edges)-1;direct=np.zeros(k);difference=np.zeros(k);uniform=_add_integral(comp,-1.,direct,difference);value=math.log(rate[event])-forward._iv(rate,prefix,comp);direct[event]+=1./rate[event]
    if td<te:
        a=np.empty(n);b=np.empty(n);p1=np.zeros(n+1);adj_a=np.zeros(n);adj_b=np.zeros(n);adj_prefix=np.zeros(n+1);adj_blag=np.zeros(n);adj_clag=np.zeros(n);blag=np.zeros(n)
        for i in range(n):
            idx=indices[i,0];plus=indices[i,1];integral=forward._iv(rate,prefix,fw[i])
            if integral>300:raise FloatingPointError('Positive path sum exceeds supported range')
            a[i]=rate[idx]*math.exp(integral);b[i]=rate[plus]-rate[idx];e0,_=forward._moments(b[i],edges[i+1]-edges[i]);p1[i+1]=p1[i]+a[i]*e0
        # F1(upper),including its prefix and partial-segment dependencies.
        f=1.
        if upper>=edges[-1]:f+=p1[-1];adj_prefix[-1]+=1.
        elif upper>edges[0]:
            h=min(n-1,np.searchsorted(edges,upper,side='right')-1);e0,e1=forward._moments(b[h],upper-edges[h]);f+=p1[h]+a[h]*e0;adj_prefix[h]+=1.;adj_a[h]+=e0;adj_b[h]+=a[h]*e1
        # F2(upper). Each C_i=F1(edge_i-Td) contributes back to the
        # preceding prefix and at most one partial segment.
        for i in range(n):
            if edges[i]<2*td or edges[i]>=upper:continue
            length=min(upper,edges[i+1])-edges[i];current=indices[i,0];minus=indices[i,2];blag[i]=rate[minus]*math.exp(forward._iv(rate,prefix,bw[i]));c=rate[current]-rate[minus];x=edges[i]-td;cc=0.;h=-1;eo=eo1=0.
            if x>edges[0]:
                h=min(n-1,np.searchsorted(edges,x,side='right')-1);eo,eo1=forward._moments(b[h],x-edges[h]);cc=p1[h]+a[h]*eo
            e0,e1=forward._moments(b[i],length);m,mb,mc=forward._mixed_moments(b[i],c,length);f+=a[i]*cc*e0+a[i]*blag[i]*m;adj_a[i]+=cc*e0+blag[i]*m;adj_b[i]+=a[i]*cc*e1+a[i]*blag[i]*mb;adj_blag[i]+=a[i]*m;adj_clag[i]+=a[i]*blag[i]*mc
            if h>=0:
                v=a[i]*e0;adj_prefix[h]+=v;adj_a[h]+=v*eo;adj_b[h]+=v*a[h]*eo1
        # Reverse prefix scan:one scalar adjoint per F1 prefix node.
        running=0.
        for i in range(n-1,-1,-1):
            running+=adj_prefix[i+1];e0,e1=forward._moments(b[i],edges[i+1]-edges[i]);adj_a[i]+=running*e0;adj_b[i]+=running*a[i]*e1
        value+=math.log(f)
        for i in range(n):
            idx=indices[i,0];plus=indices[i,1];minus=indices[i,2];aa=adj_a[i]*a[i]/f;bb=adj_b[i]/f;direct[idx]+=aa/rate[idx]-bb;direct[plus]+=bb;uniform+=_add_integral(fw[i],aa,direct,difference)
            if edges[i]>=2*td and edges[i]<upper:
                aa=adj_blag[i]*blag[i]/f;cc=adj_clag[i]/f;direct[minus]+=aa/rate[minus]-cc;direct[idx]+=cc;uniform+=_add_integral(bw[i],aa,direct,difference)
    running=uniform
    for j in range(k):running+=difference[j];direct[j]+=running
    return value,direct


@njit(cache=True)
def _record_rows(rate,prefix,phases,gaps,flags,events,all_edges,eoffset,coffset,indices,fw,bw,comp,td,te):
    n=len(phases);values=np.zeros(n);scores=np.zeros((n,len(rate)))
    for r in range(n):
        if flags[r]:continue
        if gaps[r]<max(td,te):values[r]=-np.inf;scores[r]=np.nan;continue
        edges=all_edges[eoffset[r]:eoffset[r+1]];i=coffset[r];end=coffset[r+1];values[r],row=_one(rate,prefix,edges,indices[i:end],fw[i:end],bw[i:end],comp[r],events[r],min(te,gaps[r]-td),td,te);scores[r]=row
    return values,scores


def full_factor(rates,design,geometry,order=16,return_pieces=False):
    rate=np.asarray(rates,float);j=np.asarray(design,float);g=geometry
    if rate.ndim!=1 or len(rate)!=g['bins'] or j.ndim!=2 or j.shape[0]!=len(rate) or np.any(rate<=0) or np.any(~np.isfinite(rate)) or np.any(~np.isfinite(j)):raise ValueError('Positive finite rates and matching Jacobian required')
    prefix=np.r_[0.,np.cumsum(rate)];values,scores=_record_rows(rate,prefix,g['phases'],g['gaps'],g['terminal'],g['event_bins'],g['edges'],g['edge_offsets'],g['cell_offsets'],g['indices'],g['forward'],g['backward'],g['compensator'],g['td'],g['te'])
    if len(values):
        assert np.count_nonzero(g['terminal'])==1 and g['terminal'][-1]
        tail=forward.prepare_rows(g['phases'][-1:],g['gaps'][-1:],len(rate),g['td'],g['te'],[True],[-1]);v,z=forward.evaluate_rows(rate,np.eye(len(rate)),tail,order);values[-1]=v[0];scores[-1]=z[0]
    d=np.asarray(g['initial']);initial=-forward._iv(rate,prefix,d);direct=np.zeros(len(rate));difference=np.zeros(len(rate));uniform=_add_integral(d,-1.,direct,difference);direct+=uniform+np.cumsum(difference)
    if not g['empty']:idx=g['first_index'];initial+=math.log(rate[idx]);direct[idx]+=1./rate[idx]
    allvalues=np.r_[initial,values];allscores=np.vstack([direct,scores]);value=float(initial+np.sum(values));gradient=(direct+np.sum(scores,axis=0))@j
    if return_pieces:return value,gradient,allvalues,allscores@j
    return value,gradient
