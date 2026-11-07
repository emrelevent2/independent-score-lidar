"""Continuous-time two-recovery likelihood,with at most TWO hidden avalanches.

Positive renewal/path expansion for Te<=3Td;this covers all original four
continuous regimes. Piecewise-constant periodic rates are preserved. GL
quadrature evaluates smooth segment integrals;ordinary floating arithmetic,
NOT interval certification or a new renewal theorem. Analytic sensitivities
differentiate the numerical integrals. No detection utility is assumed.
"""
from functools import lru_cache
import math
import numpy as np
from deadtime_path_bootstrap import njit


@lru_cache(maxsize=8)
def quadrature(order):
    x,w=np.polynomial.legendre.leggauss(int(order));return (x+1)/2,w/2


def descriptor(a,b,k):
    ca,cb=np.floor([a/k,b/k]);pa,pb=a-ca*k,b-cb*k;ja,jb=int(np.floor(pa)),int(np.floor(pb));return [cb-ca,ja,jb,pa-ja,pb-jb]


def _knots(phi,lower,upper,k,shifts):
    points=[lower,upper]
    for shift in shifts:
        for n in range(math.floor(lower+phi-shift)-1,math.ceil(upper+phi-shift)+2):
            v=n-phi+shift
            if lower<v<upper:points.append(v)
    return np.unique(np.asarray(points,float))


def prepare_rows(phases,gaps,bins,td,te,terminal=None,event_bins=None):
    if td<=0 or te<0 or not np.isfinite(td+te) or te>3*td:raise ValueError('Require Td>0,0<=Te<=3Td(two hidden avalanches maximum)')
    phases,gaps=np.broadcast_arrays(np.asarray(phases,float),np.asarray(gaps,float));phases=(phases.reshape(-1)%bins).copy();gaps=gaps.reshape(-1).copy()
    if np.any(~np.isfinite(phases)) or np.any(~np.isfinite(gaps)) or np.any(gaps<0):raise ValueError('Finite phases and nonnegative gaps required')
    n=len(phases);flags=np.zeros(n,dtype=np.bool_) if terminal is None else np.asarray(terminal,dtype=np.bool_).reshape(-1);events=np.floor((phases+gaps)%bins).astype(np.int64) if event_bins is None else np.asarray(event_bins,dtype=np.int64);edges=[];indices=[];forward=[];backward=[];offset=[0];celloffset=[0];tbounds=[];toffset=[0];comp=[]
    assert len(flags)==len(events)==n
    for phi,gap,flag in zip(phases,gaps,flags):
        comp.append(descriptor(phi+td,phi+gap,bins))
        if td<te:
            knots=_knots(phi,td,te,bins,[0.,-td,td]);knots=np.unique(np.r_[knots,2*td] if td<2*td<te else knots);edges.extend(knots);offset.append(len(edges));mids=(knots[:-1]+knots[1:])/2
            for a,mid in zip(knots[:-1],mids):
                indices.append([int(np.floor((phi+mid)%bins)),int(np.floor((phi+mid+td)%bins)),int(np.floor((phi+mid-td)%bins))]);forward.append(descriptor(phi+a,phi+a+td,bins));backward.append(descriptor(phi+a-td,phi+a,bins))
            celloffset.append(len(indices))
            if flag and max(td,te)<=gap<td+te:
                knots_tail=np.r_[gap,td+te,knots+td];knots_tail=knots_tail[(knots_tail>=gap)&(knots_tail<=td+te)];extra=_knots(phi,gap,td+te,bins,[0.]);tbounds.extend(np.unique(np.r_[knots_tail,extra]))
        else:offset.append(len(edges));celloffset.append(len(indices))
        toffset.append(len(tbounds))
    return dict(phases=phases,gaps=gaps,bins=int(bins),td=float(td),te=float(te),terminal=flags,event_bins=events,edges=np.asarray(edges,float),edge_offsets=np.asarray(offset,dtype=np.int64),cell_offsets=np.asarray(celloffset,dtype=np.int64),indices=np.asarray(indices,dtype=np.int64).reshape(-1,3),forward=np.asarray(forward,float).reshape(-1,5),backward=np.asarray(backward,float).reshape(-1,5),compensator=np.asarray(comp,float).reshape(-1,5),terminal_bounds=np.asarray(tbounds,float),terminal_offsets=np.asarray(toffset,dtype=np.int64))


def prepare_record(observed,total,bins,td,te):
    obs=np.asarray(observed,float).reshape(-1)
    if total<0 or not np.isfinite(total) or np.any(~np.isfinite(obs)) or np.any(obs<0) or np.any(obs>=total) or np.any(np.diff(obs)<=0) or np.any(np.diff(obs)<max(td,te)):raise ValueError('Invalid continuous record/horizon/recovery support')
    if len(obs):
        flags=np.zeros(len(obs),dtype=np.bool_);flags[-1]=True;event_bins=np.r_[np.floor(obs[1:]%bins).astype(np.int64),-1];g=prepare_rows(obs%bins,np.r_[np.diff(obs),total-obs[-1]],bins,td,te,flags,event_bins);g.update(first_index=int(np.floor(obs[0]%bins)),initial=descriptor(0.,obs[0],bins),empty=False)
    else:g=prepare_rows([],[],bins,td,te);g.update(first_index=-1,initial=descriptor(0.,total,bins),empty=True)
    return g


@njit(cache=True)
def _iv(rate,prefix,d):
    lo=int(d[1]);hi=int(d[2]);return d[0]*prefix[-1]+prefix[hi]-prefix[lo]+rate[hi]*d[4]-rate[lo]*d[3]


@njit(cache=True)
def _ivg(design,prefix,d,j):
    lo=int(d[1]);hi=int(d[2]);return d[0]*prefix[-1,j]+prefix[hi,j]-prefix[lo,j]+design[hi,j]*d[4]-design[lo,j]*d[3]


@njit(cache=True)
def _desc(a,b,k):
    ca=math.floor(a/k);cb=math.floor(b/k);pa=a-ca*k;pb=b-cb*k;ja=int(math.floor(pa));jb=int(math.floor(pb));return np.array([float(cb-ca),float(ja),float(jb),pa-ja,pb-jb])


@njit(cache=True)
def _moments(slope,length):
    z=slope*length
    if abs(z)<.1:
        term=1.;a=1.;b=.5
        for n in range(1,18):term*=z/n;a+=term/(n+1);b+=term/(n+2)
        return length*a,length*length*b
    return math.expm1(z)/slope,(math.exp(z)*(z-1)+1)/(slope*slope)


@njit(cache=True)
def _f1(x,edges,p):
    dim=p[2].shape[1]
    if x<=edges[0]:return 0.,np.zeros(dim)
    if x>=edges[-1]:return p[10][-1],p[11][-1].copy()
    i=min(len(edges)-2,np.searchsorted(edges,x,side='right')-1);length=x-edges[i];a,b=_moments(p[1][i],length);return p[10][i]+p[0][i]*a,p[11][i]+p[2][i]*a+p[0][i]*p[3][i]*b


@njit(cache=True)
def _partial2(i,length,p,nodes,weights):
    dim=p[2].shape[1];value=0.;gradient=np.zeros(dim)
    if length<=0:return value,gradient
    for q in range(len(nodes)):
        z=length*nodes[q];weight=length*weights[q];e0,e1=_moments(p[5][i],z);ex=math.exp(p[1][i]*z);inner=p[8][i]+p[4][i]*e0;value+=weight*p[0][i]*ex*inner
        for j in range(dim):gradient[j]+=weight*ex*((p[2][i,j]+p[0][i]*z*p[3][i,j])*inner+p[0][i]*(p[9][i,j]+p[6][i,j]*e0+p[4][i]*p[7][i,j]*e1))
    return value,gradient


@njit(cache=True)
def _profile(rate,design,prefix,gprefix,edges,indices,forward,backward,td,nodes,weights):
    n=len(edges)-1;dim=design.shape[1];a=np.empty(n);b=np.empty(n);da=np.empty((n,dim));db=np.empty((n,dim));bb=np.zeros(n);c=np.zeros(n);dbb=np.zeros((n,dim));dc=np.zeros((n,dim));cc=np.zeros(n);dcc=np.zeros((n,dim));p1=np.zeros(n+1);dp1=np.zeros((n+1,dim));p2=np.zeros(n+1);dp2=np.zeros((n+1,dim));p=(a,b,da,db,bb,c,dbb,dc,cc,dcc,p1,dp1,p2,dp2)
    for i in range(n):
        index=indices[i,0];plus=indices[i,1];integral=_iv(rate,prefix,forward[i])
        if integral>300:raise FloatingPointError('Prototype positive path sums exceed supported floating range')
        a[i]=rate[index]*math.exp(integral);b[i]=rate[plus]-rate[index]
        for j in range(dim):da[i,j]=a[i]*(design[index,j]/rate[index]+_ivg(design,gprefix,forward[i],j));db[i,j]=design[plus,j]-design[index,j]
        e0,e1=_moments(b[i],edges[i+1]-edges[i]);p1[i+1]=p1[i]+a[i]*e0;dp1[i+1]=dp1[i]+da[i]*e0+a[i]*db[i]*e1
    for i in range(n):
        p2[i+1]=p2[i];dp2[i+1]=dp2[i]
        if edges[i]<2*td:continue
        current=indices[i,0];minus=indices[i,2];integral=_iv(rate,prefix,backward[i]);bb[i]=rate[minus]*math.exp(integral);c[i]=rate[current]-rate[minus]
        for j in range(dim):dbb[i,j]=bb[i]*(design[minus,j]/rate[minus]+_ivg(design,gprefix,backward[i],j));dc[i,j]=design[current,j]-design[minus,j]
        cc[i],gradient=_f1(edges[i]-td,edges,p);dcc[i]=gradient;value,gradient=_partial2(i,edges[i+1]-edges[i],p,nodes,weights);p2[i+1]+=value;dp2[i+1]+=gradient
    return p


@njit(cache=True)
def _F(x,edges,p,nodes,weights,td):
    value,gradient=_f1(x,edges,p)
    if x>=edges[-1]:return 1+value+p[12][-1],gradient+p[13][-1]
    if x<=edges[0]:return 1.,gradient
    i=min(len(edges)-2,np.searchsorted(edges,x,side='right')-1);value+=p[12][i];gradient+=p[13][i]
    if edges[i]>=2*td:
        v,g=_partial2(i,x-edges[i],p,nodes,weights);value+=v;gradient+=g
    return 1+value,gradient


@njit(cache=True)
def _rows(rate,design,prefix,gprefix,phases,gaps,flags,events,all_edges,eoffset,coffset,indices,fw,bw,comp,all_tbounds,toffset,td,te,nodes,weights):
    n=len(phases);dim=design.shape[1];values=np.zeros(n);gradients=np.zeros((n,dim));k=len(rate)
    for r in range(n):
        s=gaps[r];phi=phases[r];terminal=flags[r]
        if s<max(td,te):
            if not terminal:values[r]=-np.inf;gradients[r]=np.nan
            continue
        integral=_iv(rate,prefix,comp[r]);base=-integral;gb=np.empty(dim)
        for j in range(dim):gb[j]=-_ivg(design,gprefix,comp[r],j)
        if not terminal:
            idx=events[r];base+=math.log(rate[idx]);gb+=design[idx]/rate[idx]
        if td>=te:values[r]=base;gradients[r]=gb;continue
        edges=all_edges[eoffset[r]:eoffset[r+1]];i=coffset[r];end=coffset[r+1];p=_profile(rate,design,prefix,gprefix,edges,indices[i:end],fw[i:end],bw[i:end],td,nodes,weights)
        if not terminal or s>=td+te:
            upper=te if terminal else min(te,s-td);f,g=_F(upper,edges,p,nodes,weights,td);values[r]=base+math.log(f);gradients[r]=gb+g/f;continue
        # Positive integral of the remaining density+analytic post-transienttail.
        taildesc=_desc(phi+td,phi+td+te,k);f,g=_F(te,edges,p,nodes,weights,td);maximum=-_iv(rate,prefix,taildesc)+math.log(f);sumweight=1.;gsum=g/f
        for j in range(dim):gsum[j]-=_ivg(design,gprefix,taildesc,j)
        bounds=all_tbounds[toffset[r]:toffset[r+1]]
        for cell in range(len(bounds)-1):
            length=bounds[cell+1]-bounds[cell]
            for q in range(len(nodes)):
                v=bounds[cell]+length*nodes[q];idx=int(math.floor((phi+v)%k));d=_desc(phi+td,phi+v,k);ff,gg=_F(v-td,edges,p,nodes,weights,td);logterm=math.log(rate[idx])-_iv(rate,prefix,d)+math.log(ff)+math.log(length*weights[q]);gradient=design[idx]/rate[idx]+gg/ff
                for j in range(dim):gradient[j]-=_ivg(design,gprefix,d,j)
                if logterm>maximum:
                    scale=math.exp(maximum-logterm);sumweight=sumweight*scale+1;gsum=gsum*scale+gradient;maximum=logterm
                else:
                    weight=math.exp(logterm-maximum);sumweight+=weight;gsum+=weight*gradient
        values[r]=maximum+math.log(sumweight);gradients[r]=gsum/sumweight
    return values,gradients


def evaluate_rows(rates,design,geometry,order=16):
    rate=np.asarray(rates,float);j=np.asarray(design,float)
    if rate.ndim!=1 or len(rate)!=geometry['bins'] or j.ndim!=2 or j.shape[0]!=len(rate) or np.any(rate<=0) or np.any(~np.isfinite(rate)) or np.any(~np.isfinite(j)):raise ValueError('Positive finite rates and matching Jacobian required')
    prefix=np.r_[0.,np.cumsum(rate)];gp=np.vstack([np.zeros(j.shape[1]),np.cumsum(j,axis=0)]);g=geometry;nodes,weights=quadrature(order);values,gradients=_rows(rate,j,prefix,gp,g['phases'],g['gaps'],g['terminal'],g['event_bins'],g['edges'],g['edge_offsets'],g['cell_offsets'],g['indices'],g['forward'],g['backward'],g['compensator'],g['terminal_bounds'],g['terminal_offsets'],g['td'],g['te'],nodes,weights);return values,gradients


def full_factor(rates,design,geometry,order=16,return_pieces=False):
    values,gradients=evaluate_rows(rates,design,geometry,order);rate=np.asarray(rates,float);j=np.asarray(design,float);prefix=np.r_[0.,np.cumsum(rate)];gp=np.vstack([np.zeros(j.shape[1]),np.cumsum(j,axis=0)]);d=np.asarray(geometry['initial']);initial=-_iv(rate,prefix,d);gi=np.array([-_ivg(j,gp,d,k) for k in range(j.shape[1])])
    if not geometry['empty']:index=geometry['first_index'];initial+=np.log(rate[index]);gi+=j[index]/rate[index]
    value=float(initial+np.sum(values));gradient=gi+np.sum(gradients,axis=0)
    if return_pieces:return value,gradient,np.r_[initial,values],np.vstack([gi,gradients])
    return value,gradient
