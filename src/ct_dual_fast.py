"""Analytic-segment continuous likelihood,with at most TWO hidden avalanches.

Positive renewal/path expansion for Te<=3Td;this covers all original four
continuous regimes. Piecewise-constant periodic rates are preserved. GL
segment formulas evaluate the renewal integrals;ordinary floating arithmetic,
NOT interval certification or a new renewal theorem. Analytic sensitivities
differentiate the numerical integrals. No detection utility is assumed.
"""
from functools import lru_cache
import math
import numpy as np
from deadtime_path_bootstrap import njit

_gx,_gw=np.polynomial.legendre.leggauss(32)
_FALLBACK_NODES=(_gx+1)/2
_FALLBACK_WEIGHTS=_gw/2


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
    n=len(phases);flags=np.zeros(n,dtype=np.bool_) if terminal is None else np.asarray(terminal,dtype=np.bool_).reshape(-1);events=np.floor((phases+gaps)%bins).astype(np.int64) if event_bins is None else np.asarray(event_bins,dtype=np.int64)
    assert len(flags)==len(events)==n
    edges,offset,celloffset,indices,forward,backward,comp,tbounds,toffset=_prepare_arrays(phases,gaps,flags,int(bins),float(td),float(te))
    return dict(phases=phases,gaps=gaps,bins=int(bins),td=float(td),te=float(te),terminal=flags,event_bins=events,edges=edges,edge_offsets=offset,cell_offsets=celloffset,indices=indices,forward=forward,backward=backward,compensator=comp,terminal_bounds=tbounds,terminal_offsets=toffset)


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
def _prepare_arrays(phases,gaps,flags,k,td,te):
    n=len(phases);cap=3*(max(0,int(math.ceil(te-td)))+4)+4;tcap=cap+int(math.ceil(td))+8;edges=np.empty(n*cap);indices=np.empty((n*cap,3),dtype=np.int64);fw=np.empty((n*cap,5));bw=np.empty((n*cap,5));comp=np.empty((n,5));eo=np.zeros(n+1,dtype=np.int64);co=np.zeros(n+1,dtype=np.int64);to=np.zeros(n+1,dtype=np.int64);tb=np.empty(n*tcap);ne=nc=nt=0
    for r in range(n):
        phi=phases[r];s=gaps[r];comp[r]=_desc(phi+td,phi+s,k)
        if td<te:
            points=np.empty(cap);m=2;points[0]=td;points[1]=te
            if td<2*td<te:points[m]=2*td;m+=1
            for shift in [0.,-td,td]:
                for integer in range(math.floor(td+phi-shift)-1,math.ceil(te+phi-shift)+2):
                    v=integer-phi+shift
                    if td<v<te:points[m]=v;m+=1
            points=np.sort(points[:m]);knots=np.empty(m);nk=0
            for i in range(m):
                if i==0 or points[i]!=points[i-1]:knots[nk]=points[i];nk+=1
            for i in range(nk):edges[ne]=knots[i];ne+=1
            for i in range(nk-1):
                a=knots[i];mid=(a+knots[i+1])/2;indices[nc,0]=int(math.floor((phi+mid)%k));indices[nc,1]=int(math.floor((phi+mid+td)%k));indices[nc,2]=int(math.floor((phi+mid-td)%k));fw[nc]=_desc(phi+a,phi+a+td,k);bw[nc]=_desc(phi+a-td,phi+a,k);nc+=1
            if flags[r] and max(td,te)<=s<td+te:
                points_tail=np.empty(tcap);m=2;points_tail[0]=s;points_tail[1]=td+te
                for i in range(nk):
                    v=knots[i]+td
                    if s<=v<=td+te:points_tail[m]=v;m+=1
                for integer in range(math.floor(s+phi)-1,math.ceil(td+te+phi)+2):
                    v=integer-phi
                    if s<v<td+te:points_tail[m]=v;m+=1
                points_tail=np.sort(points_tail[:m])
                for i in range(m):
                    if i==0 or points_tail[i]!=points_tail[i-1]:tb[nt]=points_tail[i];nt+=1
        eo[r+1]=ne;co[r+1]=nc;to[r+1]=nt
    return edges[:ne],eo,co,indices[:nc],fw[:nc],bw[:nc],comp,tb[:nt],to


@njit(cache=True)
def _moments(slope,length):
    z=slope*length
    if z==0:return length,length*length/2
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
def _unit_moment(x,n):
    if x==0:return 1./(n+1)
    if abs(x)<=2:
        term=1.;result=1./(n+1)
        for k in range(1,36):
            term*=x/k;result+=term/(n+k+1)
            if abs(term)<1e-18:break
        return result
    result=0.
    for k in range(32):z=_FALLBACK_NODES[k];result+=_FALLBACK_WEIGHTS[k]*z**n*math.exp(x*z)
    return result


@njit(cache=True)
def _mixed_moments(b,c,length):
    # M=integral exp(b*z)*E(c,z)dz,and exact b/c derivatives.
    x=b*length;y=c*length
    if c==0:
        _,i1=_moments(b,length);i2=length**3*_unit_moment(x,2);return i1,i2,i2/2
    if abs(y)<.1:
        value=0.;db=0.;dc=0.;coefficient=1.;derivative=0.;j1=_unit_moment(x,1);j2=_unit_moment(x,2)
        for n in range(13):
            value+=coefficient*j1;db+=coefficient*j2;dc+=derivative*j1
            if n>=1 and abs(coefficient)+abs(derivative)<1e-17:break
            derivative=(derivative*y+coefficient)/(n+2);coefficient*=y/(n+2)
            j1=j2;j2=_unit_moment(x,n+3)
        return length**2*value,length**3*db,length**3*dc
    e0,e1=_moments(b,length);f0,f1=_moments(b+c,length);return (f0-e0)/c,(f1-e1)/c,(c*f1-f0+e0)/(c*c)


@njit(cache=True)
def _partial2(i,length,p,nodes,weights,offset=0.):
    dim=p[2].shape[1];gradient=np.zeros(dim)
    if length<=0:return 0.,gradient
    b=p[1][i];c=p[5][i];exb=math.exp(b*offset);exc=math.exp(c*offset);eo,eo1=_moments(c,offset);a=p[0][i]*exb;bb=p[4][i]*exc;cc=p[8][i]+p[4][i]*eo;e0,e1=_moments(b,length);m,mb,mc=_mixed_moments(b,c,length);value=a*cc*e0+a*bb*m
    for j in range(dim):
        da=exb*(p[2][i,j]+p[0][i]*offset*p[3][i,j]);dbb=exc*(p[6][i,j]+p[4][i]*offset*p[7][i,j]);dcc=p[9][i,j]+p[6][i,j]*eo+p[4][i]*p[7][i,j]*eo1
        gradient[j]=(da*cc+a*dcc)*e0+a*cc*e1*p[3][i,j]+(da*bb+a*dbb)*m+a*bb*(mb*p[3][i,j]+mc*p[7][i,j])
    return value,gradient


@njit(cache=True)
def _profile(rate,design,prefix,gprefix,edges,indices,forward,backward,td,nodes,weights):
    n=len(edges)-1;dim=design.shape[1];a=np.empty(n);b=np.empty(n);da=np.empty((n,dim));db=np.empty((n,dim));bb=np.zeros(n);c=np.zeros(n);dbb=np.zeros((n,dim));dc=np.zeros((n,dim));cc=np.zeros(n);dcc=np.zeros((n,dim));p1=np.zeros(n+1);dp1=np.zeros((n+1,dim));p2=np.zeros(n+1);dp2=np.zeros((n+1,dim));p=(a,b,da,db,bb,c,dbb,dc,cc,dcc,p1,dp1,p2,dp2)
    for i in range(n):
        index=indices[i,0];plus=indices[i,1];integral=_iv(rate,prefix,forward[i])
        if integral>300:raise FloatingPointError('Prototype positive path sums exceed supported floating range')
        a[i]=rate[index]*math.exp(integral);b[i]=rate[plus]-rate[index]
        for j in range(dim):da[i,j]=a[i]*(design[index,j]/rate[index]+_ivg(design,gprefix,forward[i],j));db[i,j]=design[plus,j]-design[index,j]
        e0,e1=_moments(b[i],edges[i+1]-edges[i]);p1[i+1]=p1[i]+a[i]*e0
        for j in range(dim):dp1[i+1,j]=dp1[i,j]+da[i,j]*e0+a[i]*db[i,j]*e1
    for i in range(n):
        p2[i+1]=p2[i];dp2[i+1]=dp2[i]
        if edges[i]<2*td:continue
        current=indices[i,0];minus=indices[i,2];integral=_iv(rate,prefix,backward[i]);bb[i]=rate[minus]*math.exp(integral);c[i]=rate[current]-rate[minus]
        for j in range(dim):dbb[i,j]=bb[i]*(design[minus,j]/rate[minus]+_ivg(design,gprefix,backward[i],j));dc[i,j]=design[current,j]-design[minus,j]
        x=edges[i]-td
        if x>edges[0]:
            h=min(n-1,np.searchsorted(edges,x,side='right')-1);e0,e1=_moments(b[h],x-edges[h]);cc[i]=p1[h]+a[h]*e0
            for j in range(dim):dcc[i,j]=dp1[h,j]+da[h,j]*e0+a[h]*db[h,j]*e1
        value,gradient=_partial2(i,edges[i+1]-edges[i],p,nodes,weights);p2[i+1]+=value;dp2[i+1]+=gradient
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
def _latent_tail_profile(rate,design,prefix,gprefix,phi,edges,indices,p,td,nodes,weights):
    # m(v)=g(v)*(1+F1(v-Td)) for v<=Te<=3Td. All terms are positive.
    n=len(edges)-1;dim=design.shape[1];a=np.empty(n);b=np.empty(n);da=np.empty((n,dim));db=np.empty((n,dim));tail=np.zeros(n+1);dtail=np.zeros((n+1,dim))
    for i in range(n):
        idx=indices[i,0];desc=_desc(phi+td,phi+edges[i],len(rate));a[i]=rate[idx]*math.exp(-_iv(rate,prefix,desc));b[i]=-rate[idx]
        for j in range(dim):da[i,j]=a[i]*(design[idx,j]/rate[idx]-_ivg(design,gprefix,desc,j));db[i,j]=-design[idx,j]
    mp=(a,b,da,db,p[4],p[5],p[6],p[7],p[8],p[9])
    for i in range(n-1,-1,-1):
        length=edges[i+1]-edges[i];e0,e1=_moments(b[i],length);value=a[i]*e0;gradient=da[i]*e0+a[i]*db[i]*e1
        if edges[i]>=2*td:
            z,g=_partial2(i,length,mp,nodes,weights);value+=z;gradient+=g
        tail[i]=tail[i+1]+value;dtail[i]=dtail[i+1]+gradient
    return mp,tail,dtail


@njit(cache=True)
def _latent_tail(lower,edges,mp,tail,dtail,td,nodes,weights):
    dim=dtail.shape[1]
    if lower>=edges[-1]:return 0.,np.zeros(dim)
    if lower<=edges[0]:return tail[0],dtail[0].copy()
    i=min(len(edges)-2,np.searchsorted(edges,lower,side='right')-1);offset=lower-edges[i];length=edges[i+1]-lower;e0,e1=_moments(mp[1][i],length);ex=math.exp(mp[1][i]*offset);a=mp[0][i]*ex;da=ex*(mp[2][i]+mp[0][i]*offset*mp[3][i]);value=tail[i+1]+a*e0;gradient=dtail[i+1]+da*e0+a*mp[3][i]*e1
    if edges[i]>=2*td:
        z,g=_partial2(i,length,mp,nodes,weights,offset);value+=z;gradient+=g
    return value,gradient


@njit(cache=True)
def _survival(rate,design,prefix,gprefix,phi,s,edges,indices,p,td,te,nodes,weights,mp,tail,dtail):
    dim=design.shape[1]
    if s<max(td,te):return 0.,np.zeros(dim)
    desc=_desc(phi+td,phi+s,len(rate));f,g=_F(min(te,s-td),edges,p,nodes,weights,td);logactive=-_iv(rate,prefix,desc)+math.log(f);ga=g/f
    for j in range(dim):ga[j]-=_ivg(design,gprefix,desc,j)
    if s>=td+te:return logactive,ga
    dead,gd=_latent_tail(max(td,s-td),edges,mp,tail,dtail,td,nodes,weights)
    if dead<=0:return logactive,ga
    logdead=math.log(dead);maximum=max(logactive,logdead);a=math.exp(logactive-maximum);b=math.exp(logdead-maximum);return maximum+math.log(a+b),(a*ga+b*gd/dead)/(a+b)


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
        mp,tail,dtail=_latent_tail_profile(rate,design,prefix,gprefix,phi,edges,indices[i:end],p,td,nodes,weights);values[r],gradient=_survival(rate,design,prefix,gprefix,phi,s,edges,indices[i:end],p,td,te,nodes,weights,mp,tail,dtail);gradients[r]=gradient
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


def cached_profile(rates,design,phase,td,te,order=16):
    rate=np.asarray(rates,float);j=np.asarray(design,float);geo=prepare_rows([phase],[td+te+1],len(rate),td,te);prefix=np.r_[0.,np.cumsum(rate)];gp=np.vstack([np.zeros(j.shape[1]),np.cumsum(j,axis=0)]);nodes,weights=quadrature(order);p=None;latent=None
    if td<te:
        p=_profile(rate,j,prefix,gp,geo['edges'],geo['indices'],geo['forward'],geo['backward'],td,nodes,weights);latent=_latent_tail_profile(rate,j,prefix,gp,geo['phases'][0],geo['edges'],geo['indices'],p,td,nodes,weights)
    return dict(rates=rate,design=j,prefix=prefix,gprefix=gp,geometry=geo,profile=p,latent=latent,nodes=nodes,weights=weights)


@njit(cache=True)
def _cached_hidden_gaps(rate,j,prefix,gp,phi,gaps,td,te,edges,indices,p,latent,nodes,weights,survival):
    values=np.empty(len(gaps));gradients=np.zeros((len(gaps),j.shape[1]));mp,tail,dtail=latent
    for i in range(len(gaps)):
        s=gaps[i]
        if survival:values[i],g=_survival(rate,j,prefix,gp,phi,s,edges,indices,p,td,te,nodes,weights,mp,tail,dtail);gradients[i]=g
        elif s<te:values[i]=-np.inf;gradients[i]=np.nan
        else:
            desc=_desc(phi+td,phi+s,len(rate));idx=int(math.floor((phi+s)%len(rate)));f,g=_F(min(te,s-td),edges,p,nodes,weights,td);values[i]=math.log(rate[idx])-_iv(rate,prefix,desc)+math.log(f);gradients[i]=j[idx]/rate[idx]+g/f
            for k in range(j.shape[1]):gradients[i,k]-=_ivg(j,gp,desc,k)
    return values,gradients


def conditional_from_profile(profile,gaps,survival=False):
    x=np.asarray(gaps,float).reshape(-1);g=profile['geometry']
    if g['td']>=g['te']:
        return _cached_plain_gaps(profile['rates'],profile['design'],profile['prefix'],profile['gprefix'],g['phases'][0],x,g['td'],survival)
    return _cached_hidden_gaps(profile['rates'],profile['design'],profile['prefix'],profile['gprefix'],g['phases'][0],x,g['td'],g['te'],g['edges'],g['indices'],profile['profile'],profile['latent'],profile['nodes'],profile['weights'],survival)


@njit(cache=True)
def _cached_plain_gaps(rate,j,prefix,gp,phi,gaps,td,survival):
    values=np.zeros(len(gaps));gradients=np.zeros((len(gaps),j.shape[1]))
    for i in range(len(gaps)):
        s=gaps[i]
        if s<td:
            if not survival:values[i]=-np.inf;gradients[i]=np.nan
            continue
        d=_desc(phi+td,phi+s,len(rate));values[i]=-_iv(rate,prefix,d)
        for k in range(j.shape[1]):gradients[i,k]=-_ivg(j,gp,d,k)
        if not survival:
            idx=int(math.floor((phi+s)%len(rate)));values[i]+=math.log(rate[idx]);gradients[i]+=j[idx]/rate[idx]
    return values,gradients
