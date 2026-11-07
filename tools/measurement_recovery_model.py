"""Conditional recorded-gap density with independent random electronics recovery.

Detector recovery d is fixed. Electronics recovery R has a normal law truncated
to [d,2d]. At most one hidden avalanche is possible. The likelihood conditions
on the first recorded event; it includes every subsequent observed gap.
"""
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.integrate import cumulative_trapezoid
from scipy.io import loadmat


class Pulse:
    def __init__(self,path,period,bins=128):
        raw=loadmat(path);mass=raw['smoothPulseShape'].ravel().astype(float)
        mass=np.maximum(mass-mass.min(),0);mass/=mass.sum()
        dt=float(raw['binRes'].item())*1e9
        self.edges=np.arange(len(mass)+1)*dt-np.dot(np.arange(len(mass))+.5,mass)*dt
        self.cdf=np.r_[0.,np.cumsum(mass)];self.period=period;self.bins=bins
        self.width=period/bins

    def rate(self,x):
        a,tau,b=np.exp(x[0]),x[1],np.exp(x[2]);f=np.zeros(self.bins)
        edges=np.arange(self.bins+1)*self.width-tau
        for j in [-1,0,1]:f+=np.diff(np.interp(edges+j*self.period,self.edges,self.cdf,left=0,right=1))
        assert abs(f.sum()-1)<1e-10
        return (a*f+b/self.bins)/self.width


def primitive(rate,t,period):
    k=len(rate);width=period/k;x=np.asarray(t)/width
    cycle=np.floor(x/k);phase=x-cycle*k;index=np.minimum(phase.astype(np.int64),k-1)
    prefix=np.r_[0.,np.cumsum(rate)*width]
    return cycle*prefix[-1]+prefix[index]+rate[index]*(phase-index)*width


def instantaneous(rate,t,period):
    x=np.remainder(t,period)*len(rate)/period
    return rate[np.minimum(x.astype(np.int64),len(rate)-1)]


def recovery_cdf(t,mu,sd,d):
    den=ndtr((2*d-mu)/sd)-ndtr((d-mu)/sd)
    return np.clip((ndtr((np.asarray(t)-mu)/sd)-ndtr((d-mu)/sd))/den,0,1)


class GapLikelihood:
    def __init__(self,phase,gaps,period,td,step=.125):
        self.phase=np.asarray(phase);self.gaps=np.asarray(gaps)
        self.period=period;self.td=td
        assert len(phase)==len(gaps) and np.all(gaps>=td)
        self.grid=np.linspace(td,2*td,int(round(td/step))+1)
        self.v=self.phase[:,None]+self.grid
        self.u=np.minimum(2*td,self.gaps-td)
        z=np.clip((self.u-td)/(self.grid[1]-self.grid[0]),0,len(self.grid)-1)
        self.index=np.minimum(np.floor(z).astype(int),len(self.grid)-2)
        self.frac=z-self.index;self.row=np.arange(len(phase))

    def components(self,rate):
        q=self.phase;s=self.gaps;d=self.td;p=self.period
        logg=np.log(instantaneous(rate,q+s,p))-(primitive(rate,q+s,p)-primitive(rate,q+d,p))
        h=instantaneous(rate,self.v,p)*np.exp(primitive(rate,self.v+d,p)-primitive(rate,self.v,p))
        return logg,h

    def logdensity(self,rate,mu,sd,hard=False):
        logg,h=self.components(rate);G=recovery_cdf(self.gaps,mu,sd,self.td) if not hard else (self.gaps>=mu).astype(float)
        gv=recovery_cdf(self.grid,mu,sd,self.td) if not hard else (self.grid>=mu).astype(float)
        # Integration domain follows the observed gap. Negative domains have
        # zero hidden-path integral. The trapezoid mesh is checked by refinement.
        integrand=h*(G[:,None]-gv[None,:])
        cumulative=cumulative_trapezoid(integrand,self.grid,axis=1,initial=0)
        integral=cumulative[self.row,self.index]*(1-self.frac)+cumulative[self.row,self.index+1]*self.frac
        integral=np.where(self.u>self.td,integral,0)
        factor=G+np.maximum(integral,0)
        with np.errstate(divide='ignore'):return logg+np.log(factor)


def raw_pairs(path):
    raw=loadmat(path);phase=raw['detTime'].ravel().astype(float)
    frame=raw['frameNum'].ravel().astype(np.int64);period=float(raw['tr'].item())*1e9
    gaps=np.diff(frame)*period+np.diff(phase)
    assert np.all(gaps>0)
    return phase[:-1]%period,gaps,period,dict(events=len(phase),pairs=len(gaps),holdoff_ns=float(raw['T_holdoff'].item()),optical_density=float(raw['OD'].item()),raw_resolution_ns=float(raw['binRes'].item())*1e9)
