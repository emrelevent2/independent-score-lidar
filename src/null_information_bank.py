"""Fixed independent null-information score weights,not exact finite Fisher.

Convex log-flux/fractional-phase interpolation preserves PSD. Integer phase
rotation approximates the finite initially-active law;it is NOT asserted to
be exact. Complete query/bootstrap likelihoods retain the original physics.
"""
from pathlib import Path
import json
import numpy as np


class NullInformationBank:
    def __init__(self,information,amplitudes,backgrounds,fractions,regimes,base_phase_integer=15):
        self.h=np.asarray(information,float);self.a=np.asarray(amplitudes,float);self.b=np.asarray(backgrounds,float);self.f=np.asarray(fractions,float);self.regimes=list(regimes);self.origin=int(base_phase_integer)
        assert self.h.ndim==6 and self.h.shape[:4]==(len(regimes),self.a.shape[1],self.b.shape[1],len(self.f)) and self.h.shape[-2]==self.h.shape[-1]
        assert self.a.shape[0]==self.b.shape[0]==len(regimes) and np.all(np.diff(self.a,axis=1)>0) and np.all(np.diff(self.b,axis=1)>0) and np.all(self.a>0) and np.all(self.b>0)
        assert self.f[0]==0 and np.all(np.diff(self.f)>0) and self.f[-1]<1 and np.all(np.isfinite(self.h))

    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as f:
            return cls(f['information'],f['amplitudes'],f['backgrounds'],f['fractions'],f['regimes'].tolist(),int(f['base_phase_integer']))

    @staticmethod
    def bracket(grid,value):
        bounded=float(np.clip(value,grid[0],grid[-1]));i=min(len(grid)-2,max(0,int(np.searchsorted(grid,bounded,side='right')-1)));weight=float((np.log(bounded)-np.log(grid[i]))/(np.log(grid[i+1])-np.log(grid[i])))
        return i,weight,bool(value<grid[0] or value>grid[-1])

    def lookup(self,theta,regime):
        theta=np.asarray(theta,float)
        if theta.shape!=(3,) or np.any(~np.isfinite(theta)):raise ValueError('Finite a,tau,B required')
        r=self.regimes.index(int(regime));ai,aw,ac=self.bracket(self.a[r],theta[0]);bi,bw,bc=self.bracket(self.b[r],theta[2]);k=self.h.shape[-1];phase=float(theta[1]%k);whole=int(np.floor(phase));fraction=phase-whole;fi=min(len(self.f)-1,int(np.searchsorted(self.f,fraction,side='right')-1));next_f=self.f[fi+1] if fi+1<len(self.f) else 1.;fw=float((fraction-self.f[fi])/(next_f-self.f[fi]));result=np.zeros((k,k))
        for offset,phaseweight in [(0,1-fw),(1,fw)]:
            raw_index=fi+offset;index=raw_index%len(self.f);shift=whole-self.origin+(1 if raw_index==len(self.f) else 0);matrix=np.zeros((k,k))
            for da,wa in [(0,1-aw),(1,aw)]:
                for db,wb in [(0,1-bw),(1,bw)]:matrix+=wa*wb*self.h[r,ai+da,bi+db,index]
            result+=phaseweight*np.roll(matrix,(shift,shift),axis=(0,1))
        result=(result+result.T)/2
        return result,dict(amplitude_clamped=ac,background_clamped=bc,regime=int(regime),fractional_phase=fraction)
