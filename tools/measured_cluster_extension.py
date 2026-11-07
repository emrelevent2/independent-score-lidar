"""Recovery and clustered-transition working model; evaluate on untouched segments.

The extra component is a normalized transition law, not a claim to identify
afterpulsing. All original recovery-only fits and their diagnostics are retained.
"""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from measurement_recovery_model import raw_pairs,GapLikelihood,primitive,instantaneous,recovery_cdf
from measurement_recovery_predictive import predict
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/measured_cluster_v1';P=ROOT/'MEASURED_CLUSTER_PROTOCOL_20261107.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def burst_logdensity(s,mu,sd,d,tau):
    s=np.asarray(s);normalizer=ndtr((2*d-mu)/sd)-ndtr((d-mu)/sd)
    mass=ndtr((np.minimum(s,2*d)-mu-sd*sd/tau)/sd)-ndtr((d-mu-sd*sd/tau)/sd)
    with np.errstate(divide='ignore'):
        return (mu-s)/tau+sd*sd/(2*tau*tau)+np.log(mass)-np.log(tau*normalizer)


class ScaledLikelihood:
    def __init__(self,phase,gaps,period,rate,mu,sd,step=.125):
        self.gaps=np.asarray(gaps);self.mu=mu;self.sd=sd;self.d=50.
        geometry=GapLikelihood(phase,gaps,period,50.,step);v=geometry.v
        self.A=primitive(rate,phase+gaps,period)-primitive(rate,phase+50.,period)
        self.logevent=np.log(instantaneous(rate,phase+gaps,period))
        self.D=primitive(rate,v+50.,period)-primitive(rate,v,period)
        self.G=recovery_cdf(gaps,mu,sd,50.)
        j=np.arange(len(geometry.grid))[None,:];i=geometry.index[:,None];frac=geometry.frac[:,None]
        weights=np.where(j<i,1.,0.)
        weights[:,0]*=.5
        weights+=np.where(j==i,(1+frac)*.5,0.)+np.where(j==i+1,frac*.5,0.)
        # At u below the first complete cell there is no preceding interval.
        weights[:,0]=np.where(geometry.index==0,geometry.frac*.5,weights[:,0])
        weights*=geometry.grid[1]-geometry.grid[0];weights[geometry.u<=50.]=0
        self.coef=weights*(self.G[:,None]-recovery_cdf(geometry.grid,mu,sd,50.)[None,:])*instantaneous(rate,v,period)

    def physical(self,q,gradient=False):
        t=q*self.coef*np.exp(q*self.D);factor=self.G+t.sum(axis=1)
        value=np.log(q)+self.logevent-q*self.A+np.log(factor)
        if gradient:return value,1-q*self.A+(t*(1+q*self.D)).sum(axis=1)/factor
        return value

    def mixture(self,x,gradient=False):
        q,pi,tau=np.exp(x[0]),x[1],np.exp(x[2])
        lf,dq=self.physical(q,True);lb=burst_logdensity(self.gaps,self.mu,self.sd,self.d,tau)
        l=np.logaddexp(np.log1p(-pi)+lf,np.log(pi)+lb)
        if gradient:
            w=np.exp(np.log1p(-pi)+lf-l);epsilon=1e-5
            bp=burst_logdensity(self.gaps,self.mu,self.sd,self.d,tau*np.exp(epsilon));bm=burst_logdensity(self.gaps,self.mu,self.sd,self.d,tau*np.exp(-epsilon))
            grad=np.stack([w*dq,np.exp(lb-l)-np.exp(lf-l),(np.exp(np.log(pi)+bp-l)-np.exp(np.log(pi)+bm-l))/(2*epsilon)],axis=1)
            return l,grad
        return l


def freeze():
    assert not P.exists();base=ROOT/'results/measured_recovery_v1/analysis.json';pred=ROOT/'results/measured_recovery_v1/predictive.json'
    p=dict(status='FIXED_BEFORE_NEW_HELDOUT_EVALUATION',source_sha256=sha(__file__),recovery_analysis_sha256=sha(base),earlier_diagnostic_sha256=sha(pred),
      development='Recovery-only heldout CDF distances .24269 low and .03220 high motivated a clustered-transition component. Earlier analysis and all outcomes retained. The new evaluation segment has not been used for fitting or model comparison.',
      calibration_pair_range=[100,20100],evaluation_pair_range=[40400,60400],starts=[[.75,.25,40.],[1.,.05,40.]],bounds=dict(flux_scale=[.05,2.],cluster_fraction=[.000001,.8],cluster_exponential_mean_ns=[1.,1000.]),
      model='Normalized conditional mixture (1-pi)*f_random_recovery(s|phase;q*lambda)+pi*density(R+Exp(tau)). Recovery parameters, pulse phase and signal/background ratio come from the recovery-only fit on the same calibration segment. Fit q,pi,tau only; two fixed starts and analytic/checked derivatives. Mixture is a working transition model for clustering; the cause is not identified as afterpulsing.',
      endpoints='All heldout densities and mean log-score difference versus recovery-only, descriptive predictive marginal CDF distance. Fixed sample-minimum comparator coverage on same new heldout segment. All previously shown outcomes preserved; no measured two-surface detection claim.',integration_step_ns=.125,predictions=100000,seed_base=7912611047)
    P.write_text(json.dumps(p,indent=2));print('Cluster extension frozen',sha(P),flush=True)


def run():
    p=json.loads(P.read_text());assert sha(__file__)==p['source_sha256']
    base=json.loads((ROOT/'results/measured_recovery_v1/analysis.json').read_text());assert sha(ROOT/'results/measured_recovery_v1/analysis.json')==p['recovery_analysis_sha256']
    OUT.mkdir(exist_ok=True);assert not (OUT/'analysis.json').exists();rows=[]
    for index,row in enumerate(base['rows']):
        phase,gaps,period,meta=raw_pairs(ROOT/'data/highflux_raw/2019_01_28'/row['file']);par=row['parameters'];a,b=p['calibration_pair_range'];c,e=p['evaluation_pair_range']
        original=np.load(ROOT/'results/measured_recovery_v1'/(Path(row['file']).stem+'_heldout.npz'));rate=original['rate']
        fit=ScaledLikelihood(phase[a:b],gaps[a:b],period,rate,par['recovery_location_ns'],par['recovery_scale_ns'])
        # Check the accelerated weighted-integral evaluator against the frozen
        # recovery-only criterion and verify every new objective derivative.
        n=73;plain=GapLikelihood(phase[a:a+n],gaps[a:a+n],period,50.,.125).logdensity(rate,par['recovery_location_ns'],par['recovery_scale_ns'])
        assert np.max(abs(fit.physical(1.)[:n]-plain))<1e-11
        z=np.array([np.log(.8),.2,np.log(40.)]);values,derivatives=fit.mixture(z,True)
        differences=[]
        for j in range(3):
            h=2e-5;plus=z.copy();minus=z.copy();plus[j]+=h;minus[j]-=h
            numeric=(fit.mixture(plus)-fit.mixture(minus))/(2*h);differences.append(float(np.max(abs(numeric-derivatives[:,j]))))
        assert max(differences)<1e-5,differences
        candidates=[]
        for q,pi,tau in p['starts']:
            def objective(x):
                l,g=fit.mixture(x,True);return -l.mean(),-g.mean(axis=0)
            t=time.perf_counter();r=minimize(objective,[np.log(q),pi,np.log(tau)],jac=True,method='L-BFGS-B',bounds=[tuple(np.log(p['bounds']['flux_scale'])),tuple(p['bounds']['cluster_fraction']),tuple(np.log(p['bounds']['cluster_exponential_mean_ns']))],options=dict(ftol=1e-12,gtol=1e-7,maxiter=120,maxls=30))
            candidates.append(dict(x=r.x.tolist(),negative_mean_logdensity=float(r.fun),success=bool(r.success),evaluations=r.nfev,seconds=time.perf_counter()-t));print(row['file'],'cluster fit',candidates[-1],flush=True)
        best=min(candidates,key=lambda x:x['negative_mean_logdensity']);x=np.array(best['x']);del fit
        evaluation=ScaledLikelihood(phase[c:e],gaps[c:e],period,rate,par['recovery_location_ns'],par['recovery_scale_ns']);ll=evaluation.mixture(x);baseline=evaluation.physical(1.)
        q,pi,tau=np.exp(x[0]),x[1],np.exp(x[2]);sim=predict(rate*q,phase[c:e],period,50.,par['recovery_location_ns'],par['recovery_scale_ns'],p['seed_base']+index,n=p['predictions'])
        rng=np.random.default_rng(p['seed_base']+index+111);is_cluster=rng.uniform(size=len(sim))<pi
        from scipy.special import ndtri
        low=ndtr((50-par['recovery_location_ns'])/par['recovery_scale_ns']);high=ndtr((100-par['recovery_location_ns'])/par['recovery_scale_ns'])
        R=par['recovery_location_ns']+par['recovery_scale_ns']*ndtri(low+(high-low)*rng.uniform(size=int(is_cluster.sum())))
        sim[is_cluster]=R+rng.exponential(tau,size=int(is_cluster.sum()))
        baseline_sim=predict(rate,phase[c:e],period,50.,par['recovery_location_ns'],par['recovery_scale_ns'],p['seed_base']+index+222,n=p['predictions'])
        target=OUT/(Path(row['file']).stem+'.npz');np.savez_compressed(target,phase=phase[c:e],gaps=gaps[c:e],logdensity=ll,baseline_logdensity=baseline,rate=rate,predicted=sim,baseline_predicted=baseline_sim)
        result=dict(file=row['file'],parameters=dict(flux_scale=float(q),cluster_fraction=float(pi),cluster_exponential_mean_ns=float(tau)),best=best,starts=candidates,evaluation_n=e-c,evaluation_pair_range=[c,e],all_finite=bool(np.all(np.isfinite(ll))),mean_log_score_gain=float(np.mean(ll-baseline)),cdf_distance=float(ks_2samp(gaps[c:e],sim).statistic),baseline_cdf_distance=float(ks_2samp(gaps[c:e],baseline_sim).statistic),fixed_cutoff_zero_densities=int(np.count_nonzero(gaps[c:e]<par['hard_recovery_ns'])),derivative_max_errors=differences,data_sha256=sha(target))
        rows.append(result);print('Fresh measured evaluation',json.dumps(result),flush=True)
    output=dict(status='COMPLETE',protocol_sha256=sha(P),rows=rows)
    (OUT/'analysis.json').write_text(json.dumps(output,indent=2))


if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
