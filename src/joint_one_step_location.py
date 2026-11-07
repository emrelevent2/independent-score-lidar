"""Prototype joint Fisher one-step LOCATION estimator,not a new test.

Starts at the local projected-score secondary location and a query null
primary; profiles nonnegative amplitudes at those fixed centers,then takes
one damped five-parameter Fisher step. This adapts established one-step
estimation. No study has established reduced bias/error or superiority.
Detection scores,p-values and their calibration are not changed here.
"""
import numpy as np
from scipy.optimize import minimize
from deadtime_scan import pulse,likelihood_parts
from deadtime_path_bootstrap import fine_scan


def initial_amplitudes(s,e,sigma,null_fit,secondary):
    k=len(s);a,t,b=null_fit['theta'];f,_=pulse(t,k,sigma);g,_=pulse(secondary,k,sigma);design=np.column_stack([f,np.ones(k)/k,g]);scale=max(1.,sum(s))
    def objective(x):
        ll,u,_=likelihood_parts(s,e,design@x,'duration');return -ll/scale,-design.T@u/scale
    fit=minimize(objective,[a,b,.02],jac=True,method='L-BFGS-B',bounds=[(0,20),(1e-6,30),(0,20)],options=dict(ftol=1e-12,gtol=1e-8,maxiter=160))
    a,b,c=fit.x;return np.array([a,t,b,c,secondary]),dict(success=bool(fit.success),loglik=-float(fit.fun)*scale)


def likelihood_and_fisher(s,e,theta,sigma):
    a,t,b,c,d=theta;k=len(s);f,df=pulse(t,k,sigma);g,dg=pulse(d,k,sigma);rate=np.maximum(a*f+b/k+c*g,1e-14);ll,u,w=likelihood_parts(s,e,rate,'duration');j=np.column_stack([f,a*df,np.ones(k)/k,g,c*dg]);return ll,j.T@u,j.T@(w[:,None]*j)


def estimate(s,e,sigma,null_fit):
    s=np.asarray(s,float);e=np.asarray(e,float);k=len(s);pilot=fine_scan(s,e,sigma,fit=null_fit);initial,fit=initial_amplitudes(s,e,sigma,null_fit,pilot['delay']);ll,score,information=likelihood_and_fisher(s,e,initial,sigma);delta=np.linalg.pinv(information,rcond=1e-10)@score;selected=initial.copy();newll=ll;backtracks=21
    for n in range(21):
        candidate=initial+delta*2.**(-n);candidate[[0,3]]=np.clip(candidate[[0,3]],0,20);candidate[2]=np.clip(candidate[2],1e-6,30);candidate[1]=np.clip(candidate[1],initial[1]-3*sigma,initial[1]+3*sigma);candidate[4]=np.clip(candidate[4],initial[4]-3*sigma,initial[4]+3*sigma);v,_,_=likelihood_and_fisher(s,e,candidate,sigma)
        if v>=ll-1e-10:selected=candidate;newll=v;backtracks=n;break
    primary,secondary=selected[1]%k,selected[4]%k
    if selected[3]>selected[0]:primary,secondary=secondary,primary
    return dict(primary=float(primary),delay=float(secondary),theta=selected.tolist(),initial_theta=initial.tolist(),initial_loglik=float(ll),loglik=float(newll),backtracks=backtracks,amplitude_optimizer_success=fit['success'],pilot_delay=float(pilot['delay']),fisher_rank=int(np.linalg.matrix_rank(information)),detection_statistic_unchanged=True)
