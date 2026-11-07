"""Selected numerical-witness diagnostic; no new accuracy/efficacy claim."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.optimize import minimize,nnls
from scipy.special import xlogy
from flint import arb
import conditional_joint_confirmation as s
from conditional_calibration import saturated_binomial
from pilot import coates
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float

ROOT=s.ROOT;P=ROOT/'JOINT_BOUND_DIAGNOSTIC_PROTOCOL.json';OUT=ROOT/'results/joint_bound_diagnostic_v1'


def selections():
    return [r['id'] for r in s.read_records() if r['case']['secondary']>0 and r['reports']['background_ci']['declared'] and
        abs(r['reports']['background_ci']['primary']-r['truth']['primary'])<=2 and abs(r['reports']['background_ci']['secondary']-r['truth']['secondary'])<=2 and not r['reports']['joint']['declared']]


def freeze():
    if P.exists():raise FileExistsError('Immutable diagnostic')
    parent=s.check_protocol();sources=dict(parent['sources']);sources['src/joint_bound_diagnostic.py']=s.bgdev.original.sha(Path(__file__))
    p=dict(status='FROZEN_BEFORE_NUMERICAL_WITNESS_FITS',sources=sources,selected_ids=selections(),
        parent_protocol_sha256=s.bgdev.original.sha(s.P),parent_sqlite_sha256=s.bgdev.original.sha(s.OUT/'records.sqlite3'),
        purpose='Only CI-correct/joint-nondeclared positives from failed prospective study. Find certified feasible null LOWERs to bound how much further null-cover refinement could improve the SAMEfixed sparse statistic.',
        settings=dict(eta_starts=[0.,.0625,.125,.25,'training_eta'],query_delay_starts=['donor_primary','raw_max_bin'],width_starts=['donor_width'],optimizer_maxiter=500),
        inference='Outcome-selected numerical diagnosis only. Local optimizer does NOT certify a global maximum; its point is certified feasible. Upper ratio below cutoff rules out improvement by NULL refinement alone; upper above cutoff is merely unresolved.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',s.bgdev.original.sha(P),len(p['selected_ids']),'selected misses',flush=True)


def cal_profile(eta,k,r,cf):
    if not cf['informative']:return 0.,0.
    fail=r-k;lo=cf['signal_main_lower']*(eta+cf['forward_leakage_lower'])+11*cf['background_lower']
    hi=cf['signal_main_upper']*(eta+cf['forward_leakage_upper'])+11*cf['background_upper']
    mode=np.log1p(k/fail) if fail else np.inf;rate=np.clip(mode,lo,hi)
    value=float(xlogy(k,-np.expm1(-rate))-fail*rate)
    derivative=k/np.expm1(rate)-fail if rate>0 else -fail
    slope=cf['signal_main_lower'] if mode<lo else cf['signal_main_upper'] if mode>hi else 0.
    return value,derivative*slope


def cal_feasible_lower(eta,k,r,cf):
    fail=r-k
    if not cf['informative']:rate=(arb(k)/fail).log1p() if k and fail else arb(0) if not k else arb(100)
    else:
        low=arb(cf['signal_main_lower'])*(arb(eta)+arb(cf['forward_leakage_lower']))+11*arb(cf['background_lower'])
        high=arb(cf['signal_main_upper'])*(arb(eta)+arb(cf['forward_leakage_upper']))+11*arb(cf['background_upper'])
        if not fail:rate=high
        else:
            mode=(arb(k)/fail).log1p() if k else arb(0)
            if mode<low:rate=low
            elif mode>high:rate=high
            elif mode>=low and mode<=high:rate=mode
            else:rate=(low+high)/2
    value=arb(0)
    if k:value+=k*(-(-rate).expm1()).log()
    value-=fail*rate
    return outward_float(value,False)


def fit_witness(query,fold):
    t=np.arange(256);remaining=1000-np.cumsum(query);constant=4*np.log(2)
    k,r=fold['test_lobe_count'],fold['test_lobe_risk'];cf=fold['nuisance_confidence']
    def pulse(d,w):
        return np.exp(-constant*((t-d)/w)**2),np.exp(-constant*((t-(d+16))/w)**2)
    def loss(z):
        a,b,d,w,eta=z;g,side=pulse(d,w);rate=(a*(g+eta*side)+b/256)/1000
        q=float(np.sum(xlogy(query,-np.expm1(-rate))-remaining*rate));c,cgrad=cal_profile(eta,k,r,cf)
        with np.errstate(over='ignore'):derivative=query/np.expm1(rate)-remaining
        gd=2*constant*(t-d)/w**2*g;sd=2*constant*(t-(d+16))/w**2*side
        gw=2*constant*(t-d)**2/w**3*g;sw=2*constant*(t-(d+16))**2/w**3*side
        jac=[g+eta*side,np.ones(256)/256,a*(gd+eta*sd),a*(gw+eta*sw),a*side]
        grad=np.array([np.sum(derivative*x/1000) for x in jac]);grad[-1]+=cgrad
        return -q-c,-grad
    fits=[];obs=coates(query,1000)*1000
    for eta in sorted(set([0.,.0625,.125,.25,fold['training_fit']['eta']])):
        for d in sorted(set([fold['proposal']['primary'],float(np.argmax(query))])):
            w=fold['proposal']['fitted_fwhm'];g,side=pulse(d,w)
            coef,_=nnls(np.stack([g+eta*side,np.ones(256)/256],axis=1),obs)
            init=np.r_[max(coef[0],1e-8),max(coef[1],1e-8),d,w,eta]
            fit=minimize(loss,init,jac=True,method='L-BFGS-B',bounds=[(0,None),(1e-8,None),(0,255),(4.5,5.5),(0,.25)],options={'ftol':1e-12,'gtol':1e-8,'maxiter':500})
            fits.append(fit)
    best=min(fits,key=lambda f:f.fun);a,b,d,w,eta=map(float,best.x)
    if not(a>=0 and b>=0 and 0<=d<=255 and 4.5<=w<=5.5 and 0<=eta<=.25):raise ArithmeticError('Infeasible optimizer output')
    g=exact_gaussian(256,d,w,False);side=exact_gaussian(272,arb(d)+16,w,False)[:256]
    rates=[(arb(a)*(x+arb(eta)*y)+arb(b)/256)/1000 for x,y in zip(g,side)]
    qlower=certified_log_likelihood(query,rates,1000,'first');clower=cal_feasible_lower(eta,k,r,cf)
    lower=outward_float(arb(qlower)+arb(clower),False)
    assert lower<=fold['bound']['upper']
    upper=outward_float(arb(fold['screen']['joint_numerator_upper'])-arb(lower))
    return dict(parameters=[a,b,d,w,eta],feasible_joint_loglik_lower=lower,
        null_bound_gap_upper=outward_float(arb(fold['bound']['upper'])-arb(lower)),
        fixed_sparse_ratio_log_upper=upper,optimizer_success=bool(best.success),starts=len(fits))


def run():
    p=json.loads(P.read_text());s.check_protocol()
    for path,digest in p['sources'].items():assert s.bgdev.original.sha(ROOT/path)==digest
    assert s.bgdev.original.sha(s.OUT/'records.sqlite3')==p['parent_sqlite_sha256']
    rows={r['id']:r for r in s.read_records()};start=time.perf_counter();results=[]
    for id in p['selected_ids']:
        row=rows[id];joint=row['methods']['joint'];decomposition=[]
        for f in joint['folds']:
            cs=float(saturated_binomial(f['test_lobe_count'],f['test_lobe_risk']))
            decomposition.append(dict(cal_predictive_minus_saturated=f['calibration_numerator']['lower']-cs,
                query_numerator_minus_ci=f['query_numerator']['lower']-row['methods']['background_ci']['numerator']['lower'],
                equivalent_null_upper_minus_ci=None if 'bound' not in f else f['bound']['upper']-cs-row['methods']['background_ci']['bound']['upper']))
        if joint['certified_screened_out']:
            results.append(dict(id=id,already_ruled_out_by_certified_screen=True,decomposition=decomposition));continue
        folds=[fit_witness(np.array(row['query']),f) for f in joint['folds']]
        upper=s.development.method.outward_float(arb(max(f['fixed_sparse_ratio_log_upper'] for f in folds))+
            (sum(((arb(f['fixed_sparse_ratio_log_upper'])-arb(max(x['fixed_sparse_ratio_log_upper'] for x in folds))).exp() for f in folds),arb(0))/2).log())
        results.append(dict(id=id,folds=folds,decomposition=decomposition,log_average_ratio_upper=upper,
            further_null_refinement_ruled_out=upper<outward_float((arb(1000)/9).log(),False)))
        print('Witness diagnostic',len(results),'of',len(p['selected_ids']),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(status='NUMERICAL_WITNESS_DIAGNOSTIC_COMPLETE',records=results,seconds=time.perf_counter()-start,protocol_sha256=s.bgdev.original.sha(P),scope=p['inference'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);a=parser.parse_args()
    if a.action=='freeze':freeze()
    else:run()
