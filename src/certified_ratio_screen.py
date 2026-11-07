"""Certified reject-only bounds for the FIXED sparse-ratio algorithms.

The selected numerator subset and its original weights stay fixed. A sparse
numerator upper and a feasible null likelihood lower bound dominate every
subsequent finite-cover ratio lower. This does not bound an unevaluated FULL
mixture, nor authorize changing selection/precision after rejection.
"""
from fractions import Fraction
import numpy as np
from flint import arb
from envelope_evidence import exact_gaussian,certified_log_likelihood,outward_float
from joint_calibration_model import fit_training
from conditional_calibration import nuisance_confidence,predictive_log_interval,calibration_upper,saturated_binomial
from calibration_confidence import window_counts
from fixed_irf_control import fixed_proposal,fixed_components
from shoulder_alternative import numeric_and_sparse
from joint_calibration_envelope import shared_response_cover


def sparse_numerator_upper(h,models,indices,cycles=1000):
    indices=list(map(int,indices))
    if not indices or len(set(indices))!=len(indices):raise ValueError('Unique nonempty selected subset required')
    terms=[]
    for i in indices:
        w=models['weights'][i]
        if not isinstance(w,Fraction) or w<=0:raise ValueError('Positive exact original weights required')
        ell=certified_log_likelihood(h,models['rates'][i],cycles,'first',upper=True)
        terms.append(arb(ell)+(arb(w.numerator)/w.denominator).log())
    shift=max(float(x) for x in terms)
    value=arb(shift)+sum(((x-arb(shift)).exp() for x in terms),arb(0)).log()
    return outward_float(value)


def single_irf_witness(h,q,eta,cycles=1000):
    bins=len(h);d=float(q['primary']);w=float(q['fitted_fwhm'])
    if not(0<=eta<=.25 and 0<=d<=bins-1 and 4.5<=w<=5.5):raise ValueError('Invalid null witness')
    g=exact_gaussian(bins,d,w,False);side=exact_gaussian(bins+16,arb(d)+16,w,False)[:bins]
    a=arb(float(q['primary_incident']))/sum(g,arb(0));b=arb(float(q['background_incident']))
    if not(a>=0 and b>=0):raise ValueError('Negative witness amplitude')
    rates=[(a*(x+arb(float(eta))*y)+b/bins)/cycles for x,y in zip(g,side)]
    return dict(lower=certified_log_likelihood(h,rates,cycles,'first'),eta=float(eta),
                primary=d,width=w,kind='Donor point with secondary amplitude set to zero')


def calibration_box_witness(k,r,eta,cf):
    k,r=int(k),int(r);fail=r-k
    if not 0<=k<=r:raise ValueError('Invalid calibration count/risk')
    if cf['informative']:
        f=(arb(cf['forward_leakage_lower'])+arb(cf['forward_leakage_upper']))/2
        factor=arb(float(eta))+f
        low=arb(cf['signal_main_lower'])*factor+11*arb(cf['background_lower'])
        high=arb(cf['signal_main_upper'])*factor+11*arb(cf['background_upper'])
        # This interval is attainable by nonnegative nuisance values in the
        # same relaxed confidence BOX covered by the joint-null calculation.
        if fail==0:rate=high
        else:
            mode=(arb(k)/fail).log1p() if k else arb(0)
            if mode<low:rate=low
            elif mode>high:rate=high
            elif mode>=low and mode<=high:rate=mode
            else:rate=(low+high)/2
        kind='Feasible linear nuisance-box subinterval at fixed forward leakage'
    else:
        rate=(arb(k)/fail).log1p() if fail and k else arb(0) if not k else arb(100)
        kind='Finite feasible point in the uninformative saturated-calibration family'
    phi=arb(0)
    if k:phi+=k*(-(-rate).expm1()).log()
    phi-=fail*rate
    return dict(lower=outward_float(phi,False),kind=kind)


def average_log(values,upper=False):
    shift=max(values)
    value=arb(shift)+(sum(((arb(x)-arb(shift)).exp() for x in values),arb(0))/len(values)).log()
    return outward_float(value,upper)


def prepare_fold(h,train,test,ntrain,ntest):
    fitted=fit_training(train,ntrain);confidence=nuisance_confidence(train,ntrain)
    kt,rt=window_counts(train,ntrain,43,53);k,r=window_counts(test,ntest,43,53)
    calnum=predictive_log_interval(kt,rt,k,r);proposal=fixed_proposal(h,fitted['eta'])
    models=fixed_components(proposal,fitted['eta']);qnum=numeric_and_sparse(h[4,4],models)
    q_upper=sparse_numerator_upper(h[4,4],models,qnum['indices'])
    jointnum=outward_float(arb(calnum['lower'])+arb(qnum['lower']),False)
    num_upper=outward_float(arb(calnum['upper'])+arb(q_upper))
    cw=calibration_box_witness(k,r,fitted['eta'],confidence);qw=single_irf_witness(h[4,4],proposal,fitted['eta'])
    witness=outward_float(arb(cw['lower'])+arb(qw['lower']),False)
    upper=outward_float(arb(num_upper)-arb(witness))
    return dict(training_fit=fitted,nuisance_confidence=confidence,training_lobe_count=kt,training_lobe_risk=rt,
        test_lobe_count=k,test_lobe_risk=r,calibration_numerator=calnum,query_numerator=qnum,proposal=proposal,
        joint_numerator_lower=jointnum,calibration_predictive_minus_saturated=outward_float(arb(calnum['lower'])-saturated_binomial(k,r),False),
        screen=dict(query_sparse_numerator_upper=q_upper,joint_numerator_upper=num_upper,
            calibration_witness=cw,query_witness=qw,joint_null_witness_lower=witness,log_ratio_upper=upper))


def finish_fold(h,fold):
    cf=fold['nuisance_confidence'];k,r=fold['test_lobe_count'],fold['test_lobe_risk']
    threshold=outward_float((arb(2000)/9).log());level=outward_float(arb(fold['joint_numerator_lower'])-arb(threshold),False)
    provider=lambda e0,e1:calibration_upper(k,r,e0,e1,cf)
    bound=shared_response_cover(h[4,4],provider,decision_level=level,max_boxes=1024)
    fold.update(bound=bound,log_e_lower=outward_float(arb(fold['joint_numerator_lower'])-arb(bound['upper']),False))
    return fold


def score_pair(h,a,b,na,nb):
    folds=[prepare_fold(h,a,b,na,nb),prepare_fold(h,b,a,nb,na)]
    upper=average_log([f['screen']['log_ratio_upper'] for f in folds],True)
    cut_lower=outward_float((arb(1000)/9).log(),False);cut_upper=outward_float((arb(1000)/9).log())
    if upper<cut_lower:
        return dict(folds=folds,report=dict(declared=False,log_ratio_lower=None,log_ratio_upper=upper,
            log_threshold_upper=cut_upper,primary=None,secondary=None,head='certified_screen',reported_fold=None),
            certified_screened_out=True,screen_preserves_fixed_sparse_decisions=True)
    folds=[finish_fold(h,f) for f in folds];lower=average_log([f['log_e_lower'] for f in folds])
    winner=int(np.argmax([f['log_e_lower'] for f in folds]));f=folds[winner]
    return dict(folds=folds,report=dict(declared=lower>=cut_upper,log_ratio_lower=lower,log_ratio_upper=upper,
        log_threshold_upper=cut_upper,primary=f['proposal']['primary'],secondary=f['query_numerator']['location'],
        head=f['query_numerator']['head'],reported_fold=winner),certified_screened_out=False,
        screen_preserves_fixed_sparse_decisions=True)
