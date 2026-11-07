"""Ordered predictive reuse of calibration, with one confidence-set budget.

The query predictor may see ALL calibration because the independent query is
integrated first. The calibration predictor still excludes its scored count.
One full-calibration nuisance confidence set is used only in the denominator;
on its coverage event the ratio is pointwise dominated by the true-parameter
predictive ratio. This is NOT an unconditional e-value claim.
"""
from fractions import Fraction
from flint import arb
from joint_calibration_model import fit_training
from conditional_calibration import nuisance_confidence,predictive_log_interval,saturated_binomial,calibration_upper
from calibration_confidence import window_counts
from fixed_irf_control import fixed_proposal,fixed_components,fixed_null_bound
from shoulder_alternative import numeric_and_sparse
from shoulder_envelope import shoulder_null_bound
from certified_ratio_screen import sparse_numerator_upper,single_irf_witness,calibration_box_witness,average_log
from joint_calibration_envelope import shared_response_cover
from envelope_evidence import outward_float


def query_predictor(h,cal,cycles):
    fitted=fit_training(cal,cycles)
    fitted.update(scored_calibration_excluded=False,query_excluded=True,
        usage='Query predictor only: full calibration allowed by ordered normalization; not used to train the scored calibration predictor')
    proposal=fixed_proposal(h,fitted['eta']);models=fixed_components(proposal,fitted['eta'])
    num=numeric_and_sparse(h[4,4],models)
    return dict(fit=fitted,proposal=proposal,numerator=num,
        numerator_upper=sparse_numerator_upper(h[4,4],models,num['indices']))


def interval_score(query,predictor,cap,max_boxes=2048,point=False):
    """Strong crossed control: point-trained query Q with interval null."""
    eta=predictor['fit']['eta'];cutvalue=arb(100).log() if point else (arb(1000)/9).log()
    cut=outward_float(cutvalue);cutlow=outward_float(cutvalue,False)
    witness=single_irf_witness(query,predictor['proposal'],eta if point else min(eta,cap))
    upper=outward_float(arb(predictor['numerator_upper'])-arb(witness['lower']))
    report=dict(declared=False,log_ratio_upper=upper,log_ratio_lower=None,log_threshold_upper=cut,
        primary=predictor['proposal']['primary'],secondary=predictor['numerator']['location'],head=predictor['numerator']['head'])
    result=dict(certified_screened_out=upper<cutlow,null_witness=witness,bound=None,report=report)
    if upper<cutlow:return result
    level=outward_float(arb(predictor['numerator']['lower'])-arb(cut),False)
    bound=(fixed_null_bound(query,eta,decision_level=level,max_boxes=max_boxes) if point else
        shoulder_null_bound(query,cap=cap,decision_level=level,max_boxes=max_boxes))
    lower=outward_float(arb(predictor['numerator']['lower'])-arb(bound['upper']),False)
    result['bound']=bound;report.update(declared=lower>=cut,log_ratio_lower=lower)
    return result


def score_ordered(h,cal,ncal,a,b,na,nb,predictor=None,max_boxes=4096):
    pred=query_predictor(h,cal,ncal) if predictor is None else predictor
    cf=nuisance_confidence(cal,ncal,delta=Fraction(1,1000))
    cf.update(scope='ONE full-calibration confidence event, shared by both folds; denominator only')
    folds=[];eta=pred['fit']['eta'];q=pred['proposal'];num=pred['numerator']
    qw=single_irf_witness(h[4,4],q,eta)
    for train,test,nt,nv in [(a,b,na,nb),(b,a,nb,na)]:
        kt,rt=window_counts(train,nt,43,53);k,r=window_counts(test,nv,43,53)
        cn=predictive_log_interval(kt,rt,k,r)
        lower=outward_float(arb(cn['lower'])+arb(num['lower']),False)
        upper=outward_float(arb(cn['upper'])+arb(pred['numerator_upper']))
        cw=calibration_box_witness(k,r,eta,cf)
        witness=outward_float(arb(cw['lower'])+arb(qw['lower']),False)
        folds.append(dict(training_lobe_count=kt,training_lobe_risk=rt,test_lobe_count=k,test_lobe_risk=r,
            calibration_numerator=cn,joint_numerator_lower=lower,
            calibration_predictive_minus_saturated=outward_float(arb(cn['lower'])-saturated_binomial(k,r),False),
            screen=dict(calibration_witness=cw,query_witness=qw,log_ratio_upper=outward_float(arb(upper)-arb(witness)))))
    upper=average_log([f['screen']['log_ratio_upper'] for f in folds],True)
    cut=outward_float((arb(1000)/9).log());cutlow=outward_float((arb(1000)/9).log(),False)
    report=dict(declared=False,log_ratio_upper=upper,log_ratio_lower=None,log_threshold_upper=cut,
        primary=q['primary'],secondary=num['location'],head=num['head'])
    result=dict(predictor=pred,nuisance_confidence=cf,folds=folds,certified_screened_out=upper<cutlow,report=report)
    if upper<cutlow:return result
    for fold in folds:
        k,r=fold['test_lobe_count'],fold['test_lobe_risk']
        provider=lambda e0,e1:calibration_upper(k,r,e0,e1,cf)
        level=outward_float(arb(fold['joint_numerator_lower'])-(arb(2000)/9).log(),False)
        bound=shared_response_cover(h[4,4],provider,decision_level=level,max_boxes=max_boxes)
        fold.update(bound=bound,log_ratio_lower=outward_float(arb(fold['joint_numerator_lower'])-arb(bound['upper']),False))
    lower=average_log([f['log_ratio_lower'] for f in folds])
    report.update(declared=lower>=cut,log_ratio_lower=lower)
    return result
