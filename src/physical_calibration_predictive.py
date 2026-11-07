"""Proper training-only optical-model prediction of the scored lobe count."""
from flint import arb
from joint_calibration_model import fit_training
from conditional_calibration import predictive_log_interval
from calibration_confidence import window_counts
from envelope_evidence import outward_float


def fixed_hazard_log(k,r,rate):
    if not(0<=k<=r):raise ValueError('Invalid calibration count/risk')
    hazard=arb(rate)
    if not hazard>=0:raise ValueError('Nonnegative predictive hazard required')
    value=-(r-k)*hazard
    if k:value+=k*(-(-hazard).expm1()).log()
    return dict(lower=outward_float(value,False),upper=outward_float(value),binomial_constant_omitted=True)


def half_mixture(a,b):
    shift=max(a['upper'],b['upper'])
    def evaluate(endpoint,up):
        v=arb(shift)+(((arb(a[endpoint])-shift).exp()+(arb(b[endpoint])-shift).exp())/2).log()
        return outward_float(v,up)
    return dict(lower=evaluate('lower',False),upper=evaluate('upper',True),binomial_constant_omitted=True)


def predictors(train,ntrain,test_k,test_r):
    fit=fit_training(train,ntrain)
    assert fit['scored_calibration_excluded']
    # Stored floating hazards define a legitimate nonnegative predictive
    # distribution exactly; no claim these equal the unknown true hazards.
    rate=sum((arb(float(x)) for x in fit['rates'][43:54]),arb(0))
    physical=fixed_hazard_log(test_k,test_r,rate)
    kt,rt=window_counts(train,ntrain,43,53)
    beta=predictive_log_interval(kt,rt,test_k,test_r)
    return dict(training_fit=fit,training_lobe_count=kt,training_lobe_risk=rt,
        physical=physical,beta=beta,mixture=half_mixture(physical,beta),
        usage='Calibration predictor trained on opposite half ONLY; proper binomial prediction conditional on scored risk')
