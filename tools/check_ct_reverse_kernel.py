"""Frozen fresh numerical parity fixtures;not a detection experiment."""
from pathlib import Path
import sys,json,time,hashlib,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import check_ct_dual_kernel as model
import ct_dual_reverse as reverse
import ct_dual_fast as forward
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times
P=ROOT/'CT_REVERSE_KERNEL_PARITY_PROTOCOL.json';OUT=ROOT/'results/ct_reverse_kernel_parity_v1';sha=model.sha


def freeze():
    if P.exists():raise FileExistsError('Frozen reverse parity protocol exists')
    model.check();names=['src/ct_dual_reverse.py','src/test_ct_dual_reverse.py','tools/check_ct_reverse_kernel.py'];p=dict(status='FROZEN_BEFORE_FRESH_NUMERICAL_FIXTURES',sources={n:sha(ROOT/n) for n in names},continuous_model_protocol_sha256=sha(model.P),seed_base=6198432701,profiles_per_regime=16,bins=64,sigma=1.5,cycles=200,regimes=[[83.7,40.],[126.7,80.],[31.7,80.],[83.7,100.]],criterion='All64freshrecords:fullvalue,64phasegradients,eachinitial/transition/terminalscorepiece againstfrozenforwardkernel. First2profiles/regime additionallyagainst32pointquadrature5parametergradient. Maxpieceabserror<1e-6andmaxfullLLerror<1e-7. No fit/detection/threshold/utility claim.',scope='Established reverse-mode differentiation of same positive continuouspathlaw. Alloldfrozenmodel/efficacysourcesunchanged. Newversion isengineeringpreparationonly;mustnotreplaceoldstudyhalfway. DomainTe<=3Td,floatingnotintervalarithmetic.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());model.check();assert sha(model.P)==p['continuous_model_protocol_sha256']
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p


def run():
    p=check();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'analysis.json').exists():raise FileExistsError('Completed parity audit exists;use replay')
    result=compute(p);(OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


def compute(p):
    start=time.perf_counter();records=[];largest=[0.,0.,0.];qh=hashlib.sha256()
    for gi,(td,te) in enumerate(p['regimes']):
        for i in range(p['profiles_per_regime']):
            rng=np.random.default_rng(p['seed_base']+1009*(gi*16+i));theta=np.array([rng.uniform(.1,3.5),rng.uniform(0,64),rng.uniform(.05,2.),rng.uniform(0,.3),rng.uniform(0,64)]);rate,j=rates_and_design(theta,64,1.5);obs,_=simulate_two_dead_times(rate,p['cycles'],td,te,p['seed_base']+2000003*(gi*16+i));qh.update(obs.tobytes());g=reverse.prepare_record(obs,64*p['cycles'],64,td,te);a=reverse.full_factor(rate,np.eye(64),g,return_pieces=True);b=forward.full_factor(rate,np.eye(64),g,return_pieces=True);gaps=[abs(a[0]-b[0]),float(np.max(np.abs(a[1]-b[1]))),float(np.max(np.abs(a[3]-b[3])))];largest=[max(x,y) for x,y in zip(largest,gaps)];assert gaps[0]<1e-7 and max(gaps[1:])<1e-6,(gi,i,gaps);np.testing.assert_allclose(a[2],b[2],atol=1e-8,rtol=1e-10)
            qgap=None
            if i<2:
                value,gradient=reference.full_factor(rate,j,g,order=32);qgap=dict(value=abs(value-a[0]),gradient=float(np.max(np.abs(gradient-a[1]@j))));assert qgap['value']<1e-7 and qgap['gradient']<1e-6
            records.append(dict(regime=gi,profile=i,observed_events=len(obs),gaps=gaps,quadrature_gap=qgap))
        print('Reverse fullrecord parity regime',gi+1,'/4',flush=True)
    check();return dict(status='PASS',records=records,max_full_loglik_gap=largest[0],max_full64gradient_gap=largest[1],max_individual_factor64score_gap=largest[2],all_observed_record_arrays_sha256=qh.hexdigest(),protocol_sha256=sha(P),seconds=time.perf_counter()-start,scope=p['scope'])


def replay():
    p=check();saved=json.loads((OUT/'analysis.json').read_text());actual=compute(p);assert {k:v for k,v in saved.items() if k!='seconds'}=={k:v for k,v in actual.items() if k!='seconds'};r=dict(status='PASS',all64record_fixtures_and_numerical_checks_regenerated=True,analysis_sha256=sha(OUT/'analysis.json'));(OUT/'replay.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':{'freeze':freeze,'run':run,'replay':replay}[sys.argv[1]]()
