"""Freeze/train/validate an independent fixed null-information weight bank."""
from pathlib import Path
import sys,json,time,os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import check_ct_reverse_kernel as model
import ct_dual_reverse as kernel
from null_information_bank import NullInformationBank
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times
from ct_full_score import templates
P=ROOT/'NULL_INFORMATION_BANK_PROTOCOL.json';OUT=ROOT/'results/null_information_bank_v1';sha=model.sha


def freeze():
    if P.exists():raise FileExistsError('Frozen bank protocol exists')
    model.check();names=['src/null_information_bank.py','src/test_null_information_bank.py','tools/train_null_information_bank.py'];p=dict(status='FROZEN_BEFORE_INDEPENDENT_NULL_TRAINING_AND_VALIDATION',sources={n:sha(ROOT/n) for n in names},model_protocol_sha256=sha(model.P),grid_multipliers=[.25,.5,1.,2.,4.],fractional_phases=[0.,.25,.5,.75],base_phase_integer=15,records_per_cell=200,training_seed_base=8124790163,validation_seed_base=9126387041,validation_profiles_per_regime=16,validation_records_per_profile=200,validation_amplitude_multipliers=[.55,1.8],validation_background_multipliers=[.5,2.4],bins=64,sigma=1.5,regimes=[dict(index=2,name='bright_hidden',a=2.4,b=.8,cycles=1000,td=31.7,te=80.),dict(index=3,name='weak_hidden',a=.25,b=.1,cycles=1500,td=83.7,te=100.)],execution=dict(workers=4,threads={x:'1' for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),training='200gridcells:2recoveryregimesx5amplitudesx5backgroundsx4fractionalphases. Each200freshknown-nullcompletecontinuousrecords;no secondaryphotons,efficacylabelsorfittedquerydata. Pool64phaseper-factorOPG(sum/Nrecords),symmetrize. No eigenvaluefloor. Persistatomiccellfiles and hashes. Separatefreshknown-nullvalidation32profilesx200records witharbitraryfullperiodphase.',lookup='Convex trilinear weights:logamplitude,logbackground,linearfractionalphase;integerphase rotation includingfractionalwrap. Parameterclampsdeclaredandflagged,noexcludedqueries. Matrixis a numerical fixedscore-weightSURROGATE:rotationdoesNOTexactlypreservefiniteinitiallyactiveboundarylaw. Allquerylikelihoods/calibrationsretainoriginalphysics. No claimexactfiniteFisher.',validation='Independent200recordreferenceinformationper32profiles. Forall256secondarytemplatesexcludingwithin2.5sigmaofknownprimary,compare nuisance-projectedbankvariance with referencevariance. Aggregateabslogratio median<=log1.15,95quantile<=log1.35,max<=log2;allvariancespositive,PSDrelativeeigenlower>=-1e-10. This numerical-approximationgate isNOT a detection/power/Q1gate. Bothoriginalhiddenregimes/allphases/fluxrangeskept;no queryoutcome-basedtuning.',scope='Independentnullsimulation/interpolation and expected-informationweightingareestablished. FiniteMonteCarloandphase-rotation/interpolationapproximationsremain. Knownwidth/recovery/initialstate;no physicalvalidation,uniformriskguaranteeornewstatisticaltheorem. No efficacycohortproduceduntiltraining/validation/sourcechecks complete.')
    P.write_text(json.dumps(p,indent=2));print('Frozen',sha(P),flush=True)


def check():
    p=json.loads(P.read_text());model.check();assert sha(model.P)==p['model_protocol_sha256']
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
    return p


def cells(p):
    index=0
    for ri,g in enumerate(p['regimes']):
        for ai,am in enumerate(p['grid_multipliers']):
            for bi,bm in enumerate(p['grid_multipliers']):
                for fi,fraction in enumerate(p['fractional_phases']):
                    yield dict(id=f'{ri}_{ai}_{bi}_{fi}',index=index,ri=ri,ai=ai,bi=bi,fi=fi,theta=[g['a']*am,p['base_phase_integer']+fraction,g['b']*bm]);index+=1


def information(theta,g,seed_base,records,p):
    import hashlib
    rate,_=rates_and_design(theta,64,1.5);summed=np.zeros((64,64));h=hashlib.sha256();events=0
    for i in range(records):
        obs,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],seed_base+1009*i);h.update(obs.tobytes());geo=kernel.prepare_record(obs,g['cycles']*64,64,g['td'],g['te']);ll,gradient,terms,pieces=kernel.full_factor(rate,np.eye(64),geo,return_pieces=True);summed+=pieces.T@pieces;events+=len(obs)
    matrix=summed/records;matrix=(matrix+matrix.T)/2;return matrix,dict(records=records,observed_events=events,observed_arrays_sha256=h.hexdigest(),minimum_eigenvalue=float(np.linalg.eigvalsh(matrix).min()))


def train_cell(cell):
    p=check();start=time.perf_counter();h,metadata=information(cell['theta'],p['regimes'][cell['ri']],p['training_seed_base']+1000003*cell['index'],p['records_per_cell'],p);return dict(cell=cell,matrix=h,metadata=metadata,seconds=time.perf_counter()-start)


def train():
    p=check();assert {k:os.environ.get(k) for k in p['execution']['threads']}==p['execution']['threads'];OUT.mkdir(parents=True,exist_ok=True);folder=OUT/'cells';folder.mkdir(exist_ok=True);expected=list(cells(p));remaining=[]
    for cell in expected:
        file=folder/(cell['id']+'.npz')
        if file.exists():
            with np.load(file,allow_pickle=False) as saved:assert json.loads(str(saved['cell']))==cell and str(saved['protocol_sha256'])==sha(P) and saved['matrix'].shape==(64,64)
        else:remaining.append(cell)
    start=time.perf_counter();done=len(expected)-len(remaining);print('Independent null bank4workers existing',done,'remaining',len(remaining),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(train_cell,remaining,chunksize=1):
            cell=row['cell'];file=folder/(cell['id']+'.npz');temporary=folder/(cell['id']+'.tmp')
            with temporary.open('wb') as handle:np.savez_compressed(handle,matrix=row['matrix'],cell=json.dumps(cell),metadata=json.dumps(row['metadata']),seconds=row['seconds'],protocol_sha256=sha(P))
            temporary.replace(file);done+=1
            if done%4==0:print('Null bank cells',done,'/200 elapsed_s',round(time.perf_counter()-start,1),flush=True)
    matrices=np.empty((2,5,5,4,64,64));manifest=[]
    for cell in expected:
        file=folder/(cell['id']+'.npz')
        with np.load(file,allow_pickle=False) as saved:matrices[cell['ri'],cell['ai'],cell['bi'],cell['fi']]=saved['matrix'];meta=json.loads(str(saved['metadata']))
        manifest.append(dict(cell=cell,file=str(file.relative_to(OUT)),sha256=sha(file),metadata=meta))
    bank=OUT/'bank.npz';np.savez_compressed(bank,information=matrices,amplitudes=np.array([[g['a']*m for m in p['grid_multipliers']] for g in p['regimes']]),backgrounds=np.array([[g['b']*m for m in p['grid_multipliers']] for g in p['regimes']]),fractions=p['fractional_phases'],regimes=[g['index'] for g in p['regimes']],base_phase_integer=p['base_phase_integer']);result=dict(status='TRAINING_COMPLETE',protocol_sha256=sha(P),bank_sha256=sha(bank),cells=manifest,records=40000,seconds_this_invocation=time.perf_counter()-start,scope=p['scope']);(OUT/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2),flush=True)


def projected_variance(h,j,bank):
    hj=h@j;cross=bank@hj;inv=np.linalg.pinv(j.T@hj,rcond=1e-10);return np.einsum('mi,ij,mj->m',bank,h,bank)-np.einsum('mi,ij,mj->m',cross,inv,cross)


def validation_case(case):
    p=check();g=p['regimes'][case['ri']];h,meta=information(case['theta'],g,case['seed'],p['validation_records_per_profile'],p);bank=NullInformationBank.load(OUT/'bank.npz');hb,flags=bank.lookup(case['theta'],g['index']);_,j=rates_and_design(case['theta'],64,1.5);grid,templates_=templates(64,1.5);v=projected_variance(h,j,templates_);vb=projected_variance(hb,j,templates_);allowed=np.abs((grid-case['theta'][1]+32)%64-32)>=3.75;assert np.all(v[allowed]>1e-8) and np.all(vb[allowed]>1e-8);assert not flags['amplitude_clamped'] and not flags['background_clamped'];ratios=vb[allowed]/v[allowed];return dict(case=case,reference_matrix=h,metadata=meta,ratios=ratios,bank_minimum_eigenvalue=float(np.linalg.eigvalsh(hb).min()),reference_minimum_eigenvalue=float(np.linalg.eigvalsh(h).min()))


def validation_cases(p):
    for ri,g in enumerate(p['regimes']):
        for i in range(p['validation_profiles_per_regime']):
            index=ri*p['validation_profiles_per_regime']+i;rng=np.random.default_rng(p['validation_seed_base']+1009*index);yield dict(ri=ri,index=index,theta=[g['a']*rng.uniform(*p['validation_amplitude_multipliers']),rng.uniform(0,64),g['b']*rng.uniform(*p['validation_background_multipliers'])],seed=p['validation_seed_base']+2000003*(index+1))


def validate():
    p=check();c=json.loads((OUT/'completion.json').read_text());assert c['bank_sha256']==sha(OUT/'bank.npz');folder=OUT/'validation';folder.mkdir(exist_ok=True)
    if (OUT/'validation.json').exists():raise FileExistsError('Completedvalidationexists')
    start=time.perf_counter();rows=[];all_ratios=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(validation_case,validation_cases(p),chunksize=1):
            file=folder/(str(row['case']['index'])+'.npz');np.savez_compressed(file,reference_matrix=row['reference_matrix'],ratios=row['ratios'],case=json.dumps(row['case']),metadata=json.dumps(row['metadata']));rows.append({k:v for k,v in row.items() if k not in ['reference_matrix','ratios']});rows[-1].update(file=str(file.relative_to(OUT)),sha256=sha(file),variance_ratio_quantiles=np.quantile(row['ratios'],[0,.5,.95,1]).tolist());all_ratios.extend(row['ratios'].tolist());print('Null bank independent validation',len(rows),'/32',flush=True)
    error=np.abs(np.log(all_ratios));q=np.quantile(error,[.5,.95,1]);passed=q[0]<=np.log(1.15) and q[1]<=np.log(1.35) and q[2]<=np.log(2.) and all(r['bank_minimum_eigenvalue']>=-1e-10 for r in rows);result=dict(status='NUMERICAL_WEIGHT_APPROXIMATION_PASS' if passed else 'NUMERICAL_WEIGHT_APPROXIMATION_FAILED',profiles=rows,reference_records=6400,secondary_direction_checks=len(all_ratios),absolute_log_variance_ratio_quantiles=dict(median=float(q[0]),p95=float(q[1]),max=float(q[2])),bank_sha256=c['bank_sha256'],protocol_sha256=sha(P),seconds=time.perf_counter()-start,scope=p['scope']);(OUT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='profiles'},indent=2),flush=True)


if __name__=='__main__':{'freeze':freeze,'train':train,'validate':validate}[sys.argv[1]]()
