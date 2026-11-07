"""Independent interpolation/weighted-QR and validation path provenance."""
from pathlib import Path
import sys,json,time,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import train_null_information_bank as d
import ct_dual_reference as reference
from clock_pulse_likelihood import rates_and_design
from safe_deadtime_exposure import simulate_two_dead_times
from ct_full_score import templates


def interpolate(arrays,theta,regime):
    ri=list(arrays['regimes']).index(regime);a,t,b=theta;fgrid=arrays['fractions'];whole=math.floor(t%64);fraction=t%64-whole;fi=max(i for i,x in enumerate(fgrid) if x<=fraction);upper=fgrid[fi+1] if fi+1<len(fgrid) else 1.;wf=(fraction-fgrid[fi])/(upper-fgrid[fi]);result=np.zeros((64,64));weights=[]
    for name,value in [('amplitudes',a),('backgrounds',b)]:
        grid=arrays[name][ri];value=min(grid[-1],max(grid[0],value));index=max(i for i in range(len(grid)-1) if grid[i]<=value);w=(math.log(value)-math.log(grid[index]))/(math.log(grid[index+1])-math.log(grid[index]));weights.append([(index,1-w),(index+1,w)])
    for ai,wa in weights[0]:
        for bi,wb in weights[1]:
            for index,wf_ in [(fi,1-wf),(fi+1,wf)]:
                shift=whole-int(arrays['base_phase_integer'])+(1 if index==len(fgrid) else 0);permutation=(np.arange(64)-shift)%64;matrix=arrays['information'][ri,ai,bi,index%len(fgrid)];result+=wa*wb*wf_*matrix[np.ix_(permutation,permutation)]
    return (result+result.T)/2


def variance(h,j,templates_):
    design=np.linalg.cholesky(h).T;q,_=np.linalg.qr(design@j);v=design@templates_.T;v-=q@(q.T@v);return np.sum(v*v,axis=0)


def main():
    start=time.perf_counter();p=d.check();a=json.loads((d.OUT/'validation.json').read_text());assert a['status']=='NUMERICAL_WEIGHT_APPROXIMATION_PASS' and a['protocol_sha256']==d.sha(d.P);before=d.sha(d.OUT/'validation.json');assert d.sha(d.OUT/'bank.npz')==a['bank_sha256'];bank=np.load(d.OUT/'bank.npz');rows=list(d.validation_cases(p));all_ratios=[];paths=0;full=0;max_matrix_gap=0.;max_ratio_gap=0.
    for case,saved in zip(rows,a['profiles']):
        assert saved['case']==case;file=d.OUT/saved['file'];assert d.sha(file)==saved['sha256'];arrays=np.load(file);matrix=arrays['reference_matrix'];meta=json.loads(str(arrays['metadata']));assert meta==saved['metadata'];g=p['regimes'][case['ri']];rate,j=rates_and_design(case['theta'],64,1.5);h=hashlib.sha256();events=0
        for i in range(p['validation_records_per_profile']):
            obs,_=simulate_two_dead_times(rate,g['cycles'],g['td'],g['te'],case['seed']+1009*i);h.update(obs.tobytes());events+=len(obs);paths+=1
        assert h.hexdigest()==meta['observed_arrays_sha256'] and events==meta['observed_events']
        if case['index']%16 in [0,15]:
            regenerated,rm=d.information(case['theta'],g,case['seed'],p['validation_records_per_profile'],p);np.testing.assert_array_equal(regenerated,matrix);assert rm==meta;full+=1
        hb=interpolate(bank,case['theta'],g['index']);library,_=d.NullInformationBank.load(d.OUT/'bank.npz').lookup(case['theta'],g['index']);gap=float(np.max(np.abs(hb-library)));max_matrix_gap=max(max_matrix_gap,gap);np.testing.assert_allclose(hb,library,atol=1e-7,rtol=1e-12);grid,templates_=templates(64,1.5);allowed=np.abs((grid-case['theta'][1]+32)%64-32)>=3.75;ratios=variance(hb,j,templates_)[allowed]/variance(matrix,j,templates_)[allowed];gap=float(np.max(np.abs(ratios-arrays['ratios'])));max_ratio_gap=max(max_ratio_gap,gap);np.testing.assert_allclose(ratios,arrays['ratios'],atol=1e-8,rtol=1e-9);all_ratios.extend(ratios.tolist())
    q=np.quantile(np.abs(np.log(all_ratios)),[.5,.95,1]);saved=np.array(list(a['absolute_log_variance_ratio_quantiles'].values()));np.testing.assert_allclose(q,saved,atol=1e-9);assert before==d.sha(d.OUT/'validation.json');d.check();result=dict(status='PASS',all32validation_profiles_keys_and_hashes_checked=True,all6400continuous_validation_paths_regenerated=paths,whole200record_reference_matrix_replays=full,independent_phase_interpolation_checks=32,independent_Cholesky_QR_projected_variance_ratios=len(all_ratios),max_independent_interpolation_entry_gap=max_matrix_gap,max_independent_variance_ratio_gap=max_ratio_gap,validation_sha256=before,bank_sha256=a['bank_sha256'],seconds=time.perf_counter()-start,scope='Allvalidationprovenanceand32independentinterpolations/weightedQRvarianceprofiles;4fullreferencecellreplays800paths. Numericalweightapproximationonly,not calibrated detector efficacy or exact finite Fisher/physical validity.')
    (d.OUT/'validation_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
