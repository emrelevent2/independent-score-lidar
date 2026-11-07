"""Exploratory raw-time support check;not a fitted physical/efficacy result.

Written after inspecting extrema. Nominal80ns is an upstream algorithm
setting,not a calibrated hard lower bound established by these MAT files.
"""
from pathlib import Path
from fractions import Fraction
import json,hashlib,time
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/highflux_support_diagnostic_v1'
sha=lambda file:hashlib.sha256(Path(file).read_bytes()).hexdigest()


def main():
    start=time.perf_counter();rows=[];inputs=[]
    for name in ['FGS_td048_2019_01_28_acq1.mat','FGS_td198_2019_01_28_acq14.mat']:
        file=ROOT/'data/highflux_raw/2019_01_28'/name;inputs.append(dict(file=str(file.relative_to(ROOT)),sha256=sha(file)));r=loadmat(file);t=r['detTime'].ravel();f=r['frameNum'].ravel().astype(np.int64);period=float(r['tr'].item()*1e9);holdoff=float(r['T_holdoff'].item());df=np.diff(f);dt=np.diff(t);gap=df*period+dt;bad=np.flatnonzero(gap<80.);minimum=int(np.argmin(gap));maximum_bad_margin=max(0.,80.-float(gap[minimum]));exact_period=Fraction.from_float(period)
        # Separately evaluate every flagged80ns comparison in exact dyadic
        # rational arithmetic,using each original pair rather than rounded dt.
        exact=0;largest_difference=0.
        for i in bad:
            rational=int(df[i])*exact_period+Fraction.from_float(float(t[i+1]))-Fraction.from_float(float(t[i]));assert rational<80;exact+=1;largest_difference=max(largest_difference,abs(float(rational)-float(gap[i])))
        # A full independent pair computation after canonicalizing phase into
        # its actual period must preserve every gap,including overflow phases.
        carry=np.floor(t/period).astype(np.int64);phase=t-carry*period;canonical=np.diff(f+carry)*period+np.diff(phase);max_canonical=float(np.max(np.abs(canonical-gap)));assert max_canonical<1e-7
        # Computed-double operation bound for support comparisons. This is an
        # arithmetic diagnostic,not an instrument calibration certificate.
        eps=np.finfo(float).eps;bound=16*eps*(np.abs(df*period)+np.abs(t[1:])+np.abs(t[:-1])+80.);robust=int(np.count_nonzero(gap+bound<80.))
        rows.append(dict(file=name,records=len(t),consecutive_pairs=len(gap),period_ns=period,detector_holdoff_metadata_ns=holdoff,nominal_electronics_setting_ns=80.,electronics_recovery_metadata_present=False,phase_outside_actual_period=int(np.count_nonzero((t<0)|(t>=period))),minimum_gap_ns=float(gap[minimum]),minimum_pair_index=minimum,minimum_pair_frame_difference=int(df[minimum]),minimum_pair_phases_ns=[float(t[minimum]),float(t[minimum+1])],gap_quantiles_ns={str(q):float(v) for q,v in zip([0,.00001,.001,.01,.5,.99,1],np.quantile(gap,[0,.00001,.001,.01,.5,.99,1]))},pairs_below_nominal_80_ns=len(bad),all_below80_pairs_exact_dyadic_checked=exact,pairs_robustly_below80_under_arithmetic_bound=robust,largest_below80_margin_ns=maximum_bad_margin,maximum_exact_vs_double_gap_difference_ns=largest_difference,maximum_canonical_phase_reconstruction_gap_ns=max_canonical,pairs_below_metadata_holdoff=int(np.count_nonzero(gap<holdoff)),nonpositive_gaps=int(np.count_nonzero(gap<=0))))
    source=ROOT/'upstream/HighFluxSPL/main_ranging_comparison.m';assert 'Te = 80;' in source.read_text();inputs.append(dict(file=str(source.relative_to(ROOT)),sha256=sha(source)))
    result=dict(status='EXPLORATORY_SUPPORT_CHECK_COMPLETE',design_status='Postinspection descriptive diagnostic;no prospective claim or efficacy endpoint',rows=rows,inputs=inputs,seconds=time.perf_counter()-start,primary_reference=dict(url='https://doi.org/10.1364/OPTICA.403190',section='4.B Experimental Data',electronics_recovery='approximately80ns,thenassumed80inmodel',detector_recovery='approximatelyMAT holdoff+2ns;MAT holdoff isnot independently calibrated exact detector recovery'),interpretation='Foracq1,idealexact-knownTe80continuouslawassignszero probabilitytomeasuredgapsbelow80. TheMATcontainsdetectorholdoff,butnotcalibratedelectronicsrecovery;80isnominalupstreamalgorithmsetting andis explicitly approximate in the original paper. This is a limitation of treating nominal parameters as exact for an event-time likelihood,NOT a newly discovered instrument fault or refutation of the original approximate histogram method. Neitherphaseoverflowcanonicalizationnordoublearithmeticexplainstheobservedfewnsdiscrepancy. Thisdoesnotidentifythecause,jitter,actualrecoveryornewphysicalmodel. Preservealltimestamps;do notdropincompatiblepairsorfitTe=sampleminimumandcallitindependentcalibration. Need an explicitly measured timing/recovery model or documented approximate physical scope before real-data likelihood claims.',scope='Rawtimestamp/model-support diagnostic only. No secondary-surface groundtruth,benefit claim,afterpulsing diagnosisornewcalibration guarantee. Scientific synthetic protocols and upstream files unchanged. No rawdata redistribution.')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
