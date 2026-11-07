"""Independent direct-rank recount of the two CLOSED fusion diagnostics."""
import json,time
import numpy as np
import safe_donor_development as d
import safe_spatial_control_addendum as extra
from safe_donor_alternative import build_alternative
from certified_safe_components import exact_safe_statistics
from safe_deadtime_exposure import safe_lag


def direct(values):
    # All pairwise tail counts;does not use scipy.rankdata or fusion module.
    ranks=np.stack([np.count_nonzero(values[None,:,j]>=values[:,None,j],axis=1) for j in range(values.shape[1])],axis=1)
    minimum=np.min(ranks,axis=1);return float(np.count_nonzero(minimum<=minimum[0])/len(values)),int(np.argmin(ranks[0])),ranks[0]/len(values)


def main():
    start=time.perf_counter();p=d.check();extra.check();dirs=['closed_paired_fusion_diagnostic_v1','closed_omnibus_location_diagnostic_v1'];protocols=['CLOSED_PAIRED_FUSION_DIAGNOSTIC_PROTOCOL.json','CLOSED_OMNIBUS_LOCATION_DIAGNOSTIC_PROTOCOL.json'];allhashes={};records=[];analysis=[]
    for folder,protocol in zip(dirs,protocols):
        q=json.loads((d.ROOT/protocol).read_text())
        for n,h in {**q['sources'],**q['inputs']}.items():assert d.sha(d.ROOT/n)==h,n
        out=d.ROOT/'results'/folder;records.append(json.loads((out/'records.json').read_text()));analysis.append(json.loads((out/'analysis.json').read_text()));allhashes[folder]={n:d.sha(out/n) for n in ['records.json','analysis.json']}
    totals=[{m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in v['totals']} for v in analysis]
    base=d.read();controls=extra.read();assert all(len(v)==len(base) for v in records)
    for i,(row,control) in enumerate(zip(base,controls)):
        local=row['samples']['local_bootstrap'];spatial=control['sample'];mixture=row['samples']['donor_bootstrap'];parts=[local,spatial,mixture]
        for j,count in enumerate([2,3]):
            v=np.column_stack([np.r_[x['score'],x['bootstrap_scores']] for x in parts[:count]]);pv,selected,component=direct(v);saved=records[j][i]['fusion'];assert pv==saved['pvalue'] and selected==saved['selected'];np.testing.assert_array_equal(component,saved['component_pvalues'])
        case=row['case'];g=p['regimes'][case['regime']];obs,ds,de,truth,h=d.data(case,p);assert h==row['data_sha256'];alt=build_alternative(ds,de,p['settings']['sigma']);assert alt['sha256']==row['alternative_sha256'];st=exact_safe_statistics(obs,g['cycles'],p['settings']['bins'],safe_lag(g['td'],g['te']));exposure=np.array([float(z.mid()) for z in st['exposure']]);ll=st['counts']@np.log(alt['rates']).T-exposure@alt['rates'].T;index=int(np.argmax(ll));assert index==records[1][i]['maximum_likelihood_component']
        for j in [0,1]:
            r=records[j][i]
            for method,sample in r['samples'].items():
                if j==0:
                    sel=parts[r['fusion']['selected']];np.testing.assert_array_equal([sample['primary'],sample['delay']],[sel['primary'],sel['delay']]);declared=sample['pvalue']<=.01 and (method=='paired_min_rank' or r['agreement'])
                else:
                    idx=r['fusion']['selected'];sel=parts[idx];loc=alt['locations'][index] if method=='mixture_profile_location' or (method=='omnibus_profile_location' and idx==2) else [sel['primary'],sel['delay']];np.testing.assert_array_equal([sample['primary'],sample['delay']],loc);declared=sample['pvalue']<=.01
                distance=lambda x,y:abs((x-y+32)%64-32);pos=case['positive'];correct=pos and declared and distance(sample['primary'],truth['primary'])<=2 and distance(sample['delay'],truth['secondary'])<=2;ep=dict(correct=int(correct),wrong=int(pos and declared and not correct),false=int(not pos and declared),positive=int(pos),null=int(not pos))
                for k,z in ep.items():totals[j][method][k]+=z
    for j,(folder,result) in enumerate(zip(dirs,analysis)):
        assert totals[j]==result['totals'];out=d.ROOT/'results'/folder;assert all(d.sha(out/n)==h for n,h in allhashes[folder].items());(out/'integrity.json').write_text(json.dumps(dict(status='PASS',all_cases_recounted=len(base),independent_all_pairwise_rank_implementation=True,all_context_predictors_and_profile_locations_replayed=True,original_outputs_sha256=allhashes[folder],scope='Closed-development diagnostic verification only;no efficacy confirmation or nominal composite-null guarantee.'),indent=2))
    print('Both closed diagnostics verified:',len(base),'cases each;seconds',round(time.perf_counter()-start,3),flush=True)


if __name__=='__main__':main()
