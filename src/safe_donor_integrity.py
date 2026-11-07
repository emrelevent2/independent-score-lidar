"""Replay ALL certified decisions and verify arithmetic versus feasible nulls."""
import json,time
import numpy as np
from flint import arb
import safe_donor_development as d
import safe_spatial_control_addendum as extra
from safe_donor_alternative import build_alternative,mixture_scores
from certified_safe_components import exact_safe_statistics,mixture_log_lower,periodic_point_rates,log_likelihood_interval
from certified_safe_null import declare_from_fixed_mixture,null_witness
from safe_deadtime_exposure import safe_lag


def canonical(cert):return {k:v for k,v in cert.items() if k!='seconds'}


def main():
    start=time.perf_counter();p=d.check();q,_=extra.check();rows=d.read();controls=extra.read();expected=list(d.cases(p));assert [r['case'] for r in rows]==expected and [r['case'] for r in controls]==expected
    cfg=p['settings'];a=json.loads((d.OUT/'analysis.json').read_text());x=json.loads((extra.OUT/'analysis.json').read_text());hashes={n:d.sha(d.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']};xh={n:d.sha(extra.OUT/n) for n in ['records.sqlite3','analysis.json','completion.json']};c=json.loads((d.OUT/'completion.json').read_text());ec=json.loads((extra.OUT/'completion.json').read_text());assert hashes['records.sqlite3']==c['records_sha256'] and xh['records.sqlite3']==ec['records_sha256'] and ec['base_records_sha256']==hashes['records.sqlite3']
    totals={m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in d.METHODS+['spatial_prior_score']};per={};certpoints=0;events=0;boxes=0
    for i,(row,control) in enumerate(zip(rows,controls)):
        case=row['case'];g=p['regimes'][case['regime']];obs,ds,de,truth,dh=d.data(case,p);assert dh==row['data_sha256'] and truth==row['truth'];events+=len(obs);alt=build_alternative(ds,de,cfg['sigma']);assert alt['sha256']==row['alternative_sha256'];stats=exact_safe_statistics(obs,g['cycles'],cfg['bins'],safe_lag(g['td'],g['te']));numlo=mixture_log_lower(stats,alt['rates'],alt['weights']);saved=row['samples']['donor_certified']['certificate'];assert numlo==saved['log_numerator_lower'];mid=np.array([float(v.mid()) for v in stats['exposure']]);_,loc,idx=mixture_scores(stats['counts'],mid,alt);assert int(idx[0])==row['selected_component'] and alt['kinds'][int(idx[0])]==row['selected_kind']
        assert list(loc[0])==[row['samples']['donor_bootstrap']['primary'],row['samples']['donor_bootstrap']['delay']]
        replay=declare_from_fixed_mixture(stats,numlo,sigma_range=(cfg['sigma'],cfg['sigma']),initial_cells=cfg['cover_initial_cells'],max_boxes=cfg['cover_max_boxes'],fit=dict(theta=np.array(row['null_fit'])))
        assert canonical(replay)==canonical(saved);boxes+=replay['boxes']
        if replay['upper'] is not None:
            point=log_likelihood_interval(stats,periodic_point_rates(cfg['bins'],g['a'],truth['primary'],g['b'],cfg['sigma']));assert arb(replay['upper'])>=point;certpoints+=1
        assert replay['declared']==(replay['upper'] is not None and replay['upper']<=replay['decision_level_lower'])
        copy=dict(row);copy['samples']=dict(row['samples'],spatial_prior_score=control['sample']);key=(case['regime'],case['family']);per.setdefault(key,{m:dict(correct=0,wrong=0,false=0,positive=0,null=0) for m in totals})
        for m in totals:
            v=copy['samples'][m]
            if m!='donor_certified':
                boot=np.asarray(v['bootstrap_scores']);pv=float((1+np.count_nonzero(boot>=v['score']))/(len(boot)+1));assert pv==v['pvalue'] and len(boot)==199
            declared=v['declared'] if m=='donor_certified' else v['pvalue']<=.01;pos=case['positive'];distance=lambda z,y:min(abs((z-y)%64),abs((y-z)%64));correct=pos and declared and distance(v['primary'],truth['primary'])<=2 and distance(v['delay'],truth['secondary'])<=2;ep=dict(correct=int(correct),wrong=int(pos and declared and not correct),false=int(not pos and declared),positive=int(pos),null=int(not pos))
            for k,z in ep.items():totals[m][k]+=z;per[key][m][k]+=z
        if (i+1)%64==0:print('Verified full certificates and endpoints',i+1,'/',len(rows),flush=True)
    assert {m:totals[m] for m in d.METHODS}==a['totals'] and totals['spatial_prior_score']==x['totals']
    for saved in a['per_regime_family']:
        gi=next(i for i,g in enumerate(p['regimes']) if g['name']==saved['regime']);assert {m:per[(gi,saved['family'])][m] for m in d.METHODS}==saved['methods']
    proposed=totals['donor_bootstrap']
    def rule(b):return dict(strict_more_correct=proposed['correct']>b['correct'],strict_fewer_total_errors=proposed['wrong']+proposed['false']<b['wrong']+b['false'],no_more_wrong=proposed['wrong']<=b['wrong'],no_more_null=proposed['false']<=b['false'])
    assert rule(totals['local_bootstrap'])==a['primary_rule'] and all(a['primary_rule'].values())==a['continue_to_fresh_efficacy_confirmation'];assert rule(totals['spatial_prior_score'])==x['additional_superiority_gate'] and all(x['additional_superiority_gate'].values())==x['passes_additional_gate']
    assert all(d.sha(d.OUT/n)==h for n,h in hashes.items()) and all(d.sha(extra.OUT/n)==h for n,h in xh.items())
    result=dict(status='PASS',all_query_context_sets_regenerated=len(rows),all_donor_predictors_regenerated=True,all_certified_controllers_replayed=len(rows),certified_boxes_replayed=boxes,feasible_primary_only_null_points_checked=certpoints,query_recorded_events=events,all_decisions_recounted=len(rows)*len(totals),all_bootstrap_statistics_already_replayed_by_addendum=True,all_bootstrap_fits_already_replayed_by_addendum=True,base_output_sha256=hashes,control_output_sha256=xh,seconds=time.perf_counter()-start,scope='All448datasets/16donors/predictors/certifiedcontrollers and1792decisions replayed. Separate equal-context runner replays all89152bootstrap paths and all corresponding nonlinear fits/statistics. Certified arithmetic is for recorded dyadics and idealized bin-constant-rate model,not physical timestamp/recovery/IRF uncertainty.')
    (d.OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
