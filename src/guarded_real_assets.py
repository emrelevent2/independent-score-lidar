"""Tables and descriptive decomposition for the prospective real guard test."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from real_poisson_analysis_v2 import interval_match
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/guarded_real_confirmation_v1';DATA=ROOT/'data/guarded_real_confirmation_v1';M=ROOT/'manuscript'


def main():
    a=json.loads((D/'analysis.json').read_text());assert json.loads((D/'integrity.json').read_text())['status']=='PASS'
    p=json.loads((ROOT/'GUARDED_REAL_CONFIRMATORY_PROTOCOL.json').read_text());records=json.loads((D/'endpoint_records.json').read_text())
    primary=[];allrows=[];diagnostics={};frame_names=[]
    for f in p['frames']:
        name=f'Sequence{f["sequence"]}_{f["position"]}';short='Quarter' if f['position']=='quarter' else 'Three-quarter';frame_names.append(name)
        for i,prob in enumerate(p['settings']['probabilities']):
            for alpha in p['settings']['alphas']:
                gs={method:a['methods'][f'{name}_{i}_{alpha}_{method}']['groups'] for method in ['local','guarded','routed']}
                l,g,r=[gs[x] for x in ['local','guarded','routed']]
                assert l['two_interval']['declared']==g['two_interval']['declared'] and l['two_interval']['pair']==g['two_interval']['pair']
                assert l['known_one']['accepted_unmatched']==g['known_one']['accepted_unmatched']
                allrows.append(f'{short} & {100*prob:g} & {100*alpha:g} & {l["two_interval"]["pair"]} & {g["two_interval"]["pair"]} & {r["two_interval"]["pair"]} & {g["two_interval"]["actual_location_changed"]} & {g["known_one"]["accepted_unmatched"]} '+r'\\')
                if prob==p['settings']['primary_retention'] and alpha==p['settings']['primary_alpha']:
                    assert l['two_interval']['n']==500 and l['known_one']['n']==1000
                    primary.append(f'{short} & {l["two_interval"]["pair"]} & {g["two_interval"]["pair"]} & {r["two_interval"]["pair"]} & {l["two_interval"]["declared"]} & {g["two_interval"]["actual_location_changed"]} & {g["known_one"]["accepted_unmatched"]} '+r'\\')
                    fm=json.loads((DATA/name/'manifest.json').read_text());queries={q['id']:q for q in fm['queries']}
                    local=records[f'{name}_{i}_{alpha}_local'];guarded=records[f'{name}_{i}_{alpha}_guarded']
                    diag=dict(n=500,local_pairs=l['two_interval']['pair'],guarded_pairs=g['two_interval']['pair'],
                        declared=l['two_interval']['declared'],not_declared=500-l['two_interval']['declared'],
                        accepted_incomplete_pair=l['two_interval']['declared']-l['two_interval']['pair'],
                        guarded_coordinate_changed=g['two_interval']['actual_location_changed'],
                        primary_unmatched_among_declared=0,secondary_unmatched_among_declared=0,pair_endpoint_gain=0,pair_endpoint_loss=0)
                    for x,y in zip(local,guarded):
                        assert x['query_id']==y['query_id'];q=queries[x['query_id']]
                        if q['split']!='two_interval':continue
                        diag['pair_endpoint_gain']+=bool(y['pair'] and not x['pair']);diag['pair_endpoint_loss']+=bool(x['pair'] and not y['pair'])
                        if x['declared']:
                            diag['primary_unmatched_among_declared']+=not any(interval_match(x['primary'],w) for w in q['windows'])
                            diag['secondary_unmatched_among_declared']+=not any(interval_match(x['delay'],w) for w in q['windows'])
                    diagnostics[name]=diag
    header=[r'\begin{tabular}{lrrrrrr}\toprule',r'Capture & \multicolumn{3}{c}{Pairs /500} & \multicolumn{2}{c}{Two-window queries} & Known-one /1000\\',
        r' & Local & Guarded & Router & Declared L/G & Moved G & Unmatched L/G\\\midrule']
    (M/'guarded_real_table.tex').write_text('\n'.join(header+primary+[r'\bottomrule\end{tabular}'])+'\n')
    header=[r'\begin{tabular}{lrrrrrrr}\toprule',r'Capture & Retained (\%) & Nominal (\%) & Local pair & Guard pair & Router pair & Moved G & Known-one unmatched\\\midrule']
    (M/'guarded_real_all_table.tex').write_text('\n'.join(header+allrows+[r'\bottomrule\end{tabular}'])+'\n')
    (D/'diagnostics.json').write_text(json.dumps(dict(status='Post-analysis descriptive decomposition; not a method-selection experiment.',
        primary_alpha=.01,primary_retention=.025,frames=diagnostics,
        notes=['Both frames have500two-window queries. A location-only reporter cannot recover a query on which its fixed existence decision is negative.',
            'Known-one labels do not supply complete physical-negative truth. Coordinate motion inside a known window does not establish finer physical accuracy.']),indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained');x=np.arange(2);labels=['Quarter','Three-quarter']
    bottom=np.zeros(2)
    for field,label,color in [('local_pairs','Recovered pair','#176b94'),('accepted_incomplete_pair','Accepted incomplete pair','#c75d2b'),('not_declared','No secondary declaration','#c2cbd0')]:
        v=np.array([diagnostics[n][field] for n in frame_names]);axes[0].bar(x,v,bottom=bottom,color=color,label=label,width=.5);bottom+=v
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Two-window queries /500',title='(a) Fixed-decision outcome');axes[0].legend(fontsize=6.6,loc='upper left')
    for j,n in enumerate(frame_names):
        d=diagnostics[n];axes[1].bar(j,d['guarded_coordinate_changed'],width=.5,color='#176b94')
        axes[1].text(j,d['guarded_coordinate_changed']+7,str(d['guarded_coordinate_changed']),ha='center',fontsize=8)
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Accepted secondary coordinates moved',ylim=(0,300),title='(b) Motion without pair-label change')
    axes[1].text(.5,.92,'Pair gains = 0; losses = 0',ha='center',transform=axes[1].transAxes,fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf']:fig.savefig(M/f'figures/guarded_real_decomposition.{ext}',dpi=220)
    plt.close(fig)
    (M/'figures/guarded_real_assets.json').write_text(json.dumps(dict(analysis_sha256=hashlib.sha256((D/'analysis.json').read_bytes()).hexdigest(),
        protocol_sha256=a['protocol_sha256']),indent=2));print(json.dumps(diagnostics,indent=2),flush=True)


if __name__=='__main__':main()
