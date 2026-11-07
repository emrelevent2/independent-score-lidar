"""Saved-output article figures/tables. No simulation, fitting or efficacy run."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,sqlite3,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'manuscript/ded_style_article'
FIG=DEST/'figures'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
METHODS=['safe_score_common','safe_joint_common','safe_context_max_common','safe_context_transport_common',
         'safe_score_original','safe_joint_original','safe_context_max_original','safe_context_transport_original',
         'full_opg_safe','full_opg_refit','full_bank_safe','full_bank_refit',
         'localized_safe_score_common','localized_safe_score_original','localized_full_bank']
LABELS=['Censored score / common','Censored joint / common','Context maximum / common','Transported profile / common',
        'Censored score / original','Censored joint / original','Context maximum / original','Transported profile / original',
        'Full OPG / pilot','Full OPG / refit','Full bank / pilot','Full bank / refit',
        'Localized censored / common','Localized censored / original','Proposed full bank + localization']
PRIMARY=METHODS[-1]
COLORS=['#ae582c','#75618f','#176b87']


def endpoint(r,m,alpha):
    z=r['samples'][m];d=z['pvalue']<=alpha;p=r['case']['positive']
    dist=lambda a,b:abs((a-b+32)%64-32)
    c=p and d and dist(z['primary'],r['truth']['primary'])<=2 and dist(z['delay'],r['truth']['secondary'])<=2
    return np.array([int(c),int(p and d and not c),int(not p and d)],dtype=int)


def totals(rows,alpha=.01):
    return {m:np.sum([endpoint(r,m,alpha) for r in rows],axis=0).tolist() for m in METHODS}


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.04,
                metadata={'CreationDate':None,'ModDate':None,'Creator':'Two-recovery article figure builder'})
    fig.savefig(FIG/(name+'.png'),dpi=190,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def schematic():
    fig=plt.figure(figsize=(7.05,3.55));gs=fig.add_gridspec(1,2,width_ratios=[1.32,1],wspace=.40)
    ax=fig.add_subplot(gs[0]);xmax=148
    for y,label in [(4,'Incident photons'),(3,'Detector recovery'),(2,'Electronics recovery'),(1,'Avalanches'),(0,'Recorded events')]:
        ax.hlines(y,0,xmax,color='#c3c8cc',lw=.6);ax.text(-3,y,label,ha='right',va='center',fontsize=7)
    for t in [4,20,40,52,75,90,116]:ax.vlines(t,3.85,4.2,color='#686868',lw=1)
    for t in [4,40,75,116]:ax.broken_barh([(t,min(31.7,xmax-t))],(2.86,.28),facecolors='#d0d5da',edgecolors='#74818a',linewidth=.5)
    for t in [4,116]:ax.broken_barh([(t,min(80,xmax-t))],(1.86,.28),facecolors='#dbeef3',edgecolors='#176b87',linewidth=.6)
    for t in [4,116]:ax.plot(t,1,'o',color='#176b87',ms=4);ax.vlines(t,-.13,.16,color='#176b87',lw=1.3)
    for t in [40,75]:ax.plot(t,1,'o',mfc='white',mec='#ae582c',ms=4)
    ax.text(57.5,.62,'unrecorded',color='#ae582c',ha='center',fontsize=7)
    ax.set(xlim=(-2,xmax),ylim=(-.65,4.55),xlabel='Time (phase-bin units)',xticks=[0,40,80,120],yticks=[])
    ax.set_title('(a) Two hidden avalanches',fontsize=9,pad=10)
    ax.spines[['top','right','left']].set_visible(False)
    bx=fig.add_subplot(gs[1]);bx.set(xlim=(0,1),ylim=(0,1));bx.axis('off')
    boxes=[(.70,'Timestamp record'),(.52,'Single-return null fit'),(.34,'Projected full score'),(.16,'Calibrated declaration')]
    for y,label in boxes:
        bx.add_patch(FancyBboxPatch((.04,y),.9,.12,boxstyle='round,pad=.015',ec='#74818a',fc='#f4f7f8',lw=.7))
        bx.text(.49,y+.06,label,ha='center',va='center',fontsize=7.5)
        if y>.16:bx.annotate('',(.49,y-.05),(.49,y),arrowprops=dict(arrowstyle='->',lw=.7,color='#52606a'))
    bx.text(.49,.94,'Independent null-only bank',ha='center',fontsize=7.5,color='#176b87')
    bx.annotate('',(.96,.40),(.96,.92),arrowprops=dict(arrowstyle='->',color='#176b87',lw=.8))
    bx.text(.49,.04,'One observed-record location fit',ha='center',fontsize=7.5)
    bx.annotate('',(.49,.07),(.49,.15),arrowprops=dict(arrowstyle='->',lw=.7))
    bx.set_title('(b) Detection and reporting',fontsize=9,pad=10)
    save(fig,'fig01_model')


def main():
    DEST.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    dbpath=ROOT/'results/score_peak_localization_development_v1/records.sqlite3';before=sha(dbpath)
    assert before=='6a2d5d5a6c1bfdc8ca1fabbc00d929d8a286a6c93ad0dbc1fbd30b352a8dbecd'
    mainout=dbpath.parent
    for n in ['integrity.json','parent_failure_audit.json','failure_audit.json']:
        assert json.loads((mainout/n).read_text())['status']=='PASS'
    with sqlite3.connect(f'file:{dbpath.as_posix()}?mode=ro',uri=True) as db:
        rows=[json.loads(v[0]) for v in db.execute('SELECT payload FROM records ORDER BY rowid')]
    assert len(rows)==640 and sum(r['case']['positive'] for r in rows)==256
    observed=totals(rows)
    source=json.loads((mainout/'analysis.json').read_text())
    for m in METHODS:assert observed[m]==[source['totals'][m][k] for k in ['correct','wrong','false']]
    for r in rows:
        for m in METHODS:
            z=r['samples'][m]
            assert (1+sum(t>=z['score'] for t in z['bootstrap_scores']))/200==z['pvalue']
    regimes=json.loads((ROOT/'SCORE_PEAK_LOCALIZATION_DEVELOPMENT_PROTOCOL.json').read_text())['regimes']
    byregime={str(i):totals([r for r in rows if r['case']['regime']==i]) for i in range(4)}
    families=['coherent','isolated','misleading','primary_jump_positive']
    byfamily={f:totals([r for r in rows if r['case']['family']==f]) for f in families}
    paired={}
    for m in METHODS[:-1]:
        g=l=fix=new=0
        for r in rows:
            a,b=endpoint(r,PRIMARY,.01),endpoint(r,m,.01)
            g+=int(a[0]>b[0]);l+=int(a[0]<b[0]);fix+=int(a[1:].sum()<b[1:].sum());new+=int(a[1:].sum()>b[1:].sum())
        paired[m]=dict(correct_gains=g,correct_losses=l,error_fixes=fix,new_errors=new)
    operating={str(a):totals(rows,a) for a in [.005,.01,.02]}
    sys.path.insert(0,str(ROOT/'src'))
    from localization_confirmation_statistics import stratified_sums
    ci_controls=['safe_score_original','localized_safe_score_original','localized_safe_score_common','full_opg_refit']
    strata=defaultdict(list)
    for r in rows:
        a=endpoint(r,PRIMARY,.01);vector=[]
        for m in ci_controls:
            b=endpoint(r,m,.01);vector.extend([int(a[0]-b[0]),int(b[1:].sum()-a[1:].sum())])
        strata[(r['case']['regime'],r['case']['family'])].append(vector)
    assert len(strata)==28
    denominators=np.tile([256.,640.],len(ci_controls))
    draws=stratified_sums([np.asarray(strata[k]) for k in sorted(strata)],10000,71190643825)/denominators
    ci={m:dict(correct_gain_interval95=np.quantile(draws[:,2*i],[.025,.975]).tolist(),
               total_error_reduction_interval95=np.quantile(draws[:,2*i+1],[.025,.975]).tolist()) for i,m in enumerate(ci_controls)}
    validation=json.loads((ROOT/'results/null_information_bank_v1/validation.json').read_text())
    costs=json.loads((ROOT/'results/localization_test_runtime_v1.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    schematic()
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.3),gridspec_kw={'wspace':.42})
    vals=validation['absolute_log_variance_ratio_quantiles']
    # Saved quantile ordering inspected/verified below; preserve it verbatim.
    assert set(vals)=={'median','p95','max'}
    v=[vals[k] for k in ['median','p95','max']]
    assert np.allclose(v,[.05197,.13131,.38426],atol=1e-5)
    axes[0].bar(np.arange(3),v,color='#176b87',width=.5)
    axes[0].set(xticks=np.arange(3),xticklabels=['Median','95th percentile','Maximum'],ylabel='Absolute log variance ratio',ylim=(0,.48),title='(a) Null-bank approximation')
    for i,y in enumerate(v):axes[0].text(i,y+.015,f'{y:.3f}',ha='center',fontsize=7)
    information=json.loads((ROOT/'results/ct_secondary_information_v1/analysis.json').read_text())
    assert information['records_total']==2000
    directions=[next(d for d in g['directions'] if d['gap']==gap)
                for name,gap in [('bright_hidden',10.),('weak_hidden',20.)]
                for g in information['groups'] if g['regime']==name]
    assert len(directions)==2
    means=np.array([d['ratio'] for d in directions]);low=np.array([d['paired_path_bootstrap95'][0] for d in directions]);high=np.array([d['paired_path_bootstrap95'][1] for d in directions])
    axes[1].errorbar(np.arange(2),means,yerr=np.stack([means-low,high-means]),fmt='o',color='#176b87',capsize=4)
    axes[1].axhline(1,color='#90979c',lw=.8,ls='--');axes[1].set(xticks=[0,1],xticklabels=['Bright hidden','Weak hidden'],ylabel='Full / known-active information',ylim=(.9,2.15),xlim=(-.5,1.5),title='(b) Fixed secondary directions')
    save(fig,'fig02_information')
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.65),gridspec_kw={'wspace':.36})
    methods=['localized_safe_score_original','full_opg_refit',PRIMARY];names=['Localized censored','Empirical full / refit','Proposed']
    x=np.arange(4);w=.23
    for i,(m,name,color) in enumerate(zip(methods,names,COLORS)):
        axes[0].bar(x+(i-1)*w,[100*byregime[str(g)][m][0]/64 for g in range(4)],width=w,color=color,label=name)
    axes[0].set(xticks=x,xticklabels=['Moderate','Near','Bright\nhidden','Weak\nhidden'],ylabel='Correct localizations (%)',ylim=(0,110),title='(a) All recovery regimes')
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,frameon=False,loc='upper center',
               bbox_to_anchor=(.5,1.10),ncol=3,fontsize=7)
    delta=[byfamily[f][PRIMARY][0]-byfamily[f]['localized_safe_score_original'][0] for f in families]
    axes[1].bar(np.arange(4),delta,color='#176b87',width=.55)
    axes[1].set(xticks=np.arange(4),xticklabels=['Coherent','Isolated','Misleading','Primary\njump'],ylabel='Additional correct localizations',title='(b) All positive families',ylim=(min(0,min(delta))-1,max(delta)+2))
    for i,y in enumerate(delta):axes[1].text(i,y+.25,f'{y:+d}',ha='center',fontsize=8)
    save(fig,'fig03_detection')
    fig,axes=plt.subplots(1,2,figsize=(6.95,2.6),gridspec_kw={'wspace':.38});alphas=np.array([.005,.01,.02])
    for m,name,color in zip(methods,names,COLORS):
        axes[0].plot(100*alphas,[100*operating[str(a)][m][0]/256 for a in alphas],'-o',lw=1.2,ms=3,color=color,label=name)
        axes[1].plot(100*alphas,[100*operating[str(a)][m][2]/384 for a in alphas],'-o',lw=1.2,ms=3,color=color,label=name)
    for ax in axes:ax.set_xlabel('Nominal significance level (%)');ax.set_xticks([.5,1,2]);ax.axvline(1,color='#adb3b7',lw=.7,ls=':')
    axes[0].set(ylabel='Correct localizations (%)',title='(a) Detection sensitivity');axes[0].legend(frameon=False,fontsize=6.5)
    axes[1].set(ylabel='Null false alarms (%)',title='(b) Observed false-alarm rate',ylim=(0,None))
    save(fig,'fig04_operating_points')
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.65),gridspec_kw={'wspace':.42});x=np.arange(4)
    pc=[g['primary_total_charged_seconds'] for g in costs['groups']];rc=[g['reference_total_charged_seconds'] for g in costs['groups']];ratios=np.array(rc)/pc
    for g,q in zip(costs['groups'],ratios):assert np.isclose(q,g['reference_to_primary_charged_ratio'])
    axes[0].bar(x-.17,pc,width=.34,color='#176b87',label='Score + localization');axes[0].bar(x+.17,rc,width=.34,color='#8a9299',label='Full LR reference')
    axes[0].set(yscale='log',ylabel='Charged computation (s)',title='(a) Calibrated-test components');axes[0].legend(frameon=False,fontsize=6.6)
    axes[1].bar(x,ratios,width=.55,color='#176b87');axes[1].set(ylabel='Reference / proposed cost',ylim=(0,10),title='(b) Charged cost ratio')
    for i,y in enumerate(ratios):axes[1].text(i,y+.2,f'{y:.2f}',ha='center',fontsize=7)
    for ax in axes:ax.set(xticks=x,xticklabels=['Moderate','Near','Bright\nhidden','Weak\nhidden'])
    save(fig,'fig05_cost')
    table=['\\begin{table*}[t]\\centering','\\caption{Complete detection and localization results at $\\alpha=0.01$. Correct and wrong-location counts use 256 positive queries; null false alarms use 384 null queries. Percentages are calculated using their respective denominators. All methods use the same query acquisitions.}','\\label{tab:all_results}','\\begin{tabular}{lrrrrr}\\toprule Method / calibration & Correct & Correct (\\%) & Wrong location & Null false & Null false (\\%)\\\\\\midrule']
    for m,label in zip(METHODS,LABELS):
        c,w,f=observed[m];row=f'{label} & {c} & {100*c/256:.2f} & {w} & {f} & {100*f/384:.2f}'
        if m==PRIMARY:row='\\bfseries '+row
        table.append(row+'\\\\')
    table+=['\\bottomrule\\end{tabular}','\\end{table*}']
    (DEST/'table_results.tex').write_text('\n'.join(table)+'\n')
    inputs=[dbpath,*[mainout/n for n in ['completion.json','analysis.json','integrity.json','parent_failure_audit.json','failure_audit.json']],ROOT/'SCORE_PEAK_LOCALIZATION_DEVELOPMENT_PROTOCOL.json',ROOT/'results/null_information_bank_v1/validation.json',ROOT/'results/localization_test_runtime_v1.json',ROOT/'output/continuous_localization_results_source.json']
    inputs.extend([ROOT/'src/localization_confirmation_statistics.py',ROOT/'results/ct_secondary_information_v1/analysis.json'])
    result=dict(status='SAVED_COMPLETE_RESULTS_REANALYSIS_NO_NEW_EXPERIMENTS',cases=640,positive=256,null=384,methods=METHODS,labels=dict(zip(METHODS,LABELS)),totals=observed,by_regime=byregime,by_positive_family=byfamily,paired=paired,operating_points=operating,regimes=regimes,engineering_costs=[{k:g[k] for k in ['regime','primary_total_charged_seconds','reference_total_charged_seconds','reference_to_primary_charged_ratio']} for g in costs['groups']],input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},asset_builder_sha256=sha(__file__),figures={p.name:sha(p) for p in FIG.glob('*')},posthoc_descriptive_uncertainty=dict(replicates=10000,seed=71190643825,strata=28,contrasts=ci,coverage='Approximate paired stratified percentile95 intervals, posthoc article reanalysis, unadjusted. Not independent confirmation, simultaneous testing, physical/general-population or exact rare-event coverage.'),scope='Existing frozen development only. No new fits/paths, partial-full-LR efficacy, threshold promotion, confirmation or physical superiority.')
    (DEST/'article_data.json').write_text(json.dumps(result,indent=2))
    assert sha(dbpath)==before
    print(json.dumps(dict(primary=observed[PRIMARY],byregime=byregime,paired=paired['localized_safe_score_original'],figures=len(result['figures'])),indent=2))


if __name__=='__main__':main()
