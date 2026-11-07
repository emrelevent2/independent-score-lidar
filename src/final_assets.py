"""Generate manuscript numbers and figures from completed acquisition records."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_confirmation import interval

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/'confirmation_v1';M=ROOT/'manuscript'
F=M/'figures';F.mkdir(exist_ok=True)
A=json.loads((R/'analysis.json').read_text())
assert A['caspi_cases_available']==579 and 'rt3d_authors_default' not in A['methods']
names={'proposed':'Independent evidence','local_likelihood':'Local likelihood','raw':'Raw flux fit',
 'box3':'Box 3 x 3','depth_guided_3':'Depth-guided 3','depth_guided_7':'Depth-guided 7',
 'caspi':'CASPI + flux fit','rt3d_25':'RT3D native, 25','rt3d_100':'RT3D native, 100'}
order=['proposed','local_likelihood','raw','box3','depth_guided_3','depth_guided_7','rt3d_25','rt3d_100']
order=[m for m in order if m in A['methods']]
colors={'proposed':'#007f80','local_likelihood':'#ae3f32','raw':'#5068ad'}
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})

def pct(x):return f'{100*x:.1f}'
def texname(m):return names[m].replace(' x ',r'$\times$')
def table(path,methods,subset=False):
 lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & Null reports & Recall (\%) & Wrong & Cal.\\',r'\midrule']
 for m in methods:
  d=A['methods'][m];s=d['summaries']['paired_caspi_subset'] if subset else d['summaries']
  n,p=s['null'],s['positive']
  lines.append(f"{texname(m)} & {n['declared']['count']}/{n['n']} & {pct(p['localized']['rate'])} & {p['wrong_depth']['count']} & {d['calibration_n']}\\\\")
 lines +=[r'\bottomrule',r'\end{tabular}']
 (M/path).write_text('\n'.join(lines))
table('full_table.tex',order)
table('paired_table.tex',['proposed','local_likelihood','raw','box3','caspi'],True)

fig,axes=plt.subplots(1,3,figsize=(10,2.9),gridspec_kw={'width_ratios':[1.1,1.1,1.25]})
for m in ['proposed','local_likelihood','raw']:
 s=A['methods'][m]['summaries']
 p=[s['photon_levels'][str(float(n))]['localized'] for n in [3,6,10]]
 y=np.array([r['rate'] for r in p]);ci=np.array([r['ci95'] for r in p]).T
 axes[0].errorbar([3,6,10],y*100,yerr=(np.vstack([y,y])-ci)*np.array([[1],[-1]])*100,
                  marker='o',capsize=2,label=names[m],color=colors[m])
axes[0].set(xlabel='Incident secondary photons',ylabel='Localized recall (%)',ylim=(-2,102),xticks=[3,6,10],title='(a) Sensitivity')
axes[0].legend(fontsize=9,loc='upper left')
for j,m in enumerate(['proposed','local_likelihood','raw']):
 s=A['methods'][m]['summaries']['null']['declared'];lo,hi=s['ci95']
 axes[1].errorbar(j,s['rate']*100,yerr=[[100*(s['rate']-lo)],[100*(hi-s['rate'])]],
                  fmt='o',capsize=3,color=colors[m])
axes[1].axhline(1,color='0.5',ls='--',lw=1)
axes[1].set(xticks=range(3),xticklabels=['Evidence','Likelihood','Raw'],ylabel='Null declarations (%)',ylim=(-.1,3),title='(b) Local false reports')
xx=np.arange(5)
for m in ['proposed','local_likelihood']:
 s=A['methods'][m]['summaries']['positive_geometries']
 y=np.array([s[str(g)]['localized']['rate'] for g in xx]);ci=np.array([s[str(g)]['localized']['ci95'] for g in xx]).T
 axes[2].errorbar(xx+(0 if m=='proposed' else .08),y*100,
  yerr=np.vstack([y-ci[0],ci[1]-y])*100,marker='o',capsize=2,color=colors[m],label=names[m])
axes[2].set(xticks=xx,xticklabels=['Disc','Primary\nstep','Second\nstep','3 x 3','Curved\nstripe'],
 ylabel='Localized recall (%)',ylim=(-2,85),title='(c) Geometry dependence')
fig.tight_layout();fig.savefig(F/'confirmation.pdf');fig.savefig(F/'confirmation.png',dpi=160);plt.close(fig)

records=[json.loads((R/(r['id']+'.json')).read_text()) for r in A['per_case']['proposed']]
ev=[r['reports']['proposed']['evidence'] for r in records]
diag=dict(n=len(ev),gap_quantiles=dict(zip(['min','q25','median','q75','q95','max'],np.quantile([e['gap'] for e in ev],[0,.25,.5,.75,.95,1]).tolist())),
 boxes_quantiles=np.quantile([e['boxes_evaluated'] for e in ev],[0,.5,1]).tolist(),
 conservative_fallbacks=sum(e['conservative_fallbacks'] for e in ev),
 acquisitions_with_fallback=sum(e['conservative_fallbacks']>0 for e in ev),
 total_optimizer_failures=sum(e['optimizer_failures'] for e in ev),
 acquisitions_with_optimizer_failures=sum(e['optimizer_failures']>0 for e in ev),
 target_gap_reached=sum(e['gap']<=.01 for e in ev),
 invalid_proposals=sum(not r['reports']['proposed']['proposal_parameters']['valid'] for r in records),
 note='Optimizer failures count evaluated cells, not invalid numerical bounds. Conservative certificates/fallbacks remain enforced.')
(R/'diagnostics.json').write_text(json.dumps(diag,indent=2))
fig,ax=plt.subplots(1,2,figsize=(7,2.6))
for positive,label,color in [(False,'Single return','#007f80'),(True,'Two returns','#ae3f32')]:
 vals=np.array([r['reports']['proposed']['evidence']['gap'] for r,c in zip(records,A['per_case']['proposed']) if c['positive']==positive])
 vals=np.sort(vals);ax[0].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label,color=color)
ax[0].set(xlabel='Null log-likelihood bound gap',ylabel='Cumulative fraction',xscale='log',title='(a) Computational conservatism')
ax[0].legend(fontsize=9)
for positive,label,color in [(False,'Single return','#007f80'),(True,'Two returns','#ae3f32')]:
 vals=[r['reports']['proposed']['score'] for r,c in zip(records,A['per_case']['proposed']) if c['positive']==positive]
 edges=np.linspace(-100,40,45)
 ax[1].hist(vals,bins=edges,weights=np.ones(len(vals))/(len(vals)*(edges[1]-edges[0])),
            histtype='step',label=label,color=color)
ax[1].axvline(np.log(100),ls='--',color='black',lw=1)
ax[1].set(xlabel='Certified lower log evidence',ylabel='Fraction / log unit',title='(b) Fixed decision threshold')
fig.tight_layout();fig.savefig(F/'bound_diagnostics.pdf');fig.savefig(F/'bound_diagnostics.png',dpi=160);plt.close(fig)

# Exhaustive interval tables, including the adverse conditions, not a selected subset.
lines=[r'\begin{longtable}{llrrr}',r'\toprule',r'Method & Group & $n$ & Count & Rate (95\% CI)\\\midrule\endhead']
for m in order+['caspi']:
 s=A['methods'][m]['summaries']
 for group,key in [('null','declared'),('positive','localized')]:
  v=s[group];z=v[key];lo,hi=z['ci95']
  lines.append(f"{texname(m)} & {group} & {v['n']} & {z['count']} & {pct(z['rate'])} ({pct(lo)}, {pct(hi)})\\\\")
 for collection,key in [('null_contexts','declared'),('photon_levels','localized'),('positive_geometries','localized')]:
  for group,v in s[collection].items():
   if v is None:continue
   z=v[key];lo,hi=z['ci95'];group=group.replace('_',r'\_')
   category=collection.replace('_',r'\_')
   lines.append(f" & {category}: {group} & {v['n']} & {z['count']} & {pct(z['rate'])} ({pct(lo)}, {pct(hi)})\\\\")
lines +=[r'\bottomrule\end{longtable}']
(M/'interval_tables.tex').write_text('\n'.join(lines))
lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Comparator & Recall difference (pp) & Null-report difference (pp)\\\midrule']
for m,d in A['paired_differences'].items():
 parts=[]
 for key in ['recall','false_declarations']:
  v=d[key];lo,hi=v['ci95'];parts.append(f"{100*v['difference']:.1f} ({100*lo:.1f}, {100*hi:.1f})")
 lines.append(texname(m)+' & '+' & '.join(parts)+r'\\')
lines +=[r'\bottomrule\end{tabular}']
(M/'paired_intervals.tex').write_text('\n'.join(lines))
print(json.dumps(diag,indent=2))
