"""Saved-result-only scientific plot of the prospective optical-lobe study."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'results/shoulder_confirmation_v1/analysis.json').read_text())
    audit=json.loads((ROOT/'results/shoulder_confirmation_v1/integrity.json').read_text())
    assert audit['status']=='PASS' and audit['protocol_sha256']==data['protocol_sha256']
    methods=['old_gaussian','new_uncalibrated','cal_100000']
    labels=['Gaussian\ncontrol','IRF interval\nno cal.','IRF interval\n100k cal.']
    sources=['results/shoulder_confirmation_v1/analysis.json','results/shoulder_confirmation_v1/integrity.json']
    totals=dict(data['totals'])
    point_dir=ROOT/'results/point_irf_comparison_v1'
    if (point_dir/'integrity.json').exists():
        pa=json.loads((point_dir/'analysis.json').read_text());pv=json.loads((point_dir/'integrity.json').read_text())
        assert pv['status']=='PASS' and pa['protocol_sha256']==pv['protocol_sha256']
        methods.append('point_irf');labels.append('Point IRF\n100k cal.');totals['point_irf']=pa['totals']
        sources.extend(['results/point_irf_comparison_v1/analysis.json','results/point_irf_comparison_v1/integrity.json'])
    count=len(methods)
    correct=np.array([totals[m]['correct'] for m in methods]);wrong=np.array([totals[m]['wrong_location'] for m in methods])
    false=np.array([totals[m]['false_objects'] for m in methods]);missed=384-correct-wrong
    assert np.all(missed>=0) and np.all((0<=false)&(false<=384))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.6));x=np.arange(count)
    bottom=np.zeros(count)
    for values,label,color in [(correct,'Correct pair','#277da8'),(wrong,'Wrong location','#dc6944'),(missed,'No second object declared','#dde2e5')]:
        axes[0].bar(x,values,bottom=bottom,color=color,label=label,width=.66)
        for j,v in enumerate(values):
            if v>=8:axes[0].text(j,bottom[j]+v/2,str(v),ha='center',va='center',color='white' if color!='#dde2e5' else '#25313a')
            elif v>0:axes[0].annotate(str(v),(j,bottom[j]+v/2),xytext=(16,12),textcoords='offset points',ha='left',va='bottom',arrowprops={'arrowstyle':'-','color':color})
        bottom+=values
    axes[1].bar(x,false,color='#dc6944',label='False second object',width=.66)
    axes[1].bar(x,384-false,bottom=false,color='#dde2e5',label='No false second object',width=.66)
    for j,v in enumerate(false):
        axes[1].annotate(str(v),(j,max(v,3)),ha='center',va='bottom',xytext=(0,3),textcoords='offset points')
    axes[0].set_title('Two physical surfaces (384 cases)');axes[1].set_title('One physical surface (384 cases)')
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=9.5);ax.set_ylim(0,425);ax.set_ylabel('Number of acquisitions');ax.set_yticks([0,96,192,288,384]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),frameon=False,fontsize=9)
    fig.suptitle('Prospective optical-response stress test on new simulated acquisitions',fontsize=12)
    fig.text(.5,.025,'Gaussian control is misspecified on lobe scenes. Interval and point calibration each use 100,000 additional cycles.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.84,bottom=.31,wspace=.25)
    out=ROOT/'output/shoulder_confirmation';out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/f'correct_and_false_objects.{ext}',dpi=180)
    plt.close(fig)
    (out/'figure_provenance.json').write_text(json.dumps(dict(protocol_sha256=data['protocol_sha256'],sources_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Designed simulation stress test, not physical-sensor validation'),indent=2))
    print(out/'correct_and_false_objects.png')


if __name__=='__main__':main()
