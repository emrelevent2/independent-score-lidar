"""Current routing figures and manuscript source tables from frozen outputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'routing_confirmation_v1';M=ROOT/'manuscript';FIG=M/'figures'
names={'routed':'Support-routed report','local_fit':'Ordinary local fit','spatial_report':'Prior spatial report',
    'guided_fit':'Guided query refinement','naive_union':'Naive union','disagreement':'Disagreement ablation','fixed_secondary':'Fixed-secondary profile'}


def main():
    a=json.loads((OUT/'analysis.json').read_text());methods=a['methods'];records=json.loads((OUT/'endpoint_records.json').read_text())
    cases=json.loads((ROOT/'data'/'routing_confirmation_v1'/'manifest.json').read_text())['cases']
    order=['routed','spatial_report','local_fit','guided_fit','naive_union','disagreement','fixed_secondary']
    rows=[r'\begin{tabular}{lrrrrr}\toprule',r'Method & Coherent /900 & Wrong & Singleton /600 & Wrong & Null false /3000\\\midrule']
    stress=[r'\begin{tabular}{lrrrr}\toprule',r'Method & Filament /300 & Wrong & Crossing step /300 & Wrong\\\midrule']
    for name in order:
        s=methods[name+'_cal0.005']['summaries']
        values=[s['coherent']['correct']['count'],s['coherent']['wrong_depth']['count'],s['singleton']['correct']['count'],s['singleton']['wrong_depth']['count'],s['null']['declared']['count']]
        rows.append(names[name]+' & '+' & '.join(map(str,values))+r'\\')
        values=[s['filament']['correct']['count'],s['filament']['wrong_depth']['count'],s['crossing_step']['correct']['count'],s['crossing_step']['wrong_depth']['count']]
        stress.append(names[name]+' & '+' & '.join(map(str,values))+r'\\')
    for content,path in [(rows,M/'routing_table.tex'),(stress,M/'routing_stress_table.tex')]:
        content.append(r'\bottomrule\end{tabular}');path.write_text('\n'.join(content))
    plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.4),layout='constrained')
    selected=['routed','spatial_report','local_fit'];colors=['#176b94','#d35c30','#7854a1']
    for ax,family,title in [(axes[0,0],'coherent','(a) Spatially supported targets'),(axes[0,1],'singleton','(b) Isolated targets')]:
        seq=[c for c in cases if c['kind']=='positive' and c['support_family']==family]
        for name,color in zip(selected,colors):
            rr=records[name+'_cal0.005'][family]
            rates=[np.mean([r['correct'] for r,c in zip(rr,seq) if c['secondary']==level])*100 for level in [3.,6.,10.]]
            ax.plot([3,6,10],rates,'o-',color=color,label=names[name],markersize=3)
        ax.set(xlabel='Incident secondary photons',ylabel='Correct localization (%)',ylim=(0,102),xticks=[3,6,10],title=title)
        ax.legend(fontsize=6.6,loc='upper left' if family=='singleton' else 'lower right')
    x=np.arange(3)
    for i,name in enumerate(selected):
        s=methods[name+'_cal0.005']['summaries']
        axes[1,0].bar(i-.24,s['null']['declared']['count'],.24,color='#176b94',label='Null false' if i==0 else None)
        axes[1,0].bar(i,s['coherent']['wrong_depth']['count'],.24,color='#d35c30',label='Coherent wrong depth' if i==0 else None)
        axes[1,0].bar(i+.24,s['singleton']['wrong_depth']['count'],.24,color='#7854a1',label='Singleton wrong depth' if i==0 else None)
    axes[1,0].set(xticks=x,xticklabels=['Routed','Spatial','Local'],ylabel='Error count',title='(c) Distinct error types',ylim=(0,25));axes[1,0].legend(fontsize=6.3)
    for i,(name,color) in enumerate(zip(selected,colors)):
        s=methods[name+'_cal0.005']['summaries']
        axes[1,1].bar(np.array([0,1])+(i-1)*.23,[100*s[g]['correct']['rate'] for g in ['filament','crossing_step']],.23,color=color,label=names[name])
    axes[1,1].set(xticks=[0,1],xticklabels=['Filament','Crossing step'],ylabel='Correct localization (%)',ylim=(0,100),title='(d) Prespecified new geometries');axes[1,1].legend(fontsize=6.3,loc='upper right')
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf']:fig.savefig(FIG/f'routing_confirmation.{ext}',dpi=220)
    plt.close(fig)
    p=methods['routed_cal0.005']['summaries'];local=methods['local_fit_cal0.005']['summaries'];spatial=methods['spatial_report_cal0.005']['summaries']
    note=f'''# Yeni bağımsız seçim deneyi

3.999 kalibrasyon,3.000 test-null,900 komşulukta devam eden hedef,
600 tek piksellik hedef,300 ince çizgi ve300 derinlik-sınırı stresi.
Yöntem ve eşik kuralı9.099yeni veri üretilmeden donduruldu.

| Yöntem | Devam eden hedefte doğru /900 | Yanlış konum | Tek piksellik doğru /600 | Yanlış konum | Yanlış alarm /3000 |
|---|---:|---:|---:|---:|---:|
| Seçim yapan yeni yöntem |659|0|306|11|15|
| Önceki komşuluk yöntemi |644|0|8|13|13|
| Standart yerel yöntem |483|17|318|11|10|

Yeni yöntem komşulardaki ikinci-yüzey desteği yeterliyse mekânsal konumu,
yetersizse sorgulanan pikselin yerel kestirimini kullanır. Destek puanı bir
güven aralığı değildir; seçimden sonraki tüm puan yeniden kalibre edilir.

Bu yeni testte önceki yöntemle aynı görüntüler üzerinde tek piksellik doğru
sayısı8->306oldu; devam eden yüzeylerde644->659. Eski testteki702sayısı
ile karşılaştırılamaz: bunlar farklı görüntüler ve farklı kalibrasyondur.

Standart yerel yönteme göre devam eden hedeflerde176daha fazla doğru,
tek piksellik hedeflerde12daha az doğru var. Yanlış alarm10->15; bu artış
saklanmamalıdır. Devam eden hedeflerde yanlış konum17->0,tek pikselliklerde
11->11. Bu test karışımında toplam hata38->26; nüfus geneli kesinlik oranı
veya her sensör için yanlış alarm garantisi değildir.

Yeni derinlik-sınırı stresinde yeni yöntemin10yanlış konumu var; standart
yerel yöntemde3. Bu durum güçlü ama yanlış yerdeki komşu desteğinin hâlâ
zararlı olabildiğini gösterir. Önceden tanımlanmış anlaşmazlık varyantı bu
grupta daha çok doğru buldu, ancak birincil yöntem sonuçlar görüldükten
sonra değiştirilmedi.

Kaynak:results/routing_confirmation_v1/analysis.json.
SHA-256:{hashlib.sha256((OUT/'analysis.json').read_bytes()).hexdigest()}.
'''
    (ROOT/'ROUTING_RESULTS_TR.md').write_text(note,encoding='utf8')
    print('Wrote routing tables, figures and Turkish result note')


if __name__=='__main__':main()
