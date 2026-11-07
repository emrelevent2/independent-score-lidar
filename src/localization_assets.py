"""Deterministic figures, tables and an author-facing result note."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from guided_alternatives import guided_pool
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'localization_confirmation_v1'
M=ROOT/'manuscript';FIG=M/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
names={'spatial_report_cal0.005':'Spatial report (proposed)',
 'complete_guided_cal0.005':'Guided local refinement','local_likelihood_cal0.005':'Ordinary local fit',
 'guided_fast_cal0.005':'Donor-mixture score','guided_certified':'Certified evidence',
 'fixed_secondary_cal0.005':'Fixed-secondary profile','fixed_both_cal0.005':'Fixed-both profile'}
colors=['#176b94','#d35c30','#7854a1','#388668']


def main():
    a=json.loads((OUT/'analysis.json').read_text());records=json.loads((OUT/'endpoint_records.json').read_text())
    methods=a['methods'];keys=['spatial_report_cal0.005','complete_guided_cal0.005','local_likelihood_cal0.005','guided_certified']
    manifest=json.loads((ROOT/'data'/'localization_confirmation_v1'/'manifest.json').read_text())
    positive=[c for c in manifest['cases'] if c['split']=='evaluation' and c['kind']=='positive']
    stress=[c for c in manifest['cases'] if c['split']=='stress']
    fig,axes=plt.subplots(1,3,figsize=(10.3,3.5),layout='constrained')
    for key,color in zip(keys,colors):
        rr=records[key]['positive'];means=[]
        for level in [3.,6.,10.]:
            ids=[i for i,c in enumerate(positive) if c['secondary']==level]
            means.append(100*np.mean([rr[i]['correct'] for i in ids]))
        axes[0].plot([3,6,10],means,'o-',color=color,label=names[key],markersize=4)
    axes[0].set(xlabel='Incident secondary photons',ylabel='Correctly localized (%)',xticks=[3,6,10],ylim=(0,103),title='(a) Main positive queries (n=900)')
    axes[0].legend(fontsize=9,loc='lower right')
    x=np.arange(len(keys));nulls=[methods[k]['summaries']['null']['declared']['count'] for k in keys]
    wrong=[methods[k]['summaries']['positive']['wrong_depth']['count'] for k in keys]
    axes[1].bar(x-.18,nulls,.36,color='#176b94',label='Null false')
    axes[1].bar(x+.18,wrong,.36,color='#d35c30',label='Wrong depth')
    axes[1].set(xticks=x,xticklabels=['Proposed','Guided fit','Local fit','Certified'],ylabel='Errors (count)',title='(b) Distinct failure types')
    axes[1].tick_params(axis='x',labelsize=9);axes[1].set_ylim(0,max(nulls+wrong)*1.4);axes[1].legend(fontsize=9)
    for i,(key,color) in enumerate(zip(keys,colors)):
        ss=methods[key]['summaries']['singleton_stress']['correct']
        axes[2].bar(i,100*ss['rate'],color=color)
        lo,hi=np.array(ss['ci95'])*100
        axes[2].errorbar(i,100*ss['rate'],yerr=[[100*ss['rate']-lo],[hi-100*ss['rate']]],fmt='none',color='black',capsize=3,lw=.8)
    axes[2].set(xticks=x,xticklabels=['Proposed','Guided fit','Local fit','Certified'],ylabel='Correctly localized (%)',ylim=(0,103),title='(c) Singleton stress (n=300)')
    axes[2].tick_params(axis='x',labelsize=9)
    for ax in axes: ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['pdf','png']:fig.savefig(FIG/f'localization_confirmation.{ext}',dpi=220)
    plt.close(fig)
    gained=[c for i,c in enumerate(positive) if records[keys[0]]['positive'][i]['correct'] and not records[keys[1]]['positive'][i]['correct']]
    if gained:
        c=gained[0];r=json.loads((OUT/(c['id']+'.json')).read_text())
        h=np.load(ROOT/'data'/'localization_confirmation_v1'/(c['id']+'.npz'))['histogram']
        pool,_=guided_pool(h,8,8);truth=c['metadata'];d=r['reports']['complete_guided']['delay'];s=r['proposal']['secondary']
        with plt.rc_context({'font.size':8}):
            fig,axes=plt.subplots(2,1,figsize=(3.4,3.2),sharex=True,layout='constrained')
        t=np.arange(256);lo=max(0,truth['secondary_delay']-12);hi=min(255,truth['secondary_delay']+12)
        visible=(t>=lo)&(t<=hi)
        axes[0].step(t[visible],h[8,8][visible],where='mid',color='#333333',lw=1)
        axes[0].set(ylabel='Detections / bin',title=f'First gain case: {c["id"]}')
        axes[1].plot(t[visible],pool[visible]*1000,color='#176b94',lw=1.5);axes[1].set(ylabel='Donor estimate\n(incident / bin)',xlabel='Time bin')
        for ax in axes:
            ax.axvline(truth['secondary_delay'],color='#388668',lw=1.8,label='True delay')
            ax.axvline(s,color='#176b94',ls='--',lw=1.2,label='Donor delay')
            ax.axvline(d,color='#d35c30',ls=':',lw=1.5,label='Local fit')
            ax.axvspan(truth['secondary_delay']-2,truth['secondary_delay']+2,color='#388668',alpha=.09)
            ax.set_xlim(lo,hi)
            ax.tick_params(labelsize=8);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
        axes[0].title.set_size(8)
        axes[1].legend(fontsize=6.5,loc='upper right')
        for ext in ['pdf','png']:fig.savefig(FIG/f'localization_failure_example.{ext}',dpi=220)
        plt.close(fig)
        (OUT/'illustration_selection.json').write_text(json.dumps(dict(id=c['id'],rule='First main positive in input order correctly localized by proposed but not guided-refinement control.',
            purpose='Post-analysis illustration only, not a separate accuracy experiment.',truth=truth['secondary_delay'],donor=s,query_refined=d),indent=2))
    # Compact article table; complete operating points stay in summary.csv.
    order=['spatial_report_cal0.005','complete_guided_cal0.005','local_likelihood_cal0.005',
           'guided_fast_cal0.005','fixed_secondary_cal0.005','fixed_both_cal0.005','guided_certified']
    lines=[r'\begin{tabular}{lrrr}\toprule',r'Method & Correct /900 & Wrong depth & Null false /2000\\\midrule']
    for k in order:
        ss=methods[k]['summaries'];lines.append(f'{names[k]} & {ss["positive"]["correct"]["count"]} & {ss["positive"]["wrong_depth"]["count"]} & {ss["null"]["declared"]["count"]}'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}'])
    (M/'localization_table.tex').write_text('\n'.join(lines))
    all_lines=[r'\begin{tabular}{lrrrrr}\toprule',
        r'Method / nominal level & Correct & Wrong & Null false & Single correct & Single wrong\\\midrule']
    for k,entry in methods.items():
        label=names.get(k,names.get(entry['method']+'_cal0.005',entry['method']))
        label+=(' / analytic1\%' if entry['alpha'] is None else f' / {100*entry["alpha"]:g}'+r'\%')
        ss=entry['summaries'];values=[ss['positive']['correct']['count'],ss['positive']['wrong_depth']['count'],ss['null']['declared']['count'],
            ss['singleton_stress']['correct']['count'],ss['singleton_stress']['wrong_depth']['count']]
        all_lines.append(label+' & '+' & '.join(map(str,values))+r'\\')
    all_lines.append(r'\bottomrule\end{tabular}')
    (M/'localization_all_table.tex').write_text('\n'.join(all_lines))
    p=methods[keys[0]];control=methods[keys[1]];ordinary=methods[keys[2]]
    pc=p['summaries']['positive']['correct']['count'];cc=control['summaries']['positive']['correct']['count'];oc=ordinary['summaries']['positive']['correct']['count']
    pw=p['summaries']['positive']['wrong_depth']['count'];cw=control['summaries']['positive']['wrong_depth']['count'];ow=ordinary['summaries']['positive']['wrong_depth']['count']
    pf=p['summaries']['null']['declared']['count'];cf=control['summaries']['null']['declared']['count'];of=ordinary['summaries']['null']['declared']['count']
    delta=a['comparisons'][keys[1]]['positive_correct'];ci=np.array(delta['ci95'])*100
    note=f'''# Bağımsız doğrulama: doğru konumları artırma, yanlışları azaltma

Yeni yöntem, yüzeyin varlığı için sorgulanan pikselin fotonlarını kullanır;
raporlanan ikinci yüzey konumunu ise bu pikseli tamamen dışlayan komşulardan
alır. En güçlü kontrolle var/yok kararları **birebir aynıdır**. Değişen şey
konum raporudur; eşik gevşetilmemiştir.

Yöntem ve birincil eşik yeni veri üretilmeden donduruldu. 1.999 kalibrasyon
örneği; ayrı 2.000 tek yüzeyli, 900 iki yüzeyli ve 300 tek piksellik stres
örneği. Bunlar simülasyondur; fiziksel sensör doğrulaması değildir.

| Yöntem | Doğru konum /900 | Yanlış konum | Olmayan yüzeye var deme /2000 |
|---|---:|---:|---:|
| Yeni mekânsal konum raporu | {pc} | {pw} | {pf} |
| Güçlü yönlendirmeli yerel uyum | {cc} | {cw} | {cf} |
| Standart yerel uyum | {oc} | {ow} | {of} |

Güçlü kontrole göre doğru sayısı **{pc-cc:+d}**; yanlış konum sayısı
**{pw-cw:+d}**; tek yüzeyli yanlış alarm sayısı **{pf-cf:+d}**.
Eşleştirilmiş doğru oranı farkı %{100*delta['rate_difference']:.2f} puan,
%95 aralık [{ci[0]:.2f}, {ci[1]:.2f}]. Bu aralık gözlenen kalibrasyona
koşulludur; kalibrasyon belirsizliğini de örnekleyen iki aşamalı analiz
analysis.json içindedir.

Standart yerel kontrole göre doğru farkı {pc-oc:+d}, yanlış konum farkı
{pw-ow:+d}, tek yüzeyli yanlış alarm farkı {pf-of:+d}. Her hata türü ayrı
raporlanmalıdır; tasarlanmış örnek karışımındaki toplam hata, genel kullanım
ortamının kesinlik oranı değildir.

**Sınır:** tek piksellik hedeflerde önerilen yöntem
{p['summaries']['singleton_stress']['correct']['count']}/300, standart yerel
yöntem {ordinary['summaries']['singleton_stress']['correct']['count']}/300
doğru konum buldu. Bu nedenle komşulukta uzanan zayıf yüzeyler için elde
edilen sonuç, yalıtılmış hedeflere genellenemez. Stres sonuçları yöntem
seçmek veya eşik ayarlamak için kullanılmadı.

Ampirik kalibrasyonlu bu yöntem, sertifikalı yöntemin piksel bazındaki
koşullu hata garantisini devralmaz. Genel komşuluk kullanımı veya olabilirlik
testi yeni değildir. Asıl makale katkısı sürekli bileşik boş hipotezin
hesaplanabilir sertifikası ve bu pratik konum/varlık ayrımının açık sınırlarla
değerlendirilmesi olarak ele alınmalıdır. Q1 kabulü veya fiziksel üstünlük
kanıtlanmış değildir.

Kaynak: results/localization_confirmation_v1/analysis.json.
Analiz SHA-256: {hashlib.sha256((OUT/'analysis.json').read_bytes()).hexdigest()}.
'''
    (ROOT/'INDEPENDENT_IMPROVEMENT_RESULTS_TR.md').write_text(note,encoding='utf8')
    print('Wrote figures, article table, illustration audit and Turkish result note')


if __name__=='__main__':main()
