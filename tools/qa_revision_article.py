"""Independent counts, provenance, text/font and final-page checks."""
from pathlib import Path
import json,sqlite3,hashlib,re,sys,unicodedata
import numpy as np
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'manuscript/ded_style_article_v2'
PDF=ROOT/'output/pdf/weak_return_two_recovery_revised.pdf'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    with sqlite3.connect(f'file:{p.as_posix()}?mode=ro',uri=True) as db:return [json.loads(v[0]) for v in db.execute('SELECT payload FROM records ORDER BY rowid')]

def hit(r,z,a):
    pos=r['case']['positive'];decl=z['pvalue']<=a
    dist=lambda x,y:abs((x-y+32)%64-32)
    ok=pos and decl and max(dist(z['primary'],r['truth']['primary']),dist(z['delay'],r['truth']['secondary']))<=2
    return [int(ok),int(pos and decl and not ok),int(not pos and decl)]

def main():
    data=json.loads((D/'article_data.json').read_text());assert data['status']=='COMPLETE'
    for name,digest in data['input_hashes'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'tools/build_revision_article.py')==data['builder_sha256']
    for f in ['revision_censored_bank_audit.json','revision_simulation_tests.json']:
        assert json.loads((ROOT/'results'/f).read_text())['status']=='PASS'
    for kind in ['power','robust']:
        a=json.loads((ROOT/'results/revision_sensitivity_v1'/(kind+'_audit.json')).read_text());assert a['status']=='PASS'
        assert a['records_sha256']==sha(ROOT/'results/revision_sensitivity_v1'/(kind+'.sqlite3'))
        assert a['protocol_sha256']==sha(ROOT/'REVISION_SENSITIVITY_PROTOCOL_20261107.json')
    a=json.loads((ROOT/'results/revision_exposure_bank_v1/audit.json').read_text());assert a['status']=='PASS' and a['records_sha256']==sha(ROOT/'results/revision_exposure_bank_v1/records.sqlite3')
    a=json.loads((ROOT/'results/revision_safe_robustness_v1/audit.json').read_text());assert a['status']=='PASS'
    assert a['records_sha256']==sha(ROOT/'results/revision_safe_robustness_v1/records.sqlite3') and a['main_robust_records_sha256']==sha(ROOT/'results/revision_sensitivity_v1/robust.sqlite3')
    for file in ['postfailure_full_joint_integrity.json','postfailure_full_joint_failure_audit.json']:
        a=json.loads((ROOT/'results/localization_postfailure_reference_v1'/file).read_text());assert a['status']=='PASS'
        assert a['reference_records_sha256']==sha(ROOT/'results/localization_postfailure_reference_v1/equal_information_joint_reference_v2.sqlite3')
    rows=read(ROOT/'results/score_peak_localization_development_v1/records.sqlite3')
    for folder,name in [('revision_censored_bank_v1','safe_bank_original'),('revision_censored_bank_common_v1','safe_bank_common')]:
        rr=read(ROOT/'results'/folder/'records.sqlite3');assert len(rr)==640
        for r,a in zip(rows,rr):assert r['case']==a['case'];r['samples'][name]=a['sample']
    rr=read(ROOT/'results/revision_exposure_bank_v1/records.sqlite3');assert len(rr)==1280
    for branch in ['original','common']:
        for r,a in zip(rows,[x for x in rr if x['branch']==branch]):assert r['case']==a['case'];r['samples']['safe_exposure_'+branch]=a['sample']
    rr=read(ROOT/'results/localization_postfailure_reference_v1/equal_information_joint_reference_v2.sqlite3');assert len(rr)==640
    for r,a in zip(rows,rr):assert r['case']==a['case'];r['samples'].update(a['samples'])
    ranks=0
    for alpha,expected in data['operating_points'].items():
        for m,v in expected.items():
            counts=np.zeros(3,dtype=int)
            for r in rows:
                z=r['samples'][m];counts+=hit(r,z,float(alpha))
                assert len(z['bootstrap_scores'])==199 and z['pvalue']==(1+sum(b>=z['score'] for b in z['bootstrap_scores']))/200;ranks+=1
            assert counts.tolist()==v,(alpha,m)
    for i,expected in data['by_regime'].items():
        for m,v in expected.items():assert np.sum([hit(r,r['samples'][m],.01) for r in rows if r['case']['regime']==int(i)],axis=0).tolist()==v
    for m,curve in data['empirical_operating_curves'].items():
        indices=[round(r['samples'][m]['pvalue']*200) for r in rows]
        correct=[hit(r,r['samples'][m],1.)[0] for r in rows]
        false=[int(not r['case']['positive']) for r in rows]
        c=np.cumsum(np.bincount(indices,weights=correct,minlength=201))/256*100
        f=np.cumsum(np.bincount(indices,weights=false,minlength=201))/384*100
        assert np.allclose(c,curve['correct_localization_percent'],atol=1e-12)
        assert np.allclose(f,curve['false_alarm_percent'],atol=1e-12)
        assert np.all(np.diff(c)>=0) and np.all(np.diff(f)>=0)
    power=read(ROOT/'results/revision_sensitivity_v1/power.sqlite3');assert len(power)==480
    pd=json.loads((D/'power_data.json').read_text());assert len(pd['cells'])==24
    for cell in pd['cells']:
        selected=[r for r in power if r['case']['positive'] and r['case']['ratio']==cell['ratio'] and r['case']['separation']==cell['separation']]
        assert len(selected)==cell['n']==32
        counts=np.sum([hit(r,r['samples'][cell['method']],.01) for r in selected],axis=0)
        assert counts[:2].tolist()==[cell['correct'],cell['wrong']]
    for m,v in pd['null_counts'].items():assert sum(hit(r,r['samples'][m],.01)[2] for r in power if not r['case']['positive'])==v
    robust=read(ROOT/'results/revision_sensitivity_v1/robust.sqlite3');original=read(ROOT/'results/revision_safe_robustness_v1/records.sqlite3')
    assert len(robust)==len(original)==864;rd=json.loads((D/'robustness_data.json').read_text());assert len(rd)==9
    for item in rd:
        for m in ['full','safe','safe_original']:
            rr=[r for r in (original if m=='safe_original' else robust) if r['case']['condition']['name']==item['condition']]
            assert len(rr)==96
            ip=sum(r['case']['positive'] and r['status']!='EVALUATED' for r in rr);inn=sum(not r['case']['positive'] and r['status']!='EVALUATED' for r in rr)
            assert [ip,inn]==item['incompatible_by_method'][m]
            counts=np.sum([hit(r,r['samples'][m],.01) for r in rr if r['status']=='EVALUATED'],axis=0).astype(int).tolist() if any(r['status']=='EVALUATED' for r in rr) else [0,0,0]
            assert counts==item['valid_counts'][m]
    md=json.loads((D/'measured_data.json').read_text());assert md['status']=='COMPLETE'
    for name,digest in md['input_hashes'].items():assert sha(ROOT/name)==digest,name
    assert len(md['rows'])==2 and all(r['evaluation']['random_finite']==20000 and r['all_gaps_below50']==0 for r in md['rows'])
    for row in md['rows']:
        fresh=row['fresh_evaluation'];z=np.load(ROOT/'results/measured_cluster_v1'/(Path(row['file']).stem+'.npz'))
        assert fresh['all_finite'] and len(z['gaps'])==20000 and np.all(np.isfinite(z['logdensity']))
        assert abs(float(np.mean(z['logdensity']-z['baseline_logdensity']))-fresh['mean_log_score_gain'])<1e-12
        for name,key in [('predicted','cdf_distance'),('baseline_predicted','baseline_cdf_distance')]:
            pooled=np.sort(np.r_[z['gaps'],z[name]])
            distance=np.max(np.abs(np.searchsorted(np.sort(z['gaps']),pooled,side='right')/len(z['gaps'])-np.searchsorted(np.sort(z[name]),pooled,side='right')/len(z[name])))
            assert abs(distance-fresh[key])<1e-12
    assert sum(row['all_gaps'] for row in md['rows'])==md['total_measured_gaps']==10394778
    reader=PdfReader(PDF);text=' '.join(p.extract_text() or '' for p in reader.pages);plain=' '.join(unicodedata.normalize('NFKC',text).split())
    compact=re.sub(r'\s+','',plain)
    for m,v in data['totals'].items():
        calibration='Own-null' if m.endswith('_original') else 'Common'
        expected=re.sub(r'\s+','',data['labels'][m]+calibration+''.join(map(str,v)))
        assert expected in compact,(m,expected)
    for word in ['Appendix','Appendices','REVISION_IN_PROGRESS','provisional','in progress','still running','INSERT_RESULT','TODO','TBD','author review and approval','all-comparator criterion','editable manuscript','The new bright-hidden','A new amplitude','legacy solution','whole-worker','flagged draws','incompatible']:
        assert word.lower() not in plain.lower(),word
    for value in ['203','186','187','14/64','30/64','0.0002213','25','1783','3224376','76.128760','0.2449','0.0366','0.7522','10394778','40000','emre.levent@fbu.edu.tr']:
        assert value in plain,value
    assert re.search(r'V\s*ecdi Emre Levent',plain) and 'no competing interests' in plain and 'no external funding' in plain
    for i in range(1,7):assert re.search(r'Fig\.\s*'+str(i)+r'\.',plain)
    body=' '.join(p.read_text() for p in D.glob('*.tex') if p.name!='references.tex')
    assert not re.search(r'\\appendic|Appendix',body,re.I)
    # Control characters catch accidental Python string escapes in generated TeX.
    for p in D.glob('*.tex'):assert not any(ord(c)<32 and c not in '\n\r\t' for c in p.read_text()),p
    refs=(D/'references.tex').read_text();assert len(re.findall(r'\\bibitem\{',refs))==13
    master=(D/'weak_return_two_recovery_revised.tex').read_text();ordered=[]
    for filename in re.findall(r'\\input\{([^}]+)\}',master):
        if filename=='references.tex':continue
        for citation in re.findall(r'\\cite\{([^}]+)\}',(D/filename).read_text()):
            for key in citation.split(','):
                if key not in ordered:ordered.append(key)
    assert ordered==re.findall(r'\\bibitem\{([^}]+)\}',refs),ordered
    log=(ROOT/'output/pdf/weak_return_two_recovery_revised.log').read_text(errors='replace')
    assert 'Overfull' not in log and not re.search(r'LaTeX Warning: (?:Reference|Citation).*undefined',log)
    fonts=set()
    for page in reader.pages:
        for f in page['/Resources'].get_object().get('/Font',{}).get_object().values():
            fonts.add(str(f.get_object().get('/BaseFont','')))
    assert any('TimesNewRoman' in n for n in fonts),fonts
    assert not any('DejaVu' in n for n in fonts),fonts
    figure_fonts={}
    for p in sorted((D/'figures').glob('*.pdf')):
        ff=set()
        for page in PdfReader(p).pages:
            for f in page['/Resources']['/Font'].get_object().values():ff.add(str(f.get_object().get('/BaseFont','')))
        assert ff and all('TimesNewRoman' in n for n in ff),(p,ff);figure_fonts[p.name]=sorted(ff)
    reference=PdfReader(ROOT/'data/reference_ded_2026.pdf');reftext=' '.join(p.extract_text() or '' for p in reference.pages[:16])
    words=lambda s:re.findall(r'[a-z]{2,}',unicodedata.normalize('NFKC',s).lower())
    rw,bw=words(reftext),words(body);grams={tuple(rw[i:i+12]) for i in range(len(rw)-11)}
    overlaps=sorted({' '.join(bw[i:i+12]) for i in range(len(bw)-11) if tuple(bw[i:i+12]) in grams});assert not overlaps,overlaps[:5]
    images=sorted((D/'qa').glob('page-*.png'));assert len(images)==len(reader.pages)
    visual='--visual-verified' in sys.argv
    result=dict(status='PASS' if visual else 'PENDING_VISUAL_REVIEW',pages=len(reader.pages),word_count=len(plain.split()),appendices=0,figures=6,references=13,rank_checks_including_three_levels=ranks,all_complete_comparisons_recounted=True,all_regimes_and_nulls_recounted=True,source_input_hashes_verified=True,body_fonts=sorted(fonts),figure_fonts=figure_fonts,reference_body_12word_overlap=overlaps,all_final_pages_visually_inspected=visual,pdf_sha256=sha(PDF),data_sha256=sha(D/'article_data.json'),source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in D.glob('*.tex')},page_hashes={p.name:sha(p) for p in images},qa_source_sha256=sha(__file__))
    (ROOT/'output/revised_article_qa.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','pages','word_count','figures','rank_checks_including_three_levels','pdf_sha256']},indent=2))

if __name__=='__main__':main()
