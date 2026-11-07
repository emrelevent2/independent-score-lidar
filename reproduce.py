"""Public saved-output replay; no frozen scientific source is modified."""
from pathlib import Path
import argparse,hashlib,json,sqlite3,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
D=ROOT/'manuscript/ded_style_article_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with sqlite3.connect(f'file:{p.as_posix()}?mode=ro',uri=True) as c:
        return [json.loads(x[0]) for x in c.execute('SELECT payload FROM records ORDER BY rowid')]
def endpoint(r,z,a):
    positive=r['case']['positive'];decl=z['pvalue']<=a
    dist=lambda x,y:abs((x-y+32)%64-32)
    correct=positive and decl and max(dist(z['primary'],r['truth']['primary']),dist(z['delay'],r['truth']['secondary']))<=2
    return np.array([int(correct),int(positive and decl and not correct),int(not positive and decl)])
def check():
    d=json.loads((D/'article_data.json').read_text())
    for name,digest in d['input_hashes'].items():assert sha(ROOT/name)==digest,name
    rows=read(ROOT/'results/score_peak_localization_development_v1/records.sqlite3')
    assert len(rows)==640
    for folder,key in [('revision_censored_bank_v1','safe_bank_original'),('revision_censored_bank_common_v1','safe_bank_common')]:
        other=read(ROOT/'results'/folder/'records.sqlite3');assert len(other)==640
        for r,s in zip(rows,other):assert r['case']==s['case'];r['samples'][key]=s['sample']
    other=read(ROOT/'results/revision_exposure_bank_v1/records.sqlite3');assert len(other)==1280
    for branch in ['original','common']:
        for r,s in zip(rows,[v for v in other if v['branch']==branch]):
            assert r['case']==s['case'];r['samples']['safe_exposure_'+branch]=s['sample']
    other=read(ROOT/'results/localization_postfailure_reference_v1/equal_information_joint_reference_v2.sqlite3')
    assert len(other)==640
    for r,s in zip(rows,other):assert r['case']==s['case'];r['samples'].update(s['samples'])
    ranks=0
    for level,totals in d['operating_points'].items():
        for key,expected in totals.items():
            total=np.zeros(3,dtype=int)
            for r in rows:
                z=r['samples'][key];assert len(z['bootstrap_scores'])==199
                assert z['pvalue']==(1+sum(x>=z['score'] for x in z['bootstrap_scores']))/200
                total+=endpoint(r,z,float(level));ranks+=1
            assert total.tolist()==expected,(level,key,total,expected)
    for regime,totals in d['by_regime'].items():
        for key,expected in totals.items():
            total=sum((endpoint(r,r['samples'][key],.01) for r in rows if r['case']['regime']==int(regime)),np.zeros(3,dtype=int))
            assert total.tolist()==expected,(regime,key)
    power=read(ROOT/'results/revision_sensitivity_v1/power.sqlite3')
    pd=json.loads((D/'power_data.json').read_text());assert len(power)==480
    for cell in pd['cells']:
        selected=[r for r in power if r['case']['positive'] and r['case']['ratio']==cell['ratio'] and r['case']['separation']==cell['separation']]
        assert len(selected)==32
        total=sum((endpoint(r,r['samples'][cell['method']],.01) for r in selected),np.zeros(3,dtype=int))
        assert total[:2].tolist()==[cell['correct'],cell['wrong']]
    mismatch=read(ROOT/'results/revision_sensitivity_v1/robust.sqlite3')
    own=read(ROOT/'results/revision_safe_robustness_v1/records.sqlite3')
    for item in json.loads((D/'robustness_data.json').read_text()):
        for method in ['full','safe','safe_original']:
            selected=[r for r in (own if method=='safe_original' else mismatch) if r['case']['condition']['name']==item['condition']]
            assert len(selected)==96
            available=[r for r in selected if r['status']=='EVALUATED']
            total=sum((endpoint(r,r['samples'][method],.01) for r in available),np.zeros(3,dtype=int))
            assert total.tolist()==item['valid_counts'][method]
    measured=json.loads((D/'measured_data.json').read_text())
    for name,digest in measured['input_hashes'].items():assert sha(ROOT/name)==digest,name
    for row in measured['rows']:
        z=np.load(ROOT/'results/measured_cluster_v1'/(Path(row['file']).stem+'.npz'))
        assert len(z['gaps'])==20000 and np.all(np.isfinite(z['logdensity']))
        assert abs(float(np.mean(z['logdensity']-z['baseline_logdensity']))-row['fresh_evaluation']['mean_log_score_gain'])<1e-12
    result=dict(status='PASS',queries=640,rank_checks=ranks,power_records=len(power),mismatch_records=len(mismatch),measured_evaluation_gaps=40000)
    (ROOT/'output').mkdir(exist_ok=True)
    (ROOT/'output/reproduction_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--figures',action='store_true');a=p.parse_args()
    if not(a.check or a.figures):p.error('choose --check and/or --figures')
    if a.check:
        check()
        subprocess.run([sys.executable,'tools/audit_calibrated_recovery.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'tools/audit_semisynthetic_recovery.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'tools/audit_extended_measured_nulls.py'],cwd=ROOT,check=True)
    if a.figures:subprocess.run([sys.executable,'tools/build_revision_article.py','--require-complete'],cwd=ROOT,check=True)

    if a.figures:subprocess.run([sys.executable,'tools/build_calibrated_recovery_assets.py'],cwd=ROOT,check=True)
