"""Fixed pretrained 2025 comparator on the pre-specified CASPI paired subset."""
from pathlib import Path
import json,hashlib,subprocess,datetime,time,contextlib,io,argparse,importlib.metadata
import numpy as np
import torch
from pointcloud_cpu_adapter import OfficialNetwork,validate_knn,report_from_centers,UPSTREAM
from pilot import coates
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'guided_confirmation_v1'
OUT=ROOT/'results'/'guided_confirmation_v1'/'pointcloud';P=ROOT/'POINTCLOUD_COMPARATOR_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
OWN=['run_pointcloud_confirmation.py','pointcloud_cpu_adapter.py','likelihood_baseline.py','pilot.py']


def freeze():
    if P.exists():raise RuntimeError('Comparator already frozen; retain snapshot')
    commit=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    assert commit=='5895893e2b04ab3315b4c195189a688de04228fa'
    subprocess.run(['git','-C',str(UPSTREAM),'diff','--exit-code'],check=True,stdout=subprocess.DEVNULL)
    m=json.loads((DATA/'manifest.json').read_text());cases=[c['id'] for c in m['cases'] if c['caspi']]
    assert len(cases)==579
    p=dict(status='Frozen before inspecting new final query-level accuracy',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        main_protocol_sha256=sha(ROOT/'GUIDED_CONFIRMATORY_PROTOCOL.json'),commit=commit,
        own_source_sha256={n:sha(ROOT/'src'/n) for n in OWN},
        upstream_source_sha256={f.relative_to(UPSTREAM).as_posix():sha(f) for f in UPSTREAM.rglob('*.py') if '__pycache__' not in f.parts},
        weights_sha256=sha(UPSTREAM/'run'/'baseline.pth'),
        environment={n:importlib.metadata.version(n) for n in ['torch','torch-geometric','torch-sparse','torch-scatter','numpy','scipy','PyWavelets']},
        cases=cases,seed=0,model=dict(nscale=8,att=10,k=6,nblock=3,nconv=3),
        variants=['unrolled_raw','unrolled_coates'],
        notes=['Exact CPU Euclidean KNN adapter checked against independent dense distances; distance ties may have a different valid order.',
            'Original model/weights unchanged; hard-coded CUDA tensor allocations remain on CPU during inference.',
            'Seed0 reset for every forward pass because the official evaluation still uses Gumbel sampling.',
            'Raw variant uses original observed histograms. Coates variant adds known-cycle pileup correction before unchanged official initialization.',
            'Nominal supplied Gaussian width5, kernel21samples; official get_IF threshold1percent, all original spatial initialization scales retained.',
            'Fixed original gamma1024*.003 and pixelspacing.008 retained; predictions inverted using actual T256.',
            'Two-return pretrained transfer comparator; temporal window, response and small scene size differ from published training.',
            'Presence reporting is an added local calibrated reporter, not a claim that the published network supplies an absence decision.',
            'Locally fitted stronger component is primary, weaker amplitude is calibrated score; same8bin separation/width4.5--5.5/refinement1.5.',
            'All paired controls use same199 calibration acquisitions. Results are not extrapolated to full2900-case test.',
            'Single-plane fixture can produce spurious two-point reconstructions because of the forced two-return prior; this is not suppressed.',
            'Two-plane80/140 fixture max time-axis error7.7e-6bins; repeated fixed-seed single-plane inference was exactly repeatable.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen579case pretrained comparator',flush=True)


def run():
    p=json.loads(P.read_text())
    for n,d in p['own_source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    for n,d in p['upstream_source_sha256'].items():assert sha(UPSTREAM/n)==d,n
    assert sha(UPSTREAM/'run'/'baseline.pth')==p['weights_sha256']
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);validate_knn();model=OfficialNetwork();start=time.perf_counter()
    irf=np.exp(-4*np.log(2)*(np.arange(-10,11)/5.)**2)
    for i,identifier in enumerate(p['cases'],1):
        path=OUT/(identifier+'.json')
        if path.exists():
            assert json.loads(path.read_text())['protocol_sha256']==sha(P);continue
        h=np.load(DATA/(identifier+'.npz'))['histogram'];reports={};arrays={};meta={}
        for variant in p['variants']:
            inp=h if variant=='unrolled_raw' else coates(h)*1000
            t=time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):centers,info=model.predict(inp,irf,seed=p['seed'])
            reports[variant]=report_from_centers(h[8,8],centers[8,8]);reports[variant]['seconds']=time.perf_counter()-t
            arrays[variant]=centers;meta[variant]=info
        np.savez_compressed(OUT/(identifier+'_centers.npz'),**arrays)
        path.write_text(json.dumps(dict(id=identifier,protocol_sha256=sha(P),reports=reports,metadata=meta),indent=2,allow_nan=False))
        if i%50==0 or i==len(p['cases']):print('Pretrained comparator',i,'/',len(p['cases']),'seconds',round(time.perf_counter()-start,1),flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(status='Completed; accuracy not analyzed here',cases=len(p['cases']),elapsed_seconds=time.perf_counter()-start,protocol_sha256=sha(P)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.action=='freeze' else run()
