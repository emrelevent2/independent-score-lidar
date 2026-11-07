"""Recover only omitted, already pinned real members used in current audits.

No new count frame is selected. Existing files must match recorded hashes.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from fetch_real_test_reproduction import recover_frame
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def check_or_make(path,h,item,verify_only):
    if path.exists():assert sha(path)==item['sha256'];return
    assert not verify_only,f'Missing {path}; run without --verify-present to recover pinned inputs'
    child=np.random.default_rng(item['seed']).binomial(h,item['probability']).astype(np.int32)
    temp=path.with_name(path.stem+'_recovered.npz');np.savez_compressed(temp,histograms=child)
    assert sha(temp)==item['sha256'],'Use locked NumPy version for exact NPZ bytes';temp.replace(path)


def main(verify_only):
    source=ROOT/'data/stb_multireturn_development';data=ROOT/'data/real_poisson_development_v1'
    spec=json.loads((source/'download_manifest.json').read_text())['histogram']
    p=json.loads((ROOT/'REAL_POISSON_DEVELOPMENT_PROTOCOL.json').read_text());m=json.loads((data/'manifest.json').read_text())
    expected=p['input_sha256']['data/stb_multireturn_development/frame_histograms.npz']
    raw=source/Path(spec['member']).name;frame=source/'frame_histograms.npz'
    if verify_only:
        assert sha(raw)==spec['sha256'] and sha(frame)==expected
        h=None
    else:
        recover_frame(source,dict(download=spec,frame_sha256=expected,inputs=[]))
        h=np.load(frame)['histograms']
    for item in m['inputs']:check_or_make(data/f'thin_{item["index"]}.npz',h,item,verify_only)
    print('Verified/recovered fixed Sequence7development inputs',flush=True)
    guarded=ROOT/'data/guarded_real_confirmation_v1';gm=json.loads((guarded/'manifest.json').read_text())
    for entry in gm['frames']:
        directory=guarded/entry['name'];assert sha(directory/'manifest.json')==entry['manifest_sha256'];fm=json.loads((directory/'manifest.json').read_text())
        if verify_only:
            assert sha(directory/Path(fm['download']['member']).name)==fm['download']['sha256']
            assert sha(directory/'frame_histograms.npz')==fm['frame_sha256']
            for item in fm['inputs']:assert sha(directory/f'thin_{item["index"]}.npz')==item['sha256']
        else:recover_frame(directory,fm)
        print('Verified/recovered',entry['name'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify-present',action='store_true');a=ap.parse_args();main(a.verify_present)
