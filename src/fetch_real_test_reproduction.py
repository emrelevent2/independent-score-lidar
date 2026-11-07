"""Recover omitted large real tensors from pinned, hashed ZIP members.

Never changes frozen query coordinates, method settings or thresholds.
Files already present must match their recorded hashes. No downloaded code
or pickle is executed. Intended for the versioned local reproduction bundle.
"""
import json,zipfile
from pathlib import Path
import numpy as np
from real_multiframe_test import verify,ROOT,DATA,sha
from stb_remote_zip import RangeReader,BASE


def recover_frame(directory,manifest):
    spec=manifest['download'];target=directory/Path(spec['member']).name
    if target.exists():assert sha(target)==spec['sha256']
    else:
        reader=RangeReader(BASE+'Histogram/p1.zip',156925687435,budget=128*1024*1024)
        with zipfile.ZipFile(reader) as z:
            info=z.getinfo(spec['member']);assert info.file_size==spec['bytes'] and info.CRC==spec['crc32']
            partial=target.with_suffix('.partial')
            with z.open(info) as src,partial.open('wb') as dst:
                while True:
                    chunk=src.read(1024*1024)
                    if not chunk:break
                    dst.write(chunk)
            assert sha(partial)==spec['sha256'];partial.replace(target)
    frame=directory/'frame_histograms.npz'
    h=np.loadtxt(target,dtype=np.int64);assert h.shape==(49152,672) and np.all(h>=0)
    h=h.astype(np.int32).reshape(192,256,672)
    if frame.exists():assert sha(frame)==manifest['frame_sha256']
    else:
        temporary=directory/'frame_recovered.npz';np.savez_compressed(temporary,histograms=h)
        assert sha(temporary)==manifest['frame_sha256'],'Use the locked NumPy version for byte-identical NPZ output'
        temporary.replace(frame)
    for item in manifest['inputs']:
        path=directory/f'thin_{item["index"]}.npz'
        if path.exists():assert sha(path)==item['sha256'];continue
        child=np.random.default_rng(item['seed']).binomial(h,item['probability']).astype(np.int32)
        temporary=directory/f'thin_recovered_{item["index"]}.npz';np.savez_compressed(temporary,histograms=child)
        assert sha(temporary)==item['sha256'],'Use the locked NumPy version for byte-identical NPZ output'
        temporary.replace(path)


def main():
    p=verify();m=json.loads((DATA/'manifest.json').read_text());assert m['protocol_sha256']==sha(ROOT/'REAL_POISSON_TEST_PROTOCOL.json')
    for entry in m['frames']:
        directory=DATA/entry['name'];assert sha(directory/'manifest.json')==entry['manifest_sha256']
        recover_frame(directory,json.loads((directory/'manifest.json').read_text()));print('Verified/recovered',entry['name'],flush=True)


if __name__=='__main__':main()
