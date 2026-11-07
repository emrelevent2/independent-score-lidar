"""Prospective real-frame eligibility without reading method outputs or histograms."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,zipfile
import numpy as np
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];META=ROOT/'data'/'stb_metadata';OUT=ROOT/'data'/'stb_real_test_selection'
P=ROOT/'STB_REAL_TEST_SELECTION_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing selection protocol retained')
    tree=json.loads((META/'tree.json').read_text());candidates=[]
    for sequence in range(2,21):
        if sequence==7:continue
        path=f'Annotations/p1/Sequence{sequence}.zip';entry=next(r for r in tree if r['path']==path)
        candidates.append(dict(sequence=sequence,archive=path,archive_bytes=entry['size']))
    p=dict(status='Prospective annotation-eligibility selection. No test histogram or method output read.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)),range_reader_sha256=sha(ROOT/'src'/'stb_remote_zip.py'),
        candidates=candidates,frame_rule='Lexically first annotation member in each listed sequence.',
        pixel_eligibility='Exactly2known intervals1--13,gap>=8bins,query>=4pixels from image edge; >=100eligible pixels perframe.',
        selection='First4eligible sequences in numeric sequence order,or all if fewer4. Sequence7is development and excluded. Sequence1already inspected and excluded.',
        notes=['Eligibility depends only on provided annotations, not detector outcome, photon counts or an easy-looking histogram.',
            'Class14Unknown and0Unlabeled are never negative ground truth.',
            'All screened counts retained. Selected test histograms must not be downloaded/evaluated until real-method settings are frozen.',
            'Known semantic windows originate from full-histogram peak peeling, not independent physical depth measurements.',
            'This is a conditional annotated-frame cohort, not a representative random sample of all real scenes.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen',len(candidates),'candidate sequences',flush=True)


def run():
    p=json.loads(P.read_text());assert sha(Path(__file__))==p['source_sha256'];assert sha(ROOT/'src'/'stb_remote_zip.py')==p['range_reader_sha256']
    OUT.mkdir(exist_ok=True);records=[]
    for item in p['candidates']:
        save=OUT/f'Sequence{item["sequence"]}_record.json'
        if save.exists():
            r=json.loads(save.read_text());assert r['protocol_sha256']==sha(P);records.append(r);continue
        reader=RangeReader(BASE+item['archive'],item['archive_bytes'],budget=8*1024*1024)
        with zipfile.ZipFile(reader) as z:
            members=sorted([i for i in z.infolist() if i.filename.endswith('.npy')],key=lambda i:i.filename);info=members[0]
            assert info.file_size<40*1024*1024
            raw=z.read(info);assert len(raw)==info.file_size
        labels=np.load(io.BytesIO(raw),allow_pickle=False);assert labels.shape==(49152,672) and labels.dtype==np.uint8
        known=(labels>0)&(labels<14)
        starts=known&~np.pad(known[:,:-1],((0,0),(1,0)));ends=known&~np.pad(known[:,1:],((0,0),(0,1)))
        n=starts.sum(-1);pixels=np.flatnonzero(n==2);second=671-np.argmax(starts[pixels,::-1],axis=1);first_end=np.argmax(ends[pixels],axis=1)
        eligible=(second-first_end>=8)&(pixels//256>=4)&(pixels//256<188)&(pixels%256>=4)&(pixels%256<252)
        path=OUT/f'Sequence{item["sequence"]}_labels.npz';np.savez_compressed(path,labels=labels)
        r=dict(**item,protocol_sha256=sha(P),member=info.filename,crc32=info.CRC,uncompressed_bytes=info.file_size,
            original_member_sha256=hashlib.sha256(raw).hexdigest(),local_numeric_file=path.name,local_sha256=sha(path),
            eligible_two_window_pixels=int(eligible.sum()),known_window_counts={str(k):int((n==k).sum()) for k in range(int(n.max())+1)},
            remote_bytes_read=reader.transferred)
        save.write_text(json.dumps(r,indent=2));records.append(r)
        print('Sequence',item['sequence'],'eligible',r['eligible_two_window_pixels'],'remote bytes',reader.transferred,flush=True)
    selected=[r for r in records if r['eligible_two_window_pixels']>=100][:4]
    result=dict(protocol_sha256=sha(P),screened=records,selected=selected,
        status='No selected test histogram or method result has been read.')
    (OUT/'selection.json').write_text(json.dumps(result,indent=2));print('Selected sequences',[r['sequence'] for r in selected],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
