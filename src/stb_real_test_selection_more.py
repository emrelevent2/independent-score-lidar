"""Annotation-only expansion to distinct held-out real sequences.

The already selected Sequence17 frame stays primary. No histogram/method
outcome is read; up to three additional eligible sequences improve coverage.
"""
from pathlib import Path
import argparse,datetime,hashlib,io,json,zipfile
import numpy as np
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'data'/'stb_real_test_selection'
OUT=ROOT/'data'/'stb_real_test_selection_more';P=ROOT/'STB_REAL_TEST_SELECTION_MORE_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing expanded eligibility protocol retained')
    assert not (ROOT/'REAL_POISSON_TEST_PROTOCOL.json').exists(),'A held-out method protocol already exists; do not expand silently'
    tree=json.loads((ROOT/'data'/'stb_metadata'/'tree.json').read_text());items=[]
    for seq in range(2,21):
        if seq in [7,17]:continue
        archive=f'Annotations/p1/Sequence{seq}.zip';entry=next(x for x in tree if x['path']==archive)
        items.append(dict(sequence=seq,archive=archive,archive_bytes=entry['size'],positions=['middle','last']))
    p=dict(status='Prospective annotation-only coverage expansion before any real test histogram/method result.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),
        range_reader_sha256=sha(ROOT/'src'/'stb_remote_zip.py'),original_selection_sha256=sha(OLD/'selection.json'),candidates=items,
        selection='Keep originalSequence17primary; add first3other eligible sequences in numeric order, middle before last. At most1frame persequence.',
        criterion='Exactly2known intervals1--13,gap>=8bins,query>=4pixels from edge; >=100eligiblepixels/frame.',
        notes=['No method outputs, photon counts or test histograms select these frames.',
            'All candidates and failures of eligibility retained. This is an annotated multi-return cohort, not a random scene sample.',
            'Sequence7isdevelopment;1waspreviouslyscreened;17alreadyhasitsfixedprimaryframe.',
            'Zero/Unknown14are never physical negatives. Selected test histograms stay unread until final real-method freeze.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen additional annotation-only screen',flush=True)


def run():
    p=json.loads(P.read_text());assert sha(Path(__file__))==p['source_sha256'];assert sha(ROOT/'src'/'stb_remote_zip.py')==p['range_reader_sha256']
    assert sha(OLD/'selection.json')==p['original_selection_sha256'];OUT.mkdir(exist_ok=True);records=[]
    for item in p['candidates']:
        reader=RangeReader(BASE+item['archive'],item['archive_bytes'],budget=16*1024*1024)
        with zipfile.ZipFile(reader) as z:
            members=sorted([i for i in z.infolist() if i.filename.endswith('.npy')],key=lambda i:i.filename)
            for position in item['positions']:
                record_file=OUT/f'Sequence{item["sequence"]}_{position}_record.json'
                if record_file.exists():
                    record=json.loads(record_file.read_text());assert record['protocol_sha256']==sha(P);records.append(record);continue
                info=members[len(members)//2 if position=='middle' else -1];assert info.file_size<40*1024*1024
                raw=z.read(info);labels=np.load(io.BytesIO(raw),allow_pickle=False);assert labels.shape==(49152,672) and labels.dtype==np.uint8
                known=(labels>0)&(labels<14);starts=known&~np.pad(known[:,:-1],((0,0),(1,0)));ends=known&~np.pad(known[:,1:],((0,0),(0,1)))
                n=starts.sum(-1);pixels=np.flatnonzero(n==2);second=671-np.argmax(starts[pixels,::-1],axis=1);first_end=np.argmax(ends[pixels],axis=1)
                eligible=(second-first_end>=8)&(pixels//256>=4)&(pixels//256<188)&(pixels%256>=4)&(pixels%256<252)
                path=OUT/f'Sequence{item["sequence"]}_{position}_labels.npz';np.savez_compressed(path,labels=labels)
                record=dict(sequence=item['sequence'],archive=item['archive'],archive_bytes=item['archive_bytes'],position=position,
                    protocol_sha256=sha(P),member=info.filename,crc32=info.CRC,original_member_sha256=hashlib.sha256(raw).hexdigest(),
                    label_file_relative=path.relative_to(ROOT).as_posix(),local_sha256=sha(path),eligible_two_window_pixels=int(eligible.sum()))
                record_file.write_text(json.dumps(record,indent=2));records.append(record)
                print('Sequence',item['sequence'],position,'eligible',record['eligible_two_window_pixels'],flush=True)
    primary=json.loads((OLD/'selection.json').read_text())['selected'][0]
    primary=dict(primary,label_file_relative=(OLD/primary['local_numeric_file']).relative_to(ROOT).as_posix(),position='first')
    selected=[primary];seen={17}
    for r in records:
        if r['eligible_two_window_pixels']>=100 and r['sequence'] not in seen:
            selected.append(r);seen.add(r['sequence'])
        if len(selected)==4:break
    (OUT/'selection.json').write_text(json.dumps(dict(protocol_sha256=sha(P),selected=selected,screened=records,
        status='No selected test histogram or method output read.'),indent=2))
    print('Final candidate test sequences',[(r['sequence'],r['position']) for r in selected],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
