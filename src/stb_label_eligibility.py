"""Prespecified annotation-only eligibility screen, before method testing."""
from pathlib import Path
import argparse,datetime,hashlib,json,zipfile
import numpy as np
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data'/'stb_label_eligibility';P=ROOT/'STB_LABEL_ELIGIBILITY_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing eligibility protocol retained')
    p=dict(status='Annotation-only screen; no method output has been evaluated on these frames.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'src'/n) for n in ['stb_label_eligibility.py','stb_remote_zip.py']},
        candidates=[dict(archive='Annotations/p2/Sequence21.zip',size=30602679,positions=['middle','last']),
            dict(archive='Annotations/p2/Sequence22.zip',size=36242239,positions=['first','middle']),
            dict(archive='Annotations/p1/Sequence1.zip',size=100558372,positions=['first','middle']),
            dict(archive='Annotations/p1/Sequence7.zip',size=5610003,positions=['first','middle'])],
        rule='Lexically first archive/member with at least100pixels having exactly2known annotation intervals separated by at least8bins.',
        known_class_ids=list(range(1,14)),unknown_class_id=14,
        notes=['Author label tool identifies14asUnknown, not a fourteenth known semantic class.',
            'Other labels/zero are not evidence of absence of a physical secondary return.',
            'Eligibility uses annotations only, never proposed-method accuracy, calibration score or favorable photon noise.',
            'All screened candidates and counts remain in the record; only eligible histogram files need to be downloaded.',
            'Selected frames are development. Fresh real evaluation selection must be frozen separately.'])
    P.write_text(json.dumps(p,indent=2))


def run():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    OUT.mkdir(exist_ok=True);records=[]
    for item in p['candidates']:
        reader=RangeReader(BASE+item['archive'],item['size'],budget=32*1024*1024)
        with zipfile.ZipFile(reader) as z:
            members=sorted([i for i in z.infolist() if i.filename.endswith('.npy')],key=lambda i:i.filename)
            for position in item['positions']:
                index=0 if position=='first' else len(members)//2 if position=='middle' else len(members)-1
                info=members[index];path=OUT/(item['archive'].split('/')[-2]+'_'+Path(info.filename).name)
                if not path.exists():
                    assert info.file_size<40*1024*1024
                    with z.open(info) as source,path.open('wb') as target:
                        while True:
                            data=source.read(1024*1024)
                            if not data:break
                            target.write(data)
                labels=np.load(path,allow_pickle=False,mmap_mode='r');assert labels.shape==(49152,672) and labels.dtype==np.uint8
                known=(labels>0)&(labels<14)
                starts=known&~np.pad(known[:,:-1],((0,0),(1,0)));ends=known&~np.pad(known[:,1:],((0,0),(0,1)))
                n=starts.sum(-1);pixels=np.flatnonzero(n==2);eligible=0
                # Vectorized starts/ends for two-window pixels; unknown intervals
                # do not become negative labels and cannot validate a null.
                first=np.argmax(starts[pixels],axis=1);second=671-np.argmax(starts[pixels,::-1],axis=1)
                first_end=np.argmax(ends[pixels],axis=1)
                eligible=int(((second-first_end)>=8).sum())
                record=dict(archive=item['archive'],archive_bytes=item['size'],member=info.filename,
                    bytes=info.file_size,compressed_bytes=info.compress_size,crc32=info.CRC,
                    source_position=position,local=path.name,sha256=sha(path),
                    pixels_with_known_windows={str(k):int((n==k).sum()) for k in range(int(n.max())+1)},eligible_two_window_pixels=eligible)
                records.append(record);print(item['archive'],position,'known-window counts',record['pixels_with_known_windows'],'eligible',eligible,flush=True)
    candidates=sorted([r for r in records if r['eligible_two_window_pixels']>=100],key=lambda r:(r['archive'],r['member']))
    result=dict(protocol_sha256=sha(P),candidates=records,selected=candidates[0] if candidates else None,
        note='No method was run or selected on these annotations.')
    (OUT/'screen.json').write_text(json.dumps(result,indent=2));print('Eligible selected',None if not candidates else (candidates[0]['archive'],candidates[0]['member']),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
