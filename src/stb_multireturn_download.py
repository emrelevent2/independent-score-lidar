"""Acquire the annotation-eligible real development frame, not full archives."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,zipfile
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];ELIG=ROOT/'data'/'stb_label_eligibility'
OUT=ROOT/'data'/'stb_multireturn_development';P=ROOT/'STB_MULTIRETURN_DATA_PROTOCOL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing acquisition snapshot retained')
    screen=json.loads((ELIG/'screen.json').read_text());selected=screen['selected'];assert selected
    assert selected['archive']=='Annotations/p1/Sequence7.zip'
    target=Path(selected['member']).name.replace('_semantic.npy','.txt')
    p=dict(status='Development acquisition selected from annotation eligibility only, before any method result.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),
        range_reader_sha256=sha(ROOT/'src'/'stb_remote_zip.py'),eligibility_screen_sha256=sha(ELIG/'screen.json'),
        annotation=selected,histogram_archive='Histogram/p1.zip',archive_bytes=156925687435,
        histogram_member='p1/Sequence7/'+target,
        notes=['Only one selected ZIP member is read from the156GBarchive using HTTPRange; full archive is never downloaded.',
            'MemberCRC and downloaded bytes SHA recorded; whole remote archive SHA is not claimed verified.',
            'P1device: no P1-specific per-pixel time-shift/ray calibration has been verified. Do not silently reuse P2calibration.',
            'References are known semantic intervals1--13;14isUnknown,0isUnlabeled,neither proves absence of another return.',
            'Labels use full-measurement histogram peak peeling; not an independent physical reference for every weak return.'])
    P.write_text(json.dumps(p,indent=2))


def run():
    p=json.loads(P.read_text());assert sha(Path(__file__))==p['source_sha256'];assert sha(ROOT/'src'/'stb_remote_zip.py')==p['range_reader_sha256']
    OUT.mkdir(exist_ok=True)
    if (OUT/'download_manifest.json').exists():raise RuntimeError('Existing download retained')
    annotation=ELIG/p['annotation']['local'];assert sha(annotation)==p['annotation']['sha256']
    label=OUT/Path(p['annotation']['member']).name;shutil.copyfile(annotation,label)
    reader=RangeReader(BASE+p['histogram_archive'],p['archive_bytes'],budget=128*1024*1024)
    target=OUT/Path(p['histogram_member']).name
    if target.exists():raise RuntimeError('Existing histogram retained')
    with zipfile.ZipFile(reader) as z:
        info=z.getinfo(p['histogram_member']);assert info.file_size<=150*1024*1024 and info.compress_size<=100*1024*1024
        partial=target.with_suffix('.partial')
        with z.open(info) as source,partial.open('wb') as out:
            while True:
                chunk=source.read(1024*1024)
                if not chunk:break
                out.write(chunk)
        assert partial.stat().st_size==info.file_size;partial.replace(target)
        hist_meta=dict(member=info.filename,bytes=info.file_size,compressed_bytes=info.compress_size,crc32=info.CRC,
            sha256=sha(target),local=target.name)
    result=dict(protocol_sha256=sha(P),annotation=dict(local=label.name,sha256=sha(label)),histogram=hist_meta,
        archive=p['histogram_archive'],archive_bytes=p['archive_bytes'],remote_bytes_read=reader.transferred,requests=reader.requests,
        license='CC BY-NC-SA4.0; cite Dong et al.,SP-TransientBench2026.')
    (OUT/'download_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='requests'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
