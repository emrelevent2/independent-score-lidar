"""Download a fixed annotated development pair via bounded ZIP-member reads."""
from pathlib import Path
import argparse,datetime,hashlib,json,zipfile
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];META=ROOT/'data'/'stb_metadata';OUT=ROOT/'data'/'stb_semantic_development'
P=ROOT/'STB_SEMANTIC_DATA_PROTOCOL.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if P.exists():raise RuntimeError('Existing data-selection snapshot retained')
    ann=json.loads((META/'Annotations_p2_Sequence21.zip_members.json').read_text())
    hist=json.loads((META/'Histogram_p2.zip_members.json').read_text())
    label=sorted([r for r in ann['members'] if r['name'].endswith('.npy')],key=lambda r:r['name'])[0]
    target=Path(label['name']).name.replace('_semantic.npy','.txt')
    histogram=next(r for r in hist['members'] if Path(r['name']).name==target)
    p=dict(status='Fixed development frame selected without inspecting histograms or annotations.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        commit='cadc5d7a0fa0cf5e6f49973ba3633198c4c3c02d',selection='Lexically first labeled frame of deviceP2 Sequence21, matching histogram by timestamp.',
        source_sha256={n:sha(ROOT/'src'/n) for n in ['stb_semantic_download.py','stb_remote_zip.py']},
        files=[dict(archive=ann['archive'],archive_bytes=ann['archive_bytes'],member=label),dict(archive=hist['archive'],archive_bytes=hist['archive_bytes'],member=histogram)],
        notes=['Only selected members, never the full7GBhistogram archive, are downloaded.',
            'CRC32 and downloaded-member SHA256 recorded. Entire remote archive SHA is not claimed locally verified.',
            'Public semantic windows were labeled by histogram peak peeling; they are not independent physical measurements of every hidden surface.',
            'Unlabeled bins/pixels are not automatically true background or proof of no secondary surface.',
            'Development only. No real-method performance is claimed by this acquisition.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen pair',target,flush=True)


def run():
    p=json.loads(P.read_text())
    for n,d in p['source_sha256'].items():assert sha(ROOT/'src'/n)==d,n
    OUT.mkdir(exist_ok=True)
    if (OUT/'download_manifest.json').exists():raise RuntimeError('Existing download retained')
    files=[]
    for item in p['files']:
        member=item['member'];path=OUT/Path(member['name']).name
        if path.exists():raise RuntimeError('Existing member needs explicit partial-download inspection')
        assert member['bytes']<=150*1024*1024
        reader=RangeReader(BASE+item['archive'],item['archive_bytes'],budget=64*1024*1024)
        with zipfile.ZipFile(reader) as z:
            info=z.getinfo(member['name']);assert info.CRC==member['crc32'] and info.file_size==member['bytes']
            with z.open(info) as source,path.open('wb') as target:
                while True:
                    data=source.read(1024*1024)
                    if not data:break
                    target.write(data)
        assert path.stat().st_size==member['bytes']
        files.append(dict(**item,local=path.name,sha256=sha(path),remote_bytes_read=reader.transferred,requests=reader.requests))
        print('Downloaded',path.name,path.stat().st_size,'bytes; remote bytes',reader.transferred,flush=True)
    (OUT/'download_manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(P),files=files,license='CC BY-NC-SA4.0; cite Dong et al.,SP-TransientBench2026.'),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
