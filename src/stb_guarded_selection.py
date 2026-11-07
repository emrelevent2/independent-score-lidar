"""Prospective annotation-only frame selection for the fixed guarded rule.

Prefer previously untested sequences, then new frames of the two closed
test sequences. All eligibility screens are retained. No count file is read.
"""
from pathlib import Path
import argparse,datetime,hashlib,io,json,zipfile
import numpy as np
from stb_remote_zip import RangeReader,BASE
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/stb_guarded_selection_v1';P=ROOT/'STB_GUARDED_SELECTION_PROTOCOL.json'
RULE=ROOT/'REAL_GUARDED_RULE_FREEZE.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    assert not P.exists() and not RULE.exists(),'Existing prospective snapshots retained'
    files=['guarded_real_policy.py','real_poisson_analysis_v2.py','real_poisson_methods_v2.py','likelihood_baseline.py']
    rule=dict(status='Fixed before additional annotation selection or any new count acquisition.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={n:sha(ROOT/'src'/n) for n in files},
        simulation_protocol_sha256=sha(ROOT/'GUARDED_LOCALIZATION_CONFIRMATORY_PROTOCOL.json'),
        development_calibration_sha256=sha(ROOT/'results/real_poisson_development_v2/calibration.json'),
        settings=dict(distance_bins=4.,guided_local_raw_score_ratio=.5,donor_weight=.5,support_proxy=3.5,
            primary_alpha=.01,primary_retention=.025,retentions=[.01,.025,.05],pulse_kind='gaussian'),
        notes=['Direct fixed-parameter transfer, not a new real-data tuning search.',
            'Use working-Poisson donor support, not simulator first-event exponential correction.',
            'Exact normalized local score and cutoff identity; only secondary coordinate can change.',
            'Primary endpoint: mean framewise paired two-window agreement difference at2.5%retention; frame-conditional block bootstrap.',
            'Earlier Sequence17first/Sequence3middle results are closed; no retrospective untouched claim.'])
    RULE.write_text(json.dumps(rule,indent=2))
    tree=json.loads((ROOT/'data/stb_metadata/tree.json').read_text());items=[]
    order=[s for s in range(1,21) if s not in [3,7,17]]+[3,17]
    for seq in order:
        archive=f'Annotations/p1/Sequence{seq}.zip';entry=next(x for x in tree if x['path']==archive)
        items.append(dict(sequence=seq,archive=archive,archive_bytes=entry['size'],positions=['quarter','three_quarters']+(['middle','last'] if seq==17 else [])))
    old=json.loads((ROOT/'data/stb_real_test_selection_more/selection.json').read_text())['selected']
    p=dict(status='Annotation-only selection AFTER fixed guarded-policy freeze; BEFORE all new histograms/outcomes.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),
        range_reader_sha256=sha(ROOT/'src/stb_remote_zip.py'),rule_sha256=sha(RULE),candidates=items,
        excluded_count_frames=[r['member'] for r in old],
        selection='First eligible frame per sequence in stated order, up to4sequences. If fewer than4sequences qualify, fill to4frames from remaining eligible distinct frames in the same order.',
        criterion='Exactly2known intervals1--13,gap>=8bins,margin4pixels; at least100eligible two-window and1000known-one pixels.',
        notes=['Sequence7excluded entirely as development. Untested sequences precede closed test sequences3and17.',
            'No histogram intensity, method score or performance metric participates in eligibility.',
            'Frames from3/17are new captures within already seen sequences, not independent unseen-scene validation.',
            'All noneligible screens and member identities retained. Unknown14/Unlabeled0never become physical negatives.'])
    P.write_text(json.dumps(p,indent=2));print('Frozen rule and',sum(len(i['positions']) for i in items),'annotation-only candidates',flush=True)


def eligibility(labels):
    assert labels.shape==(49152,672) and labels.dtype==np.uint8
    known=(labels>0)&(labels<14);starts=known&~np.pad(known[:,:-1],((0,0),(1,0)));ends=known&~np.pad(known[:,1:],((0,0),(0,1)))
    n=starts.sum(-1);interior=np.zeros(49152,bool);interior.reshape(192,256)[4:-4,4:-4]=True
    pixels=np.flatnonzero((n==2)&interior);second=671-np.argmax(starts[pixels,::-1],axis=1);first_end=np.argmax(ends[pixels],axis=1)
    return int(((second-first_end)>=8).sum()),int(((n==1)&interior).sum())


def run():
    p=json.loads(P.read_text());assert sha(Path(__file__))==p['source_sha256'] and sha(RULE)==p['rule_sha256']
    assert sha(ROOT/'src/stb_remote_zip.py')==p['range_reader_sha256'];OUT.mkdir(exist_ok=True);records=[]
    excluded=set(p['excluded_count_frames'])
    for item in p['candidates']:
        reader=RangeReader(BASE+item['archive'],item['archive_bytes'],budget=24*1024*1024)
        with zipfile.ZipFile(reader) as z:
            members=sorted([x for x in z.infolist() if x.filename.endswith('.npy')],key=lambda x:x.filename)
            assert members
            positions=dict(quarter=len(members)//4,three_quarters=3*len(members)//4,middle=len(members)//2,last=len(members)-1)
            for position in item['positions']:
                record_path=OUT/f'Sequence{item["sequence"]}_{position}_record.json'
                if record_path.exists():
                    record=json.loads(record_path.read_text());assert record['protocol_sha256']==sha(P);records.append(record);continue
                info=members[positions[position]];record=dict(sequence=item['sequence'],position=position,member=info.filename,
                    member_index=positions[position],members=len(members),protocol_sha256=sha(P),archive=item['archive'],archive_bytes=item['archive_bytes'])
                if info.filename in excluded:
                    record.update(status='Excluded: count frame previously acquired',eligible=False)
                else:
                    assert info.file_size<40*1024*1024;raw=z.read(info);labels=np.load(io.BytesIO(raw),allow_pickle=False)
                    two,one=eligibility(labels);path=OUT/f'Sequence{item["sequence"]}_{position}_labels.npz';np.savez_compressed(path,labels=labels)
                    record.update(status='Annotation only; counts unread',eligible=two>=100 and one>=1000,
                        eligible_two_window_pixels=two,known_one_pixels=one,crc32=info.CRC,original_member_sha256=hashlib.sha256(raw).hexdigest(),
                        label_file_relative=path.relative_to(ROOT).as_posix(),local_sha256=sha(path))
                record_path.write_text(json.dumps(record,indent=2));records.append(record)
                print('Eligibility',item['sequence'],position,record.get('eligible_two_window_pixels'),record['eligible'],flush=True)
    eligible=[r for r in records if r['eligible']];selected=[];seen_sequences=set();seen_members=set()
    for r in eligible:
        if r['sequence'] in seen_sequences or r['member'] in seen_members:continue
        selected.append(r);seen_sequences.add(r['sequence']);seen_members.add(r['member'])
        if len(selected)==4:break
    if len(selected)<4:
        for r in eligible:
            if r['member'] in seen_members:continue
            selected.append(r);seen_members.add(r['member'])
            if len(selected)==4:break
    (OUT/'selection.json').write_text(json.dumps(dict(protocol_sha256=sha(P),rule_sha256=sha(RULE),selected=selected,screened=records,
        status='All selected new count tensors unread; no method outcomes used.'),indent=2))
    print('Selected',[(r['sequence'],r['position'],r['eligible_two_window_pixels']) for r in selected],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
