"""Complete refit and independent endpoint recount for the closed split trial."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,math,sqlite3
from thinned_predictive_development import verify,ROOT,DATA,OLD,OUT,P,sha,worker,POLICIES


def main():
    p=verify();m=json.loads((DATA/'manifest.json').read_text());db=sqlite3.connect(OUT/'records.sqlite3');rows={}
    for key,digest,raw,h in db.execute('SELECT * FROM records'):
        assert digest==sha(P) and hashlib.sha256(raw.encode()).hexdigest()==h;rows[key]=json.loads(raw)
    db.close();assert len(rows)==2697;done=0
    for entry in m['inputs']:
        i=entry['index'];assert sha(DATA/f'thin_{i}.npz')==entry['sha256']
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(worker,q,i,p['settings'],p['old_records_sha256'][f'gaussian_{i}_{q["id"]}.json']) for q in m['queries']]
            for job in as_completed(jobs):
                r=job.result();assert r==rows[f'{i}_{r["query_id"]}'];done+=1
        print('Audited thinned predictive',done,flush=True)
    a=json.loads((OUT/'analysis.json').read_text());checked=0
    match=lambda d,w:w[0]-2<=d<=w[1]+2
    for i in range(3):
        for method in POLICIES+p['controls']:
            values=sorted(rows[f'{i}_{q["id"]}']['reports'][method]['score'] for q in m['queries'] if q['split']=='calibration')
            for alpha in p['settings']['alphas']:
                cut=max(0.,values[math.ceil(400*(1-alpha))-1]);saved=a['methods'][f'{i}_{alpha}_{method}'];assert cut==saved['threshold']
                for split in ['evaluation_one','evaluation_two']:
                    pair=unmatched=declared=0
                    for q in m['queries']:
                        if q['split']!=split:continue
                        r=rows[f'{i}_{q["id"]}']['reports'][method];d=r['score']>cut;declared+=d;windows=q['windows'];x,y=r['primary'],r['delay']
                        pair+=bool(d and len(windows)==2 and ((match(x,windows[0]) and match(y,windows[1])) or (match(x,windows[1]) and match(y,windows[0]))))
                        unmatched+=bool(d and (not any(match(x,w) for w in windows) or not any(match(y,w) for w in windows)))
                    s=saved['groups'][split];assert (declared,pair,unmatched)==(s['declared'],s['pair'],s['accepted_unmatched']);checked+=1
    result=dict(status='PASS',protocol_sha256=sha(P),records_fully_refit=done,endpoint_group_configurations_independently_recounted=checked,
        sqlite_sha256=sha(OUT/'records.sqlite3'),
        scope='All sources,parent input/record hashes,exact splits,training fits,heldout fits,predictors,scores,cutoffs,pair/unmatched decisions; no sensor-law validation or independent efficacy.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
