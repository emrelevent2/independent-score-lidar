"""Recompute every saved sparse/full numerical enclosure, without reusing times."""
import json
from coverage_sparse_benchmark import verify,worker,OUT,P,sha


def main():
    p=verify();a=json.loads((OUT/'analysis.json').read_text());assert a['protocol_sha256']==sha(P)
    saved={r['id']:r for r in a['records']};checked=0
    for i,c in enumerate(p['cases']):
        r=worker(c,p['records_sha256'][c['id']],i);s=saved[c['id']]
        for key in r:
            if key not in ['seconds','speed_ratio_full_over_sparse']:assert r[key]==s[key],(c['id'],key)
        checked+=1
    result=dict(status='PASS',protocol_sha256=sha(P),cases=checked,
        scope='Every full/sparse Arb log interval,retained-mass gap,component count,existing-bound decision and numeric consistency recomputed. Original timings are observations,not deterministic replay targets.')
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
