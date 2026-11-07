import numpy as np
from guided_alternatives import guided_pool,fit_pool
from likelihood_baseline import local_likelihood_report
from complete_guided_likelihood import complete_guided_likelihood
from support_routing import route_report


def test_routing_preserves_supported_location_and_recovers_an_isolated_fixture():
    t=np.arange(256)
    def pulse(d):
        g=np.exp(-4*np.log(2)*((t-d)/5.)**2);return g/g.sum()
    for supported in [True,False]:
        h=np.tile(np.round(100*pulse(80)+20/256).astype(int),(16,16,1))
        if supported:h+=np.round(30*pulse(120)).astype(int)
        else:h[8,8]+=np.round(30*pulse(120)).astype(int)
        pool,info=guided_pool(h,8,8);q=dict(**fit_pool(pool),**info)
        local=local_likelihood_report(h[8,8]);spatial=complete_guided_likelihood(h[8,8],q['primary'],q['secondary'])
        r=route_report(q,spatial,local,dict(spatial=1.,local=1.),'route',2.5)
        assert r['head']==('spatial' if supported else 'local')
        assert abs(r['delay']-120)<1 and r['calibration_required']
