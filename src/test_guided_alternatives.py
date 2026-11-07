import numpy as np
from confirmatory_scenes import make_case
from guided_alternatives import guided_pool,candidates


def test_every_candidate_parameter_and_weight_exactly_excludes_raw_center():
    h,_,_=make_case(619900,719900,'positive',6.)
    a=candidates(h);changed=h.copy();changed[8,8]=0;changed[8,8,170]=800;b=candidates(changed)
    for name in a:
        assert a[name]['parameters']==b[name]['parameters']
        assert a[name]['component_delays']==b[name]['component_delays']
        assert len(a[name]['means'])==len(a[name]['component_delays'])
        assert all(np.array_equal(x,y) and np.all(x>0) for x,y in zip(a[name]['means'],b[name]['means']))


def test_foreground_guidance_rejects_donors_across_a_depth_step():
    t=np.arange(256);h=np.zeros((16,16,256),int)
    for row in range(16):
        for col in range(16):
            center=80 if col<10 else 95
            pulse=np.exp(-4*np.log(2)*((t-center)/5.)**2);pulse/=pulse.sum()
            h[row,col]=np.round(200*pulse)
    pool,info=guided_pool(h,8,8,spatial_sigma=2.5)
    assert abs(info['anchor']-80)<1 and np.argmax(pool)==80
    assert pool[95]<pool[80]*.01 and info['effective_donors']>=8
