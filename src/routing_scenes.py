"""Fresh routing study, including prespecified untrained geometries."""
import numpy as np
from guided_confirmatory_scenes import make_case
from pilot import first_detection_probabilities


def make_routing_case(item):
    limits=(12.,24.) if item['separation_family']==0 else (48.,80.)
    h,m,truth=make_case(item['scene_seed'],item['noise_seed'],item['kind'],item['secondary'],separation_range=limits)
    family=item.get('support_family','coherent')
    if family=='coherent':return h,m,truth
    y,x=np.mgrid[:16,:16];x=x-8;y=y-8
    rng=np.random.default_rng(item['scene_seed']+20000000)
    phi=rng.uniform(0,2*np.pi);normal=x*np.cos(phi)+y*np.sin(phi);transverse=-x*np.sin(phi)+y*np.cos(phi)
    if family=='singleton':
        mask=np.zeros((16,16),bool);mask[8,8]=True;secondary_delay=truth['secondary_delay']
    elif family=='filament':
        mask=np.abs(normal)<=.55
        secondary_delay=truth['primary_delay']+rng.uniform(*limits)+rng.uniform(-.12,.12)*transverse
    elif family=='crossing_step':
        mask=np.ones((16,16),bool);jump=rng.uniform(6,12)
        offset=rng.uniform(-.45,-.05)
        secondary_delay=truth['primary_delay']+rng.uniform(*limits)+rng.uniform(-.12,.12)*transverse+jump*(normal>offset)
        m['crossing_jump']=float(jump);m['crossing_offset']=float(offset)
    else:raise ValueError('Unknown support family')
    t=np.arange(256);w=truth['fwhm']
    def pulse(d):
        g=np.exp(-4*np.log(2)*((t-d[...,None])/w[...,None])**2);return g/g.sum(-1,keepdims=True)
    rate=(truth['primary_incident'][...,None]*pulse(truth['primary_delay'])+
        (truth['secondary_incident']*mask)[...,None]*pulse(secondary_delay)+truth['background'][...,None]/256)/1000
    noise=np.random.default_rng(item['noise_seed'])
    for r in range(16):
        for c in range(16):h[r,c]=noise.multinomial(1000,first_detection_probabilities(rate[r,c]))[:-1]
    truth=dict(truth,secondary_delay=secondary_delay,secondary_mask=mask)
    m.update(context=family,secondary_delay=float(secondary_delay[8,8]),
        secondary_incident=float(truth['secondary_incident'][8,8]),positive=True,
        support_family=family,secondary_support_pixels=int(mask.sum()))
    return h,m,truth
