"""Random heterogeneous scenes for the separately frozen confirmatory study.

This generator receives no detector outputs. Geometry/physics metadata are
available to evaluation only. Test seed ranges must not be used to tune code.
"""
import numpy as np
from pilot import first_detection_probabilities


def make_case(scene_seed,noise_seed,kind='null',secondary=6.,size=16):
    rng=np.random.default_rng(scene_seed)
    yy,xx=np.mgrid[:size,:size];row=col=size//2
    x=xx-col;y=yy-row
    phi=rng.uniform(0,2*np.pi);normal=x*np.cos(phi)+y*np.sin(phi)
    transverse=-x*np.sin(phi)+y*np.cos(phi)
    geometry=int(rng.integers(0,5))
    base=rng.uniform(65,100)
    primary_delay=base+rng.uniform(-.25,.25)*x+rng.uniform(-.25,.25)*y
    if geometry==0:primary_delay+=rng.uniform(-.015,.015)*(x*x+y*y)
    elif geometry==1:primary_delay+=rng.uniform(6,12)*(normal>rng.uniform(1,3))
    separation=rng.uniform(12,24)
    secondary_delay=primary_delay+separation+rng.uniform(-.12,.12)*transverse
    if geometry==2:secondary_delay+=rng.uniform(4,8)*(normal>rng.uniform(1,3))
    if geometry==0:mask=x*x+y*y<=rng.uniform(4,6)**2
    elif geometry==1:mask=normal>-rng.uniform(1,3)
    elif geometry==2:mask=np.abs(normal)<rng.uniform(1.5,2.5)
    elif geometry==3:mask=(np.abs(x)<=1)&(np.abs(y)<=1)
    else:mask=np.abs(transverse-.08*normal*normal)<rng.uniform(1.5,2.5)
    context='positive'
    if kind=='null':
        category=int(rng.integers(0,3))
        if category==0:mask[:]=False;context='globally_single'
        elif category==1:
            mask[:]=True
            radius=int(rng.integers(0,2));mask[row-radius:row+radius+1,col-radius:col+radius+1]=False
            context='hole'
        else:mask=normal>rng.uniform(.25,1.5);context='edge'
        secondary=float(rng.choice([3.,6.,10.,20.]))
        mask[row,col]=False
    elif kind=='positive':mask[row,col]=True
    else:raise ValueError(kind)
    primary_total=rng.uniform(20,100)
    if kind=='null' and rng.random()<.1:primary_total=rng.uniform(300,600)
    primary_count=primary_total*np.exp(rng.uniform(-.012,.012)*x+rng.uniform(-.012,.012)*y)
    background_total=float(np.exp(rng.uniform(np.log(5),np.log(60))))
    background=background_total*rng.uniform(.85,1.15,(size,size))
    fwhm_base=rng.uniform(4.6,5.4)
    fwhm=np.clip(fwhm_base+rng.uniform(-.05,.05,(size,size)),4.5,5.5)
    weak_count=float(secondary)*np.exp(rng.uniform(-.02,.02)*x+rng.uniform(-.02,.02)*y)
    t=np.arange(256)
    def pulse(center):
        z=np.exp(-4*np.log(2)*((t-center[...,None])/fwhm[...,None])**2)
        return z/z.sum(-1,keepdims=True)
    rate=(primary_count[...,None]*pulse(primary_delay)+
          (weak_count*mask)[...,None]*pulse(secondary_delay)+background[...,None]/256)/1000
    observation=np.empty((size,size,256),dtype=np.int32)
    noise=np.random.default_rng(noise_seed)
    for r in range(size):
        for c in range(size):observation[r,c]=noise.multinomial(1000,first_detection_probabilities(rate[r,c]))[:-1]
    metadata=dict(scene_seed=int(scene_seed),noise_seed=int(noise_seed),kind=kind,
        context=context,geometry=geometry,row=row,column=col,
        primary_delay=float(primary_delay[row,col]),secondary_delay=float(secondary_delay[row,col]),
        primary_incident=float(primary_count[row,col]),secondary_incident=float(weak_count[row,col]) if mask[row,col] else 0.,
        background_incident=float(background[row,col]),fwhm=float(fwhm[row,col]),
        spatial_scale_ratio=.25,cycles=1000,positive=bool(mask[row,col]))
    truth=dict(primary_delay=primary_delay,secondary_delay=secondary_delay,secondary_mask=mask,
               primary_incident=primary_count,secondary_incident=weak_count,background=background,fwhm=fwhm)
    return observation,metadata,truth
