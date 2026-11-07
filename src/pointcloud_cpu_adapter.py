"""Technical CPU adapter for the unchanged official 2025 network.

Only PyTorch3D's inference-time Euclidean KNN kernel is replaced by exact
SciPy cKDTree queries. Model source and pretrained weights remain untouched.
This adapter is not a training implementation. Distance ties may use a
different valid neighbor ordering from PyTorch3D; record that limitation.
"""
from pathlib import Path
import sys,types,collections,hashlib,time
from contextlib import contextmanager
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.optimize import least_squares,nnls
from likelihood_baseline import pulse_derivatives
from pilot import coates
ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT.parent/'pointcloud_unrolling_original'


def exact_cpu_knn(x,y,K=1,**kwargs):
    if x.device.type!='cpu' or y.device.type!='cpu' or x.requires_grad or y.requires_grad:
        raise ValueError('CPU no-gradient inference only')
    if x.ndim!=3 or y.ndim!=3 or x.shape[0]!=y.shape[0] or x.shape[2]!=y.shape[2] or not 1<=K<=y.shape[1]:
        raise ValueError('Invalid batched KNN input')
    unsupported={k:v for k,v in kwargs.items() if v is not None and k not in ['return_nn','return_sorted','norm','version']}
    if unsupported or kwargs.get('return_nn',False) or not kwargs.get('return_sorted',True) or kwargs.get('norm',2)!=2:
        raise ValueError('Unsupported KNN mode')
    distances=[];indices=[]
    for xb,yb in zip(x.detach().numpy(),y.detach().numpy()):
        d,i=cKDTree(yb.astype(np.float64)).query(xb.astype(np.float64),k=K,eps=0.,p=2,workers=1)
        if K==1:d=d[:,None];i=i[:,None]
        distances.append(d*d);indices.append(i)
    result=collections.namedtuple('KNN','dists idx knn')
    return result(torch.as_tensor(np.stack(distances),dtype=x.dtype),torch.as_tensor(np.stack(indices),dtype=torch.long),None)


def install_knn_adapter():
    package=types.ModuleType('pytorch3d');ops=types.ModuleType('pytorch3d.ops');ops.knn_points=exact_cpu_knn
    package.ops=ops;sys.modules['pytorch3d']=package;sys.modules['pytorch3d.ops']=ops


@contextmanager
def cpu_tensor_allocations():
    """Port hard-coded fresh-tensor .cuda() allocations; restore on exit."""
    previous=torch.Tensor.cuda
    def remain_cpu(tensor,*args,**kwargs):
        if tensor.device.type!='cpu':raise ValueError('Only CPU tensors accepted by device adapter')
        return tensor
    torch.Tensor.cuda=remain_cpu
    try:yield
    finally:torch.Tensor.cuda=previous


def validate_knn():
    generator=torch.Generator().manual_seed(20261106)
    x=torch.randn((2,31,10),generator=generator,dtype=torch.float64)
    y=torch.randn((2,37,10),generator=generator,dtype=torch.float64)
    actual=exact_cpu_knn(x,y,K=7);dense=torch.cdist(x,y)**2;values,indices=torch.topk(dense,7,largest=False,sorted=True)
    assert torch.equal(actual.idx,indices) and torch.allclose(actual.dists,values,rtol=1e-12,atol=1e-12)
    same=exact_cpu_knn(x,x,K=7)
    assert torch.equal(same.idx[:,:,0],torch.arange(31).expand(2,-1))
    return dict(random_dense_distance_check=True,batch_and_self_index_check=True)


def image_to_points(depths):
    # Matches official test_middlebury.py, including its 0.008m pixel spacing.
    B,L,H,W=depths.shape
    if B!=1 or L!=8:raise ValueError('Published single-image eight-scale input required')
    pp=depths.reshape(B,L,H*W).transpose(2,1)*(1024*.003)
    pc=pp.clone();pc[:,:,::2]=pp[:,:,:4];pc[:,:,1::2]=pp[:,:,4:]
    grid=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
    xy=torch.stack([grid[0].flipud().reshape(-1)*.008,grid[1].reshape(-1)*.008],dim=-1).float()
    return torch.cat([xy.unsqueeze(0),pc],dim=-1)


class OfficialNetwork:
    def __init__(self):
        validate_knn();install_knn_adapter();sys.path.insert(0,str(UPSTREAM));sys.path.insert(0,str(UPSTREAM/'run'))
        from model.vanilla import Model
        from gen_initial_multiscale_2obj import gen_initial_multiscale,get_IF,shift_h
        self.gen,self.get_if,self.shift=gen_initial_multiscale,get_IF,shift_h
        self.model=Model(nscale=8,att=10,k=6,nblock=3,nconv=3).cpu()
        checkpoint=torch.load(UPSTREAM/'run'/'baseline.pth',map_location='cpu',weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True);self.model.eval()

    def predict(self,histograms,irf,seed=0):
        h=np.asarray(histograms);H,W,T=h.shape;irf=np.asarray(irf,float).copy()
        if len(irf)>T:raise ValueError('IRF longer than timing window')
        source=self.get_if(irf,T,(T-len(irf))//2)
        response=source[:,T//2];shifted=self.shift(response,T)
        trailing=int(np.argmax(shifted<1e-5)-1);attack=int(np.argmax(np.flip(shifted)<1e-5)-1)
        if trailing<0 or attack<0:raise ValueError('Invalid official thresholded IRF support')
        depths,_=self.gen(torch.as_tensor(h.copy(),dtype=torch.float32),torch.as_tensor(np.flip(shifted).copy(),dtype=torch.float32),
            attack=attack,trailing=trailing,use_cuda=False)
        # Official image_to_pcl uses gamma=1024*.003 even for a different T.
        # Retain that scaling for network input; invert it using this actual T.
        points=image_to_points(depths.unsqueeze(0))
        torch.manual_seed(seed)
        with torch.no_grad(),cpu_tensor_allocations():result=self.model(points,debug=0)
        final=result[0][-1].detach().cpu().numpy()[0]
        bins=final/(1024*.003)*T-1
        if not np.all(np.isfinite(bins)):raise ValueError('Nonfinite network output')
        return bins.reshape(H,W,2),dict(seed=seed,trailing_bins=trailing,attack_bins=attack,
            initial_center_bins=(depths[:,H//2,W//2]*T-1).tolist(),
            network_gamma_meter=1024*.003,time_bins=T,source_unmodified=True,knn='exact CPU cKDTree, eps0',
            device_adapter='Hard-coded fresh tensor CUDA allocations retained on CPU during inference; restored afterwards.',
            transfer_note='Pretrained model retained. Different temporal resolution, nominal Gaussian and small spatial support are transfer limitations.')


def report_from_centers(h,centers,cycles=1000):
    centers=np.asarray(centers,float)
    if centers.shape!=(2,) or not np.all(np.isfinite(centers)):raise ValueError('Two finite centers required')
    centers=np.clip(centers,0,len(h)-1)
    if abs(centers[0]-centers[1])<8:return dict(score=None,delay=None,reason='unresolved centers below shared8bin exclusion')
    flux=coates(h,cycles);t=np.arange(len(h));g=[pulse_derivatives(t,d,5.)[0] for d in centers]
    coeff,_=nnls(np.stack([*g,np.ones(len(t))/len(t)],axis=-1),flux*cycles)
    order=np.argsort(coeff[:2])[::-1];p,s=centers[order];a,c=coeff[:2][order];b=max(coeff[2],1e-10)
    def model(q):
        a,c,b,p,s,w=q;return a*pulse_derivatives(t,p,w)[0]+c*pulse_derivatives(t,s,w)[0]+b/len(t)
    low=np.array([0.,0.,1e-10,max(0,p-1.5),max(0,s-1.5),4.5]);high=np.array([np.inf,np.inf,np.inf,min(len(t)-1,p+1.5),min(len(t)-1,s+1.5),5.5])
    initial=np.minimum(np.maximum([a,c,b,p,s,5.],low+1e-9),high-1e-9)
    fit=least_squares(lambda q:model(q)-flux*cycles,initial,bounds=(low,high),ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=120)
    a,c,b,p,s,w=fit.x
    if c>a:a,c,p,s=c,a,s,p
    return dict(score=float(c),delay=float(s),primary=float(p),parameters=[float(x) for x in [a,c,b,p,s,w]],
        optimizer_success=bool(fit.success),report='Strongest locally fitted component is primary; common two-Gaussian flux amplitude score.')
