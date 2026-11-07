"""Explicit formats and coordinate mapping for the unmodified RT3D executable."""
from pathlib import Path
import struct
import numpy as np


def write_rbin(path,histograms,irf,scale_ratio=1.):
    h=np.asarray(histograms);g=np.asarray(irf,dtype='<f4')
    if h.ndim!=4 or np.any(h<0) or np.any(h!=np.floor(h)) or np.any(h>65535):
        raise ValueError('Nonnegative uint16-range frames,row,col,time counts required')
    frames,nr,nc,bins=h.shape
    if not all(0<x<65535 for x in [frames,nr,nc,bins,len(g)]):raise ValueError('Invalid dimensions')
    if g.ndim!=1 or np.any(g<0) or g.sum()<=0:raise ValueError('Invalid IRF')
    with Path(path).open('wb') as f:
        f.write(struct.pack('<HHHfHH',nr,nc,bins,scale_ratio,0,len(g)))
        f.write(g.tobytes());f.write(struct.pack('<H',frames))
        for frame in h:
            for col in range(nc):
                for row in range(nr):
                    ids=np.flatnonzero(frame[row,col])
                    data=np.empty((len(ids),2),dtype='<u2')
                    data[:,0]=ids+1;data[:,1]=frame[row,col,ids]
                    f.write(data.tobytes());f.write(struct.pack('<H',65535))


def read_pcl_ply(path):
    types={'float':'<f4','double':'<f8','int':'<i4','uint':'<u4','uchar':'u1'}
    fields=[];vertices=None;active=False;format_name=None
    with Path(path).open('rb') as f:
        if f.readline().strip()!=b'ply':raise ValueError('Not a PLY')
        while True:
            raw=f.readline()
            if not raw:raise ValueError('Truncated header')
            tokens=raw.decode('ascii').strip().split()
            if not tokens:continue
            if tokens[0]=='format':format_name=tokens[1]
            elif tokens[0]=='element':
                active=tokens[1]=='vertex'
                if active:vertices=int(tokens[2])
            elif tokens[0]=='property' and active:
                if tokens[1] not in types:raise ValueError('Unsupported vertex type')
                fields.append((tokens[2],types[tokens[1]]))
            elif tokens[0]=='end_header':break
        if format_name!='binary_little_endian' or vertices is None:raise ValueError('Unexpected PLY format')
        data=np.fromfile(f,dtype=np.dtype(fields),count=vertices)
        if len(data)!=vertices:raise ValueError('Truncated vertex array')
    return data


def native_pixel_points(points,height,width,scale_ratio=1.):
    # Source ReadPoint and visualization::loadPointCloud fix transverse grid
    # coordinates, reverse row/column axes, and put temporal delay in PLY x.
    row=height-points['z']/scale_ratio;col=width-points['y']/scale_ratio
    if not np.allclose(row,np.rint(row),atol=1e-4) or not np.allclose(col,np.rint(col),atol=1e-4):
        raise ValueError('Non-native output grid; run RT3D with upsampling=1')
    row=np.rint(row).astype(int);col=np.rint(col).astype(int)
    if np.any(row<0)|np.any(row>=height)|np.any(col<0)|np.any(col>=width):raise ValueError('Point outside native grid')
    return row,col,points['x'].astype(float),points['intensity'].astype(float)
