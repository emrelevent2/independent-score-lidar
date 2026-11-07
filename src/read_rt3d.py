"""Read the sparse .rbin format documented by the upstream mat_to_RT3D.m.

Do not infer acquisition cycles, physical bin duration, or ground truth from
this container: those quantities are not stored in its header.
"""
from pathlib import Path
import struct
import numpy as np


def read_rbin(path):
    data = Path(path).read_bytes()
    pos = 0

    def scalar(fmt):
        nonlocal pos
        val = struct.unpack_from('<'+fmt, data, pos)[0]
        pos += struct.calcsize('<'+fmt)
        return val

    nr, nc, bins = scalar('H'), scalar('H'), scalar('H')
    ratio = scalar('f')
    flag, irf_len = scalar('H'), scalar('H')
    if min(nr, nc, bins, irf_len) <= 0 or flag not in (0, 65535):
        raise ValueError('Invalid .rbin header')
    count = nr*nc*irf_len if flag else irf_len
    dtype = '<u2' if flag else '<f4'
    irf = np.frombuffer(data, dtype=dtype, count=count, offset=pos).copy()
    pos += count*np.dtype(dtype).itemsize
    if flag:
        # Upstream writes column, row, time as nested loops.
        irf = irf.reshape(nc, nr, irf_len).transpose(1, 0, 2)
    frames = scalar('H')
    if frames <= 0:
        raise ValueError('No frames in container')
    h = np.zeros((frames, nr, nc, bins), np.uint32)
    for f in range(frames):
        for col in range(nc):
            for row in range(nr):
                while True:
                    idx = scalar('H')
                    if idx == 65535:
                        break
                    if not 1 <= idx <= bins:
                        raise ValueError(f'Out-of-range bin {idx}')
                    n = scalar('H')
                    if h[f, row, col, idx-1] != 0:
                        raise ValueError('Repeated nonzero bin')
                    h[f, row, col, idx-1] = n
    if pos != len(data):
        raise ValueError(f'Unexpected trailing bytes: {len(data)-pos}')
    return dict(histograms=h, irf=irf, scale_ratio=ratio,
                per_pixel_irf=bool(flag), frames=frames)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    for path in (root/'data'/'rt3d_measured').glob('*.rbin'):
        d = read_rbin(path)
        h = d['histograms']
        print(path.name, 'shape', h.shape, 'IRF', d['irf'].shape,
              'counts/pixel', np.quantile(h.sum(-1), [0, .25, .5, .75, 1]),
              'IRF maximum', np.argmax(d['irf']), flush=True)
