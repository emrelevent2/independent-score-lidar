"""Checks required before freezing previously untouched evaluation seeds."""
from pathlib import Path
import numpy as np
from pilot import first_detection_probabilities
from likelihood_baseline import pulse_derivatives,local_likelihood_report
from read_rt3d import read_rbin
from rt3d_io import write_rbin,read_pcl_ply,native_pixel_points

ROOT=Path(__file__).resolve().parents[1]


def test_pulse_derivatives_against_finite_differences_at_edges():
    t=np.arange(64)
    for d,w in [(1.3,4.5),(31.37,5.2),(62.1,5.5)]:
        g,dd,dw=pulse_derivatives(t,d,w);eps=1e-5
        fd=(pulse_derivatives(t,d+eps,w)[0]-pulse_derivatives(t,d-eps,w)[0])/(2*eps)
        fw=(pulse_derivatives(t,d,w+eps)[0]-pulse_derivatives(t,d,w-eps)[0])/(2*eps)
        assert np.allclose(dd,fd,rtol=1e-6,atol=1e-9)
        assert np.allclose(dw,fw,rtol=1e-6,atol=1e-9)
        assert np.isclose(g.sum(),1) and abs(dd.sum())<1e-12 and abs(dw.sum())<1e-12


def test_local_likelihood_fit_on_exact_expected_photon_histograms():
    t=np.arange(256);primary=pulse_derivatives(t,80.3,5.2)[0]
    secondary=pulse_derivatives(t,96.2,5.2)[0]
    for count in [0.,6.]:
        rates=(50*primary+count*secondary+20/256)/1000
        h=1000*first_detection_probabilities(rates)[:-1]
        report=local_likelihood_report(h)
        if count==0:assert report['score']<1e-4
        else:
            assert abs(report['primary']-80.3)<.05
            assert abs(report['delay']-96.2)<.05
            assert report['score']>1.


def test_writer_matches_official_matlab_fixture_byte_for_byte(tmp_path):
    original=ROOT/'data'/'rt3d_measured'/'format_fixture.rbin'
    d=read_rbin(original);path=tmp_path/'regenerated.rbin'
    write_rbin(path,d['histograms'],d['irf'],d['scale_ratio'])
    assert path.read_bytes()==original.read_bytes()


def test_official_rt3d_native_grid_and_time_coordinate():
    path=ROOT/'tools'/'rt3d_adapter_check'/'output_batch'/'frame0_w0.ply'
    points=read_pcl_ply(path);row,col,delay,intensity=native_pixel_points(points,16,16,.25)
    assert len(points)==256 and len(set(zip(row,col)))==256
    assert np.max(np.abs(delay-80))<.02
    path=path.with_name('frame1_w0.ply');points=read_pcl_ply(path)
    row,col,delay,_=native_pixel_points(points,16,16,.25)
    interior=(row>2)&(row<13)&(col>2)&(col<13)
    fit=np.linalg.lstsq(np.stack([np.ones(len(row)),row,col],axis=-1)[interior],delay[interior],rcond=None)[0]
    # Algorithmic smoothing bias is not confused with an axis swap.
    assert abs(fit[1]-.25)<.04 and abs(fit[2]-.5)<.04
