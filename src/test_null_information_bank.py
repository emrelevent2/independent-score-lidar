import numpy as np
from null_information_bank import NullInformationBank


def fixture():
    rng=np.random.default_rng(5182307);z=rng.normal(size=(1,3,3,4,12,8));h=np.einsum('...ki,...kj->...ij',z,z)+np.eye(8);return NullInformationBank(h,[[1,2,4]],[[.25,.5,1]],[0,.25,.5,.75],[2],base_phase_integer=3)


def test_grid_nodes_and_integer_phase_rotation_are_preserved():
    b=fixture()
    for ai in range(3):
        for bi in range(3):
            for fi in range(4):
                a,d=b.lookup([b.a[0,ai],3+b.f[fi],b.b[0,bi]],2);np.testing.assert_allclose(a,b.h[0,ai,bi,fi],atol=1e-13);assert not d['amplitude_clamped'] and not d['background_clamped'];shifted,_=b.lookup([b.a[0,ai],4+b.f[fi],b.b[0,bi]],2);np.testing.assert_allclose(shifted,np.roll(a,(1,1),axis=(0,1)),atol=1e-13)


def test_log_flux_interpolation_is_convex_and_positive():
    b=fixture();actual,_=b.lookup([np.sqrt(2),3.125,np.sqrt(.125)],2);expected=b.h[0,:2,:2,:2].mean(axis=(0,1,2));np.testing.assert_allclose(actual,expected,atol=1e-13);assert np.linalg.eigvalsh(actual).min()>0


def test_fractional_wrap_rotates_the_upper_neighbor_and_period_is_invariant():
    b=fixture();actual,_=b.lookup([2,3.875,.5],2);expected=.5*b.h[0,1,1,3]+.5*np.roll(b.h[0,1,1,0],(1,1),axis=(0,1));np.testing.assert_allclose(actual,expected,atol=1e-13);other,_=b.lookup([2,11.875,.5],2);np.testing.assert_array_equal(actual,other)


def test_outside_bank_values_are_flagged_without_dropping_queries():
    b=fixture();actual,flags=b.lookup([0.,3.,4.],2);np.testing.assert_array_equal(actual,b.h[0,0,-1,0]);assert flags['amplitude_clamped'] and flags['background_clamped']
