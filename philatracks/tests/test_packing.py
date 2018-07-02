from warnings import warn
import pytest

from os import path
import numpy as np
import pandas

from philatracks import packing
# Run with "py.test"

datasource = '/nfsbigdata1/keimlab/colldata/140612/223147-mov'
tracksfile = path.join(datasource, 'tracks.h5')
data_missing = not path.exists(tracksfile)

@pytest.fixture
def realdata():
    import trackpy
    return trackpy.PandasHDFStoreBig(tracksfile, mode='r')


@pytest.fixture
def fakedata():
    x, y = np.mgrid[:30,:30].astype(float)
    ftr0 = pandas.DataFrame({'x': x.flat, 'y': y.flat, 
        'frame': 0, 'particle': range(len(x.flat))})
    strain = 0.1
    ftr1 = pandas.DataFrame({'x': (x + strain * y).flat, 'y': y.flat, 
        'frame': 0, 'particle': range(len(x.flat))})
    return ftr0, ftr1


rdcutoff = 8.2


def _check_scalar(series, value):
    vals = series.dropna().values
    assert len(vals)
    assert np.allclose(vals, value)
def _perturb_ftr(ftr):
    """Displace a single particle. 
    
    Return the ID of the particle, and the perturbed DataFrame."""
    ftr = ftr.copy()
    pid = ftr.particle[(ftr.x == 15) & (ftr.y == 15)].values[0]
    ftr.y[ftr.particle == pid] += 0.1
    return pid, ftr

@pytest.mark.skipif("data_missing")
def test_psi6_rd(realdata):
    ftr = realdata[1]
    bop = packing.psi6(ftr, cutoff=rdcutoff)
    assert bop.bopmag.mean() > 0.5
    assert bop.bopmag.mean() < 1
def test_psi6_fd(fakedata):
    ftr = fakedata[0]
    bop = packing.psi6(ftr, cutoff=1.6)
    # psi6 of square lattice is zero
    _check_scalar(bop.bopmag, 0)
    
@pytest.mark.skipif("data_missing")
def test_affine_field_rd(realdata):
    afff = packing.affine_field(realdata[1], realdata[100], rdcutoff/1.5*2.5,
            d2min_scale=rdcutoff/1.5)
    assert abs(abs(afff.hstrain.mean()) - 0.01) < 0.005*11
    assert afff.d2min.mean() < 0.005
def test_affine_field_fd(fakedata):
    ftr0, ftr1 = fakedata
    afff = packing.affine_field(ftr0, ftr1, cutoff=2.5)
    assert np.allclose((ftr1.set_index('particle').x \
            - afff.set_index('particle').x).dropna(), 0)
    _check_scalar(afff.hstrain, -0.1)
    _check_scalar(afff.vstrain, 0)
    _check_scalar(afff.xdil, 1)
    _check_scalar(afff.ydil, 1)
    _check_scalar(afff.d2min, 0)
def test_affine_field_perturbed(fakedata):
    ftr0, ftr1 = fakedata
    pid, ftr1p = _perturb_ftr(ftr0)
    afff = packing.affine_field(ftr1p, ftr0, cutoff=2.5)
    assert np.isclose(afff.hstrain.median(), 0)
    assert np.isclose(afff.d2min.median(), 0)
    pd2min = afff.d2min[afff.particle == pid]
    # Upper bound on D2min is set by
    # (N * (0.1**2)) / N, where N is number of neighbors
    assert np.isclose(pd2min, 0.1**2, rtol=1e-2, atol=1e-3)
def test_local_displacements(fakedata):
    ftr0, ftr1 = fakedata
    pid, ftr1p = _perturb_ftr(ftr0)
    ld = packing.local_displacements(ftr0, ftr1p, cutoff=8.5)
    # Check cropping
    assert ld.dropna().x.max() == 20
    assert ld.dropna().y.max() == 20
    assert ld.dropna().x.min() == 9
    assert ld.dropna().y.min() == 9
    # Check displacements
    assert np.allclose(ld.dxlocal.dropna(), 0)
    assert np.all((ld.dylocal.median() < 0) & (ld.dylocal.median() > -0.1))
    assert np.isclose(ld.dylocal[ld.particle == pid], 0.1)


@pytest.mark.skipif("data_missing")
def test_gofr_rd(realdata):
    gofr, bin_edges = packing.pairCorrelationR(realdata[1], fast=True)
    assert abs(1 - gofr[-1]) < 0.05
    assert max(gofr) > 2
    assert min(gofr) >= 0


@pytest.mark.skipif("data_missing")
def test_gofr_rd_slow(realdata):
    gofr, bin_edges = packing.pairCorrelationR(realdata[1])
    assert abs(1 - gofr[-1]) < 0.05
    assert max(gofr) > 2
    assert min(gofr) >= 0
def test_gofr_fd(fakedata):
    gofr, bin_edges = packing.pairCorrelationR(fakedata[0], dr=0.1, cutoff=10)
    assert abs(1 - np.mean(gofr[-10:])) < 0.15
    assert max(gofr) > 2
    assert min(gofr) >= 0

@pytest.mark.skipif("data_missing")
def test_gofrvec_rd(realdata):
    gofr, bin_edges = packing.pairCorrelationVector(realdata[1], fast=True)
    assert gofr[0,0] == -0.1 # Outside cutoff
    assert max(gofr.flat) > 2
    flatvals = gofr.flatten()
    assert min(flatvals.compress(flatvals >= 0)) < 0.05
def test_gofrvec_fd(fakedata):
    gofr, bin_edges = packing.pairCorrelationVector(fakedata[0], cutoff=10,
            dx=0.25, fast=True)
    assert gofr[0,0] == -0.1 # Outside cutoff
    assert max(gofr.flat) > 2
    flatvals = gofr.flatten()
    assert min(flatvals.compress(flatvals >= 0)) < 0.05


