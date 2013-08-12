from warnings import warn
import pytest
from path import path
from .. import packing

# Run with "py.test"

@pytest.fixture
def realdata():
    from pantracks import BigTracks
    datasource = path('/Users/nkeim/colldata/120908/183542-mov')
    tracksfile = datasource / 'bigtracks.h5'
    return BigTracks(tracksfile)
rdcutoff = 8.2

def test_psi6_rd(realdata):
    ftr = realdata[1]
    bop = packing.psi6(ftr, cutoff=rdcutoff)
    assert bop.bopmag.mean() > 0.5
    assert bop.bopmag.mean() < 1
    
def test_affine_field_rd(realdata):
    afff = packing.affine_field(realdata[1], realdata[100], rdcutoff/1.5*2.5,
            d2min_scale=rdcutoff/1.5)
    assert abs(abs(afff.hstrain.mean()) - 0.01) < 0.005
    assert afff.d2min.mean() < 0.002

def test_gofr_rd(realdata):
    gofr, bin_edges = packing.pairCorrelationR(realdata[1], fast=True)
    assert abs(1 - gofr[-1]) < 0.05
    assert max(gofr) > 2
    assert min(gofr) >= 0

def test_gofr_rd_slow(realdata):
    gofr, bin_edges = packing.pairCorrelationR(realdata[1])
    assert abs(1 - gofr[-1]) < 0.05
    assert max(gofr) > 2
    assert min(gofr) >= 0

def test_gofrvec_rd(realdata):
    gofr, bin_edges = packing.pairCorrelationVector(realdata[1], fast=True)
    assert gofr[0,0] == -0.1 # Outside cutoff
    assert max(gofr.flat) > 2
    flatvals = gofr.flatten()
    assert min(flatvals.compress(flatvals >= 0)) < 0.05
