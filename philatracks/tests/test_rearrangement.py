import pytest
from warnings import warn
from path import path
from pantracks import BigTracks
from .. import rearrangement

datasource = path('/Users/nkeim/colldata/120908/183542-mov')
tracksfile = datasource / 'bigtracks.h5'

@pytest.mark.skipif("not tracksfile.exists()")
def test_T1_realdata():
    bt = BigTracks(tracksfile)

    frames = [1, 100, 101, 1] # So t1cat should contain rearrangements for 100 and 101 only.
    t1cat = rearrangement.build_T1_catalog((bt[i] for i in frames))
    assert set(t1cat.frame.unique()) == set((100, 101))

    n100_expect = 92
    n100 = len(t1cat[t1cat.frame == 100])
    if n100 != n100_expect:
        warn('Expected %i events; got %i' % (n100_expect, n100))

    p100_expect = 291
    p100 = len(rearrangement.particles(t1cat))
    if p100 != p100_expect:
        warn('Expected %i rearranging particles; got %i' % (p100_expect, p100))
