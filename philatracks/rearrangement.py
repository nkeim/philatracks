"""Detect topological rearrangements among many particles"""
import numexpr
import numpy as np
from scipy.spatial import Delaunay
import pandas

def edgeDF_fast(ftr, maxpid=10000000):
    """Extracts the set of edges in the Delaunay triangulation of tracks DataFrame
    'ftr', which is indexed by particle ID. 
    
    'maxpid' is the highest expected particle ID number, used to construct a globally 
    unique identifier for an edge."""
    pts = np.vstack([ftr.x.values, ftr.y.values]).T
    pids = np.array(ftr.index.values, dtype=float)
    tri = Delaunay(pts)
    # Idea: use the fact that we have a limited number of particles to quickly generate
    # a "hash" (unique number that represents an edge)
    def sortverts(va, vb, maxpid):
        return np.vstack([numexpr.evaluate('where(va > vb, va, vb)'),
           numexpr.evaluate('where(va > vb, vb, va)'), \
           numexpr.evaluate('where(va > vb, va, vb) + maxpid * where(va > vb, vb, va)')]).T
    def edges_raw(n, m):
        return sortverts(tri.vertices[:,n], tri.vertices[:,m], maxpid)
    alledges = np.vstack([edges_raw(n, m) for n, m in [(0, 1), (0, 2), (1, 2)]])
    uni, unidx = np.unique(alledges[:,2], return_index=True)
    uni_edges = np.take(alledges[:,0:2], unidx, axis=0)
    edgepids = np.take(pids, uni_edges[:,0:2])
    #jpids = np.take(pids, uni_edges[:,1])
    sel = edgepids[:,0] > edgepids[:,1]
    ipids = np.where(sel, edgepids[:,0], edgepids[:,1])
    jpids = np.where(sel, edgepids[:,1], edgepids[:,0])
    return pandas.DataFrame({'i': ipids, 'j': jpids}, 
            index=numexpr.evaluate('ipids + maxpid*jpids'))
def changedEdges_fast(ftr0, ftr1, maxEdgeLength=1e20, edgeData=None, edgeSetData=None, 
        diag=False):
    """Compares edges in 2 frames. Outputs DataFrames for lost and gained edges.

    Returned DataFrames have columns i, j giving global particle IDs, and
    xi, yi, xj, yj giving endpoints.

    'maxEdgeLength' filters out spurious edges. Set it to O(1) particle spacing using
    g(r) data.

    Supply 'edgeData', 'edgeSetData' to accelerate.
    """
    if edgeData is None:
        ed0, ed1 = edgeDF_fast(ftr0), edgeDF_fast(ftr1)
    else:
        ed0, ed1 = edgeData
    if edgeSetData is None:
        ed0s = set(ed0.index.values)
        ed1s = set(ed1.index.values)
    else:
        ed0s, ed1s = edgeSetData
    lostids = ed0s - ed1s
    gainedids = ed1s - ed0s
    def _prepareEdgeDifferences(frameTracks, frameTracks1, frameEdges, diffIDs, maxEdgeLength):
        ediff = frameEdges.reindex(diffIDs)
        # Use original frame's coordinates for future calculations, but blend in the
        # alternate frame, to exclude particles that are not in both.
        ediffWithCoords = ediff.join(frameTracks, on='i').join(
                frameTracks, on='j', lsuffix='i', rsuffix='j').join(
                frameTracks1, on='i').join(
                frameTracks1, on='j', lsuffix='i1', rsuffix='j1').dropna()
        #for colname in ['framei', 'framej', 'framei1', 'framej1']:
        #    del ediffWithCoords[colname]
        ediffCut = ediffWithCoords[(ediffWithCoords.xi - ediffWithCoords.xj)**2 + \
                (ediffWithCoords.yi - ediffWithCoords.yj)**2 < maxEdgeLength**2]
        return ediffCut
    lost = _prepareEdgeDifferences(ftr0, ftr1, ed0, lostids, maxEdgeLength)
    gained = _prepareEdgeDifferences(ftr0, ftr1, ed1, gainedids, maxEdgeLength)
    if diag:
        return (gainedids, lostids), (lost, gained)
    else:
        return lost, gained
def findT1(lost, gained):
    """Find T1-like events: lost and gained edges that intersect.

    Works on output of changedEdges_fast(). Returns a pair of arrays,
    where each pair of elements indexes 'lost' and 'gained', respectively, 
    identifying the 2 edges associated with a T1 event.
    """
    # Make a line a + bx - y = 0 out of each edge.
    def lignify(ed):
        ed['b'] = (ed.yj - ed.yi) / (ed.xj - ed.xi)
        ed['a'] = ed.yi - ed.b * ed.xi
    lignify(lost)
    lignify(gained)
    # Test each point against each edge
    def testStraddle(e1, x, y):
        n = len(x)
        m = len(e1.a.values)
        xT = np.tile(x, (m, 1)) # Columns -> particles
        yT = np.tile(y, (m, 1))
        aT = np.tile(e1.a.values.reshape((m, 1)), (1, n)) # Rows -> edges
        bT = np.tile(e1.b.values.reshape((m, 1)), (1, n))
        return aT + bT * xT - yT
    tsi = testStraddle(lost, gained.xi.values, gained.yi.values)
    tsj = testStraddle(lost, gained.xj.values, gained.yj.values)
    gtsi = testStraddle(gained, lost.xi.values, lost.yi.values)
    gtsj = testStraddle(gained, lost.xj.values, lost.yj.values)
    # If the i and j points fall on opposite sides of a given line, that element
    # in (tsi * tsj) with index (lost edge, gained edge) will be negative.
    # We perform the same operation on the reverse case and transpose.
    # Then if (lost edge, gained edge) is negative in both arrays, the edges
    # cross and we have probably found a T1 event.
    xfinder = ((tsi * tsj) < 0) & ((gtsi * gtsj).T < 0)
    return xfinder.nonzero()
def particles(t1cat):
    """Return a list of the particles in a T1 catalog DataFrame.

    Use it to find the individual particles involved in a group of events."""
    return particles_fromlist(t1cat.particles.tolist())
def particles_fromlist(particle_list):
    return list(set(np.array(particle_list).flat))
def build_T1_catalog(frameiter, win=None, maxEdgeLength=16., minLengthChange=0.005, 
        limit=None, criterion='2013a'):
    """Identifies T1 rearrangements in a sequence of frames. 
    Returns a DataFrame with a structure similar to tracks data.

    'frameiter': Iterable of tracks DataFrames. First element is used as reference topology.
    'win' - dict with crop region. None = full frame
    'maxEdgeLength' - longest Delaunay edge to consider. Should be ~2x first g(r) trough.
    'minLengthChange' - Minimum factor by which aspect ratio of T1 group must change
            to register as an event.
    'limit' - stop after this many unique 4-particle rearrangements have been identified. 
        1e5 might be a good choice.
    'criterion' selects how 'minLengthChange' is applied:
        "2012": Looks at proportional difference of aspect ratios before and after.
            Has the drawback that it is not equal under exchange of particle pairs.
        "2013a": Fully symmetric test.
        When short edge length is ~6.8, a 2012 threshold of 0.1 ~ 2013a threshold of 0.005.
    """
    t1fnums, t1ids, t1len_changes = [], [], []
    # For calculating aspect ratio below
    # Potential speedup: use the coordinates that are already in 'lost', 'gained'
    edlen = lambda ftr, pids: np.sqrt((ftr.x[pids[0]] - ftr.x[pids[1]])**2 + \
                                   (ftr.y[pids[0]] - ftr.y[pids[1]])**2)
    ed0 = None
    for ftr1_oidx in frameiter:
        assert len(ftr1_oidx) # Otherwise error msg will be confusing
        fnum = ftr1_oidx.frame.values[0]
        ftr1 = ftr1_oidx[['particle', 'x', 'y']].set_index('particle')
        ed1 = edgeDF_fast(ftr1)
        ed1s = set(ed1.index.values)
        if ed0 is None:
            # Establish the first frame, with which others will be compared.
            ed0, ed0s, ftr0, ftr0_oidx = ed1, ed1s, ftr1, ftr1_oidx
            continue # Need more data
        lost, gained = changedEdges_fast(ftr0, ftr1, maxEdgeLength=maxEdgeLength,
                                                       edgeData=(ed0, ed1),
                                                       edgeSetData=(ed0s, ed1s))
        t1events = findT1(lost, gained)
        for lidx, gidx in zip(*t1events):
            idtup = (lost.i[lidx], lost.j[lidx], gained.i[gidx], gained.j[gidx])
            if len(set(idtup)) < 4: continue # Must contain 4 different particles
            if criterion == '2012':
                # Significance cut: aspect ratio must have changed by a minimum amount
                ar0, ar1 = [edlen(ftr, idtup[0:2]) / edlen(ftr, idtup[2:4]) \
                        for ftr in (ftr0, ftr1)]
                len_change = abs(ar0 - ar1) / (ar0 + ar1)
                if len_change < minLengthChange: continue
            elif criterion == '2013a':
                rab0, rcd0, rab1, rcd1 = [edlen(ftr, edids) \
                        for ftr in (ftr0, ftr1) \
                        for edids in (idtup[0:2], idtup[2:4])]
                len_change = -(rab0 - rab1) * (rcd0 - rcd1) / \
                        ((rab0 + rab1) * (rcd0 + rcd1))
                if len_change < minLengthChange: continue
            else:
                raise ValueError('Invalid threshold criterion choice.')
            t1fnums.append(fnum)
            t1ids.append(idtup)
            t1len_changes.append(len_change)
        if limit and len(set(t1ids)) >= limit: break
    t1cat = pandas.DataFrame({'frame': t1fnums, 'particles': t1ids, 
        'len_change': t1len_changes})
    return t1cat
