"""Generic edge-finding tools, based on getfromtif.py from the bubbles project.

These are just the basic algorithms; important speedups and features like 
dirtmasks have not been ported.
"""
import numpy as np
import pylab as pl

def getDirtMask(imageSize, options=None):
    """Returns an array with 1's for clean, 0's for dirt.
    
    For now, returns all clean.
    """
    return np.ones(imageSize, dtype=int)
def tracePinchoffRaw2(rawdata, dirtmask=None, invertImage=False, options={}, 
	transform=lambda x:x):
    """Trace pinchoff, but with raw array data.

    Takes a dict of [image] options. Unlike with e.g. tracePinchoff2, values
    are NOT filled in with those from the current directory.

    'transform' is a function that will be applied to all image-like arrays
    passed as parameters to, or loaded by, this function 
    (currently image data, dirtmasks and backgrounds).
    """
    # Decide the threshold
    threshold = float(options.get('threshold', 0.5))
    rawdataNoTransform = rawdata
    rawdata = transform(rawdataNoTransform)
    rows, cols = rawdata.shape
    if dirtmask == None:
	dirtmask = transform(getDirtMask(rawdataNoTransform.shape, options))
    else:
	dirtmask = transform(dirtmask)
    if options.get('outputmask', None):
	pl.save(options.get('outputmask', 'mask.tif'), dirtmask)
    if options.get('backgroundimage'):
	if '__backgroundimage_cached' in options:
	    backgroundImage = transform(options['__backgroundimage_cached'])
	else:
	    backgroundImage = transform(pl.imread(options['backgroundimage']))
	blackpoint = float(options.get('blackpoint', 0.0))
	rawdata = (rawdata - blackpoint) / (backgroundImage - blackpoint)
	threshold = 0.5
    else:
	backgroundImage = None
    # Convert to binary image
    if invertImage:
	binary = np.array(rawdata < threshold, int)
    else:
	binary = np.array(rawdata >= threshold, int)
    # Locate all unmasked dark pixels
    darkpix = (binary == 0) * dirtmask
    # Prepare the diffmask: masks elements of diff (see below) 
    # where a bad pixel is involved.
    diffmask = dirtmask[:,1:] & dirtmask[:,:-1]
    # Each element in the diff array is x_n+1 - x_n in that row
    # -1: left edge between n and n+1
    #  0: no change
    # +1: right edge
    diff = (binary[:,1:] - binary[:,:-1]) * diffmask
    ylist = []
    lefts = []
    rights = []
    # This function does the following:
    # Given values at adjacent pixels a and b, returns the distance
    # from a where the value crosses t.
    lininterp = lambda a, b, t=0: abs((a-t) / (a-b))
    for i in range(rows):
	r = diff[i]
	# Count from 1, because 0 has a special meaning (i.e., no edge)
	indices = r * np.arange(1, len(r) + 1)
	# Nonzero elements represent edges
	# Value m means there is an edge between m-1 and m in the original image data
	edges = np.compress(indices, indices)
	if not len(edges): continue
	# Edge list must begin with a left edge and end with 
	# a right edge; otherwise, ignore.
	# Believe it or not, this takes care of rows where 
	# the actual edge falls within a dirtbox.
	if not (edges[0] < 0 and edges[-1] > 0): continue
	if int(options.get('microbubbles', 0)):
	    # Look for left/right edge pairs that are too close to each other
	    microbubbles = ((edges[1:] + edges[:-1]) <= int(options['microbubbles'])) * (edges[1:] > 0)
	    if max(microbubbles):
		# Zap the offending pairs
		shl = np.concatenate((microbubbles, np.array((0,))))
		shr = np.concatenate((np.array((0,)), microbubbles))
		np.putmask(edges, shl, 0)
		np.putmask(edges, shr, 0)
		edges = np.compress(edges, edges)
	    if not edges: continue
	    if not (edges[0] < 0 and edges[-1] > 0): continue
	left = -edges[0]
	right = edges[-1]
	# Final check: there should be no unmasked dark pixels outside the proposed edges.
	# Skip this check if we are trying to exclude microbubbles.
	if not int(options.get('microbubbles', 0)):
	    allwater = darkpix[i]
	    allwater[left-1:right+1] = np.zeros((right - left + 2,), int)
	    if max(allwater) != 0: continue
	# Results are legit
	ylist.append(i)
	lefts.append(float(left) - 1. + lininterp(rawdata[i,left-1], rawdata[i,left], threshold))
	rights.append(float(right) - 1. + lininterp(rawdata[i,right-1], rawdata[i,right], threshold))
    return tuple([np.array(l, float) for l in (ylist, lefts, rights)])
def traceFast(rawdata, invertImage=False, threshold=0.5, mode='both'):
    """Finds edges. 'mode' may be "left", "right", or "both".

    Specify 'invertImage' if you are looking for a bright feature on a dark
    background.

    Returns a y, leftedges, rightedges tuple. (If 'mode' is not "both", ignore
    one of the edge lists).
    """
    rows, cols = rawdata.shape
    # Convert to binary image
    if invertImage:
	binary = np.array(rawdata < threshold, int)
    else:
	binary = np.array(rawdata >= threshold, int)
    darkpix = (binary == 0)
    # Each element in the diff array is x_n+1 - x_n in that row
    # -1: left edge between n and n+1
    #  0: no change
    # +1: right edge
    diff = (binary[:,1:] - binary[:,:-1])
    ylist = []
    lefts = []
    rights = []
    # This function does the following:
    # Given values at adjacent pixels a and b, returns the distance
    # from a where the value crosses t.
    lininterp = lambda a, b, t=0: abs((a-t) / (a-b))
    for i in range(rows):
	r = diff[i]
	# Count from 1, because 0 has a special meaning (i.e., no edge)
	indices = r * np.arange(1, len(r) + 1)
	# Nonzero elements represent edges
	# Value m means there is an edge between m-1 and m in the original image data
	edges = np.compress(indices, indices)
	if not len(edges): continue
	# Edge list must begin with a left edge and end with 
	# a right edge; otherwise, ignore.
	# Believe it or not, this takes care of rows where 
	# the actual edge falls within a dirtbox.
        left, right = None, None
        if edges[0] < 0: left = -edges[0]
        if edges[-1] > 0: right = edges[-1]
        if mode == 'both':
            if (left is None) or (right is None): continue
            # Final check: there should be no unmasked dark pixels outside the proposed edges.
            allwater = darkpix[i]
            allwater[left-1:right+1] = np.zeros((right - left + 2,), int)
            if max(allwater) != 0: continue
        elif mode == 'left':
            if left is None: continue
        elif mode == 'right':
            if right is None: continue
        else:
            raise ValueError
	# Results are legit
	ylist.append(i)
        if left is not None and mode != 'right':
            lefts.append(float(left) - 1. + \
                    lininterp(rawdata[i,left-1], rawdata[i,left], threshold))
        if right is not None and mode != 'left':
            rights.append(float(right) - 1. + \
                    lininterp(rawdata[i,right-1], rawdata[i,right], threshold))
    return tuple([np.array(l, float) for l in (ylist, lefts, rights)])
def traceEdge(a, ground=1, threshold=0.5):
    """Just trace an edge contour, starting from the left side of the image.
    Takes an array as its argument.
    """
    if ground: direction = 0
    else: direction = 1
    rows, cols = a.shape
    ylist = []
    edgelist = []
    for y in range(0, rows):
        row = a[y]
        # Reject lines that begin with a non-ground pixel.
        if (ground and (row[0] <= threshold)) or \
                (not ground and (row[0] >= threshold)):
            continue
        edge = firstTransition(row, direction=direction)
        if edge:
            ylist.append(y)
            edgelist.append(edge)
    return np.array(ylist, float), np.array(edgelist, float)
def scanRow(row, ground=1, adaptive=None):
    """Crude but effective: scan from left to right, identifying
    first fall. Then scan from right to left, identifying first
    fall. Returns a leftedge, rightedge tuple.

    ground=1: white background.
    adaptive=x: set threshold to be min + (max - min) * x
    """
    if ground: direction = 0
    else: direction = 1
    centergs = None
    if adaptive != None:
        rowmin = float(min(row))
        rowmax = float(max(row))
        centergs = rowmin + (rowmax - rowmin) * adaptive
    rowl = list(row)
    leftedge = firstTransition(rowl, direction=direction, centergs=centergs)
    rowl.reverse()
    rightedge = float(len(row)) - firstTransition(rowl, direction=direction, centergs=centergs) - 1.0
    return leftedge, rightedge

def firstTransition(row, direction=0, threshold=0.5, fallgs=None, risegs=None):
    """Find the first rise or fall in a sequence.
    direction = 1 for rise, 0 for fall. 
    
    Uses linear interpolation.
    """
    centergs = threshold
    if fallgs == None:
        fallgs = centergs
    if risegs == None:
        risegs = centergs
    # Of course, this won't work if the row does not begin 
    # below (above) the threshold.
    up = row[0] >= centergs
    if direction == 0:
        for x in range(1, len(row)):
            if row[x-1] >= fallgs and row [x] < fallgs and up:
                # fall
                xt = float(x-1) + abs((fallgs - row[x-1]) / (row[x-1] - row[x]))
                return xt
    else:
        for x in range(1, len(row)):
            if row[x-1] <= risegs and row[x] > risegs and not up:
                # rise
                xt = float(x-1) + abs((risegs - row[x-1]) / (row[x-1] - row[x]))
                return xt
    return 0

def firstGeneralizedTransition(xlist, ylist, direction=0):
    """Find the first rise or fall in a sequence.
    direction = 1 for rise, 0 for fall. 
    
    Threshold value is 0.
    Uses linear interpolation.
    """
    rise_t = 0.0
    fall_t = 0.0
    center_t = 0.0
    assert len(ylist) == len(xlist)
    # Of course, this won't work if the row does not begin 
    # below (above) the threshold.
    up = ylist[0] >= center_t
    if direction == 0:
        for i in range(1, len(ylist)):
            if ylist[i-1] >= fall_t and ylist[i] < fall_t and up:
                # fall
                xt = float(xlist[i-1]) + float(xlist[i] - xlist[i-1]) * \
                        abs((fall_t - ylist[i-1]) / (ylist[i-1] - ylist[i]))
                return xt
    else:
        for i in range(1, len(ylist)):
            if ylist[i-1] <= rise_t and ylist[i] > rise_t and not up:
                # rise
                xt = float(xlist[i-1]) + float(xlist[i] - xlist[i-1]) * \
                        abs((rise_t - ylist[i-1]) / (ylist[i-1] - ylist[i]))
                return xt
    return 0
def twoPoints(ixlist, iylist, splitpoint=0.0, level=0.0):
    """Take an xlist and ylist, which represent an upward-curving parabola y(x)
    centered near splitpoint. Find the points to the right and left of
    splitpoint where the curve crosses "level", using linear interpolation.
    """
    pairlist = zip(ixlist, iylist - level)
    botlist = [p for p in pairlist if p[0] >= splitpoint]
    toplist = [p for p in pairlist if p[0] <= splitpoint]
    if not botlist or not toplist: return None, None
    # First, the lower rows.
    ybotlist, hbotlist = apply(zip, botlist)
    botpoint = firstGeneralizedTransition(ybotlist, hbotlist, direction=1)
    # Now, the upper rows.
    toplist.reverse()
    ytoplist, htoplist = apply(zip, toplist)
    toppoint = firstGeneralizedTransition(ytoplist, htoplist, direction=1)
    return toppoint, botpoint

