"""Generic edge-finding tool, useful for calibrating ISR with clean interface.

Use this to extract the edges of the rheometer tool from an image, which if
you're lucky will be slightly dirty or bumpy. You can track those features to
obtain the tool position.

The state of the documentation and code here are a little weak.
"""
import numpy as np
import six

def traceEdges(rawdata, invertImage=False, threshold=0.5, mode='both'):
    """Traces outer edges of a dark central vertical body, one per row.
    
    'mode' may be "left", "right", or "both".

    Specify 'invertImage' if you are looking for a bright feature on a dark
    background.

    Returns a y, leftedges, rightedges tuple. (If 'mode' is not "both", only one
    edge list will be valid).

    The source code is a boiled-down version of a more featureful function, so it
    is both needlessly complex and very hackable.
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
            six.raise_from(ValueError)
        # Results are legit
        ylist.append(i)
        if left is not None and mode != 'right':
            lefts.append(float(left) - 1. + \
                    lininterp(rawdata[i,left-1], rawdata[i,left], threshold))
        if right is not None and mode != 'left':
            rights.append(float(right) - 1. + \
                    lininterp(rawdata[i,right-1], rawdata[i,right], threshold))
    return tuple([np.array(l, float) for l in (ylist, lefts, rights)])
