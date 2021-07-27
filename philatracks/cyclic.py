"""Sample a movie by cycle, instead of by frame."""
import numpy as np
import pandas
from . import signalsmooth

def timing(framenumbers, fpc):
    """Returns DataFrame, indexed by frame, containing useful clock variables
    and their differentials.

    Takes 'framenumbers', a sequence of frame numbers.
    'fpc' is the number of frames per cycle of driving.

    Returned DataFrame has column 'cycle', which counts up in units of cycles,
    and 'lockin', which is the fractional part of 'cycle'.
    """
    fnums = np.array(framenumbers)
    fr = pandas.DataFrame({'cycle': (fnums - 1) / float(fpc),
        'lockin': ((fnums - 1) % fpc) / float(fpc)},
        index=fnums)
    return fr
def cycframes(fpc, cycnum):
    """Frames in a given cycle, for purposes of e.g. T1 statistics. For purposes
    of stroboscopic comparison, this includes the first frame of the following cycle.
    """
    return np.arange(1 + cycnum * fpc, 2 + (cycnum + 1)*fpc)
def assign_cycle(fpc, frames):
    """From one or more frame numbers, return the corresponding cycle numbers.
    
    Note that this is NOT an inverse of cycframes(); that function returns an extra
    frame at the end.
    """
    return np.int_(np.floor((frames - 1) / float(fpc)))
def find_cycle_extrema(series, samples_per_cycle):
    """Find minima and maxima for each cycle in an oscillatory signal.
    'series' is a pandas Series.
    
    Does *not* directly find successive minima and maxima; instead, 
    looks for 1 of each per cycle.
    
    Returns a boolean Series ("ismax"), indexed by frame number.
    """
    cycles = range(int(np.floor((max(series.index) - 1) / samples_per_cycle)))
    cyc_extrema = pandas.DataFrame({'cycmin': np.nan, 'cycmax': np.nan}, index=cycles)
    smoothed = pandas.Series(signalsmooth.smooth(series.values, samples_per_cycle // 10), 
            index=series.index)
    for cycnum in cyc_extrema.index:
        cycseries = smoothed[(smoothed.index > cycnum * samples_per_cycle) & \
                (smoothed.index <= (cycnum + 1) * samples_per_cycle)]
        cyc_extrema.cycmin[cycnum] = cycseries.index.values[cycseries.values.argmin()]
        cyc_extrema.cycmax[cycnum] = cycseries.index.values[cycseries.values.argmax()]
    # Reformat the DataFrame, indexed by cycle, into a Series indexed by frame number.
    # The values are booleans which indicate whether the entry is a max or min.
    extrema = pandas.Series(False, index=cyc_extrema.cycmin).append( \
        pandas.Series(True, index=cyc_extrema.cycmax)).sort_index()
    extrema.name = 'ismax'
    extrema.index.name = 'frame'
    return extrema
def find_sample_positions(strainseries, samples_per_cycle, maxframe=np.inf,
        method='minfirst'):
    """Identify frames for sampling maximal and stroboscopic deformation in each cycle.

    Default is to return pairs of minima and immediately following maxima. Specify
    'method' as "maxfirst" to switch.

    'strainseries' may be obtained as rheo.dynamicResponse().strain
    
    If sampling a cycle requires accessing a frame numbered < 1 or > maxframe,
    it will be omitted.
    """
    extrema = find_cycle_extrema(strainseries, samples_per_cycle)
    exfr = extrema.reset_index()
    if method == 'minfirst':
        after_frames = exfr[(exfr.ismax) & (exfr.index > 0)].frame
    elif method == 'maxfirst':
        after_frames = exfr[(~exfr.ismax) & (exfr.index > 0)].frame
    else: raise ValueError()
    before_frames = exfr.frame[list(after_frames.index.values - 1)]
    extrema = pandas.DataFrame({'before': before_frames.values, 
                    'after': after_frames.values},
                            index=assign_cycle(samples_per_cycle, after_frames))
    extrema.index.name = 'cycle'
    midpoints = ((extrema.after + extrema.before) / 2).astype(int)
    # Throw out cycles that use nonexistent frames
    extrema['midbefore'] = midpoints - int(samples_per_cycle / 2)
    extrema['midafter'] = midpoints + int(samples_per_cycle / 2)
    return extrema[(extrema.midbefore > 0) & (extrema.midafter <= maxframe)]
