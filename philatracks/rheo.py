from __future__ import division
"""Interpret data from a linear oscillatory rheometer, particularly an interfacial stress rheometer.

The ``params`` dictionaries that are used in this module can have the following entries
(all in SI units):

'm'
    Tool (i.e. needle) mass
'a'
    Needle diameter
'R'
    Gap between needle and wall
'L'
    Channel length
'k'
    Magnetic restoring force (N/m)
'd'
    Clean-interface drag coefficient (N s/m)
'F0'
    newtons per unit of driving strength (e.g. amperes)
'freq'
    Driving frequency (for modeling response) 
'visc'
    Mean of superphase and subphase viscosity (for computing Re only)
'rho'
    Mean of superphase and subphase density (for computing Re only)

Not all parameters are required for every operation.

References

1.    Brooks, C. F., Fuller, G. G., Frank, C. W. & Robertson, C. R. An Interfacial Stress Rheometer To Study Rheological Transitions in Monolayers at the Air-Water Interface. Langmuir 15, 2450-2459 (1999). 
2.    Reynaert, S., Brooks, C. F., Moldenaers, P., Vermant, J. & Fuller, G. G. Analysis of the magnetic rod interfacial stress rheometer. J. Rheol. 52, 261-285 (2008).
"""
#   Copyright 2013 Nathan C. Keim
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from warnings import warn
import numpy as np
from scipy.optimize import curve_fit
from . import signalsmooth
import pandas
from . import cyclic
import six

# NOTE: All units are SI unless otherwise specified.

def fit_response(partab, toolparams):
    """Model clean-interface response from data.

    :param partab: DataFrame describing response of the needle on a clean interface
        for some range of parameters, with columns

        freq
            driving frequency
        amp
            amplitude of driving, in e.g. amperes
        delta
            measured phase lag angle (degrees)
        ampl_m
            measured amplitude of motion, in meters

    :param toolparams: dict providing data about the rheometer itself.
        Only ``toolparams["m"]`` is used here.

    Returns a copy of ``toolparams``, with new elements corresponding to
    coefficients in the ISR equation of motion:

    d
        Drag coefficient, N/(m/s)
    k
        Magnetic "spring constant", N/m
    F0
        Conversion from the driving amplitude parameter (in, e.g., amperes) to newtons.
    
    The method bootstraps from a specified mass *m* (in ``toolparams``),
    using phase angle to compute *k* and *d*, and then using response amplitude to
    find the normalization for units of force, *F0*.
    """
    m = toolparams['m']
    # Phase difference
    fit_delta = lambda x, d, k: np.arccos((k - m * x*x) / np.sqrt(x*x*d*d + (k - m*x*x)**2))
    popt_delta, pcov = curve_fit(fit_delta,
                           partab.freq.values * 2 * np.pi,
                           partab.delta.values / 180 * np.pi,
                           )
    # Response amplitude
    # Since m is already a known dimensional quantity, we create
    # a calibration factor F0.
    # F0 encapsulates the magnetic force (N/A)
    # The sign of d is undetermined by our fitting process, so we define it as positive.
    d = abs(popt_delta[0])
    k = popt_delta[1]
    fit_ampl = lambda x, F0: F0 / np.sqrt((k - m*x*x)**2 + (x*x*d*d))
    popt_ampl, pcov = curve_fit(fit_ampl, partab.freq.values * 2 * np.pi,
                                partab.ampl_m.values / partab.amp.values)
    #print popt_ampl
    F0 = popt_ampl[0]
    six.print_('Inertial mass m:', m, 'kg')
    six.print_('Fit spring constant k (from phase angle):', k, 'N/m')
    six.print_('Fit drag coefficient d (from phase angle):', d, 'N s/m')
    six.print_('Fit force coefficient F0', F0, 'N/A')
    params = toolparams.copy()
    params.update(dict(k=k, d=d, F0=F0))
    return params


def measure_response(toolpos, fpc, t_trans=0, flipsign=False):
    """Compares FFTs of driving and response.

    'toolpos' is a pandas DataFrame, indexed by frame number, with columns
        't' - time in seconds (only meaningful if specifying t_trans > 0)
        'resp' - Displacement of needle (units arbitrary)
        'current' - Driving current (used for phase only)

    'fpc' is the number of movie frames per cycle of driving.
    
    Returns phase angle difference (deg), response amplitude, and diagnostic dict.

    Discards first 't_trans' seconds of the movie, rounded up to an integer
    number of cycles.

    Specify 'flipsign' if displacement and forcing signals have opposite polarity.

    Note that reported amplitudes have been de-normalized and so approximate the actual
    amplitude of the original signal (not as reported by FFT).
    """
    # For consistency, make sure sample contains an integer number of cycles.
    # We will apply a Hanning window as well.
    # Find the number of frames in the transient
    try:
        frames_trans = (toolpos.t >= t_trans).nonzero()[0][0]
    except IndexError:
        raise ValueError('Tool trajectory is shorter than transient duration.')
    cycles_to_discard = int(np.ceil(frames_trans / fpc))
    frames_to_discard = int(cycles_to_discard * fpc)
    n = (len(toolpos) - frames_to_discard)
    # Need an even number of samples.
    n = int(2 * np.floor(n / 2.))
    resptab = toolpos.iloc[-n:]
    assert len(resptab) == n

    cycles_after_transient = n / fpc
    # FFT to compute phase difference
    rft = np.fft.rfft((resptab.resp.values - resptab.resp.mean()) * np.hanning(n))
    dft = np.fft.rfft(resptab.current.values * np.hanning(n))
    freqs = np.fft.fftfreq(n)[:n/2 + 1]
    # Pick out peaks only
    absrft = np.abs(rft)
    absdft = np.abs(dft)
    drive_maxidx = np.nonzero(absdft == max(absdft))[0][0]
    drive_peakfreq = freqs[drive_maxidx]
    drive_phase = np.angle(dft[drive_maxidx]) / np.pi * 180
    drive_ampl = absdft[drive_maxidx]
    # We'll find the peak in the response, but use the peak frequency from driving 
    # to characterize it.
    resp_maxidx = np.nonzero(absrft == max(absrft))[0][0]
    resp_peakfreq = freqs[drive_maxidx]
    resp_phase = np.angle(rft[drive_maxidx]) / np.pi * 180
    resp_ampl = absrft[drive_maxidx]
    if (resp_maxidx - drive_maxidx) / (resp_maxidx + drive_maxidx) > 0.002:
        warn('Drive and response are not detected at the same \nfrequency: %f vs. %f. Using drive frequency.'
                % (drive_peakfreq, resp_peakfreq))
    diag = {'cycles_discarded': cycles_to_discard, 'cycles_after_transient': cycles_after_transient,
            'signals': resptab, 'resp_fft': rft, 'drive_fft': dft, 'freqs': freqs,
            'drive_phase': drive_phase, 'resp_phase': resp_phase, 
            'drive_ampl': drive_ampl / n * 4,
            'n': n, 'drive_peakfreq': drive_peakfreq, 'resp_peakfreq': resp_peakfreq}
    phaseangle = (drive_phase + (180 if flipsign else 0) - resp_phase) % 360
    return phaseangle, resp_ampl / n * 4, diag


def dynamic_response(params, toolpos, mpp=1.0, smoothwindow=7, flipsign=False):
    """Instantaneous measurements in rheometer.

    'rheoparams' is a dict of rheometer properties (see module-level documentation).
    'toolpos' is as for measure_response(), with 'toolpos.resp' in units of pixels.
    'mpp' is the magnification, in meters per pixel
    'smoothwindow' - window in which to smooth strain timeseries
    'flipsign' - true if displacement and forcing signals have opposite polarity.

    Returns a DataFrame with quantities such as strain, stress, strainrate for entire movie.
    """
    magnetflip = -1 if flipsign else 1
    data = toolpos.copy()
    # Global shear strain and tool position
    # Raw response data is either tool position, or strain from particle tracking
    data['resp_m'] = data.resp * mpp # Tool displacement (m)
    data['resp_smooth'] = pandas.Series(signalsmooth.smooth(data.resp_m.values, int(smoothwindow)),
            index=data.index)
    data['strain'] = data.resp_smooth / params['R']
    data['F'] = data.current * params['F0'] * magnetflip # Force from secondary coil (N)
    dt = data.dropna().t.diff()
    # Next we explicitly implement the equation of motion (as opposed to implicitly,
    # as for the FFT-based oscillatory rheometry)
    # This is m \ddot x = A I_\text{drive} - kx - d \dot x - F_\text{interface}
    # where $m$ is the needle mass, $A I_\text{drive}$ is the force from the 
    # computer-controlled driving current, $k$ is the spring constant for 
    # central potential of the Helmholtz field, $d$ represents drag from the 
    # bulk fluid, and $F_\text{interface}$ is due to any material adsorbed at 
    # the surface.
    #
    # Force corrected for position of tool (N)
    Fcorrected = data.F - data.resp_smooth * params['k']
    # Stress calculation
    # Assume we have lots of (smoothed) samples so that we can use 
    # this simple method of discrete derivatives.
    data['toolvel'] = data.resp_smooth.diff() / dt
    data['strainrate'] = (data.toolvel / params['R'])
    data['toolaccel'] = -data.toolvel.diff(-1) / dt
    # With this sign convention, positive stress <-> positive strain
    Finter = -data.toolaccel * params['m'] + Fcorrected - (data.toolvel 
                                                           * abs(params['d']))
    data['Finter'] = Finter
    # Double the needle length, because there is material on both sides.
    data['stress'] = Finter / (2*params['L']) # N/m
    return data.dropna()


def measure_rheology(params, delta, ampl_px, mpp, fpc, freq, drivecurrent):
    """Measures rheology of a sample.
    
    'params' describes the rheometer and was obtained from fit_response()
    'delta' and 'ampl_px' are obtained from measure_response()
    'mpp' is the magnification, in meters per pixel
    'freq' is the driving frequency (Hz)
    'drivecurrent' is the driving amplitude, in "native" units (e.g. amperes)

    Returns a dict with measured quantities including G' and G''.
    """
    params = params.copy()
    params['freq'] = freq
    p = params
    ampl_m = ampl_px * mpp
    ampl_strain = ampl_m / p['R']
    ampl_stress = abs(drivecurrent) * p['F0'] / p['L']
    # We must use abs(drivecurrent) because computeResponse always
    # reports a positive amplitude.
    # This is OK --- the sign information is recovered when we multiply by
    # e^(i*delta).
    # The "2" in the next line is because stress is exerted on both
    # sides of the needle; it is found in the original equation in Reynaert et al.
    G_apparent = (p['R'] - p['a']) / (2*p['L']) * p['F0'] * abs(drivecurrent) / \
        (ampl_m * np.exp(0-1j * delta / 180. * np.pi))
    # Compute system modulus at this frequency
    G_system = Gsys(params) 
    G = G_apparent - G_system
    visc = G.imag / (params['freq'] * 2 * np.pi)
    bulkvisc = params.get('visc', 1e-3)
    Bo = (abs(G) / params['freq'] * 2 * np.pi) / (params['a'] * bulkvisc)
    r = dict(delta=delta, ampl_px=ampl_px, ampl_m=ampl_m, ampl_strain=ampl_strain,
            ampl_stress=ampl_stress,
            G_apparent=repr(G_apparent), G_system=repr(G_system), 
            G=repr(G), Gp=G.real, Gpp=G.imag, visc=visc,
            Bo=Bo)
    return r


# Formulae for derived quantities
def AR(params):
    """Ratio of measured response to forcing. Equal to z / F"""
    p = params
    w = p['freq'] * 2 * np.pi
    return 1. / np.sqrt((p['k'] - p['m'] * w**2)**2 + (w**2 * p['d']**2))


def delta(params):
    """In radians"""
    p = params
    w = p['freq'] * 2 * np.pi
    return np.arccos((p['k'] - p['m'] * w**2) / \
                     np.sqrt(w**2 * p['d']**2 + (p['k'] - p['m'] * w**2)**2))


def Gsys(params):
    p = params
    return (p['R'] - p['a']) / (2*p['L']) \
            / AR(p) / np.exp(0-1j * delta(p))


def Re(params, ampl_m):
    """Reynolds number calculated in the simplest way possible.
    To be more physical, we could use the fitted drag coefficient."""
    visc = params.get('visc', 1e-3)
    rho = params.get('rho', 1e3)
    p = params
    return p['R'] * p['freq'] * 2*np.pi * ampl_m * rho / visc
