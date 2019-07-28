from __future__ import division

import numpy as np
import pandas
from philatracks import rheo

fpc = 200.
mpp = 1e-6
params = {'F0': 7.6654459070818698e-09,
    'L': 0.022,
    'R': 0.001144,
    'a': 0.0001143,
    'd': 5.8578048295125918e-06,
    'k': 1.8007027827533076e-05,
    'm': 9.513e-07}
def sineseries(ampl, phase_deg):
    fnums = np.arange(10000, dtype=float)
    return pandas.Series(ampl * np.sin((fnums / fpc) * 2*np.pi - \
            (phase_deg / 180. * np.pi)), index=fnums)
def toolpos(ampl, phase_deg):
    r = pandas.DataFrame({'resp': sineseries(ampl, phase_deg + 20),
        'current': sineseries(1, 20)})
    r['t'] = r.index.values.astype(float) / fpc / 0.2 # 0.2 Hz
    return r

def test_measure():
    delta, ampl, diag = rheo.measure_response(toolpos(0.1, 10), fpc)
    assert abs(delta - 10) < 0.01
    assert abs(ampl - 0.1) < 0.01

def test_fit():
    # Idea: We should be able to use the model to make a fake response curve
    # as a function of frequency, then use fit_response() to recover the model parameters.
    freqs = np.arange(0.1, 3, 0.1)
    partab = pandas.DataFrame({'freq': freqs, 'amp': 0.1, 'delta': np.nan, 'ampl_m': np.nan})
    for i in partab.index:
        p = params.copy()
        p['freq'] = partab.freq[i]
        partab.delta[i] = rheo.delta(p) / np.pi * 180.
        partab.ampl_m = rheo.AR(p) * params['F0'] * 0.1
    fitp = rheo.fit_response(partab, dict(m=params['m']))
    assert np.isclose(params['d'], fitp['d'])
    assert np.isclose(params['k'], fitp['k'])
    assert np.isclose(params['F0'], fitp['F0'])

def test_dynamic():
    dr = rheo.dynamic_response(params, toolpos(0.1, 10))
    # FIXME: This value changed after the bugfix.
    # These tests need to be redone carefully with contrived values.
    #assert abs(-dr.stress.min() - 4.15e-5) < 2e-7

def test_rheology():
    delta, ampl_px, diag = rheo.measure_response(toolpos(10, 10), fpc)
    rh = rheo.measure_rheology(params, delta, ampl_px, mpp, fpc, 0.2, 1)
    assert abs(rh['delta'] - 10) < 0.01
    assert np.isclose(rh['ampl_m'], 10 * mpp)
    assert np.isclose(rh['Gp'], 1.728187076218136e-05)
    assert np.isclose(rh['Gpp'], 2.9430990972078236e-06)
    assert np.isclose(rh['Bo'], 4818.3962176983159)

