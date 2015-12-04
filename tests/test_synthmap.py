#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the Q spectrum processing using a synthesised image as specified
in Section 5 of the "Fitting freeform shapes with orthogonal bases" document.
"""

from __future__ import print_function, absolute_import, division

import os
import numpy as np
from numpy.polynomial.polynomial import polyval2d

from skqfit.qspectre import QSpectrum

def test_synthmap(as_map=False):
    def sag_fn(rhov, thetav):
        x = rhov*np.cos(thetav)
        y = rhov*np.sin(thetav)
        return sag_xy(x, y)

    def sag_xy(x, y):
        cx = -1/452.62713
        cy = -1/443.43539
        kx = ky = 0.0
        x2 = x*x
        y2 = y*y
        z = (cx*x2 + cy*y2)/(1 + np.sqrt(1-((1+kx)*cx*cx)*x2 - ((1+ky)*cy*cy)*y2))
        return z + polyval2d(x, y, sag_xy.coeff)

    sag_xy.rmax = 174.2
    sag_xy.coeff = np.zeros((11,11), dtype=np.float)
    sag_xy.coeff[0,:]  = [0, 0, 8.446692E-05,	-1.773111E-08,	2.103339E-10,	-4.450410E-14,	1.204820E-15,	-3.751270E-18,	1.243271E-20,	-1.671689E-23,	2.740074E-26]
    sag_xy.coeff[2,:9] = [6.086975E-05,	-9.657166E-08,	3.881972E-10,	-5.340721E-13,	1.962740E-15,	-3.972902E-18,	2.276418E-20,	-6.515923E-23,	1.259617E-25]
    sag_xy.coeff[4,:7] = [1.345443E-10,	-4.424293E-13,	1.672236E-15,	-4.286471E-18,	1.613314E-20,	-4.548523E-23,	9.938038E-26]
    sag_xy.coeff[6,:5] = [3.310262E-16,	-1.749391E-18,	7.515349E-21,	-2.305324E-23,	1.939290E-26]
    sag_xy.coeff[8,:3] = [1.020537E-21,	-6.739667E-24,	-3.800397E-27]
    sag_xy.coeff[10,0] = 1.653756E-28

    def build_map():
        x = np.linspace(-1.02*sag_xy.rmax, 1.02*sag_xy.rmax, 200)
        y = np.linspace(-1.02*sag_xy.rmax, 1.02*sag_xy.rmax, 200)

        xv = np.repeat(x, y.size)
        yv = np.repeat(y.reshape((1,y.size)), x.size, axis=0).flatten()

        z = sag_xy(xv, yv)

        return x, y, z.reshape((x.size, y.size))


    exp_ispec = np.array([[70531,   225291, 25895,  199399, 3583,   2651,   1886,   339,    55, 41, 5],
                          [43,	    223995,	11377,	198,	2604,	801,	46,	    37,	    5,	0,	0],
                          [82,	    12916,	3592,	994,	158,	10,	    5,	    0,	    0,	0,	0],
                          [10,	    1568,	256,	10,	    2,	    0,	    0,	    0,	    0,	0,	0],
                          [1,	    20,	    3,	    1,	    0,	    0,	    0,	    0,	    0,	0,	0]], dtype=np.int)


    bfs_c  = None
    m_max = 10
    n_max = 9
    qfit = QSpectrum(m_max, n_max)

    if as_map:
        x, y, zmap = build_map()
        qfit.data_map(x, y, zmap, centre=(0.,0.), radius=sag_xy.rmax)
    else:
        qfit.set_sag_fn(sag_fn, sag_xy.rmax, bfs_c)
    qspec = qfit.build_q_spectrum()

    ispec = np.round(1e6*qspec).astype(int)
    idiff = ispec[:5,:] - exp_ispec
    errors = np.count_nonzero(idiff)
    assert errors == 0

if __name__ == "__main__":
    test_synthmap(False)
    test_synthmap(True)
