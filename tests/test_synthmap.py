#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the Q spectrum processing using a synthesised image as specified
in Section 5 of the "Fitting freeform shapes with orthogonal bases" document.
"""

from __future__ import print_function, absolute_import, division

import math
import time
import numpy as np
from numpy.polynomial.polynomial import polyval2d
from skqfit.qspectre import QSpectrum

def test_synthmap(as_map=False, inverse=True):
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
    sag_xy.curv = -1/478.12597
    sag_xy.coeff = np.zeros((11,11), dtype=np.float)
    sag_xy.coeff[0,:]  = [0, 0, 8.446692E-05,	-1.773111E-08,	2.103339E-10,	-4.450410E-14,	1.204820E-15,	-3.751270E-18,	1.243271E-20,	-1.671689E-23,	2.740074E-26]
    sag_xy.coeff[2,:9] = [6.086975E-05,	-9.657166E-08,	3.881972E-10,	-5.340721E-13,	1.962740E-15,	-3.972902E-18,	2.276418E-20,	-6.515923E-23,	1.259617E-25]
    sag_xy.coeff[4,:7] = [1.345443E-10,	-4.424293E-13,	1.672236E-15,	-4.286471E-18,	1.613314E-20,	-4.548523E-23,	9.938038E-26]
    sag_xy.coeff[6,:5] = [3.310262E-16,	-1.749391E-18,	7.515349E-21,	-2.305324E-23,	1.939290E-26]
    sag_xy.coeff[8,:3] = [1.020537E-21,	-6.739667E-24,	-3.800397E-27]
    sag_xy.coeff[10,0] = 1.653756E-28

    def build_map(pts, slice=False):
        x = np.linspace(-1.02*sag_xy.rmax, 1.02*sag_xy.rmax, pts)
        if slice:
            y = np.linspace(0.0, 0.0, 1)
        else:
            y = np.linspace(-1.02*sag_xy.rmax, 1.02*sag_xy.rmax, pts)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        z = sag_xy(xv, yv)
        return x, y, z.reshape((x.size, y.size))

    def test_xy_gradient(zmap, dfdx, dfdy, x, y):
        grad = np.gradient(zmap)
        gx = grad[0] / (x[1] - x[0])
        dy = grad[1] / (y[1] - y[0])

        err_dx = dfdx - gx
        gx_err = np.nanmax(err_dx[1:-1,1:-1]) - np.nanmin(err_dx[1:-1,1:-1])
        err_dy = dfdy - dy
        gy_err = np.nanmax(err_dy[1:-1,1:-1]) - np.nanmin(err_dy[1:-1,1:-1])

        #display_map(err_dx[1:-1,1:-1])
        #display_map(err_dy[1:-1,1:-1])
        return max(gx_err, gy_err)

    exp_ispec = np.array([[70531,   225291, 25895,  199399, 3583,   2651,   1886,   339,    55, 41, 5],
                          [43,	    223995,	11377,	198,	2604,	801,	46,	    37,	    5,	0,	0],
                          [82,	    12916,	3592,	994,	158,	10,	    5,	    0,	    0,	0,	0],
                          [10,	    1568,	256,	10,	    2,	    0,	    0,	    0,	    0,	0,	0],
                          [1,	    20,	    3,	    1,	    0,	    0,	    0,	    0,	    0,	0,	0]], dtype=np.int)


    bfs_c  = sag_xy.curv
    points = 501
    if False:
        m_max = 200
        n_max = 200
    else:
        m_max = 10
        n_max = 9
    qfit = QSpectrum(m_max, n_max)

    if as_map:
        x, y, zmap = build_map(points)
        qfit.data_map(x, y, zmap, centre=(0.,0.), radius=sag_xy.rmax)
        #display_map(zmap)
    else:
        qfit.set_sag_fn(sag_fn, sag_xy.rmax, bfs_c)

    start = time.time()
    a_nm, b_nm = qfit.q_fit(m_max, n_max)
    print('fit done, time %.3fs' % (time.time() - start))
    qspec = np.sqrt(np.square(a_nm) + np.square(b_nm))
    #disp_qspec(qspec)

    ispec = np.round(1e6*qspec).astype(int)
    idiff = ispec[:5,:11] - exp_ispec
    errors = np.count_nonzero(idiff)
    inv_err, grad_err = 0.0, 0.0

    if inverse:
        if not as_map:
            x, y, zmap = build_map(points)
        start = time.time()
        zinv, dfdx, dfdy = qfit.build_map(x, y, radius=sag_xy.rmax, centre=(0.0,0.0), a_nm=a_nm, b_nm=b_nm, interpolated=True, inc_deriv=True)
        print('inverse done, time %.3fs' % (time.time() - start))
        grad_err = test_xy_gradient(zinv, dfdx, dfdy, x, y)
        cond = zinv != 0.0
        diff = np.extract(cond, zmap - zinv)
        inv_err = max(math.fabs(np.nanmax(diff)), math.fabs(np.nanmin(diff)))

    assert errors == 0 and inv_err < 1.0e-7 and grad_err < 1.0e-5

if __name__ == "__main__":
    test_synthmap(as_map=False, inverse=True)
    test_synthmap(as_map=True, inverse=True)
