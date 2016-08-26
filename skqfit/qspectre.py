"""
Reference documents

[1] G W Forbes, "Fitting freeform shapes with orthogonal bases", Opt. Express 21, 19061-19081 (2013)
[2] G W Forbes, "Characterizing the shape of freeform optics", Opt. Express 20(3), 2483-2499 (2012)
[3] G W Forbes, "Robust, efficient computational methods for axially symmetric optical aspheres", Opt. Express 18(19), 19700-19712 (2010)
"""

from __future__ import print_function, absolute_import, division

import math
import numpy as np

from scipy import interpolate

from scipy.misc import factorial, factorial2
from scipy import ndimage

from skqfit.asmjacp import AsymJacobiP

# trap on warnings when debugging
#np.seterr(over='raise')

class QSpectrum(object):
    """
    Performs precomputation if Q spectrum limits are passed, otherwise it is
    delayed until the data is loaded.
    The class supports processing a data map or a pointer to a sag function that
    can be used for analytic testing.

    Parameters:
    m_max, n_max:  int
        The azimuthal and radial spectrum order. Setting values above 1500 may lead to overflow
        events.

    """

    def __init__(self, m_max=None, n_max=None):
        self.interpolate = None
        self.m_disp = m_max
        self.n_disp = n_max
        self.m_max = None
        self.n_max = None
        if m_max is not None and n_max is not None:
            self._precompute_factors(m_max, n_max)
        self.shrink_pixels = 7
        self.centre_sag = 0.0
        self.centre = (0.0, 0.0)
        self.polar_sag_fn = None

    def _precompute_factors(self, m_max, n_max):

        self.m_disp = m_max
        self.n_disp = n_max
        if (m_max == self.m_max) and (n_max == self.n_max):
            return

        self.m_max = max(m_max, 3)
        self.n_max = max(n_max, 3)
        self.k_max = self.n_max + 2
        self.j_max = self.m_max + 1
        self.jacobi_p = AsymJacobiP(self.n_max)

        # The unit vector corresponds to the radial sample space
        self.phi_kvec = np.array([(2.0 * k - 1) * math.pi / (4.0 * self.k_max) for k in range(1, self.k_max + 1)])
        self.u_vec = np.sin(self.phi_kvec)
        self.u_vec_sqr = np.square(self.u_vec)

        # pre compute the tables from [1] A.5, A.6, A.7a, A.7b, A.11 and A.12
        self.bgK, self.bgH, self.smK, self.smH, self.smS, self.smT = self._compute_qfit_tables(self.m_max, self.n_max)

        # pre compute tables from [2] A.13, A.15, A.18a, A.18b
        self.bgF, self.bgG, self.smF, self.smG = self._compute_freeform_tables(self.m_max, self.n_max)

        # pre compute tables from [3] A.14, A.15, A.16
        self.smFn, self.smGn, self.smHn = self._compute_qbfs_tables(self.n_max)

        # pre compute tables from [2] A.3a,b,c
        self.bgA, self.bgB, self.bgC = self._compute_qinv_tables(self.m_max, self.n_max)

    def _compute_qbfs_tables(self, max_n):
        """
        pre compute tables from [3] A.14, A.15, A.16
        """
        smFn = np.zeros(max_n + 3, dtype=np.float)
        smGn = np.zeros(max_n + 2, dtype=np.float)
        smHn = np.zeros(max_n + 1, dtype=np.float)

        smFn[0] = 2.0
        smFn[1] = math.sqrt(19.0) / 2
        smGn[0] = -0.5
        for n in range(2, max_n + 3):
            smHn[n - 2] = -n * (n - 1) / (2 * smFn[n - 2])
            smGn[n - 1] = -(1 + smGn[n - 2] * smHn[n - 2]) / smFn[n - 1]
            smFn[n] = math.sqrt(n * (n + 1) + 3 - smGn[n - 1] ** 2 - smHn[n - 2] ** 2)

        return smFn, smGn, smHn

    def _compute_qfit_tables(self, m_max, n_max):
        """
        Build the big H and K tables as described in ([1] A.6, A.5)
        """
        bgK = np.zeros((m_max + 1, n_max + 1), dtype=np.float)
        bgH = np.zeros((m_max + 1, n_max + 1), dtype=np.float)

        bgK[0, 0] = 3.0 / 8.0
        bgK[0, 1] = 1.0 / 24.0
        bgH[0, 0] = 1.0 / 4.0
        bgH[0, 1] = 19.0 / 32.0

        mv = np.arange(1, m_max + 1, dtype=np.float)
        nv = np.arange(2, n_max + 1, dtype=np.float)
        nv2 = nv * nv

        # build the first row
        bgK[0, 2:] = (nv2 - 1) / (32 * nv2 - 8)
        bgH[0, 2:] = (1. + 1 / (1 - 2 * nv) ** 2) / 16

        # recursively build factorial terms and complete the first two columns
        nfv = np.arange(m_max + 1, dtype=np.float)
        nfact = 0.5
        for m in range(1, m_max + 1):
            num = float(2 * m + 1)
            den = float(2 * m + 2)
            nfact = num / den * nfact
            nfv[m] = nfact

        bgK[1:, 0] = 0.5 * nfv[1:]
        bgK[1:, 1] = ((2.0 * mv * (2 * mv + 3)) / (3.0 * (mv + 3.) * (mv + 2))) * 0.5 * nfv[1:]
        bgH[1:, 0] = ((mv + 1.) / (2 * mv + 1)) * 0.5 * nfv[1:]
        bgH[1:, 1] = ((3 * mv + 2.) / (mv + 2)) * 0.5 * nfv[1:]

        v = bgK[1:, 1]
        w = bgH[1:, 1]
        for n in range(2, n_max + 1):
            bgH[1:, n] = (((mv + (2 * n - 3)) * ((mv + (n - 2)) * (4 * n - 1) + 5 * n)) / (
                (mv + (n - 2)) * (2 * n - 1) * (mv + 2 * n))) * v
            v = (((n + 1) * (mv + (2 * n - 2)) * (mv + (2 * n - 3)) * (2 * mv + (2 * n + 1))) / (
                (2 * n + 1) * (mv + (n - 2)) * (mv + (2 * n + 1)) * (mv + 2 * n))) * v
            bgK[1:, n] = v

        # Build the small H and K tables (A.7a, A.7b)
        smK = np.zeros((m_max + 1, n_max + 1), dtype=np.float)
        smH = np.zeros((m_max + 1, n_max + 1), dtype=np.float)

        smH[:, 0] = np.sqrt(bgH[:, 0])
        for n in range(1, n_max + 1):
            smK[:, n - 1] = bgK[:, n - 1] / smH[:, n - 1]
            smH[:, n] = np.sqrt(bgH[:, n] - smK[:, n - 1] ** 2)

        # Build the small S and T tables (A.11, A.12)
        smS = np.zeros((m_max + 1, n_max + 1), dtype=np.float)
        smT = np.zeros((m_max + 1, n_max + 1), dtype=np.float)
        nv = np.arange(1, n_max + 1, dtype=np.float)
        n2v = 2.0 * nv
        for m in range(1, m_max + 1):
            smS[m, 0] = 1
            smT[m, 0] = 1.0 / m
            smS[m, 1:] = (nv + (m - 2)) / (n2v + (m - 2))
            smT[m, 1:] = ((1 - n2v) * (nv + 1)) / ((m + n2v) * (n2v + 1))
        smS[1, 1] = 0.5
        smT[1, 0] = 0.5

        return bgK, bgH, smK, smH, smS, smT

    def _compute_qinv_tables(self, m_max, n_max):
        """
        Build the big A, B and C tables as described in [2] (A.3a,b,c,d)
        """
        A = np.zeros((m_max + 1, n_max + 1), dtype=np.float)
        B = np.zeros((m_max + 1, n_max + 1), dtype=np.float)
        C = np.zeros((m_max + 1, n_max + 1), dtype=np.float)
        mv = np.arange(1, m_max + 1, dtype=np.float)

        for n in range(2, n_max + 1):
            dv = (4.0 * n * n - 1) * (mv + n - 2) * (mv + 2 * n - 3)
            A[1:, n] = (2.0 * n - 1) * (mv + 2 * n - 2) * (4 * n * (mv + n - 2) + (mv - 3) * (2 * mv - 1)) / dv
            B[1:, n] = -2.0 * (2 * n - 1) * (mv + 2 * n - 1) * (mv + 2 * n - 2) * (mv + 2 * n - 3) / dv
            C[1:, n] = n * (2.0 * n - 3) * (mv + 2 * n - 1) * (2 * mv + 2 * n - 3) / dv

        # Initialze the special cases using [2] B.7 and B.8
        for m in range(2, m_max + 1):
            A[m, 0] = 2 * m - 1
            B[m, 0] = 2 * (1 - m)
            d = 3.0 * (m - 1) ** 2
            A[m, 1] = m * (4.0 * (m - 1) + (m - 3) * (2 * m - 1)) / d
            B[m, 1] = -2 * (m - 1) * m * (m + 1) / d
            C[m, 1] = -(m + 1) * (2 * m - 1) / d
        A[1, 0] = 2
        B[1, 0] = -1
        A[1, 1] = -4.0 / 3
        B[1, 1] = -8.0 / 3
        C[1, 1] = -11.0 / 3
        C[1, 2] = 0

        return A, B, C

    def _compute_freeform_tables(self, max_m, max_n):
        """
        Pre compute tables from [2] A.13, A.15, A.18a, A.18b
        """

        def gamma_factorial(m, n):
            return factorial(n) * factorial2(2 * m + 2 * n - 3) / (
            2.0 ** (m + 1) * factorial(m + n - 3) * factorial2(2 * n - 1))

        def kron_delta(i, j):
            return 1 if i == j else 0

        bgF = np.zeros((max_m + 1, max_n + 1), dtype=np.float)
        bgG = np.zeros((max_m + 1, max_n + 1), dtype=np.float)

        mv = np.arange(max_m + 1, dtype=np.float)
        mv2 = np.arange(2, max_m + 1, dtype=np.float)
        mv2_sqrd = mv2 * mv2
        fvF = np.ones(max_m + 1, dtype=np.float)
        fvG = np.ones(max_m + 1, dtype=np.float)
        gv_m = np.ones(max_m + 1, dtype=np.float)

        for m in range(1, max_m + 1):
            if m == 1:
                facF = 0.25
                facG = 0.25
                g_m = 0.25
            else:
                facF = 0.5 * ((2 * m - 3.) / (m - 1)) * facF  # (2m-3)!!/(m-1)!2^(m+1)
                facG = 0.5 * ((2 * m - 1.) / (m - 1)) * facG  # (2m-1)!!/(m-1)!2^(m+1)
                g_m = g_m * (2 * m - 3.) / (2.0 * (m - 3)) if m > 3 else 3.0 / (2 ** 4)
                fvF[m] = facF
                fvG[m] = facG
                gv_m[m] = g_m

        gamma = np.zeros(max_m + 1, dtype=np.float)
        for n in range(0, max_n + 1):
            if n == 0:
                gamma[3] = gamma_factorial(3, 0)
                gamma[4:] = gv_m[4:]
            else:
                i = max(0, 4 - n)
                if i > 0:
                    gamma[i - 1] = gamma_factorial(i - 1, n)
                gamma[i:] = (n * (2 * mv[i:] + (2 * n - 3)) / ((mv[i:] + (n - 3)) * (2 * n - 1))) * gamma[i:]

            if n == 0:
                bgF[1, n] = 0.25
                bgG[1, n] = 0.25
                bgF[2:, n] = mv2_sqrd * fvF[2:]
                bgG[2:, n] = fvG[2:]
            else:
                bgF[1, n] = (4 * ((n - 1) * n) ** 2 + 1.) / (8 * (2 * n - 1) ** 2) + kron_delta(n, 1) * 11.0 / 32
                bgG[1, n] = -(((2 * n * n - 1.) * (n * n - 1)) / (8 * (4 * n * n - 1))) - kron_delta(n, 1) / 24.0
                bgF[2:, n] = ((2 * n * (mv2 + (n - 2.)) * (3 - 5 * mv2 + 4 * n * (mv2 + (n - 2))) + mv2_sqrd * (
                3 - mv2 + 4 * n * (mv2 + (n - 2)))) / (
                              (2 * n - 1) * (mv2 + (2 * n - 3)) * (mv2 + (2 * n - 2)) * (mv2 + (2 * n - 1)))) * gamma[
                                                                                                                2:]
                bgG[2:, n] = -(((2 * n * (mv2 + (n - 1.)) - mv2) * (n + 1) * (2 * mv2 + (2 * n - 1))) / (
                (mv2 + (2 * n - 2)) * (mv2 + (2 * n - 1)) * (mv2 + 2 * n) * (2 * n + 1))) * gamma[2:]

        smF = np.zeros((max_m + 1, max_n + 1), dtype=np.float)
        smG = np.zeros((max_m + 1, max_n + 1), dtype=np.float)
        smF[:, 0] = np.sqrt(bgF[:, 0])
        for n in range(1, max_n + 1):
            smG[1:, n - 1] = bgG[1:, n - 1] / smF[1:, n - 1]
            smF[1:, n] = np.sqrt(bgF[1:, n] - smG[1:, n - 1] ** 2)

        return bgF, bgG, smF, smG

    def _dct_iv(self, data):
        """
        Implements a Discrete Cosine Transform DCT-IV which is not supported in scipy.
        The code is sufficiently fast for what is needed in the Q-fitting routine.
        """
        N = len(data)
        nv = np.arange(N, dtype=np.float) + 0.5
        xk = np.zeros(N, dtype=np.float)
        for k in range(N):
            xk[k] = np.sum(data * np.cos((math.pi * (k + 0.5) / N) * nv))
        xk *= math.sqrt(2.0 / N)
        return xk

    def _sag_polar(self, rho, theta):
        if self.polar_sag_fn is None:
            rv = rho * np.cos(theta) + self.centre[0]
            cv = rho * np.sin(theta) + self.centre[1]
            return np.array(self.interpolate.ev(rv, cv)) - self.centre_sag
        else:
            return self.polar_sag_fn(rho, theta) - self.centre_sag

    def _normal_departure(self, rho, theta):
        """ 
        Uses the rho theta vector to return an array of normal departures
        based on the polar sag function and the best fit sphere curvature.
        """
        intp = self._sag_polar(rho, theta)

        rho2 = rho * rho
        fact = np.sqrt(1.0 - self.bfs_curv ** 2 * rho2)
        ndp = fact * (intp - self.bfs_curv * rho2 / (1.0 + fact))

        return ndp

    def _build_abr_bar(self):

        scan_theta = np.linspace(0.0, 2 * np.pi, 2 * self.j_max, endpoint=False)
        rv = self.radius * np.repeat(self.u_vec, scan_theta.size)
        thv = np.repeat(scan_theta.reshape((1, scan_theta.size)), self.u_vec.size, axis=0).flatten()
        intp = self._normal_departure(rv, thv).reshape((self.u_vec.size, scan_theta.size))
        intp = np.insert(intp, 0, 0.0, axis=0)

        # Build the A(m,n) and B(m,n) terms [1] 2.9a
        scan_m_0 = range(self.m_max + 1)
        abar = np.zeros((self.m_max + 1, self.k_max + 1), dtype=np.float)
        bbar = np.zeros((self.m_max + 1, self.k_max + 1), dtype=np.float)

        # The FFT results for the lower values of k can be dropped progressively as the data is heavily oversampled
        # in the centre.
        kn = self.m_max + 1
        for k in range(1, self.k_max + 1):
            xfft = np.fft.fft(intp[k, :]) / self.j_max
            abar[:kn, k] = np.real(xfft)[:kn]
            bbar[:kn, k] = -np.imag(xfft)[:kn]

        # Build the r(n) terms [1] 4.8 
        jmat = np.zeros((self.n_max + 1, self.k_max), dtype=np.float)
        arbar = np.zeros((self.m_max + 1, self.n_max + 1), dtype=np.float)
        brbar = np.zeros((self.m_max + 1, self.n_max + 1), dtype=np.float)
        for m in scan_m_0:
            self.jacobi_p.build_recursion(m + 1)
            self.jacobi_p.jmat_u_x(jmat, self.u_vec, self.u_vec_sqr)
            awm = abar[m, 1:]
            bwm = bbar[m, 1:]
            arbar[m, :] = np.dot(jmat, awm) / self.k_max
            brbar[m, :] = np.dot(jmat, bwm) / self.k_max

        return arbar, brbar

    def _rbar_to_cbar(self, rbar):
        """
        Build the equation [1] 4.7 progressively from 
        the rbar result and the precomputed terms. 
        """
        mlim = self.m_max + 1
        scan_m_0 = range(self.m_max + 1)
        sigma_bar = np.zeros((self.m_max + 1, self.n_max + 1), dtype=np.float)
        sigma_bar[:, 0] = rbar[:, 0] / self.smH[:mlim, 0]
        for n in range(1, self.n_max + 1):
            sigma_bar[:, n] = (rbar[:, n] - self.smK[:mlim, n - 1] * sigma_bar[:, n - 1]) / self.smH[:mlim, n]

        e_bar = np.zeros((self.m_max + 1, self.n_max + 1), dtype=np.float)
        e_bar[:, self.n_max] = sigma_bar[:, self.n_max] / self.smH[:mlim, self.n_max]
        for n in range(self.n_max - 1, -1, -1):
            e_bar[:, n] = (sigma_bar[:, n] - self.smK[:mlim, n] * e_bar[:, n + 1]) / self.smH[:mlim, n]
        self.e_bar_0 = e_bar[0, :]

        d_bar = np.zeros((self.m_max + 1, self.n_max + 1), dtype=np.float)
        d_bar[1:, self.n_max] = e_bar[1:, self.n_max] / self.smS[1:mlim, self.n_max]
        for n in range(self.n_max - 1, -1, -1):
            d_bar[1:, n] = (e_bar[1:, n] - self.smT[1:mlim, n] * d_bar[1:, n + 1]) / self.smS[1:mlim, n]

        c_bar = np.zeros((self.m_max + 1, self.n_max + 1), dtype=np.float)
        for n in range(self.n_max):
            c_bar[1:, n] = self.smF[1:mlim, n] * d_bar[1:, n] + self.smG[1:mlim, n] * d_bar[1:, n + 1]
        c_bar[1:, self.n_max] = self.smF[1:mlim, self.n_max] * d_bar[1:, self.n_max]

        return c_bar

    def _e_rot_sym_fit(self, jvec, u):
        self.jacobi_p.jvec_x(jvec, u * u)
        return np.dot(jvec, self.e_bar_0) / 2.0

    def _refit_parabola(self, u):
        jvec = np.zeros(self.n_max + 1, dtype=np.float)
        return self._e_rot_sym_fit(jvec, 0.0) + (
                                                self._e_rot_sym_fit(jvec, 1.0) - self._e_rot_sym_fit(jvec, 0.0)) * u * u

    def _build_avec(self):
        """
        Builds the rotational symmetric radial fit. 
        """
        jmat = np.zeros((self.n_max + 1, self.k_max), dtype=np.float)
        self.jacobi_p.build_recursion(1)
        self.jacobi_p.jmat_x(jmat, self.u_vec_sqr)
        svec = (np.dot(self.e_bar_0, jmat) / 2.0 - self._refit_parabola(self.u_vec))[::-1]

        svec[:self.k_max] *= np.cos(self.phi_kvec) / (self.u_vec_sqr * (1 - self.u_vec_sqr))
        dct = self._dct_iv(svec)

        bv = np.zeros(self.k_max + 1, dtype=np.float)
        for k in range(self.k_max):
            bv[k] = (-1) ** k * dct[k]
        bv *= 1.0 / math.sqrt(2 * self.k_max)
        av = np.zeros(self.n_max + 1, dtype=np.float)
        for k in range(self.n_max + 1):
            av[k] = bv[k] * self.smFn[k] + bv[k + 1] * self.smGn[k] + bv[k + 2] * self.smHn[k]

        return av

    def _est_bfs(self):
        points = 500
        theta = np.linspace(0.0, 2 * np.pi, points, endpoint=False)
        rho = np.linspace(self.radius, self.radius, points)
        sag_rim = np.sum(self._sag_polar(rho, theta)) / points
        self.bfs_curv = 2.0 * sag_rim / (sag_rim ** 2 + self.radius ** 2)

    def _valid_radius(self, mask):
        cpix = self.centre_pixel
        rows = mask.shape[0]
        cols = mask.shape[1]

        xv = np.arange(rows, dtype=np.float) - cpix[0]
        yv = np.arange(cols, dtype=np.float) - cpix[1]
        m1 = np.ones((rows, cols), dtype=np.float)
        rsq = m1 * (yv * yv) + np.transpose(m1.transpose() * (xv * xv))
        if len(mask[mask == 0]) == 0:
            r_max = rows
        else:
            r_max = math.sqrt(np.min(rsq[mask == 0]))

        # Check that radius is contained inside the mask boundary
        r_max = min(r_max, cpix[0])
        r_max = min(r_max, rows - 1 - cpix[0])
        r_max = min(r_max, cpix[1])
        r_max = min(r_max, cols - 1 - cpix[1])

        return r_max

    def _radial_sum(self, av, x, inc_deriv=False):
        # The implementation follows equations 3.4 - 3.9 of [3]
        # The deivative is based on 3.10 and 3.11 of [3]

        t_4x = 2.0 - 4.0 * x
        n = self.n_max
        if hasattr(x, '__len__'):
            fact = np.ones_like(x)
        else:
            fact = 1.0
        zero = 0.0 * fact

        try:
            b_ = fact * (av[n] / self.smFn[n])
            b = (av[n - 1] - self.smGn[n - 1] * b_) / self.smFn[n - 1]
            alpha_ = b_
            alpha = b + t_4x * alpha_
            afp_ = zero
            afp = -4 * alpha_

        except (IndexError):
            if n == 0:
                return fact * (av[0] / self.smFn[0]), zero
            else:
                return zero, zero

        for i in reversed(range(n - 1)):
            b_, b = b, (av[i] - self.smGn[i] * b - self.smHn[i] * b_) / self.smFn[i]
            alpha_, alpha = alpha, b + t_4x * alpha - alpha_
            if inc_deriv:
                afp_, afp = afp, t_4x * afp - afp_ - 4 * alpha_

        if inc_deriv:
            return 2 * (alpha + alpha_), 2 * (afp + afp_)
        else:
            return 2 * (alpha + alpha_), None

    def _azimuthal_sum_centre(self, c_mn, K):
        # This is a special case of the azimuthal sum for u**2 = 0
        # that does not include the u**m scaling or the derivative.
        # It is only evaluated for m = 1 and x = 0

        ones = np.ones(K, np.float)
        cnm = c_mn[1]
        smFt, smGt = self.smF[1], self.smG[1]
        bgAt, bgBt, bgCt = self.bgA[1].transpose(), self.bgB[1], self.bgC[1]
        n = self.n_max
        dv_ = ones * (cnm[n] / smFt[n])
        alpha_ = dv_

        if n > 0:
            alpha_2, alpha_3 = None, None
            dv = (cnm[n - 1] - smGt[n - 1] * dv_) / smFt[n - 1]
            alpha = dv + bgAt[n - 1] * alpha_
            for i in reversed(range(n - 1)):
                dv = (cnm[i] - dv * smGt[i]) / smFt[i]
                alpha_3, alpha_2, alpha_, alpha = alpha_2, alpha_, alpha, dv + \
                                                  bgAt[i] * alpha - bgCt[i + 1] * alpha_
            if n > 2:
                alpha -= 0.8 * alpha_3  # scaled by 0.5 before returned for m = 1, n > 2
        else:
            alpha = alpha_

        return 0.5 * alpha

    def _azimuthal_sum(self, c_mn, x, inc_deriv=False):
        # The implementation follows equations B.4 and B.6 of [2]
        # with the derivative based on B.10 and B.11 of [2]
        # The process generates some large number in the recursion and
        # these terms are controlled when the u^m term is applied. This
        # function progressively applies the u^m term to reduce the
        # chance of the sum overflowing

        # This solution transposes the data structure so that the numpy
        # broadcast can be applied when performing the element wise multiplication
        # as it is about 25% faster than extending the matrices.
        def roll_u(u_cnt, u, upj):
            if u_cnt > 0:
                u = np.roll(u, 1, axis=1)
                u[:, 0] = ones
                if upj is not None:
                    upj *= u
                return u_cnt - 1, u, upj
            return u_cnt, u, upj

        ones = np.ones_like(x)
        upj = np.ones((len(x), self.m_max), dtype=np.float)
        u = np.outer(np.sqrt(x), upj[0])
        upj *= u
        uix = self.m_max

        cnm = c_mn[1:].transpose()
        smFt, smGt = self.smF[1:].transpose(), self.smG[1:].transpose()
        bgAt, bgBt, bgCt = self.bgA[1:].transpose(), self.bgB[1:].transpose(), self.bgC[1:].transpose()
        n = self.n_max
        dv_ = np.outer(ones, cnm[n] / smFt[n])
        alpha_ = upj * dv_
        if inc_deriv:
            afp_ = np.zeros_like(u)

        if n > 0:
            alpha_2, alpha_3 = None, None
            uix, u, upj = roll_u(uix, u, upj)
            dv = (cnm[n - 1] - smGt[n - 1] * dv_) / smFt[n - 1]
            alpha = upj * dv + u * (bgAt[n - 1] + np.outer(x, bgBt[n - 1])) * alpha_
            if inc_deriv:
                afp_2, afp_3 = None, None
                afp = u * bgBt[n - 1] * alpha_
            for i in reversed(range(n - 1)):
                uix, u, upj = roll_u(uix, u, upj)
                u2 = u * u
                dv = (cnm[i] - dv * smGt[i]) / smFt[i]
                alpha_3, alpha_2, alpha_, alpha = alpha_2, alpha_, alpha, dv * upj + u * (
                    bgAt[i] + np.outer(x, bgBt[i])) * alpha - u2 * bgCt[i + 1] * alpha_
                if inc_deriv:
                    afp_3, afp_2, afp_, afp = afp_2, afp_, afp, u * bgBt[i] * alpha_ + u * (
                    bgAt[i] + np.outer(x, bgBt[i])) * afp - u2 * bgCt[i + 1] * afp_
            if n > 2:
                alpha[:, 0] -= 0.8 * alpha_3[:, 0]  # scaled by 0.5 before returned for m = 1, n > 2
                if inc_deriv:
                    afp[:, 0] -= 0.8 * afp_3[:, 0]

            # if n < m then complete the u^m scaling of the data
            while uix > 0:
                uix, u, _ = roll_u(uix, u, None)
                alpha *= u
                if inc_deriv:
                    afp *= u
        else:
            alpha = alpha_
            afp = afp_

        if inc_deriv:
            return 0.5 * alpha.transpose(), 0.5 * afp.transpose()
        else:
            return 0.5 * alpha.transpose(), None

    def _azimuthal_term(self, a_mn, b_mn, u, theta):
        # a and b are m by n matrices of coefficients
        # and u is the normalized radius rho/rho_max
        mv = np.arange(0, self.m_max + 1, dtype=np.float)
        mv_theta = np.outer(mv, theta)
        cosf = np.cos(mv_theta)
        sinf = np.sin(mv_theta)

        if hasattr(u, '__len__'):
            usq = u * u
        else:
            usq = np.array([u * u])

        av, _ = self._azimuthal_sum(a_mn, usq)
        bv, _ = self._azimuthal_sum(b_mn, usq)
        vec = cosf[1:] * av + sinf[1:] * bv
        return np.sum(vec, axis=0)

    def _fitted_sag(self, a_mn, b_mn, rho, theta, radius, curv, inc_deriv):

        # # calculates the conic result along with the radial and azimuthal
        # # contributions
        # u = rho / rho_max
        # val, _ = self._radial_sum(a_mn[0, :], u ** 2)
        # val *= u ** 2 * (1 - u ** 2)
        # val += self._azimuthal_term(a_mn, b_mn, u, theta)
        #
        # # add the spherical section
        # sqf = np.sqrt(1 - curv ** 2 * rho ** 2)
        # val /= sqf
        # val += curv * rho ** 2 / (1 + sqf)
        # return val, None, None

        # Build the radial component first and expand for the theta values
        u = rho / radius
        if hasattr(u, '__len__'):
            u_2 = u ** 2
        else:
            u_2 = np.array([u**2])
        R, Rp = self._radial_sum(a_mn[0, :], u_2, inc_deriv=inc_deriv)
        radial = R * u_2 * (1 - u_2)

        # The asymmetric terms as [m,k] matrices
        mv = np.arange(0, self.m_max + 1, dtype=np.float)
        mv_theta = np.outer(mv, theta)
        sinf = np.sin(mv_theta)
        cosf = np.cos(mv_theta)
        av, avp = self._azimuthal_sum(a_mn, u_2, inc_deriv=inc_deriv)
        bv, bvp = self._azimuthal_sum(b_mn, u_2, inc_deriv=inc_deriv)
        vec = cosf[1:] * av + sinf[1:] * bv
        asym = np.sum(vec, axis=0)

        # Add the spherical factors
        psi = np.sqrt(1 - curv ** 2 * rho ** 2)
        sum = (radial + asym) / psi
        sum += curv * rho ** 2 / (1 + psi)

        if inc_deriv:
            # Build the derivative in rho and theta maps
            psi_2 = psi * psi
            radialp  = R * u * (1 + psi_2 - u_2*(1 + 3*psi_2)) / (radius * psi * psi_2)
            radialp += Rp * 2 * u_2 * u * (1 - u_2) / (radius * psi)
            radialp += curv * rho / psi

            # Add the azimuthmal contrib to the radial derivative. u = 0 is a special
            # case as the division by zero can be avoided as av is scaled by u**m and
            # divided by u to re-use a previous sum. The actual product is u**(m-1)
            ones = np.ones_like(u)
            mvs = np.outer(mv, ones)
            mcosf = cosf * mvs
            msinf = sinf * mvs
            azt_rp = 2 * u * np.sum((cosf[1:] * avp + sinf[1:] * bvp), axis=0)
            if np.min(u) > 0.0:
                azt_rp += np.sum((mcosf[1:] * av + msinf[1:] * bv), axis=0) / u
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    azt_rp += np.sum((mcosf[1:] * av + msinf[1:] * bv), axis=0) / u
                    # Fix up the points where u = 0.
                    cond = (u == 0.0)
                    uc = np.extract(cond, u)
                    thc = np.extract(cond, theta)
                    av_0 = self._azimuthal_sum_centre(a_mn, len(uc))
                    bv_0 = self._azimuthal_sum_centre(b_mn, len(uc))
                    sumc = np.cos(thc) * av_0 + np.sin(thc) * bv_0
                    np.place(azt_rp, cond, sumc)
            azt_rp /= (radius * psi)
            azt_rp += (curv**2 * rho / psi_2) * asym / psi
            radialp += azt_rp

            # Now add the azimuthal term
            azt_thp = np.sum((-msinf[1:] * av + mcosf[1:] * bv), axis=0) / psi

            return sum, radialp, azt_thp
        else:
            return sum, None, None

    def _build_regular_map(self, rho, theta, curv, radius, a_mn, b_mn, inc_deriv):

        # Builds a regular map where rho and theta are the axis values.
        ones = np.ones_like(theta)

        # Build the radial component first and expand for the theta values
        u = rho / radius
        u_2 = u ** 2
        R, Rp = self._radial_sum(a_mn[0, :], u_2, inc_deriv=inc_deriv)
        val = R * u_2 * (1 - u_2)
        radial = np.outer(ones, val)  # [j,m]

        # The asymmetric terms as [m,k] matrices
        mv = np.arange(0, self.m_max + 1, dtype=np.float)
        theta_mv = np.outer(theta, mv)
        sinf = np.sin(theta_mv)
        cosf = np.cos(theta_mv)
        av, avp = self._azimuthal_sum(a_mn, u_2, inc_deriv=inc_deriv)
        bv, bvp = self._azimuthal_sum(b_mn, u_2, inc_deriv=inc_deriv)
        as_jk = cosf[:, 1:].dot(av) + sinf[:, 1:].dot(bv)

        # Add the spherical factors
        psi = np.sqrt(1 - curv ** 2 * rho ** 2)
        sum_jk = (radial + as_jk) / psi
        sum_jk += curv * rho ** 2 / (1 + psi)

        if inc_deriv:
            # Build the derivative in rho and theta maps
            psi_2 = psi * psi
            valp  = R * u * (1 + psi_2 - u_2*(1 + 3*psi_2)) / (radius * psi * psi_2)
            valp += Rp * 2 * u_2 * u * (1 - u_2) / (radius * psi)
            valp += curv * rho / psi
            radialp = np.outer(ones, valp)

            # Add the azimuthmal contrib to the radial derivative. u = 0 is a special
            # case as the division by zero can be avoided as av is scaled by u**m and
            # divided by u to re-use a previous sum. The actual product is u**(m-1)
            mcosf = cosf * mv
            msinf = sinf * mv
            azt_rp = 2 * u * (cosf[:, 1:].dot(avp) + sinf[:, 1:].dot(bvp))

            with np.errstate(divide='ignore', invalid='ignore'):
                azt_rp += (mcosf[:, 1:].dot(av) + msinf[:, 1:].dot(bv)) / u
                # Fix up the columns where u = 0.
                cond = (u == 0.0)
                cix = np.extract(cond, np.arange(len(u)))
                if len(cix) > 0:
                    av_0 = self._azimuthal_sum_centre(a_mn, len(cix))
                    bv_0 = self._azimuthal_sum_centre(b_mn, len(cix))
                    sumc = np.outer(av_0, np.cos(theta)) + np.outer(bv_0, np.sin(theta))
                    for i in range(len(cix)):
                        azt_rp[:,cix[i]] = sumc[i]

            azt_rp /= (radius * psi)
            azt_rp += (curv**2 * rho / psi_2) * as_jk / psi
            radialp += azt_rp

            # Now add the azimuthal term
            azt_thp = (-msinf[:, 1:].dot(av) + mcosf[:, 1:].dot(bv)) / psi

            return sum_jk.transpose(), radialp.transpose(), azt_thp.transpose()
        else:
            return sum_jk.transpose(), None, None

    def _build_map(self, rho, theta, curv, radius, a_mn, b_mn, inc_deriv):

        block_size = 500
        zc = np.zeros_like(rho)
        if inc_deriv:
            rhop, thetap = np.zeros_like(rho), np.zeros_like(rho)
        else:
            rhop, thetap = None, None
        blocks = int(len(zc) / block_size) + 1
        for i in range(blocks):
            lo = i * block_size
            hi = (i + 1) * block_size
            zc[lo:hi], df_dr, df_dth = self._fitted_sag(a_mn, b_mn, rho[lo:hi], theta[lo:hi], radius, curv, inc_deriv)
            if inc_deriv:
                rhop[lo:hi] = df_dr
                thetap[lo:hi] = df_dth

        return zc, rhop, thetap

    def _check_transpose(self, a_nm, b_nm):
        n, m = a_nm.shape
        self._precompute_factors(m_max=m-1, n_max=n-1)
        if self.n_disp != self.n_max or self.m_disp != self.m_max:
            a = np.zeros((self.n_max+1, self.m_max+1))
            b = np.zeros_like(a)
            a[:n,:m], b[:n,:m] = a_nm, b_nm
            return a.transpose(), b.transpose()
        return a_nm.transpose(), b_nm.transpose()

    def _build_cartesian_gradient(self, dfdr, dfdth, rho_xy, theta_xy, curv, radius, a_mn, b_mn):

        """
        dfdx  = cos(theta)df/dr - (1/r)sin(theta)df/dth
        df/dy = sin(theta)df/dr + (1/r)cos(theta)df/dth

        Need to handle division by zero
        """
        cos_theta = np.cos(theta_xy)
        sin_theta = np.sin(theta_xy)

        with np.errstate(divide='ignore', invalid='ignore'):
            dfdx = cos_theta * dfdr - sin_theta * dfdth / rho_xy
            dfdy = sin_theta * dfdr + cos_theta * dfdth / rho_xy

            # Determine the correction when rho == 0 and replace value
            if 0.0 in rho_xy:
                zinv, _dfdr, _dfdth = self._build_map(np.array([0.0, 0.0]), np.array([0.0, 0.5*math.pi]),
                                                    curv, radius, a_mn, b_mn, True)
                dfdx[rho_xy == 0.0] = _dfdr[0]
                dfdy[rho_xy == 0.0] = _dfdr[1]

        return dfdx, dfdy

    def build_profile(self, xv, yv, a_nm, b_nm, curv=None, radius=None, centre=None, extend=1.0, inc_deriv=False):
        """
        Returns the nominal sag and optional x and y derivatives along a 1D trajectory of (x, y) coordinates.

        Parameters:
            x, y:   arrays
                    Arrays of values representing the (x, y) coordinates.
            a_nm, b_nm: 2D array
                    The cosine and sine terms for the Q freeform polynominal
            curv:   float
                    Nominal curvature for the part. If None uses the estimated value from the previous fit.
            radius: float
                    Defines the circular domain from the centre. If None uses the estimated value from the previous fit.
            centre: (cx, cy)
                    The centre of the part in axis coordinates. If None uses the estimated value from previous fit.
            extend: float
                    Generate a map over extend * radius from the centre
            inc_deriv: boolean
                    Return the X and Y derivatives as additional maps
        Returns:
            zval:   array
                    Sag values for the (x, y) sequence
            xder:   array
                    X derivative map for the (x, y) sequence if inc_deriv is True, else None
            yder:   array
                    Y derivative map for the (x, y) sequence if inc_deriv is True, else None
        """
        if curv is None:
            curv = self.bfs_curv
        if radius is None:
            radius = self.radius
        if centre is None:
            centre = self.centre

        a_mn, b_mn = self._check_transpose(a_nm, b_nm)
        xx, yy = xv - centre[0], yv - centre[1]
        rv = np.hypot(xv, yv)
        thv = np.arctan2(yv, xv)
        cond = rv <= extend * radius
        rvc = np.extract(cond, rv)
        thc = np.extract(cond, thv)

        def remap(vec):
            zv = np.zeros_like(xv)
            zv.fill(np.nan)
            np.place(zv, cond, vec)
            return zv

        dfdx, dfdy = None, None
        zv, dfdr, dfdth = self._build_map(rvc, thc, curv, radius, a_mn, b_mn, inc_deriv)
        zval = remap(zv)
        if inc_deriv:
            _dfdx, _dfdy = self._build_cartesian_gradient(dfdr, dfdth, rvc, thc, curv, radius, a_mn, b_mn)
            dfdx, dfdy = remap(_dfdx), remap(_dfdy)

        return zval, dfdx, dfdy

    def build_map(self, x, y, a_nm, b_nm, curv=None, radius=None, centre=None, extend=1.0, interpolated=True, inc_deriv=False):
        """
        Creates a 2D topography map and optional x and y derivate maps using the x and y axis vectors and the Q-freeform parameters.

        Parameters:
            x, y:   array
                    X, and Y axis values for the map to be created.
                    The arrays must be sorted to increasing order.
            a_nm, b_nm: 2D array
                    The cosine and sine terms for the Q freeform polynominal
            curv:   float
                    Nominal curvature for the part. If None uses the estimated value from the previous fit.
            radius: float
                    Defines the circular domain from the centre. If None uses the estimated value from the previous fit.
            centre: (cx, cy)
                    The centre of the part in axis coordinates. If None uses the estimated value from previous fit.
            extend: float
                    Generate a map over extend * radius from the centre
            interpolated: boolean
                    If True uses a high resolution regular polar grid to build the underlying
                    data and a spline interpolation to extract the (x, y) grid, otherwise it evaluates
                    each (x, y) point exactly. The non-interpolated solution is significanatly slower and
                    only practical for smaller array sizes.
            inc_deriv: boolean
                    Return the X and Y derivatives as additional maps
        Returns:
            zmap:   2-D array
                    Data map with shape (x.size, y.size)
            xder:   2-D array
                    X derivative map with shape (x.size, y.size) if inc_deriv is True, else None
            yder:   2-D array
                    Y derivative map with shape (x.size, y.size) if inc_deriv is True, else None
        """
        def remap(map):
            zv = np.zeros_like(xv)
            zv.fill(np.nan)
            np.place(zv, cond, map)
            return zv.reshape((len(x), len(y)))

        if curv is None:
            curv = self.bfs_curv
        if radius is None:
            radius = self.radius
        if centre is None:
            centre = self.centre

        a_mn, b_mn = self._check_transpose(a_nm, b_nm)
        xx, yy = np.meshgrid(x - centre[0], y - centre[1], indexing='ij')
        xv, yv = xx.flatten(), yy.flatten()
        rv = np.hypot(xv, yv)
        thv = np.arctan2(yv, xv)
        cond = rv <= extend * radius
        rvc = np.extract(cond, rv)
        thc = np.extract(cond, thv)
        dfdx, dfdy = None, None
        if interpolated:
            # Builds a regular polar map and then interpolates to the [x,y] grid
            # First find the range of the polar grid that covers the rectangle
            # and adds additional points in the polar grid to account for the
            # spline interpolation
            K = max(300, int(len(x) / 2))
            J = 6 * K
            rmin, rmax = rv.min(), rv.max()
            rdel = 0.0 #self.shrink_pixels * (rmax - rmin) / K
            rmin = rmin - rdel
            rmax = min(rmax + rdel, extend * radius)
            rho = np.linspace(rmin, rmax, K + 2 * self.shrink_pixels)
            tdel = self.shrink_pixels * (thv.max() - thv.min()) / J
            theta = np.linspace(thv.min() - tdel, thv.max() + tdel, J + 2 * self.shrink_pixels)

            # Build the regular polar map and initialize the interpolation function
            zpv, d_dr, d_dtheta = self._build_regular_map(rho, theta, curv, radius, a_mn=a_mn, b_mn=b_mn, inc_deriv=inc_deriv)
            interp = interpolate.RectBivariateSpline(rho, theta, zpv, kx=3, ky=3)
            zinv = interp.ev(rvc, thc)
            if inc_deriv:
                dfdr = interpolate.RectBivariateSpline(rho, theta, d_dr, kx=3, ky=3).ev(rvc, thc)
                dfdth = interpolate.RectBivariateSpline(rho, theta, d_dtheta, kx=3, ky=3).ev(rvc, thc)
        else:
            zinv, dfdr, dfdth = self._build_map(rvc, thc, curv, radius, a_mn, b_mn, inc_deriv)

        zmap = remap(zinv)
        if inc_deriv:
            _dfdx, _dfdy = self._build_cartesian_gradient(dfdr, dfdth, rvc, thc, curv, radius, a_mn, b_mn)
            dfdx = remap(_dfdx)
            dfdy = remap(_dfdy)
        return zmap, dfdx, dfdy

    def data_map(self, x, y, zmap, centre=None, radius=None, shrink_pixels=7, bfs_curv=None):
        """
        Creates the spline interpolator for the map, determines the best fit sphere
        and minimum valid radius.

        Parameters:
            x, y:   arrays
                    The arrays are the X and Y axis values for the data map.
                    The arrays must be sorted to increasing order.
            zmap:   array_like
                    2-D array of data with shape (x.size,y.size).
            centre: (cx, cy)
                    The centre of the part in axis coordinates. If None the centre is estimated
                    by a centre of mass calculation 
            radius: float
                    Defines the circular domain from the centre. If None it determines the 
                    maximum radius from the centre that contains no invalids (NAN).
            shrink_pixels: int
                    The estimated radius is reduced by 7 pixels to avoid edge effects with the 
                    spline interpolation. Ignored if the radius is specified.
        """
        self.polar_sag_fn = None
        self.shrink_pixels = shrink_pixels
        pixel_spacing = 0.5 * (x[1] - x[0] + y[1] - y[0])

        if centre is None or radius is None:
            mask = np.zeros_like(zmap, dtype=np.int)
            np.isfinite(zmap, out=mask)

        if centre is None:
            cpix = ndimage.measurements.center_of_mass(mask)
            cx = np.interp(cpix[0], range(len(x)), x)
            cy = np.interp(cpix[1], range(len(y)), y)
            self.centre = (cx, cy)
            self.centre_pixel = cpix
        else:
            self.centre = centre
            cpx = np.interp(centre[0], x, range(len(x)))
            cpy = np.interp(centre[1], y, range(len(y)))
            self.centre_pixel = (cpx, cpy)

        if radius is None:
            mask = np.zeros_like(zmap, dtype=np.int)
            np.isfinite(zmap, out=mask)
            self.pixel_radius = self._valid_radius(mask) - shrink_pixels
            self.radius = (self.pixel_radius) * pixel_spacing
        else:
            self.radius = radius
            self.pixel_radius = radius / pixel_spacing

        z = zmap.copy()
        np.place(z, np.isnan(z), 0.0)
        self.interpolate = interpolate.RectBivariateSpline(x, y, z, kx=3, ky=3)
        self.centre_sag = np.float(self.interpolate(self.centre[0], self.centre[1]))
        if bfs_curv is None:
            self._est_bfs()
        else:
            self.bfs_curv = bfs_curv

    def set_sag_fn(self, sag_fn, radius, bfs_curv=None):
        """
        A vectorized polar sag function that takes rho and theta as arguments.

        This function is used to pass an analytic sag function to test the performance
        of the algorithm to a higher precision but can also be used to define the input map
        and bypass data_map().

        """
        self.polar_sag_fn = sag_fn
        self.radius = radius
        self.pixel_radius = None
        self.centre = (0.0, 0.0)
        self.centre_sag = float(sag_fn(0.0, 0.0))
        if bfs_curv is None:
            self._est_bfs()
        else:
            self.bfs_curv = bfs_curv

    def q_fit(self, m_max=None, n_max=None):
        """
        Fits the departure from a best fit sphere to the Q-freeform polynominals as defined
        in [1](1.1) and returns the individual sine and cosine terms.

        Parameters:
            m_max, n_max:  int
                    The azimuthal and radial spectrum order. If None it uses the previous values
                    and if not defined it matches the values to the pixel resolution. The maximum
                    resolution supported is (1500, 1500)
        Returns:
            a_nm, b_nm:   2D array
                    The (n,m) matrix representation of the cosine and sine terms
        """
        if self.interpolate is None and self.polar_sag_fn is None:
            print("No data file or sag function available!")
            return None

        if m_max is None or n_max is None:
            if self.m_max is None:
                m_max = 500 if self.pixel_radius == None else np.int(np.round(math.pi * self.pixel_radius / 50) * 50)
                m_max = min(m_max, 1500)
                n_max = m_max
                self._precompute_factors(m_max, n_max)
        else:
            self._precompute_factors(m_max, n_max)

        arbar, brbar = self._build_abr_bar()
        a_mn = self._rbar_to_cbar(arbar)
        a_mn[0, :] = self._build_avec()
        b_mn = self._rbar_to_cbar(brbar)

        if self.m_disp != self.m_max or self.n_disp != self.n_max:
            a = a_mn[:self.m_disp+1,:self.n_disp+1]
            b = b_mn[:self.m_disp+1,:self.n_disp+1]
            return a.transpose(), b.transpose()
        else:
            return a_mn.transpose(), b_mn.transpose()

    def build_q_spectrum(self, m_max=None, n_max=None):
        """
        Fits the departure from a best fit sphere to the Q-polynominals as defined
        in [1](1.1) and returns the root sum square of the azimuthal terms.

        Parameters:
            m_max, n_max:  int
                    The azimuthal and radial spectrum order. If None it uses the previous values
                    and if not defined it matches the values to the pixel resolution.
        Returns:
            2D array
                    The (m,n) matrix representation of the spectrum (sqrt(cos^2 + sin^2)) terms
        """
        a_mn, b_mn = self.q_fit(m_max, n_max)
        q_spec = np.sqrt(np.square(a_mn) + np.square(b_mn))
        return q_spec

    def bfs_param(self):
        """
        Returns the fitted radius, curvature and centre.

        Returns:
            radius, curvature, centre:  float, float, (x,y) tuple
        """
        return self.radius, self.bfs_curv, self.centre

def qspec(x, y, zmap, m_max=None, n_max=None, centre=None, radius=None, shrink_pixels=7):
    """
    A wrapper function that creates the Q-spectrum object, loads and performs the
    fit of the data to the Q polynomials.

    Parameters:
        x, y:   array_like
                The interpolator uses grid points defined by the coordinate arrays x, y.
                The arrays must be sorted to increasing order.
        zmap:   array_like
                2-D array of data with shape (x.size,y.size)
        m_max, n_max:  int
                The azimuthal and radial spectrum order. If None, it uses the previous values
                and if not defined it matches the values to the pixel resolution.
        centre: (cx, cy)
                The centre of the part in axis coordinates. If None the centre is estimated
                by a centre of mass calculation
        radius: float
                Defines the circular domain from the centre. If None it determines the
                maximum radius from the centre that contains no invalids (NAN).
        shrink_pixels: int
                The estimated radius is reduced by 7 pixels to avoid edge effects with the
                spline interpolation. Ignored if the radius is specified.

        Returns:
            2D array
                The (m,n) matrix representation of the spectrum (sqrt(cos^2 + sin^2)) terms
    """
    qfit = QSpectrum(m_max=m_max, n_max=n_max)
    qfit.data_map(x=x, y=y, zmap=zmap, centre=centre, radius=radius, shrink_pixels=shrink_pixels)
    return qfit.build_q_spectrum()
