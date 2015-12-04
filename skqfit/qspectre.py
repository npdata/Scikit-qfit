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
        self.m_max = None
        self.n_max = None
        if not m_max is None and not n_max is None:
            self._precompute_factors(m_max, n_max)
        self.shrink_pixels = 7
        self.centre_sag = 0.0
        self.polar_sag_fn = None

    def _precompute_factors(self, m_max, n_max):

        if (not self.m_max is None) and (m_max <= self.m_max) and (not self.n_max is None) and (n_max <= self.n_max):
            return

        self.m_max = m_max
        self.n_max = n_max
        self.k_max = n_max + 2
        self.j_max = m_max + 1
        self.jacobi_p = AsymJacobiP(n_max)

        # The unit vector corresponds to the radial sample space
        self.phi_kvec = np.array([(2.0*k - 1)*math.pi/(4.0*self.k_max) for k in range(1,self.k_max+1)])
        self.u_vec = np.sin(self.phi_kvec)
        self.u_vec_sqr = np.square(self.u_vec)

        # pre compute the tables from [1] A.5, A.6, A.7a, A.7b, A.11 and A.12
        self.bgK, self.bgH, self.smK, self.smH, self.smS, self.smT = self._compute_qfit_tables(m_max, n_max)

        # pre compute tables from [2] A.13, A.15, A.18a, A.18b
        self.bgF, self.bgG, self.smF, self.smG = self._compute_freeform_tables(m_max, n_max)

        # pre compute tables from [3] A.14, A.15, A.16
        self.smFn, self.smGn, self.smHn = self._compute_qbfs_tables(n_max)

    def _compute_qbfs_tables(self, max_n):
        """
        pre compute tables from [3] A.14, A.15, A.16
        """
        smFn = np.zeros(max_n+3,dtype=np.float)
        smGn = np.zeros(max_n+2,dtype=np.float)
        smHn = np.zeros(max_n+1,dtype=np.float)

        smFn[0] = 2.0
        smFn[1] = math.sqrt(19.0)/2
        smGn[0] = -0.5
        for n in range(2, max_n+3):
            smHn[n - 2] = -n*(n - 1)/(2*smFn[n - 2])
            smGn[n - 1] = -(1 + smGn[n - 2]*smHn[n - 2])/smFn[n - 1]
            smFn[n] = math.sqrt(n*(n + 1) + 3 - smGn[n - 1]**2 - smHn[n - 2]**2)

        return smFn, smGn, smHn

    def _compute_qfit_tables(self, m_max, n_max):
        """
        Build the big H and K tables as described in ([1] A.6, A.5)
        """
        bgK = np.zeros((m_max+1,n_max+1),dtype=np.float)
        bgH = np.zeros((m_max+1,n_max+1), dtype=np.float)

        bgK[0,0] = 3.0/8.0
        bgK[0,1] = 1.0/24.0
        bgH[0,0] = 1.0/4.0
        bgH[0,1] = 19.0/32.0

        mv = np.arange(1, m_max+1, dtype=np.float)
        nv = np.arange(2, n_max+1, dtype=np.float)
        nv2 = nv*nv

        # build the first row
        bgK[0,2:] = (nv2 - 1)/(32*nv2 - 8)
        bgH[0,2:] = (1. + 1/(1 - 2*nv)**2)/16

        # recursively build factorial terms and complete the first two columns
        nfv = np.arange(m_max+1, dtype=np.float)
        nfact = 0.5
        for m in range(1,m_max+1):
            num = float(2*m+1)
            den = float(2*m+2)
            nfact = num/den*nfact
            nfv[m] = nfact
        
        bgK[1:, 0] = 0.5*nfv[1:]
        bgK[1:, 1] = ((2.0*mv*(2*mv + 3))/(3.0*(mv + 3.)*(mv + 2)))*0.5*nfv[1:]
        bgH[1:, 0] = ((mv + 1.)/(2*mv + 1))*0.5*nfv[1:]
        bgH[1:, 1] = ((3*mv + 2.)/(mv + 2))*0.5*nfv[1:]

        v = bgK[1:, 1]
        w = bgH[1:, 1]
        for n in range(2,n_max+1):
            bgH[1:, n] = (((mv + (2*n - 3))*((mv + (n - 2))*(4*n - 1) + 5*n))/((mv + (n - 2))*(2*n - 1)*(mv + 2*n)))*v
            v = (((n + 1)*(mv + (2*n - 2))*(mv + (2*n - 3))*(2*mv + (2*n + 1)))/((2*n + 1)*(mv + (n - 2))*(mv + (2*n + 1))*(mv + 2*n)))*v
            bgK[1:, n] = v

        # Build the small H and K tables (A.7a, A.7b)
        smK = np.zeros((m_max+1,n_max+1),dtype=np.float)
        smH = np.zeros((m_max+1,n_max+1), dtype=np.float)

        smH[:, 0] = np.sqrt(bgH[:, 0])        
        for n in range(1,n_max+1):
            smK[:, n - 1] = bgK[:, n - 1]/smH[:, n - 1]
            smH[:, n] = np.sqrt(bgH[:, n] - smK[:, n - 1]**2)
       
        # Build the small S and T tables (A.11, A.12)
        smS = np.zeros((m_max+1,n_max+1),dtype=np.float)
        smT = np.zeros((m_max+1,n_max+1), dtype=np.float)
        nv = np.arange(1, n_max+1, dtype=np.float)
        n2v = 2.0*nv 
        for m in range(1,m_max+1):
            smS[m, 0] = 1
            smT[m, 0] = 1.0/m
            smS[m, 1:] = (nv + (m - 2))/(n2v + (m - 2))
            smT[m, 1:] = ((1 - n2v)*(nv + 1))/((m + n2v)*(n2v + 1))
        smS[1, 1] = 0.5
        smT[1, 0] = 0.5

        return bgK, bgH, smK, smH, smS, smT

    def _compute_freeform_tables(self, max_m, max_n):
        """
        Pre compute tables from [2] A.13, A.15, A.18a, A.18b
        """
        def gamma_factorial(m, n):
            return factorial(n)*factorial2(2*m + 2*n - 3)/(2.0**(m + 1)*factorial(m + n - 3)*factorial2(2*n - 1))

        def kron_delta(i, j):
            return 1 if i == j else 0

        bgF = np.zeros((max_m+1,max_n+1),dtype=np.float)
        bgG = np.zeros((max_m+1,max_n+1),dtype=np.float)

        mv = np.arange(max_m+1, dtype=np.float)
        mv2 = np.arange(2, max_m+1, dtype=np.float)
        mv2_sqrd = mv2*mv2
        fvF = np.ones(max_m+1, dtype=np.float)
        fvG = np.ones(max_m+1, dtype=np.float)
        gv_m = np.ones(max_m+1, dtype=np.float)

        for m in range(1, max_m+1):
            if m == 1:
                facF = 0.25
                facG = 0.25
                g_m = 0.25
            else:
                facF = 0.5*((2*m - 3.)/(m - 1))*facF     # (2m-3)!!/(m-1)!2^(m+1)
                facG = 0.5*((2*m - 1.)/(m - 1))*facG     # (2m-1)!!/(m-1)!2^(m+1)
                g_m = g_m * (2*m - 3.)/(2.0*(m - 3)) if m > 3 else 3.0/(2**4)
                fvF[m] = facF
                fvG[m] = facG
                gv_m[m] = g_m

        gamma = np.zeros(max_m+1, dtype=np.float)
        for n in range(0, max_n+1):
            if n == 0:
                gamma[3] = gamma_factorial(3, 0)
                gamma[4:] = gv_m[4:]
            else:
                i = max(0, 4 - n)
                if i > 0:
                    gamma[i-1] = gamma_factorial(i-1, n)
                gamma[i:] = (n*(2*mv[i:] + (2*n - 3))/((mv[i:] + (n - 3))*(2*n - 1)))*gamma[i:]

            if n == 0:
                bgF[1, n] = 0.25
                bgG[1, n] = 0.25
                bgF[2:, n] = mv2_sqrd*fvF[2:]
                bgG[2:, n] = fvG[2:]
            else:
                bgF[1, n] = (4*((n - 1)*n)**2 + 1.)/(8*(2*n - 1)**2) + kron_delta(n,1)*11.0/32
                bgG[1, n] = -(((2*n*n - 1.)*(n*n - 1))/(8*(4*n*n - 1))) - kron_delta(n,1)/24.0
                bgF[2:, n] = ((2*n*(mv2 + (n - 2.))*(3 - 5*mv2 + 4*n*(mv2 + (n - 2))) + mv2_sqrd*(3 - mv2 + 4*n*(mv2 + (n - 2))))/((2*n - 1)*(mv2 + (2*n - 3))*(mv2 + (2*n - 2))*(mv2 + (2*n - 1))))*gamma[2:]
                bgG[2:, n] = -(((2*n*(mv2 + (n - 1.)) - mv2)*(n + 1)*(2*mv2 + (2*n - 1)))/((mv2 + (2*n - 2))*(mv2 + (2*n - 1))*(mv2 + 2*n)*(2*n + 1)))*gamma[2:]

        smF = np.zeros((max_m+1,max_n+1),dtype=np.float)
        smG = np.zeros((max_m+1,max_n+1),dtype=np.float)
        smF[:, 0] = np.sqrt(bgF[:, 0])
        for n in range(1, max_n+1):
            smG[1:, n-1] = bgG[1:, n-1] / smF[1:, n-1]
            smF[1:, n] = np.sqrt(bgF[1:, n] - smG[1:, n - 1]**2)

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
            xk[k] = np.sum(data*np.cos((math.pi*(k+0.5)/N)*nv))
        xk *= math.sqrt(2.0/N)
        return xk

    def _sag_polar(self, rho, theta):
        if self.polar_sag_fn is None:
            rv = rho*np.cos(theta) + self.centre[0]
            cv = rho*np.sin(theta) + self.centre[1]
            return np.array(self.interpolate.ev(rv, cv)) - self.centre_sag
        else:
            return self.polar_sag_fn(rho, theta) - self.centre_sag

    def _normal_departure(self, rho, theta):
        """ 
        Uses the rho theta vector to return an array of normal departures
        based on the polar sag function and the best fit sphere curvature.
        """
        intp = self._sag_polar(rho, theta)

        rho2 = rho*rho
        fact = np.sqrt(1.0 - self.bfs_curv**2*rho2)
        ndp = fact*(intp - self.bfs_curv*rho2/(1.0 + fact)) 

        return ndp

    def _build_abr_bar(self):

        scan_theta = np.linspace(0.0, 2*np.pi, 2*self.j_max, endpoint=False)
        rv = self.radius*np.repeat(self.u_vec, scan_theta.size)
        thv = np.repeat(scan_theta.reshape((1,scan_theta.size)), self.u_vec.size, axis=0).flatten()
        intp = self._normal_departure(rv, thv).reshape((self.u_vec.size,scan_theta.size)) 
        intp = np.insert(intp, 0, 0.0, axis=0)

        # Build the A(m,n) and B(m,n) terms [1] 2.9a
        scan_m_0 = range(self.m_max+1)
        abar = np.zeros((self.m_max+1, self.k_max+1), dtype=np.float)
        bbar = np.zeros((self.m_max+1, self.k_max+1), dtype=np.float)

        # The FFT results for the lower values of k can be dropped progressively as the data is heavily oversampled
        # in the centre.
        kn = self.m_max+1
        for k in range(1, self.k_max+1):
            xfft = np.fft.fft(intp[k,:])/self.j_max
            abar[:kn,k] = np.real(xfft)[:kn]
            bbar[:kn,k] = -np.imag(xfft)[:kn]

        # Build the r(n) terms [1] 4.8 
        jmat = np.zeros((self.n_max+1, self.k_max), dtype=np.float)
        arbar = np.zeros((self.m_max+1, self.n_max+1), dtype=np.float)
        brbar = np.zeros((self.m_max+1, self.n_max+1), dtype=np.float)
        for m in scan_m_0:
            self.jacobi_p.build_recursion(m+1)
            self.jacobi_p.jmat_u_x(jmat, self.u_vec, self.u_vec_sqr)
            awm = abar[m,1:]
            bwm = bbar[m,1:]
            arbar[m,:] = np.dot(jmat,awm)/self.k_max   
            brbar[m,:] = np.dot(jmat,bwm)/self.k_max

        return arbar, brbar

    def _rbar_to_cbar(self, rbar):
        """
        Build the equation [1] 4.7 progressively from 
        the rbar result and the precomputed terms. 
        """
        mlim = self.m_max+1
        scan_m_0 = range(self.m_max+1)
        sigma_bar = np.zeros((self.m_max+1, self.n_max+1), dtype=np.float)
        sigma_bar[:, 0] = rbar[:, 0]/self.smH[:mlim, 0]
        for n in range(1, self.n_max+1):
            sigma_bar[:, n] = (rbar[:, n] - self.smK[:mlim, n-1]*sigma_bar[:, n-1])/self.smH[:mlim, n]

        e_bar = np.zeros((self.m_max+1, self.n_max+1), dtype=np.float)
        e_bar[:, self.n_max] = sigma_bar[:, self.n_max]/self.smH[:mlim, self.n_max]
        for n in range(self.n_max-1,-1,-1):
            e_bar[:, n] = (sigma_bar[:, n] - self.smK[:mlim, n]*e_bar[:, n+1])/self.smH[:mlim, n]
        self.e_bar_0 = e_bar[0,:]

        d_bar = np.zeros((self.m_max+1, self.n_max+1), dtype=np.float)
        d_bar[1:, self.n_max] = e_bar[1:, self.n_max]/self.smS[1:mlim, self.n_max]
        for n in range(self.n_max-1,-1,-1):
            d_bar[1:, n] = (e_bar[1:, n] - self.smT[1:mlim, n]*d_bar[1:, n+1])/self.smS[1:mlim, n]

        c_bar = np.zeros((self.m_max+1, self.n_max+1), dtype=np.float)
        for n in range(self.n_max):
            c_bar[1:, n] = self.smF[1:mlim, n]*d_bar[1:, n] + self.smG[1:mlim, n]*d_bar[1:, n+1]
        c_bar[1:, self.n_max] = self.smF[1:mlim, self.n_max]*d_bar[1:, self.n_max]

        return c_bar

    def _e_rot_sym_fit(self, jvec, u):
        self.jacobi_p.jvec_x(jvec, u*u)
        return np.dot(jvec,self.e_bar_0)/2.0

    def _refit_parabola(self, u):
        jvec = np.zeros(self.n_max+1, dtype=np.float)
        return self._e_rot_sym_fit(jvec, 0.0) + (self._e_rot_sym_fit(jvec, 1.0) - self._e_rot_sym_fit(jvec, 0.0))*u*u

    def _build_avec(self):
        """
        Builds the rotational symmetric radial fit. 
        """
        jmat = np.zeros((self.n_max+1, self.k_max), dtype=np.float)
        self.jacobi_p.build_recursion(1)
        self.jacobi_p.jmat_x(jmat, self.u_vec_sqr)
        svec = (np.dot(self.e_bar_0, jmat)/2.0 - self._refit_parabola(self.u_vec))[::-1]

        svec[:self.k_max] *= np.cos(self.phi_kvec)/(self.u_vec_sqr*(1 - self.u_vec_sqr))
        dct = self._dct_iv(svec)

        bv = np.zeros(self.k_max,dtype=np.float)
        for k in range(self.k_max):
            bv[k] = (-1)**k*dct[k]
        bv *= 1.0/math.sqrt(2*self.k_max)
        av = np.zeros(self.n_max+1, dtype=np.float)
        for k in range(self.n_max-1):
            av[k] = bv[k]*self.smFn[k] + bv[k+1]*self.smGn[k] + bv[k+2]*self.smHn[k]
        av[self.n_max-1] = bv[self.n_max-1]*self.smFn[self.n_max-1] + bv[self.n_max]*self.smGn[self.n_max-1]
        av[self.n_max] = bv[self.n_max]*self.smFn[self.n_max]

        return av

    def _est_bfs(self):
        points = 500
        theta = np.linspace(0.0, 2*np.pi, points, endpoint=False) 
        rho = np.linspace(self.radius, self.radius, points)
        sag_rim = np.sum(self._sag_polar(rho, theta))/points
        self.bfs_curv = 2.0*sag_rim/(sag_rim**2 + self.radius**2)

    def _valid_radius(self, mask):
        cpix = self.centre_pixel
        rows = mask.shape[0]
        cols = mask.shape[1]

        xv = np.arange(rows, dtype=np.float) - cpix[0]
        yv = np.arange(cols, dtype=np.float) - cpix[1]
        m1 = np.ones((rows,cols), dtype=np.float)
        rsq = m1*(yv*yv) + np.transpose(m1.transpose()*(xv*xv))
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

    def data_map(self, x, y, zmap, centre=None, radius=None, shrink_pixels=7):
        """
        Creates the spline interpolator for the map, determines the best fit sphere
        and minimum valid radius.

        Parameters:
            x, y:   array_like
                    The interpolator uses grid points defined by the coordinate arrays x, y. 
                    The arrays must be sorted to increasing order.
            zmap:   array_like
                    2-D array of data with shape (x.size,y.size)
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
        pixel_spacing = 0.5*(x[1]-x[0] + y[1]-y[0])

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
            self.radius = (self.pixel_radius)*pixel_spacing
        else:
            self.radius = radius
            self.pixel_radius = radius / pixel_spacing

        z = zmap.copy()
        np.place(z, np.isnan(z), 0.0)
        self.interpolate = interpolate.RectBivariateSpline(x, y, z, kx=3,ky=3)
        self.centre_sag = np.float(self.interpolate(self.centre[0], self.centre[1]))
        self._est_bfs()
                      
    def set_sag_fn(self, sag_fn, radius, bfs_curv=None):
        """
        A vectorized polar sag function that takes rho and theta as arguments.

        This function is used to pass an analytic sag function to test the performance
        of the algorithm to a higher precision.

        """
        self.polar_sag_fn = sag_fn
        self.radius = radius
        self.pixel_radius = None
        self.centre_sag = float(sag_fn(0.0, 0.0))
        if bfs_curv is None:
            self._est_bfs()
        else:
            self.bfs_curv = bfs_curv

    def q_fit(self, m_max=None, n_max=None):
        """
        Fits the departure from a best fit sphere to the Q-polynominals as defined
        in [1](1.1)

        Parameters:
            m_max, n_max:  int
                    The azimuthal and radial spectrum order. If None it uses the previous values
                    and if not defined it matches the values to the pixel resolution. The maximum
                    resolution supported is (1500, 1500)
        Returns:
            a, b:   2D array
                    The (m,n) matrix representation of the cosine and sine terms
        """
        if self.interpolate is None and self.polar_sag_fn is None:
            print("No data file or sag function available!")
            return None

        if m_max is None or n_max is None:
           if self.m_max is None:
            m_max = 500 if self.pixel_radius == None else np.int(np.round(math.pi*self.pixel_radius/50)*50)
            m_max = min(m_max, 1500)
            n_max = m_max
            self._precompute_factors(m_max, n_max)
        else:
            self._precompute_factors(m_max, n_max)

        arbar, brbar = self._build_abr_bar()
        a_mn = self._rbar_to_cbar(arbar)
        a_mn[0,:] = self._build_avec()
        b_mn = self._rbar_to_cbar(brbar)

        return a_mn.transpose(), b_mn.transpose()

    def build_q_spectrum(self, m_max=None, n_max=None):
        """
        Fits the departure from a best fit sphere to the Q-polynominals as defined
        in [1](1.1)

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