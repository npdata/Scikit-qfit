"""
 References:

 G W Forbes, "Characterizing the shape of freeform optics", Opt. Express 20(3), 2483-2499 (2012)

"""


from __future__ import print_function, absolute_import, division

import numpy as np

class AsymJacobiP(object):
    """
    Generate asymmetric Jacobi like polynominals needed for the freeform fit as
    defined in the reference document [A.1]

    
    *** Warning
    The Jacobi P polynominals can generate large values that lead to a 
    double overflow for large m and n values. It is tested to (1500, 1500) for (m,n)

    *** Note
    The scipi.special.jacobi function is has a 1.5:1 performance advantage over the
    current implementation but it doesn't support the normalization and can lead 
    to an overflow condition for n or m over 500. 
    """

    def __init__(self, nmax):
        self.scan_n = None
        self.a_mn = np.zeros(nmax, dtype=np.float)
        self.b_mn = np.zeros_like(self.a_mn)
        self.c_mn = np.zeros_like(self.a_mn)
        self.abc_mn = None
        self.m = None
        self.nmax = nmax

    def build_recursion(self, m):
        """
        Build the recurssion coefficients and saves them as a sequence of tuples.
        These are the coefficients to build the m type polynominals up to order n.
        """
        start = 1 if m > 1 else 3
        self.scan_n = range(start, self.nmax)
        self.a_mn.fill(0.0)
        self.b_mn.fill(0.0)
        self.c_mn.fill(0.0)
        for n in self.scan_n:
            d_mn = (4.0*n*n-1)*(m+n-2)*(m+2*n-3)
            self.a_mn[n] = (2.0*n-1)*(m+2*n-2)*(4*n*(m+n-2)+(m-3)*(2*m-1))/d_mn
            self.b_mn[n] = -2.0*(2*n-1)*(m+2*n-1)*(m+2*n-2)*(m+2*n-3)/d_mn
            self.c_mn[n] = n*(2.0*n-3)*(m+2*n-1)*(2*m+2*n-3)/d_mn
        self.abc_mn = list(zip(self.scan_n, self.a_mn[start:], self.b_mn[start:], self.c_mn[start:]))
        self.m = m

    def jvec_x(self, jvec, x):
        """
        Builds the sequence of jacobi polynomials for the value x
        based on [A.2-5]
        """
        m = self.m
        if m == 1:
            jvec[0] = 0.5
            jvec[1] = 1.0 - x/2
            jvec[2] = (3.0 + x*(-12 + 8*x))/6 
            jvec[3] = (5.0 + x*(-60 + (120 - 64*x)*x))/10.0
            vn_ = jvec[2]
            vn  = jvec[3]
        else:
            jvec[0] = 0.5
            jvec[1] = (m - 0.5) + (1 - m)*x
            vn_ = jvec[0]
            vn  = jvec[1]
        for (n,a,b,c) in self.abc_mn:
            vn_, vn = vn, (a + b*x)*vn - c*vn_
            jvec[n+1] = vn
        return jvec

    def jmat_x(self, jmat, xv):
        """
        Builds the asymmetric Jacobi P polynomial as defined in [2] A.1 for 
        all a vector of x values.
        """
        m = self.m
        if m == 1:
            jmat[0,:].fill(0.5)
            jmat[1,:] = 1.0 - xv/2
            jmat[2,:] = (3.0 + xv*(-12 + 8*xv))/6 
            jmat[3,:] = (5.0 + xv*(-60 + (120 - 64*xv)*xv))/10.0
            vn_ = jmat[2,:]
            vn  = jmat[3,:]
        else:
            jmat[0,:].fill(0.5)
            jmat[1,:] = (m - 0.5) + (1 - m)*xv
            vn_ = jmat[0,:]
            vn  = jmat[1,:]
        for (n,a,b,c) in self.abc_mn:
            vn_, vn = vn, (a + b*xv)*vn - c*vn_
            jmat[n+1,:] = vn
        return jmat

    def jmat_u_x(self, jmat, uv, xv):
        """
        Builds the asymmetric Jacobi P polynomial as defined in [2] A.1 for all of the
        x values with the scaling factor of u**m which avoids the overflow condition. 
        """
        m = self.m
        if m == 1:
            jmat[0,:].fill(0.5)
            jmat[1,:] = (1.0 - xv/2)
            jmat[2,:] = ((3.0 + xv*(-12 + 8*xv))/6) 
            jmat[3,:] = ((5.0 + xv*(-60 + (120 - 64*xv)*xv))/10.0)
            vn_ = jmat[2,:]
            vn  = jmat[3,:]
        else:
            jmat[0,:].fill(0.5)
            jmat[1,:] = uv*((m - 0.5) + (1 - m)*xv)
            vn_ = jmat[0,:]
            vn  = jmat[1,:]
        for (n,a,b,c) in self.abc_mn:
            n_ = n+1
            if n_ < m:
                vn_, vn = vn, uv*((a + b*xv)*vn - uv*c*vn_)
            elif n_ == m:
                vn_, vn = vn, (a + b*xv)*vn - uv*c*vn_                  
            else:
                vn_, vn = vn, (a + b*xv)*vn - c*vn_                  
            jmat[n_,:] = vn

        # Complete the uv**m scaling of the data 
        n = max(0, m - 1 - (jmat.shape[0]-1))
        upm = np.power(uv, n)
        for i in range(min(jmat.shape[0]-1, m-1), -1, -1):
            jmat[i,:] *= upm
            upm *= uv
        return jmat

    def jvec_u_x(self, jvec, u, x):
        """
        Builds the asymmetric Jacobi P polynomial for a single x as defined in [2] A.1 for all of the
        x values with the scaling factor of u**m which avoids the overflow condition. 
        """
        m = self.m
        if m == 1:
            jvec[0] = 0.5
            jvec[1] = (1.0 - x/2)
            jvec[2] = ((3.0 + x*(-12 + 8*x))/6) 
            jvec[3] = ((5.0 + x*(-60 + (120 - 64*x)*x))/10.0)
            vn_ = jvec[2]
            vn  = jvec[3]
        else:
            jvec[0] = 0.5
            jvec[1] = u*((m - 0.5) + (1 - m)*x)
            vn_ = jvec[0]
            vn  = jvec[1]
        for (n,a,b,c) in self.abc_mn:
            n_ = n+1
            if n_ < m:
                vn_, vn = vn, u*((a + b*x)*vn - u*c*vn_)
            elif n_ == m:
                vn_, vn = vn, (a + b*x)*vn - u*c*vn_                  
            else:
                vn_, vn = vn, (a + b*x)*vn - c*vn_      
                           
            jvec[n_] = vn

        # Complete the u**m scaling of the data 
        n = max(0, m - 1 - (jvec.size-1))
        upm = u**n
        for i in range(min(jvec.size-1, m-1), -1, -1):
            jvec[i] *= upm
            upm *= u
        return jvec
