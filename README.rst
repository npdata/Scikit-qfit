===========
Scikit-Qfit
===========


Scikit-Qfit is a Python that supports fitting gradient orthogonal Q-polynomials to 2D data.
Specifically it implements the algorithm defined in the "Fitting freeform shapes with orthogonal
bases", G W Forbes.


Description
===========

Gradient orthogonal Q-polynomials to represent rotational optical surfaces have been used for
several years now by designers and has shown to have superior performance to a the standard monomial
representation because it requires less terms to adequately define the surface and offers
quicker convergence in design optimization.

 * G. W. Forbes, "Shape specification for axially symmetric optical surfaces", Opt. Express 15, 5218-5226 (2007)

The idea was extended by the original author to freeform shapes through the following articles:

 * "Fitting freeform shapes with orthogonal bases", Opt. Express 21, 19061-19081 (2013)
 * "Characterizing the shape of freeform optics", Opt. Express 20(3), 2483-2499 (2012)
 * "Robust, efficient computational methods for axially symmetric optical aspheres", Opt. Express 18(19), 19700-19712 (2010)

Usage
=====

After loading the data map to be processed, pass the coordinate arrays x and y and 2-D array of
data with shape (x.size,y.size) as arguments to the method qspec(). The azimuthal and radial spectrum
limits are set by m_max and n_max respectively.

  >>> import skqfit.qspectre as qf
  >>> ...
  >>> qspec = qf.qspec(x, y, zmap, m_max=500, n_max=500)

Limitations
===========
A double float representation limits the Jacobian polynomial calculation to a maximum of 1500 for the radially and
azimuthal terms (n, m). Using values greater than this can lead to an overflow.

The algorithm is an N^2 process, so doubling the number of radial and azimuthal terms takes four times as long.

The combination of the above two limitations may stop the spectral and pixel resolution matching. In this case the data
should be filtered to avoid aliasing.

Dependencies
============

The package requires numpy and scipy and was tested on Linux with:
 * Python 2.7.6
 * numpy 1.8.2
 * scipy 0.13.3

These python, numpy and scipy versions were available on the Ubuntu 14.04 Linux release at the time of testing.
The package has been informally tested with python 3.4 successfully and am not aware of reason it should not with
later releases of these packages.

Note
====

This project has been set up using PyScaffold 2.4.4. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
