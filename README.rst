===========
Scikit-Qfit
===========


Scikit-Qfit is a package that supports fitting gradient orthogonal Q-polynomials to 2D data.


Description
===========

Gradient orthogonal Q-polynomial representation of rotational optical surfaces have been used for
several years now by designers and have shown superior performance to the standard monomial form
because they require less terms to adequately define the surface and offer quicker convergence in design optimization.
Q-polynomials were first introduced with the publication of:

 * G W Forbes, "Shape specification for axially symmetric optical surfaces", Opt. Express 15, 5218-5226 (2007)

The use Q-polynomials was extended by the original author to address freeform shapes through the following articles:

 * "Fitting freeform shapes with orthogonal bases", Opt. Express 21, 19061-19081 (2013)
 * "Characterizing the shape of freeform optics", Opt. Express 20(3), 2483-2499 (2012)
 * "Robust, efficient computational methods for axially symmetric optical aspheres", Opt. Express 18(19), 19700-19712 (2010)

The implementation of this package follows the description in "Fitting freeform shapes with orthogonal
bases".

Additional project documentation can be found at
`<http://scikit-qfit.readthedocs.org/>`_.

Installation
============

The package can be installed through pip:

  > pip install scikit-qfit

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

The Jacobian polynomial calculation required by the algorithm can generate very large numbers which limits spectral resolution
to a maximum of 1500 for the radial and azimuthal terms (n, m). Using values greater than this can lead to an overflow.
If the nominal spectral resolution for a datamap is greater than this limit the data should be filtered prior to processing
to avoid aliasing.


Note that the process is an N^2 algorithm, so doubling the number of radial and azimuthal terms takes four times as long.


Dependencies
============

The package requires numpy and scipy and was tested on Linux with:
 * Python 2.7.6
 * numpy 1.8.2
 * scipy 0.13.3

These python, numpy and scipy versions were available on the Ubuntu 14.04 Linux release at the time of testing.
The package has been informally tested with python 3.4 successfully and I am not aware of reason it should not work with
later releases of these packages.

Note
====

This project has been set up using PyScaffold 2.4.4. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
