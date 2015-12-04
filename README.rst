===========
Scikit-Qfit
===========


Scikit-Qfit is a package that supports fitting gradient orthogonal Q-polynomials to 2D data.


Description
===========

This package implements the algorithm described in:

* G W Fobes, `Fitting freeform shapes with orthogonal bases <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-21-16-19061>`_, Opt. Express 21, 19061-19081 (2013)

Additional project documentation and references for Q-polynomials can be found at:
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

Acknowledge
===========

* Greg Forbes for support with the implementation and validation of the algorithm.
* Andreas Beutler, `Mahr GmbH <http://www.mahr.com/>`_, for choosing to make this work available as open source.

Note
====

This project has been set up using PyScaffold 2.4.4. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
