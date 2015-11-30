===========
Scikit-Qfit
===========


Scikit-Qfit is a Python that supports fitting gradient orthogonal Q-polynomials to 2D data.
Specifically it implements the algorithm defined in the "Fitting freeform shapes with orthogonal
bases", G W Forbes.


Description
===========

TODO...

Usage
=====

After loading the data map to be processed, pass the coordinate arrays x and y and 2-D array of
data with shape (x.size,y.size) as arguments to the method qspec(). The azimuthal and radial spectrum
limits are set by m_max and n_max respectively.

  >>> import skqfit as qf
  >>> ...
  >>> qspec = qf.qspec(x, y, zmap, m_max=500, n_max=500)


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
