===========
Scikit-Qfit
===========

.. note::
   Work in progress
   
Gradient orthogonal Q-polynomial representation of rotational optical surfaces have been used for
several years now by designers and have shown superior performance to the standard monomial form
because they require less terms to adequately define the surface and offer quicker convergence in design optimization.
Q-polynomials were first introduced with the publication of:

 * G W Forbes,  `Shape specification for axially symmetric optical surfaces <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-8-5218>`_, Opt. Express 15, 5218-5226 (2007)

The use Q-polynomials was extended by the original author to address freeform shapes through the following articles:

 * `Fitting freeform shapes with orthogonal bases <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-21-16-19061>`_, Opt. Express 21, 19061-19081 (2013)
 * `Characterizing the shape of freeform optics <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-20-3-2483>`_, Opt. Express 20(3), 2483-2499 (2012)
 * `Robust, efficient computational methods for axially symmetric optical aspheres <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-18-19-19700>`_, Opt. Express 18(19), 19700-19712 (2010)

The implementation of this package follows the description in "Fitting freeform shapes with orthogonal
bases".

Q-spectrum examples
===================

The following images were generated from data supplied by `Mahr GmbH <http://www.mahr.com/>`_.

.. image:: ./images/ring_src_qspec.PNG
   :width: 500px
   :align: center

The first of these

.. image:: ./images/radial_src_qspec.PNG
   :width: 500px
   :align: center



Contents
========

.. toctree::
   :maxdepth: 2

   License <license>
   Authors <authors>
   Changelog <changes>

..      Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
