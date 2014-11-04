.. image:: https://travis-ci.org/PNProductions/py-seam-carving.svg?branch=master
    :target: https://travis-ci.org/PNProductions/py-seam-carving

Python Seam Carving
===================

This is a Python implementation of `Seam carving algorithm`_ method both for images and for videos using graph cut algorithm

Requirements
------------

To run this code you need the following packages:

-  Python `2.6`_ and `2.7`_
-  `Numpy`_
-  `Cython`_
-  `numexpr`_
-  `OpenCV`_ (optional, only to run examples)
-  `Scipy`_ (optional, only to tests)

Maybe it should work also on other version of python, but it’s untested.

**Everything but OpenCV can be installed via
``pip install -r requirements``**

Installation
------------

To install everything just run:

.. code:: shell

    python setup.py install

Maybe you have to run it with ``sudo``.

Testing
-------

Test are provided via `unittest`_.

To run them all:

.. code:: shell

    nosetests

Examples
--------

Check ``examples`` folder for some examples.

Final Notes
-----------

This library is already in development, so don’t use it for **real**
purposes.

.. _Seam carving algorithm: http://www.eng.tau.ac.il/~avidan/papers/vidret.pdf
.. _2.6: https://www.python.org/download/releases/2.6/
.. _2.7: https://www.python.org/download/releases/2.7/
.. _Numpy: http://www.numpy.org/
.. _Cython: https://github.com/pmneila/PyMaxflow
.. _numexpr: https://github.com/pydata/numexpr
.. _OpenCV: http://opencv.org/
.. _Scipy: http://www.scipy.org/
.. _unittest: https://docs.python.org/2/library/unittest.html
