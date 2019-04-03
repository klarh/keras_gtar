Welcome to keras_gtar's documentation!
======================================

`keras_gtar <https://github.com/klarh/keras_gtar>`_ is an
in-development library for saving and restoring keras models inside
`libgetar <https://github.com/glotzerlab/libgetar>`_ files. By using a
trajectory-based format, we can save multiple versions of a model's
weights.

Installation
------------

Install `keras_gtar` from source on github::

  pip install git+https://github.com/klarh/keras_gtar.git#egg=keras_gtar

API Documentation
-----------------

.. autoclass:: keras_gtar.Trajectory
   :members:

.. autoclass:: keras_gtar.GTARLogger
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
