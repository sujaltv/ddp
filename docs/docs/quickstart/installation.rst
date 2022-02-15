Installation
############

.. warning::
  This wrapper is tested only on Linux arch-64.

With ``conda``
==============

From Anadonda
-------------

.. image:: https://img.shields.io/conda/v/tvsujal/ddpw
  :target: https://anaconda.org/tvsujal/ddpw
  :width: 125
  :alt: Conda publication

.. code:: bash

  conda install -c tvsujal ddpw

From source
-----------

.. code:: bash

  > git clone https://github.com/sujaltv/ddpw
  > cd ddpw/conda

  > conda-build . # generates a binary distribution
  > conda install  <distribution_path>

With ``pip``
============

From the PyPI registry
----------------------

.. image:: https://img.shields.io/pypi/v/ddpw
  :target: https://pypi.org/project/ddpw/
  :width: 75
  :alt: PyPI publication

.. code:: bash

  pip install ddpw

From GitHub
-----------

.. image:: https://img.shields.io/badge/github-ddpw-skyblue
  :target: https://github.com/sujaltv/ddpw
  :width: 75
  :alt: PyPI publication

.. code:: bash

  pip install git+'https://github.com/sujaltv/ddpw'

From source
------------------------

.. code:: bash

  > git clone https://github.com/sujaltv/ddpw
  > cd ddpw

  > pip install .
