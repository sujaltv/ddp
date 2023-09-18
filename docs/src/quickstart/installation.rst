Installation
############

.. warning::

    This wrapper is released for all architectures but is tested only on Linux
    arch-64 and Apple SoC.

DDPW is distributed on PyPI and Anaconda. The source is available on GitHub, so
can also be manually built from the source code.

.. tab:: Binaries

    |

    Anaconda:

    .. image:: https://img.shields.io/conda/v/tvsujal/ddpw
        :target: https://anaconda.org/tvsujal/ddpw
        :width: 125
        :alt: Conda publication

    .. code:: bash

        conda install -c tvsujal ddpw


    PyPI:

    .. image:: https://img.shields.io/pypi/v/ddpw
        :target: https://pypi.org/project/ddpw/
        :width: 75
        :alt: PyPI publication

    .. code:: bash

        pip install ddpw

    GitHub:

    .. image:: https://img.shields.io/badge/github-ddpw-skyblue
        :target: https://github.com/sujaltv/ddpw
        :width: 75
        :alt: PyPI publication

    .. code:: bash

        pip install git+'https://github.com/sujaltv/ddpw'

.. tab:: Manual build

    |

    With ``pip``:

    .. code:: bash

        > git clone https://github.com/sujaltv/ddpw
        > cd ddpw

        > pip install .

    With ``conda``:

    .. code:: bash

        > git clone https://github.com/sujaltv/ddpw
        > cd ddpw/conda

        > conda-build . # generates a binary distribution
        > conda install  <distribution_path>

