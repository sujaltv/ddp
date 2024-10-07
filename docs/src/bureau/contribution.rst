Contribution
#############

.. _sec:dependencies:

.. tab:: Development

    |

    All packages are installed with ``conda`` or ``pip`` and listed in
    ``environment.yml`` and ``requirements.txt``, respectively, and can be
    installed with the following commands:

    .. code-block:: bash

        # with conda
        conda env create --file environment.yml
        conda activate ddpw

        # with pip
        pip install -r requirements.txt

    An existing environment can be update with ``conda env update``:

    .. code-block:: bash

        conda env update --file environment.yml

.. tab:: Documentation

    |

    This documentation is written in Sphinx with Furo; the dependencies can be
    installed with the following commands:

    .. code-block:: bash

        # with conda
        > conda env create --file environment.yml # root folder
        > conda activate ddpw
        > pip install -r requirements.txt

    The ``Make`` file compiles and builds the documentation into HTML and the
    following commands can be used to do so:

    .. code-block:: bash

        > cd docs
        > make clean html

