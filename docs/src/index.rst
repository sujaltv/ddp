Introduction
^^^^^^^^^^^^

Distributed Data Parallel Wrapper (DDPW) is a lightweight Python wrapper
relevant for `PyTorch <https://pytorch.org/>`_ users. It is written in Python
3.10.

DDPW enables writing compute-intensive tasks (such as training models) without
deeply worrying about the underlying compute platform (CPU, Apple SoC, GPUs, or
SLURM (uses `Submitit <https://github.com/facebookincubator/submitit>`_)) and
instead allows specifying it simply as an argument. This considerably minimises
the need to change the code for each type of platform.

DDPW handles basic logistical tasks such as creating threads on GPUs/SLURM
nodes, setting up inter-process communication, `etc.`, and provides simple,
default utility methods to move modules to devices and get dataset samplers,
allowing the user to focus on the main aspects of the task.

Installation
============

DDPW is distributed on PyPI. The source code is available on GitHub and can be
used to manually build the package.

.. admonition:: Target platforms
   :class: warning

   This wrapper is released for all architectures but is tested only on Linux
   arch-64 and Apple SoC.

.. tab:: Wheels

    |

    PyPI
        .. code:: bash

            pip install ddpw

        .. image:: https://img.shields.io/pypi/v/ddpw
            :target: https://pypi.org/project/ddpw/
            :alt: PyPI publication

    GitHub
        .. code:: bash

            pip install git+'https://github.com/sujaltv/ddpw'

        .. image:: https://img.shields.io/badge/github-ddpw-skyblue
            :target: https://github.com/sujaltv/ddpw
            :alt: PyPI publication

.. tab:: Manual build

    |

    With ``pip``
        .. code:: bash

            > git clone https://github.com/sujaltv/ddpw
            > cd ddpw

            > pip install .

Usage
=====

.. code-block:: python
    :linenos:
    :emphasize-lines: 1,8,11,14

    from ddpw import Platform, Wrapper

    # some task
    def task(global_rank, local_rank, process_group, args):
        print(f'This is GPU {global_rank}(G)/{local_rank}(L); args = {args}') 

    # platform (e.g., 4 GPUs)
    platform = Platform(device='gpu', n_gpus=4)

    # wrapper
    wrapper = Wrapper(platform=platform)

    # start
    wrapper.start(task, ('example',))

As a decorator
______________

.. code:: python

    from ddpw import Platform, wrapper

    @wrapper(Platform(device='gpu', n_gpus=2, n_cpus=2))
    def run(a, b):
        # some task
        pass

Refer to the :ref:`API <sec:api>` for more configuration options or the :ref:`example with
MNIST <sec:mnist-example>` for an illustration.

.. toctree::
   :caption: API
   :glob:
   :hidden:
   :titlesonly:

   api/core
   api/utils

.. toctree::
   :caption: Bureau
   :glob:
   :hidden:
   :titlesonly:

   bureau/contribution
   LICENCE

