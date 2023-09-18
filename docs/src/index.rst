DDPW
####

**Distributed Data Parallel Wrapper (DDPW)** is a lightweight PyTorch-based
wrapper that makes it easy to run tasks on various compute platforms. DDPW
handles scaffolding tasks like creating threads on GPUs/nodes, setting up
inter-process communication, `etc.`, and provides simple, default utility
methods to move modules to devices and get dataset samplers, allowing the user
to focus on the main aspects of the task.

This wrapper offers setups for different compute platforms including:

#. CPU/Apple SoC,
#. one or more CUDA-supported GPUs, and
#. SLURM-managed nodes (using `Submitit <https://github.com/facebookincubator/submitit>`_)

Example
=======

.. code-block:: python
    :linenos:
    :emphasize-lines: 1,8,11,14

    from ddpw import Platform, Wrapper

    # some task
    def task(global_rank, local_rank, group, args):
        print(f'This is GPU {global_rank}(G)/{local_rank}(L); args = {args}') 

    # platform (e.g., 4 GPUs)
    platform = Platform(device='gpu', n_gpus=4)

    # wrapper
    wrapper = Wrapper(platform=platform)

    # start
    wrapper.start(task, ('example',))

Refer to :ref:`the platform API <platform api>` for more platform-related
configurations and the :ref:`example with MNIST <MNIST example>` for a more
detailed example.

.. toctree::
   :caption: Introduction
   :glob:
   :hidden:
   :titlesonly:

   quickstart/installation

.. toctree::
   :caption: API
   :glob:
   :hidden:
   :titlesonly:

   api/wrapper
   api/platform
   api/functional

.. toctree::
   :caption: Miscellaneous
   :glob:
   :hidden:
   :titlesonly:

   quickstart/mnist

.. toctree::
   :caption: Development
   :glob:
   :hidden:
   :titlesonly:

   development/contribution
   LICENCE

