DDPW
####

**Distributed Data Parallel Wrapper (DDPW)** is a minimal PyTorch-based wrapper
that makes it easy to run deep learning models on various compute platforms.
DDPW handles scaffolding tasks like creating threads on GPUs/nodes, moving the
models and datasets to devices, setting up inter-process communication, `etc.`,
and allows the user to focus on the main aspects of modelling and training.

DDPW includes support for training on CPUs, GPUs (CUDA and Apple M1 SoC) and in
the SLURM environment.

Features
========

1. **Multiple platforms**. This wrapper allows models to be trained different
   compute platforms:

   a. CPU,
   b. one or more GPUs (CUDA and Apple M1 SoC), and
   c. SLURM (by using `Submitit
      <https://github.com/facebookincubator/submitit>`_)

2. **Non-training tasks**. Tasks that are not essentially of the model-based or
   training-based paradigms but those that simply need to be run across devices
   can also be executed.

Example
=======

.. code-block:: python
    :linenos:
    :emphasize-lines: 12,15,18

    from ddpw.platform import Platform, PlatformConfig
    from ddpw.artefacts import ArtefactsConfig
    from ddpw.job import JobConfig, JobMode
    from ddpw.wrapper import Wrapper

    from src import MyDataset, MyModel, MyTrainer

    # datasets
    train_set = MyDataset(train=True)

    # configure the platform
    p_config = PlatformConfig(platform=Platform.GPU, n_gpus=4, cpus_per_task=2)

    # configure the artefacts (model, dataset, optimiser, etc.)
    a_config = ArtefactsConfig(train_set=train_set, model=MyModel())

    # call the job
    Wrapper(p_config, a_config).start(MyTrainer())

Refer to the :ref:`example with MNIST <MNIST example>` to see how ``MyDataset``,
``MyModel``, and ``MyTrainer`` are implemented.

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
   api/job
   api/types
   api/utils

.. toctree::
   :caption: Contribution
   :glob:
   :hidden:
   :titlesonly:

   contribution/source
   contribution/documentation

.. toctree::
   :caption: Miscellaneous
   :glob:
   :hidden:
   :titlesonly:

   quickstart/mnist

.. toctree::
   :caption: Bureau
   :glob:
   :hidden:
   :titlesonly:

   LICENCE
