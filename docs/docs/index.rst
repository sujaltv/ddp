DDPW
####

**Distributed Data Parallel Wrapper (DDPW)** is a minimal PyTorch-based wrapper
that makes it easy to run deep learning models on various compute platforms.
DDPW handles scaffolding tasks like creating threads on GPUs/nodes, moving the
models and optimisers to devices, setting up inter-process communication, `etc.`
and allows the user to focus on the main aspects of modelling and training.

DDPW includes support for training on CPUs, GPUs (CUDA and Apple M1 SoC) and in
the SLURM environment.

Features
========

1. **Multiple platforms**. This wrapper allows models to be trained different compute platforms:

   a. CPU,
   b. one or more GPUs (CUDA and Apple M1 SoC), and
   c. SLURM (by using `Submitit <https://github.com/facebookincubator/submitit>`_)

2. **Flexible training and evaluation**. This wrapper allows easy training,
   pausing, resuming, and evaluating of models

3. **Non-training tasks**. Tasks that are not essentially of the model-based or
   training-based paradigms but those that simply need to be run across devices
   can also be executed.

.. toctree::
   :caption: Introduction
   :glob:
   :hidden:
   :titlesonly:

   quickstart/installation
   quickstart/example

.. toctree::
   :caption: API
   :glob:
   :hidden:
   :titlesonly:

   api/wrapper
   api/job
   api/optimiser
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

