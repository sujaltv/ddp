DDPW
####

The **Distributed Data Parallel wrapper** is a minimal PyTorch-based wrapper
written in Python 3.8 that makes it easy to run deep learning models on multiple
nodes and GPUs as well as on a single CPU.

This is a "DDP" wrapper; it supports data parallelism across GPUs, not model
parallelism. To learn more about DDP, refer to PyTorch documentation on DDP or
`this nice Medium article
<https://medium.com/mlearning-ai/distributed-data-parallel-with-slurm-submitit-pytorch-168c1004b2ca>`_.

Features
========

1. **Multiple platforms**. This wrapper allows models to be trained on:

   a. a single CPU,
   b. a single GPU,
   c. multiple cluster-less GPUs, and
   d. SLURM-based multi-node (clustered) single/multiple GPUs (by using `Submitit <https://github.com/facebookincubator/submitit>`_)

2. **Flexible training and evaluation**. This wrapper allows easy training, pausing, resuming, and evaluation

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
   api/trainer
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

