# DDPW

[![AWS S3](https://img.shields.io/badge/documentation-sphinx-blue?link=https://ddpw.projects.sujal.tv)](https://ddpw.projects.sujal.tv)
[![Conda](https://img.shields.io/conda/v/tvsujal/ddpw)](https://anaconda.org/tvsujal/ddpw)
[![PyPI](https://img.shields.io/pypi/v/ddpw)](https://pypi.org/project/ddpw/)

[![Publish documentation to AWS S3](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yaml)
[![Publish to Anaconda](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yaml)
[![Publish to PyPI](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml)

---

**Distributed Data Parallel Wrapper (DDPW)** is created as a utility package to
encapsulate the scaffolding for PyTorch's Distributed Data Parallel.

This code is written in Python. The [DDPW
documentation](https://ddpw.projects.sujal.tv) contains details on how to use
this package.

## Overview

### Installation

```bash
conda install -c tvsujal ddpw # with conda
pip install ddpw # with pip from PyPI
```

### Usage

```python
from ddpw.platform import Platform, PlatformConfig
from ddpw.artefacts import ArtefactsConfig
from ddpw.job import JobConfig, JobMode
from ddpw.wrapper import Wrapper

from src import MyDataset, MyModel, MyOptimiser, MyTrainer

# datasets
train_set = MyDataset(train=True)

# configure the platform
p_config = PlatformConfig(platform=Platform.GPU, n_gpus=4, cpus_per_task=2)

# configure the artefacts (model, dataset, optimiser, etc.)
a_config = ArtefactsConfig(train_set=train_set, model=MyModel())

# call the job
Wrapper(p_config, a_config).start(MyTrainer())
```
