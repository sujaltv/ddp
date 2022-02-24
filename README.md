# DDPW

[![AWS S3](https://img.shields.io/badge/documentation-sphinx-blue?link=https://ddpw.projects.sujal.tv)](https://ddpw.projects.sujal.tv)
[![Conda](https://img.shields.io/conda/v/tvsujal/ddpw)](https://anaconda.org/tvsujal/ddpw)
[![PyPI](https://img.shields.io/pypi/v/ddpw)](https://pypi.org/project/ddpw/)

[![Publish documentation to AWS S3](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yaml)
[![Publish to Anaconda](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yaml)
[![Publish to PyPI](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml)

---

The Distributed Data Parallel wrapper (DDPW) is created as a utility package to
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
  from ddpw.platform import PlatformConfig
  from ddpw.artefacts import ArtefactsConfig
  from ddpw.job import JobConfig
  from ddpw.wrapper import Wrapper

  from src import MyTrainer

  p = PlatformConfig(...)
  a = ArtefactsConfig(...)
  t = JobConfig(...)

  d = Wrapper(p, a)
  j = MyTrainer(t)

  d.start(j)
```
