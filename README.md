<h1 align="center">DDPW</h1>

<div align="center">

[![AWS S3](https://img.shields.io/badge/Documentation-blue?link=https://ddpw.projects.sujal.tv)](https://ddpw.projects.sujal.tv)
[![Conda](https://img.shields.io/conda/v/tvsujal/ddpw)](https://anaconda.org/tvsujal/ddpw)
[![PyPI](https://img.shields.io/pypi/v/ddpw)](https://pypi.org/project/ddpw/)

[![Publish to Anaconda](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yaml)
[![Publish to PyPI](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml)
[![Publish documentation](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yaml)

</div>

---

**Distributed Data Parallel Wrapper (DDPW)** is a lightweight wrapper that
scaffolds PyTorch's (Distributed Data) Parallel.

This code is written in Python 3.10. The [DDPW
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
```

