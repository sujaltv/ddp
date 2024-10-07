<h1 align="center">DDPW</h1>

**Distributed Data Parallel Wrapper (DDPW)** is a lightweight Python wrapper
relevant for [PyTorch](https://pytorch.org/) users.

DDPW handles basic logistical tasks such as creating threads on GPUs/SLURM
nodes, setting up inter-process communication, _etc._, and provides simple,
default utility methods to move modules to devices and get dataset samplers,
allowing the user to focus on the main aspects of the task. It is written in
Python 3.10. The [documentation](https://ddpw.projects.sujal.tv) contains
details on how to use this package.

## Overview

### Installation

[![PyPI](https://img.shields.io/pypi/v/ddpw)](https://pypi.org/project/ddpw/)

```bash
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

---

###### Status

[![Publish to PyPI](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml)
[![Publish documentation](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yaml)

