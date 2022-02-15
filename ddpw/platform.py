from enum import Enum
from typing import final
from dataclasses import dataclass

import torch.distributed as dist


@final
class Platform(Enum):
  r"""The platform on which to train or evaluate."""

  CPU = 0
  r"""The platform to run on is a CPU."""

  GPU = 1
  r"""The platform to run on is one or more GPUs."""

  SLURM = 2
  r"""The platform to run on is a SLURM-based cluster of GPU nodes."""


@final
@dataclass
class PlatformConfig(object):
  r"""
  Platform-related configurations such as the environment, communication IP
  address and port, world size, `etc.`
  """

  name: str = 'DDPW'
  r"""Name of the platform job. Used for SLURM. Default: ``DDPW``."""

  verbose: bool = True
  r"""Whether to run the wrapper in a verbose mode or not. Default: ``True``."""

  spawn_method: str = 'spawn'
  r"""The way in which to start a new process, one for each GPU. This
  corresponds to the arguments passed on to :py:meth:`set_start_method` in
  Python's (and PyTorch's) ``multiprocessing`` module. Default: ``spawn``."""

  partition: str = 'general'
  r"""Name of the partition. Used for SLURM. Default: ``general``."""

  platform: Platform = Platform.GPU
  r"""The type of platform. Default: ``Platform.GPU``."""

  n_nodes: int = 1
  r"""The total number of nodes. For training on a single cluster of GPUs, this
  property is ``1``. Default: ``1``."""

  n_gpus: int = 1
  r"""The total number of GPUs allotted in each node. Default: ``1``."""

  master_addr: str = 'localhost'
  r"""The IP address of the master GPU through which interprocess communication
  happens. Default: ``localhost``."""

  master_port: str = '1889'
  r"""The port at which IPC happens. Default: ``1889``."""

  backend: dist.Backend = dist.Backend.GLOO
  r"""The PyTorch-supported backend to used for distributed data parallel.
  Default: ``torch.distributed.Backend.GLOO``."""

  seed: int = 1889
  r"""Seed with which to initialise the various random number generators.
  Default: ``1889``."""

  timeout_min: int = 2880
  r"""Minimum timeout (in minutes) for SLURM-based jobs. Used only on SLURM
  platforms. Default: ``2880`` (two days)."""

  cpus_per_task: int = 1
  r"""Number of CPUs per task. Default: ``1``."""

  @property
  def world_size(self):
    r"""World size. This is the total number of GPUs across nodes. For SLURM,
    this is implicitly the number of GPUs allotted on each node multiplied by
    the total number of nodes. Default: ``1``."""

    return self.n_nodes * self.n_gpus

  @property
  def requires_ipc(self):
    r"""Needs communication. This property tells whether the setup requires IPC.
    IPC is not required for a single CPU or a single GPU."""

    return self.platform != Platform.CPU and self.world_size > 1
