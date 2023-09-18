from enum import Enum
from random import randint
from typing import List, final, Optional, Callable
from dataclasses import dataclass, field

import torch
from torch import distributed as dist

from .io import IO


@final
class Device(Enum):
    r"""The device on which to run the task."""

    CPU = 'cpu'
    r"""The device to run on is a CPU."""

    GPU = 'gpu'
    r"""The device to run on is one or more GPUs."""

    SLURM = 'slurm'
    r"""The device to run on is a cluster of GPU nodes managed by SLURM."""

    MPS = 'mps'
    r"""The device to run on is Apple SoC."""

    @staticmethod
    def from_str(device: str) -> 'Device':
        r"""
        This method returns a :py:class:`Device` object given a valid device
        string.

        :param str device: The type of the device. Supported values: ``cpu``,
            ``gpu``, ``slurm``, and ``mps`` (case insensitive).

        :returns Device: :py:class:`Device` corresponds to the device type
            string.
        
        :raises ValueError: Raises an error if the string is invalid.
        """

        match device.lower():
            case 'cpu': return Device.CPU
            case 'gpu': return Device.GPU
            case 'slurm': return Device.SLURM
            case 'mps': return Device.MPS
            case _: raise ValueError("Invalid device string specified.")


@final
@dataclass
class Platform:
    r"""
    Platform-related configurations such as the device, environment,
    communication IP address and port, world size, `etc.`
    """

    name: str = 'ddpw'
    r"""Name of the platform job. Used by SLURM. Default: ``ddpw``."""

    device: Device | str = Device.GPU
    r"""The type of device. Default: ``Device.GPU``."""

    partition: str = 'general'
    r"""Name of the partition. Used by SLURM. Default: ``general``."""

    n_nodes: int = 1
    r"""The total number of nodes (used by SLURM).  Default: ``1``."""

    n_gpus: int = 1
    r"""The number of GPUs (per node). Default: ``1``."""

    n_cpus: int = 1
    r"""The total number of CPUs (used by SLURM). Default: ``1``."""

    ram: int = 32
    r"""Total RAM (in GB) (used by SLURM). Default: ``32``."""

    spawn_method: Optional[str] = 'fork'
    r"""This property corresponds to that passed to
    :meth:`mp.set_start_method`. Default: ``fork``."""

    ipc_protocol: str = 'tcp'
    r"""IPC protocol. Accepted values: ``tcp`` and ``file``. Default:
    ``tcp``."""

    master_addr: str = 'localhost'
    r"""IPC address. Default: ``localhost``."""

    master_port: Optional[str] = str(randint(1024, 49151))
    r"""The port at which IPC happens. Default: a random port between 1024 and
    49151."""

    ipc_groups: Optional[List[List[int]]] = field(default_factory=lambda: [])

    r"""A list of list of non-overlaping global ranks of devices. If ``None``,
    every device will be its own group, and no IPC will take place. If an empty
    list is passed, all devices are grouped into one process group. Default:
    ``[]``.

    .. admonition:: Examples
        :class: note

        .. code:: python

            # no IPC between devices; each device is its own group
            platform = Platform(device='gpu', n_gpus=4, ipc_groups=None)

            # all devices under one group: default behaviour
            platform = Platform(device='gpu', n_gpus=4)
            platform = Platform(device='gpu', n_gpus=4, ipc_groups=[])

            # custom groups
            platform = Platform(device='gpu', n_gpus=4, ipc_groups=[[0, 2], [1], [3]])
            platform = Platform(device='gpu', n_gpus=4, ipc_groups=[[0, 2], [1, 3]])

    .. admonition:: Variable groups unstable
        :class: warning

        PyTorch behaviour seems to be inconsistent when using variable process
        groups. An `open bug <https://github.com/pytorch/pytorch/issues/29115>`_
        issue is on GitHub.
    """

    backend: Optional[dist.Backend] = dist.Backend.GLOO if \
        hasattr(dist, 'Backend') else None
    r"""The PyTorch-supported backend to use for distributed data parallel.
    Default: ``torch.distributed.Backend.GLOO``."""

    seed: int = 1889
    r"""Seed with which to initialise the various [pseudo]random number
    generators. Default: ``1889``."""

    timeout_min: int = 2880
    r"""Minimum timeout (in minutes) for jobs (used by SLURM). Default:
    ``2880`` (two days)."""

    slurm_additional_parameters: Optional[dict] = None
    r"""Additional SLURM parameters; this dictionary corresponds to
    ``submitit``'s ``slurm_additional_parameters`` `argument
    <https://github.com/facebookincubator/submitit/issues/23>`_. Default:
    ``None``."""

    console_logs: str = './logs'
    r"""Location of console logs (used by SLURM). Default: ``./logs``"""

    verbose: Optional[bool] = True
    r"""Whether or not to output updates during setup. Default: ``True``."""

    upon_finish: Optional[Callable] = None
    r"""A callable task to be invoked upon completion. Default: ``None``."""

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = Device.from_str(self.device)

    @property
    def world_size(self):
        r"""World size. This is the total number of GPUs across all nodes.
        Default: ``1``."""

        n_nodes = self.n_nodes if self.device == Device.SLURM else 1 
        n_gpus = min(self.n_gpus, torch.cuda.device_count()) if self.device == Device.GPU else self.n_gpus

        return n_nodes * n_gpus

    @property
    def requires_ipc(self):
        r"""Needs communication. This property determines whether or not the
        setup requires IPC. IPC is not required for a single device."""

        if self.device in [Device.CPU, Device.MPS]: return False

        if self.ipc_groups is None: return False

        if self.device == Device.GPU:
            return torch.cuda.device_count() > 1 and self.world_size > 1

        return self.world_size > 1

    def print(self):
        r"""
        This method serialised this object in a human readable format and
        prints it.
        """

        details = f"""
        \r Platform details:

        \r • Name:\t\t\t\t{self.name}
        \r • Device:\t\t\t\t{self.device.value.upper()}
        \r • CPUs:\t\t\t{self.n_cpus}
        \r • Total RAM:\t\t\t{self.ram}GB
        \r • GPUs (per node):\t\t\t{self.n_gpus} (requested)
        \r • GPUs (per node):\t\t\t{torch.cuda.device_count()} (available)
        \r • PyTorch backend:\t\t\t{self.backend}
        \r • Seed (random number generators):\t{self.seed}
        """

        if self.device == Device.SLURM or True:
            details += f"""\
            \r • Nodes:\t\t\t\t{self.n_nodes}
            \r • SLURM partition:\t\t\t{self.partition}
            \r • SLURM timeout:\t\t\t{self.timeout_min} minutes
            """

        if not self.requires_ipc:
            details += f"""\
            \r • Spawn method:\t\t\t{self.spawn_method}
            \r • Master IP address:\t\t\t{self.master_addr}
            \r • Master port:\t\t\t\t{self.master_port}
            """

        details += f"""\
        \r • World size:\t\t\t\t{self.world_size}
        """

        IO.print(details)

