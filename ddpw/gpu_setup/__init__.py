import os

import torch
import torch.distributed as dist

from ..utils import Utils
from ..job import Job
from ..artefacts import ArtefactsConfig
from ..platform import Platform, PlatformConfig

from .__model import model_setup as __model_setup
from .__dataset import dataset_setup as __dataset_setup
from .__seed import seed_generators as __seed_generators


def init_process(global_rank: int, local_rank: int, run: Job,
                 p_config: PlatformConfig, a_config: ArtefactsConfig):
  r"""
  This function is called at the beginning of the process in each device
  (CPU/GPU). Depending on the needs, this function establishes DDP communication
  protocols, seeds random number generators, selects a portion of the data for
  the current GPU, moves a copy of the model to the device, and starts the given
  task.

  :param int global_rank: Global rank of the GPU.
  :param int local_rank: Local rank of the GPU.
  :param Job run: The task to run when once the setup is complete.
  :param PlatformConfig p_config: Platform-related configurations.
  :param ArtefactsConfig a_config: Model-related configurations.
  """

  Utils.print(f'[Device {global_rank}] Initialising the process.')

  if p_config.requires_ipc:
      os.environ['MASTER_ADDR'] = p_config.master_addr
      os.environ['MASTER_PORT'] = f'{p_config.master_port}'

      Utils.print(f'[Device {global_rank}] IPC at ' +
            f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}.')

      dist.init_process_group(backend=p_config.backend, rank=global_rank,
                              world_size=p_config.world_size)

  # 0. Seed random number generators
  Utils.print(f'[Device {global_rank}] Seeding random number generators.')
  __seed_generators(p_config.seed)

  # 1. organise the dataset into splits
  Utils.print(f'[Device {global_rank}] ' +
              f'Selecting portion of the dataset to local GPU {local_rank}.')
  a_config.train_set, a_config.validation_set, a_config.test_set = \
    __dataset_setup(global_rank, p_config, a_config)

  # 2. Set up the model on the current device
  if p_config.platform not in [Platform.CPU, Platform.MPS]:
    Utils.print(f'[Device {global_rank}] ' +
                f'Copying the model to local GPU {local_rank}.')
    if (a_config.model is not None):
      a_config.model = __model_setup(a_config.model, local_rank,
                                      a_config.model_has_batch_norm,
                                      p_config.requires_ipc)
  elif p_config.platform == Platform.MPS:
    a_config.model.to(torch.device('mps'))

  # 3. Wait for all processes to synchronise and then start the task
  Utils.print(f'[Device {global_rank}] Training model on device {local_rank}.')
  if p_config.requires_ipc:
    dist.barrier()
  run.a_config = a_config
  run.p_config = p_config

  Utils.print(f'[Device {global_rank}] All setup finished.')
  run(global_rank, local_rank)

  if p_config.requires_ipc:
    dist.destroy_process_group()

  Utils.print(f'[Device {global_rank}] Tasks on device complete.')
