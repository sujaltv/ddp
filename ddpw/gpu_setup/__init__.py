import os

import torch.distributed as dist

from ..utils import Utils
from ..trainer import Trainer
from ..artefacts import ArtefactsConfig
from ..platform import Platform, PlatformConfig

from .__model import model_setup as __model_setup
from .__dataset import dataset_setup as __dataset_setup
from .__seed import seed_generators as __seed_generators


def init_process(global_rank: int, local_rank: int, run: Trainer,
                 p_config: PlatformConfig, artefacts: ArtefactsConfig):
  r"""
  This function is called at the beginning of the process in each GPU. This
  function establishes DDP communication protocols, selects a portion of the
  data for the current GPU, moves the model to the device, and starts the given
  task.

  :param int global_rank: The global rank of the GPU.
  :param int local_rank: The local rank of the GPU.
  :param Job run: The task to run when once the setup is complete.
  :param PlatformConfig p_config: Platform-related configurations.
  :param ArtefactsConfig artefacts: Model-related configurations.
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
  artefacts.train_set, artefacts.validation_set, artefacts.test_set = \
    __dataset_setup(global_rank, p_config, artefacts)

  # 2. Set up the model on the current device
  if p_config.platform != Platform.CPU:
    Utils.print(f'[Device {global_rank}] ' +
                f'Copying the model to local GPU {local_rank}.')
    artefacts.model = __model_setup(artefacts.model, local_rank,
                                    artefacts.model_has_batch_norm,
                                    p_config.requires_ipc)

  # 3. Wait for all processes to synchronise and then start the task
  Utils.print(f'[Device {global_rank}] Training model on device {local_rank}.')
  if p_config.requires_ipc:
    dist.barrier()
  run.p_config = p_config
  run.artefacts = artefacts

  Utils.print(f'[Device {global_rank}] All setup finished.')
  run(global_rank)

  if p_config.requires_ipc:
    dist.destroy_process_group()

  Utils.print('Tasks on device complete.')
