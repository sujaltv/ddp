from typing import Optional, Dict

from torch.utils import data
from torch.utils.data import DistributedSampler, DataLoader

from ..utils import Utils
from ..artefacts import ArtefactsConfig
from ..platform import Platform, PlatformConfig


def sampler(dataset: data.Dataset, world_size: int, global_rank: int,
            is_cpu: bool = False, data_loader_args: Optional[Dict] = {}):
  r"""
  This function selects a portion of the original dataset shared by other
  devices. If the device being trained on is a CPU, no sharing is necessary.

  :param data.Dataset dataset: The dataset from which to sample for the current
    device.
  :param int world_size: World size.
  :param int global_rank: Global rank of the current GPU.
  :param bool is_cpu: Specifies if the device is a CPU or Apple M1. Default:
    `False`.

  :returns data.Dataset: A dataloader with portion of the dataset selected for
      the current process.
  """

  smplr = None if is_cpu else DistributedSampler(dataset, world_size,
                                                 rank=global_rank)
  return DataLoader(dataset, sampler=smplr, pin_memory=True, **data_loader_args)


def dataset_setup(global_rank: int, p_config: PlatformConfig,
                  a_config: ArtefactsConfig):
  r"""
  This function selects a portion of the training dataset for validation if
  specified. In case of training/testing on multiple devices, it then allocates
  a portion each of the training, test, and validation datasets (if available)
  to the current device and returns a dataloader for each.

  :param int global_rank: Global rank of the current GPU.
  :param PlatformConfig p_config: Platform configurations.
  :param ArtefactsConfig a_config: Job configurations.

  :returns tuple: A triplet of dataloaders for the training, validation, and
      test datasets respectively.
  """

  train_loader = None
  validation_loader = None
  test_loader = None

  is_cpu = p_config.platform in [Platform.CPU, Platform.MPS]

  args = (p_config.world_size, global_rank, is_cpu, a_config.dataloader_args)

  # if a train split is available
  if (train_set := a_config.train_set) is not None:
    train_loader = sampler(train_set, *args)
    Utils.print(f'[Device {global_rank}] ' +
                f'Received test set portion: {len(train_loader)}')

  # if a validation split is available
  if (validation_set := a_config.validation_set) is not None:
    validation_loader = sampler(validation_set, *args)
    Utils.print(f'[Device {global_rank}] ' +
                f'Received test set portion: {len(validation_loader)}')

  # if a test split is available
  if (test_set := a_config.test_set) is not None:
    test_loader = sampler(test_set, *args)
    Utils.print(f'[Device {global_rank}] ' +
                f'Received test set portion: {len(test_loader)}')

  return train_loader, validation_loader, test_loader
