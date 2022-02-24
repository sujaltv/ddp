from torch.utils import data
from torch.utils.data import DistributedSampler, DataLoader, random_split

from ..utils import Utils
from ..artefacts import ArtefactsConfig
from ..platform import Platform, PlatformConfig


def sampler(dataset: data.Dataset, world_size: int, global_rank: int,
            batch_size: int, is_cpu: bool = False):
  r"""
  This function selects a portion of the original dataset shared by other
  devices. If the device being trained on is a CPU, no sharing is necessary.

  :param data.Dataset dataset: The dataset from which to sample for the current
      device.
  :param int world_size: World size.
  :param int global_rank: Global rank of the current GPU.
  :param int batch_size: Batch size.
  :param bool is_cpu: Specifies if the device is a CPU. Default: `False`.

  :returns data.Dataset: A dataloader with portion of the dataset selected for
      the current process.
  """

  smplr = None if is_cpu else DistributedSampler(dataset, world_size,
                                                 rank=global_rank)
  return DataLoader(dataset, batch_size, sampler=smplr, pin_memory=True)


def dataset_setup(global_rank: int, p_config: PlatformConfig,
                  artefacts: ArtefactsConfig):
  r"""
  This function selects a portion of the training dataset for validation if
  specified. In case of training/testing on multiple devices, it then allocates
  a portion each of the training, test, and validation datasets (if available)
  to the current device and returns a dataloader for each.

  :param int global_rank: Global rank of the current GPU.
  :param PlatformConfig p_config: Platform configurations.
  :param ArtefactsConfig artefacts: Job configurations.

  :returns tuple: A triplet of dataloaders for the training, validation, and
      test datasets respectively.
  """

  train_loader = None
  val_loader = None
  test_loader = None

  is_cpu = p_config.platform == Platform.CPU
  batch_size = artefacts.batch_size

  args = (p_config.world_size, global_rank, batch_size, is_cpu)

  # if the training dataset is provided
  if (train_set := artefacts.train_set) is not None:

    # if requested to set aside a portion of the training set for validation
    if artefacts.needs_validation:
      dataset_size = len(train_set)
      v_size = (dataset_size * artefacts.validation_percentage) // 100
      t_size = dataset_size - v_size
      Utils.print(
        f'\tTrain size = {t_size}; validation size = {v_size}.')
      [train_set, val_set] = random_split(train_set, [t_size, v_size])
      val_loader = sampler(val_set, *args)

    train_loader = sampler(train_set, *args)

  # if the test dataset is provided
  if (test_set := artefacts.test_set) is not None:
    Utils.print(f'\tTest size  {len(test_set)}.')
    test_loader = sampler(test_set, *args)

  return train_loader, val_loader, test_loader
