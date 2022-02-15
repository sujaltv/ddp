from torch.utils import data
from torch.utils.data import DistributedSampler, DataLoader, random_split

from ..utils import Utils
from ..artefacts import ArtefactsConfig
from ..platform import Platform, PlatformConfig


def sampler(dataset: data.Dataset, world_size: int, global_rank: int,
            batch_size: int, is_cpu: bool = False):
  r"""
  This function creates a sampler for the given process (i.e., rank) if
  necessary (i.e., if not CPU) and creates a dataloder from the sampler.

  :param data.Dataset dataset: The dataset from which to sample.
  :param int world_size: The world size.
  :param int global_rank: Current GPU's global rank.
  :param int batch_size: Batch size.
  :param bool is_cpu: Is the dataset for CPU. Default: `False`.

  :returns data.Dataset: The dataset for the current process.
  """

  smplr = None if is_cpu else DistributedSampler(dataset, world_size,
                                                 rank=global_rank)
  result = DataLoader(dataset, batch_size, sampler=smplr, pin_memory=True)

  return result


def dataset_setup(global_rank: int, p_config: PlatformConfig,
                  artefacts: ArtefactsConfig):
  r"""
  This function selects a portion of the dataset for the current GPU (i.e., the
  rank) and splits it into train and validation in case training.

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
