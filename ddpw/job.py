import os
import abc
from enum import Enum
from typing import final, Optional, Callable
from dataclasses import dataclass

import torch
from torch.utils import data

from .utils import Utils
from .platform import PlatformConfig
from .artefacts import ArtefactsConfig


@final
class JobMode(Enum):
  r"""Modes of running the task."""

  TRAIN = 0
  r"""The task is to train."""

  RESUME = 1
  r"""The task is to resume training from a checkpoint."""

  EVALUATE = 2
  r"""The task is to evaluate a model checkpoint (`i.e.`, a trained model)."""


@final
@dataclass
class JobConfig(object):
  job_type: JobMode = JobMode.TRAIN
  r"""Type of job. Default: ``JobMode.TRAIN``."""

  start_at: int = 0
  r"""
  Epoch number from which to start training. Default: ``0``.

  Setting this to a negative value skips model/state restoration in case of
  testing or restoration. This is useful if there it no model to perform the
  task on, _i.e._, this is useful for running jobs that are to be done on the
  GPU that are not necessarily training or evaluating a model.
  """

  epochs: int = 25
  r"""Epoch number at which to stop training. Default: ``25``."""

  checkpoint_name_prefix: str = 'ckpt'
  r"""Prefix for saving the checkpoint. Default: ``ckpt``."""

  checkpoint_path: str = './checkpoint'
  r"""Location at which to store the checkpoint. Default: ``./checkpoint``."""

  console_logs_path: str = './logs/console_logs'
  r"""Location at which to store console logs. (`e.g.`, logs generated by
    SLURM). Default: ``./logs/console_logs``."""

  training_logs_path: str = './logs/training_logs'
  r"""Location at which to store training logs (`e.g.`, logs from Tensorboard).
  Default: ``./logs/training_logs``."""

  save_every: int = 5
  r"""Save a checkpoint every few epochs. If 0, ignored. Default: ``5``."""

  upon_finish: Optional[Callable] = None
  r"""A function to be called upon finishing the job. Default: ``None``."""

  def print(self):
    r"""
    This method prints this object in a readable format.
    """

    Utils.print('Job details:')
    Utils.print(f' • Job type:                            {self.job_type}')
    Utils.print(f' • Starting epoch:                      {self.start_at}')
    Utils.print(f' • Epochs:                              {self.epochs}')
    Utils.print(' • State/checkpoint save location:      ' +
                f'{self.checkpoint_path}')
    Utils.print(' • State/checkpoint name prefix:        ' +
                f'{self.checkpoint_name_prefix}')
    Utils.print(' • Console logs path:                   ' +
                f'{self.console_logs_path}')
    Utils.print(' • Training logs path:                  ' +
                f'{self.training_logs_path}')
    Utils.print(' • Save state/models every...           ' +
                f'{self.save_every} epochs')


@dataclass
class Job(object):
  r"""
  This is a template class with abstract methods to be defined by the user. This
  class provides methods to define training and evaluation procedures. Once the
  wrapper has moved the model and the dataset to the appropriate device, it
  calls :py:meth:`.train()` or :py:meth:`.evaluate()` as configured.
  """

  p_config: PlatformConfig = None
  r"""
  Platform-related configuration. This property may be used in the training
  and evaluation methods to access platform-related information such as if the
  platform is on CUDA, how big the world is, whether synchronisation across
  devices is needed, `etc`.

  .. admonition:: Definition not required
   :class: note

   This property need not be specified by the user and will be automatically
   updated by the wrapper right before training or evaluation. This can be
   directly accessed in the :py:meth:`~Job.train` and
   :py:meth:`~Job.evaluate` methods.
  """

  j_config: JobConfig = None
  r"""
  Training-related configurations. This property may be used in the training
  and evaluation methods to access training-specific aspects such as the number
  of training epochs, epoch interval to store the training state, `etc`.
  """

  a_config: ArtefactsConfig = None
  r"""
  Model-related configuration. This property may be used in the training and
  evaluation methods to access models, datasets, optimisation stragegy (the
  optimiser) `etc.`

  .. admonition:: Definition not required
   :class: note

   This property need not be specified by the user and will be automatically
   updated by the wrapper right before training or evaluation. This can be
   directly accessed in the :py:meth:`~Job.train` and
   :py:meth:`~Job.evaluate` methods.
  """

  @abc.abstractmethod
  def train(self, global_rank: int, local_rank: int):
    r"""
    .. admonition:: Definition required
      :class: important

      This method needs to be explicitly defined by the user.

    This method provides definition for the training procedure.

    :param int global_rank: Global rank of the current device.
    :param int local_rank: Local rank of the current device.

    :raises NotImplementedError: Training has not been implemented.
    """

    raise NotImplementedError

  @abc.abstractmethod
  def evaluate(self, global_rank: int, local_rank: int,
               dataset: data.DataLoader):
    r"""
    .. admonition:: Definition required
      :class: important

      This method needs to be explicitly defined by the user.

    This method provides definition for the evaluation procedure.

    :param int global_rank: Global rank of the current device.
    :param int local_rank: Local rank of the current device.
    :param data.DataLoader dataset: The dataset to use for evaluation.

    :raises NotImplementedError: Evaluation has not been implemented.
    """

    raise NotImplementedError

  def save_state(self, epoch: int):
    r"""
    This method saves the state of the training, such as the model parameters,
    optimiser gradients, the current epoch, `etc`., and can be called at every
    few epochs (as specified in :py:attr:`.JobConfig.save_every`) or as
    needed. Override this method to save more information. The state so saved is
    used by the :py:meth:`.restore_state` method that is called to resume from
    a checkpoint or evaluate a model.

    :param int epoch: The epoch number at which to save the training state.
    """

    checkpoint = {
      'stopped_at': epoch,
      'model': self.a_config.model.state_dict(),
      'optimiser': self.a_config.optimiser.state_dict()
    }
    torch.save(checkpoint, os.path.join(self.j_config.checkpoint_path,
                          f'{self.j_config.checkpoint_name_prefix}_{epoch}.pt'))

  def restore_state(self, resume_at: int):
    r"""
    Restore training from a saved state.

    :param int resume_at: The epoch checkpoint whence to resume training.
    """

    filename = f'{self.j_config.checkpoint_name_prefix}_{resume_at}.pt'
    file_path = os.path.join(self.j_config.checkpoint_path, filename)
    print(file_path)
    assert os.path.isfile(file_path)

    Utils.print(f'Loading model at {file_path}.')
    checkpoint = torch.load(file_path)
    self.j_config.start_at = checkpoint['stopped_at']
    self.a_config.model.load_state_dict(checkpoint['model'])
    self.a_config.optimiser.load_state_dict(checkpoint['optimiser'])

  def __call__(self, global_rank: int, local_rank: int):
    r"""
    When once the distributed data parallel setups are completed by the wrapper,
    this method is called. This method locally updates the dataset and model
    allotted for the current GPU in case of GPU- and SLURM-based platforms.

    :param int global_rank: The global rank of the device.
    :param int local_rank: Local rank of the current device.
    """

    Utils.print(
      f'[Device {global_rank}] Copying model parameters to the optimiser.')
    if self.a_config.optimiser_loader is not None:
      self.a_config.optimiser = self.a_config.optimiser_loader(
        self.a_config.model)

    # if this task is resumption from or evaluation of a saved model, load it
    if self.j_config.job_type in [JobMode.RESUME, JobMode.EVALUATE]:
      Utils.print(f'[Device {global_rank}] Model load setup underway.')
      if self.j_config.start_at >= 0:
        self.restore_state(self.j_config.start_at)

    # whether to training (or resumption) or evaluate
    if self.j_config.job_type in [JobMode.TRAIN, JobMode.RESUME]:
      self.train(global_rank, local_rank)
    else:
      self.evaluate(global_rank, local_rank, self.a_config.test_set)
