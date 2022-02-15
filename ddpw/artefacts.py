import abc
from typing import final
from dataclasses import dataclass

from torch import nn, optim
from torch.utils import data
from torch.nn.modules.loss import _Loss as Loss


class OptimiserLoader(object):
  r"""This class contains the optimiser with required configurations."""

  @abc.abstractmethod
  def __call__(self, model: nn.Module) -> optim.Optimizer:
    r"""
    .. admonition:: Definition required
      :class: important

      This method needs to be explicitly defined by the user.

    This method receives the model whose parameters are to be loaded into the
    optimiser.

    :param nn.Module model: The model whose parameters are to be loaded into the
      optimiser.
    :returns optim.Optimizer: The desired optimiser with model parameters loaded
    :raises NotImplementedError: Method not implemented.
    """

    raise NotImplementedError


@final
@dataclass
class ArtefactsConfig(object):
  r"""Configurations relating to the dataset and the model."""

  dataset_root: str = '/data/dataset'
  r"""Location of the dataset to be used to training or evaluation."""

  train_set: data.DataLoader = None
  r"""The dataset to use for training. Default: ``None``."""

  test_set: data.DataLoader = None
  r"""The dataset to use for evaluation. Default: ``None``."""

  validation_set: data.DataLoader = None
  r"""The dataset to use for validation. Default: ``None``."""

  validation_percentage: float = 20
  r"""The percentage of training dataset to be used for validation. If this
  property has a value of ``0``, it is assumed that no validation is required.
  This property is ignored for evaluation. Range: ``0`` to ``50``, inclusive.
  Default: ``20``. Range enforced by the :py:attr:`.needs_validation`
  property."""

  batch_size: int = 64
  r"""Batch size for training and testing. Default: ``64``."""

  model: nn.Module = None
  r"""An instance of the model to train. Default: ``None``."""

  model_has_batch_norm: bool = False
  r"""Specifies if the model to be trained has batch normalisation in it.
  Default: ``False``."""

  loss_fn: Loss = None
  r"""An instance of the :class:`Loss` module. Default: ``None``."""

  optimiser_config: OptimiserLoader = None
  r"""Optimiser setup to be passed by the user. Default: ``None``."""

  optimiser: optim.Optimizer = None
  r"""
  .. admonition:: Definition not required
   :class: note

   This property need not be specified by the user and will be automatically
   updated by the wrapper right before training. This can be directly in the
   :py:meth:`.Trainer.train` method.

  The wrapper loads model parameters into the optimiser with the specified
  configs in :py:attr:`optimiser_config` and updates this parameter. This could
  be accessed in :py:meth:`.Trainer.train` or :py:meth:`.Trainer.evaluate`.
  Default: ``None``.
  """

  @property
  def needs_validation(self):
    r"""
    This property tells if the current configuration requires validation or not.

    :returns bool: Whether validation is required or not.
    """

    return 0 < self.validation_percentage <= 50
