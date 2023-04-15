from typing import final, Optional, Dict
from dataclasses import dataclass

from torch import nn
from torch.utils import data

from .utils import Utils


@final
@dataclass
class ArtefactsConfig(object):
  r"""Configurations relating to the dataset and the model."""

  train_set: data.Dataset = None
  r"""The dataset to use for training. Default: ``None``."""

  validation_set: data.Dataset = None
  r"""The dataset to use for validation. Default: ``None``."""

  test_set: data.Dataset = None
  r"""The dataset to use for evaluation. Default: ``None``."""

  dataloader_args: Optional[Dict] = None
  r"""Arguments to be passed to the dataloader. Default: ``None``."""

  model: nn.Module = None
  r"""An instance of the model to train. Default: ``None``."""

  model_has_batch_norm: bool = False
  r"""Specifies if the model to be trained has batch normalisation in it.
  Default: ``False``."""

  def print(self):
    r"""
    This method prints this object in a readable format.
    """

    Utils.print('Artefacts details:')
    Utils.print(' • Train split:                         ' +
                f'{len(self.train_set) if self.train_set is not None else 0}')
    Utils.print(' • Validation split:                    ' +
                f'{len(self.validation_set) if self.validation_set else 0}')
    Utils.print(' • Test dataset:                        ' +
                f'{len(self.test_set) if self.test_set is not None else 0}')
    Utils.print(' • Model has batch normalisation?       ' +
                f'{"Yes" if self.model_has_batch_norm else "No"}')
