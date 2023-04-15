import abc

from .artefacts import ArtefactsConfig
from .platform import PlatformConfig


class Job(object):
  r"""
  This is a template class with abstract methods to be defined by the user. This
  class provides methods to define training and evaluation procedures. Once the
  wrapper has moved the model and the dataset to the appropriate device, it
  calls :py:meth:`.train()` or :py:meth:`.evaluate()` as configured.
  """

  p_config: PlatformConfig = None
  r"""
  Platform-related configuration. This property may be used in
  :py:meth:`~Job.__call__` other methods to access the models and the datasets.

  .. admonition:: Definition not required
   :class: note

   This property need not be specified and will be automatically updated by the
   wrapper right before training or evaluation. This can be directly accessed in
   the :py:meth:`~Job.__call__` and other methods.
  """

  a_config: ArtefactsConfig = None
  r"""
  Model-related configuration. This property may be used in
  :py:meth:`~Job.__call__` other methods to access the models and the datasets.

  .. admonition:: Definition not required
   :class: note

   This property need not be specified and will be automatically updated by the
   wrapper right before training or evaluation. This can be directly accessed in
   the :py:meth:`~Job.__call__` and other methods.
  """

  @abc.abstractmethod
  def __call__(self, global_rank: int, local_rank: int):
    r"""
    .. admonition:: Definition required
      :class: important

      This method needs to be explicitly defined.

    When once the distributed data parallel setups are completed by the wrapper,
    this method is called. This method locally updates the dataset and model
    allotted for the current GPU in case of GPU- and SLURM-based platforms.

    :param int global_rank: The global rank of the device.
    :param int local_rank: Local rank of the current device.
    :raises NotImplementedError: Evaluation has not been implemented.
    """

    raise NotImplementedError
