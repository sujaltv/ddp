import torch
import torch.distributed as dist


class Utils(object):
  verbose: bool = True
  r"""A global boolean property to specify if the wrapper must print contents or
  not."""

  @staticmethod
  def print(*args, **kwargs):
    r"""
    A custom print method that prints the contents if verbose.

    :param bool verbose: Whether to print or not. Default: ``None``.
    """

    if 'flush' not in kwargs:
      kwargs['flush'] = True

    if kwargs.get('verbose', Utils.verbose): print(*args, **kwargs)

  @staticmethod
  def all_average_gradients(model: torch.nn.Module):
    r"""
    Given a model, this method averages the gradients of the model across all
    the GPUs in the world. Copied and modified from `PyTorch Blog
    <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`_.

    :param nn.Module model: The model whose gradients are to be averaged.
    """

    world_size = float(dist.get_world_size())

    for params in model.parameters():
      if params.grad is None: continue
      dist.all_reduce(params.grad, op=dist.ReduceOp.SUM)
      params.grad /= world_size


  @staticmethod
  def optimiser_to(optimiser: torch.optim.Optimizer, device: torch.device):
    r"""
    This function offers a simple way to move all parameters of an optimiser or,
    effectively the optimiser itself, to the specified device. This method has
    been taken as is from a `solution
    <https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3>`_
    suggested on PyTorch's Discuss forum.

    :param torch.optim.Optimizer optimiser: The optimiser to move to a device.
    :param torch.device device: The device to which to move the optimiser.
    """

    for param in optimiser.state.values():
      # if any global tensors in the state dict
      if isinstance(param, torch.Tensor):
        param.data = param.data.to(device)
        if param._grad is not None:
          param._grad.data = param._grad.data.to(device)
      elif isinstance(param, dict):
        for subparam in param.values():
          if isinstance(subparam, torch.Tensor):
            subparam.data = subparam.data.to(device)
            if subparam._grad is not None:
              subparam._grad.data = subparam._grad.data.to(device)
