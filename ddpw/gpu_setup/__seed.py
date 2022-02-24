import random

import torch
import numpy as np


def seed_generators(seed: int):
  r"""
  Seed random number generators from various packages.

  :param int seed: The seed to initialise.
  """

  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
