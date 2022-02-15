Example with MNIST
##################
.. _MNIST example:

1. Custom dataset
=================

.. code-block:: python

  # src/dataset.py

  from torchvision import transforms as T
  from torchvision.datasets.mnist import MNIST


  class CustomDataset(MNIST):
    def __init__(self, root: str, train: bool = True, download: bool = False):
      transforms = T.Compose([
        T.ToTensor(),
        T.Resize((28,28)),
        T.Normalize((0.1307,), (0.3081,))
      ])

      super(CustomDataset, self).__init__(root, train, transforms,
                                          download=download).__init__()


2. Custom model
===============

.. code-block:: python

  # src/model.py

  import torch
  from torch.nn import functional as F


  class CustomModel(torch.nn.Module):
    def __init__(self):
      super(CustomModel, self).__init__()

      self.conv1 = torch.nn.Conv2d(1, 32, (3,3))
      self.conv2 = torch.nn.Conv2d(32, 64, (3,3))

      self.bn = torch.nn.BatchNorm2d(64)

      self.drop1 = torch.nn.Dropout2d(0.25)
      self.drop2 = torch.nn.Dropout2d(0.5)

      self.fc1 = torch.nn.Linear(9216, 128)
      self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))

      x = self.bn(x)

      x = self.drop1(F.max_pool2d(x, 2))

      x = x.flatten(1)
      x = F.relu(self.fc1(x))

      x = self.drop2(x)
      x = self.fc2(x)

      return F.log_softmax(x, 1)


3. Custom loss
==============

.. code-block:: python

  # src/loss.py

  from torch.nn import NLLLoss

  class CustomLoss(NLLLoss):
    pass


4. Custom optimiser
===================
.. _MNIST custom optimiser:

.. code-block:: python

  # src/optimiser.py

  import torch
  from ddpw.artefacts import OptimiserLoader


  class CustomOptimiser(OptimiserLoader):
    def __init__(self, lr=0.1):
      self.lr = lr

    def __call__(self, model: torch.nn.Module) -> torch.optim.Optimizer:
      return torch.optim.Adadelta(params=model.parameters(), lr=self.lr)


5. Custom trainer
=================
.. _MNIST custom trainer:

.. code-block:: python

  # src/train.py

  import torch
  from torch.utils import data
  import torch.distributed as dist

  from ddpw.utils import Utils
  from ddpw.platform import Platform
  from ddpw.trainer import Trainer, TrainingConfig


  class CustomTrainer(Trainer):
    def __init__(self, t_config: TrainingConfig):
      super(CustomTrainer, self).__init__(t_config=t_config)

    def train(self, global_rank: int):
      train_set = self.artefacts.train_set

      # for every epoch
      for e in range(self.t_config.start_at, self.t_config.epochs):
        self.artefacts.model.train()

        training_loss = torch.Tensor([0])
        training_accuracy = torch.Tensor([0])

        model_device = next(self.artefacts.model.parameters()).device

        # training
        for _, (datapoints, labels) in enumerate(train_set):
          self.artefacts.optimiser.zero_grad()

          preds = self.artefacts.model(datapoints.to(model_device))
          loss = self.artefacts.loss_fn(preds, labels.to(model_device))
          training_loss += loss.item()
          loss.backward()

          # average and synchronise the gradients at the end of each batch
          if self.p_config.requires_ipc:
            Utils.all_average_gradients(self.artefacts.model)

          self.artefacts.optimiser.step()

        training_loss /= len(train_set)

        # synchronise metrics
        if self.p_config.requires_ipc:
          dist.all_reduce(training_loss, dist.ReduceOp.SUM)
          training_loss /= dist.get_world_size()

        training_accuracy = self.evaluate(global_rank, train_set)

        if global_rank == 0:
          # code for storing logs and saving state
          pass

    def evaluate(self, global_rank: int, dataset: data.DataLoader = None):
      if dataset is None:
        dataset = self.artefacts.test_set
      assert dataset is not None

      accuracy = torch.Tensor([0])
      self.artefacts.model.eval()
      model_device = next(self.artefacts.model.parameters()).device
      with torch.no_grad():
        for _, (datapoints, labels) in enumerate(dataset):
          preds = self.artefacts.model(datapoints.to(model_device))
          num_correct = (preds.argmax(1) == labels.to(model_device)).sum().item()
          [num_samples, *_] = datapoints.shape
          accuracy += (num_correct / num_samples)
        accuracy *= 100/len(dataset)

        if self.p_config.requires_ipc:
          dist.all_reduce(accuracy)
          accuracy /= dist.get_world_size()

      if global_rank == 0: print(f'\tAccuracy: {accuracy.item()}')

      return accuracy
