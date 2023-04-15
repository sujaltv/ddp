Example with MNIST
##################
.. _MNIST example:

1. Custom dataset
=================

.. code-block:: python
  :caption: src/dataset.py

  from torchvision import transforms as T
  from torchvision.datasets.mnist import MNIST


  class MyDataset(MNIST):
    def __init__(self, root: str, train: bool = True, download: bool = False):
      transforms = T.Compose([
        T.ToTensor(),
        T.Resize((28,28)),
        T.Normalize((0.1307,), (0.3081,))
      ])

      super(MyDataset, self).__init__(root, train, transforms,
                                          download=download).__init__()


2. Custom model
===============

.. code-block:: python
  :caption: src/model.py

  import torch
  from torch.nn import functional as F


  class MyModel(torch.nn.Module):
    def __init__(self):
      super(MyModel, self).__init__()

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


3. Custom job
=================
.. _MNIST custom job:

.. code-block:: python
  :caption: src/train.py

  import torch
  from torch.utils import data
  import torch.distributed as dist
  import torch.nn.functional as F

  from ddpw.utils import Utils
  from ddpw.platform import Platform
  from ddpw.job import Job, JobConfig


  class MyTrainer(Job):
    start_at = 0
    epochs = 10

    def __call__(self, global_rank: int, local_rank: int):
      train_set = self.a_config.train_set
      model = self.a_config.model.train()
      optimiser = torch.optim.SGD(model.parameters(), lr=1e-2)

      # for every epoch
      for e in range(self.start_at, self.epochs):
        # training
        for _, (datapoints, labels) in enumerate(train_set):
          optimiser.zero_grad()

          preds = model(datapoints.to(model_device))
          loss = F.nll_loss(preds, labels.to(model_device))
          training_loss += loss.item()
          loss.backward()

          # average and synchronise the gradients at the end of each batch
          if self.p_config.requires_ipc:
            Utils.average_params_grads(model)

          optimiser.step()

        training_loss /= len(train_set)

        # synchronise metrics
        if self.p_config.requires_ipc:
          dist.all_reduce(training_loss, dist.ReduceOp.SUM)
          training_loss /= dist.get_world_size()

        training_accuracy = self.evaluate(global_rank, train_set)

        if global_rank == 0:
          # code for storing logs and saving state
          pass
