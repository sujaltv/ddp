Example with MNIST
##################
.. _MNIST example:

.. code-block:: python
    :caption: main.py
    :linenos:
    :emphasize-lines: 1,14,17,18

    from ddpw import Platform, Wrapper

    from torchvision.datasets.mnist import MNIST
    from torchvision import transforms as T

    from src.model import MyModel
    from src.example import Example


    model = MyModel()
    t = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    dataset = MNIST(root='./datasets/MNSIT', train=True, download=True, transform=t)

    platform = Platform(device='gpu', n_gpus=4)
    example = Example(model, dataset, platform=platform, batch_size=32, epochs=2)

    wrapper = Wrapper(platform=platform)
    wrapper.start(example)


.. code-block:: python
    :caption: src/model.py
    :linenos:

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

            x = F.dropout(F.max_pool2d(x, 2), 0.25)

            x = x.flatten(1)
            x = F.relu(self.fc1(x))

            x = F.dropout(x, 0.5)
            x = self.fc2(x)

            return F.log_softmax(x, 1)


.. code-block:: python
    :caption: src/example.py
    :linenos:

    from tqdm import tqdm
    import torch
    from torch import distributed as dist
    from torch.nn import functional as F

    from ddpw import functional as DF
    from torch.utils.data import DataLoader


    class Example:
        def __init__(self, model, dataset, platform, batch_size, epochs):
            self.model = model
            self.dataset = dataset
            self.platform = platform

            self.batch_size = batch_size
            self.epochs = epochs

        def __call__(self, global_rank, local_rank):
            print(f'Global rank {global_rank}; local rank {local_rank}')
            model = DF.to(self.model, local_rank, device=self.platform.device)
            dataloader = DataLoader(
                self.dataset,
                sampler=DF.get_dataset_sampler(self.dataset, global_rank, self.platform),
                batch_size=self.batch_size,
                pin_memory=True
            )
            optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)

            training_loss = torch.Tensor([0.0]).to(DF.device(model))
            torch.cuda.set_device(local_rank)
            print(f'Model on device {DF.device(model)}; dataset size: {len(dataloader) * self.batch_size}')

            # for every epoch
            for e in range(self.epochs):
                print(f'Epoch {e} of {self.epochs}')

                for _, (imgs, labels) in enumerate(tqdm(dataloader, position=local_rank)):
                    optimiser.zero_grad()

                    preds = model(imgs.to(DF.device(model)))
                    loss = F.nll_loss(preds, labels.to(DF.device(model)))
                    training_loss += loss
                    loss.backward()

                    optimiser.step()

                training_loss /= len(dataloader)

                # synchronise metrics
                if self.platform.requires_ipc:
                  dist.all_reduce(training_loss, dist.ReduceOp.SUM)
                  training_loss /= dist.get_world_size()

                if global_rank == 0:
                    # code for storing logs and saving state
                    print(training_loss.item())

