import typing as tp

import torch
import numpy as np
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .__logger import Logger
from .__trainer import Trainer

class AutoExecutor(object):
  nprocs = 1
  init_method: str = 'tcp://localhost:1640'
  seed = 1640

  def update_parameters(self, init_method, trainer: Trainer, nprocs: int = 1):
    assert nprocs > 0
    self.nprocs = nprocs
    self.init_method = init_method
    self.trainer = trainer

  def dist_init(self, pid, args):
    print('Communication through ', self.init_method)
    dist.init_process_group(
      backend=dist.Backend.GLOO,
      init_method=self.init_method,
      world_size=self.nprocs,
      rank=pid
    )

    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    torch.cuda.manual_seed_all(self.seed)

    print('op')
    dist.barrier()

    gpu = args.get('local_rank', pid)

    torch.cuda.set_device(gpu)

    model = args['model']
    dataset = args['dataset']
    optimiser = args['optimiser']
    loss_fn = args['loss_fn']
    optimiser_step = args.get('optimiser_step', None)
    epochs = args['epochs']
    ckpt_every = args['ckpt_every']

    model.to(gpu)
    model = DistributedDataParallel(model, [pid])
    smplr = DistributedSampler(dataset, num_replicas=self.nprocs, rank=pid)
    dataloader = DataLoader(dataset, batch_size=100, pin_memory=True,
                         sampler=smplr)

    optimiser.params = model.parameters()
    print('pre-barrier')
    dist.barrier()
    # dist.barrier(device_ids=[pid])

    logger = Logger(args['logdir']) if pid == 0 else None

    self.trainer(model, dataloader, optimiser, loss_fn, optimiser_step, epochs,
                 ckpt_every, pid=pid, logger=logger)

    dist.barrier(device_ids=[pid])
    print('post-trainer')
    dist.destroy_process_group()
    print('post-destruction')

  def submit(self, args: tp.Dict) -> None:
    mp.spawn(self.dist_init, (args,), nprocs=self.nprocs)
