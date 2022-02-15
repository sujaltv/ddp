import os

import torch.multiprocessing as mp
from submitit import AutoExecutor, JobEnvironment

from .utils import Utils
from .trainer import Trainer
from .gpu_setup import init_process
from .artefacts import ArtefactsConfig
from .platform import Platform, PlatformConfig


class Wrapper(object):
  r"""
  This class provides encapsulation for training the model on a CPU, GPU, or a
  SLURM-based cluster of GPU nodes.

  :param PlatformConfig p_config: Platform-related configurations.
  :param ArtefactsConfig a_config: Dataset- and model-related configurations.
  """

  def __init__(self, p_config: PlatformConfig, a_config: ArtefactsConfig):
    Utils.verbose = p_config.verbose

    Utils.print('Initialising the DDP Wrapper.')
    self.p_config = p_config
    self.a_config = a_config

    if p_config.requires_ipc:
      try:
        mp.set_start_method(p_config.spawn_method)
      except RuntimeError as e:
        Utils.print(
          f'Warning: {e}. Skipping setting the start method for forks.')

  def __gpu(self, run: Trainer):
    r"""
    This method sets up the training setup for cluster-less GPUs.
    """

    if self.p_config.world_size == 1:
      Utils.print('[Device 0] Task starting on GPU.')
      init_process(0, 0, run, self.p_config, self.a_config)
      return

    Utils.print(f'Spawning {self.p_config.world_size} processes.')
    processes = []

    # create a process for each GPU in the world
    for global_rank in range(self.p_config.world_size):
      p = mp.Process(target=init_process, args=(global_rank, global_rank, run,
                                                self.p_config, self.a_config))
      processes.append(p)
      p.start()

    for p in processes:
      p.join()

    Utils.print('All processes complete.')

  def __slurm(self, individual_gpu, console_logs_path: str = './logs'):
    r"""
    This method sets up the training setup for SLURM-based clusters of GPU
    nodes.

    :param Trainer run: Custom training/evaluation task.
    :param str console_logs_path: Location to save console logs. Default:
        ``./logs``.
    """

    Utils.print('Setting up the SLURM platform.')

    executor = AutoExecutor(folder=console_logs_path)
    executor.update_parameters(
      name=self.p_config.name,
      mem_gb=12*self.p_config.n_nodes,
      gpus_per_node=self.p_config.n_gpus,
      tasks_per_node=self.p_config.n_gpus,
      cpus_per_task=self.p_config.cpus_per_task,
      nodes=self.p_config.n_nodes,
      timeout_min=self.p_config.timeout_min,
      slurm_partition=self.p_config.partition
    )

    return executor.submit(individual_gpu)

  def start(self, run: Trainer):
    r"""
    This method begins the setup process for CPU/GPU/SLURM-based training and
    starts the task.

    :param Trainer run: Custom training/evaluation definitions.
    """

    Utils.print(f'Selected platform: {self.p_config.platform.name}.')
    Utils.print('Starting process(es).')

    if self.p_config.platform == Platform.CPU:
      init_process(0, 0, run, self.p_config, self.a_config)
      Utils.print('CPU process finished.')

    elif self.p_config.platform == Platform.GPU:
      self.__gpu(run)
      Utils.print('GPU processes finished.')

    elif self.p_config.platform == Platform.SLURM:
      def individual_gpu():
        r"""
        This nested function is the starting point for each SLURM-based GPU.
        """

        self.p_config.master_addr = os.environ['HOSTNAME']
        job_env = JobEnvironment()

        Utils.print(f'Node {job_env.node}: Local rank: {job_env.local_rank};' +
                    f'Gloal rank: {job_env.global_rank}.')

        init_process(job_env.global_rank, job_env.local_rank, run,
                     self.p_config, self.a_config)

      job = self.__slurm(individual_gpu, run.t_config.console_logs_path)
      Utils.print(f'SLURM job "{self.p_config.name}" scheduled; ' +
                  f'job ID: {job.job_id}.')
