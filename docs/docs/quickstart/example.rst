Example
#######

1. Configure the platform
=========================

.. code-block:: python

    from ddpw.platform import Platform, PlatformConfig

    p_config = PlatformConfig(
        name='DDPW Example,
        verbose=True,
        spawn_method='spawn',
        partition='general,
        platform=Platform.SLURM,
        n_nodes=2,
        n_gpus=4,
        master_addr='localhost,
        master_port='1889,
        seed=1889,
        timeout_min=2880,
        cpus_per_task=1
    )

2. Configure the artefacts
==========================

.. code-block:: python

    import torch
    from ddpw.artefacts import ArtefactsConfig

    from src.dataset import CustomDataset # the dataset to train the model on
    from src.model import CustomModel # the model to train
    from src.optimiser import CustomOptimiser # the optimiser to use in training
    from src.loss import CustomLoss # the loss function to use

    a_config = ArtefactsConfig(
        train_set=CustomDataset(train=True),
        test_set=CustomDataset(train=False),
        validation_percentage=25,
        batch_size=64,
        model=CustomModel(),
        model_has_batch_norm=True,
        loss_fn=CustomLoss(),
        optimiser_config=CustomOptimiser(lr=0.1)
    )

Refer to the :ref:`example with MNIST <MNIST example>` to see how the custom
artefacts are implemented.

3. Configure the job
====================

.. code-block:: python

    from ddpw.trainer import TrainingConfig, TrainingMode

    t_config = TrainingConfig(
        job_type=TrainingMode.TRAIN,
        start_at=0,
        epochs=50,
        model_name_prefix='model',
        model_path='./model,
        checkpoint_name_prefix='ckpt',
        checkpoint_path='./checkpoint',
        console_logs_path='./logs/console_logs',
        training_logs_path='./logs/tensorboard_logs',
        save_every=5
    )

4. Call the job
===============

.. code-block:: python

    from ddpw.wrapper import Wrapper

    from src.train import CustomTrainer

    w = Wrapper(p_config, a_config)
    task = CustomTrainer(t_config)

    w.start(task)
