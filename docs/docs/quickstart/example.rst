Example
#######

.. code-block:: python
    :linenos:
    :emphasize-lines: 9,12,21,26,27

    from ddpw.platform import Platform, PlatformConfig
    from ddpw.artefacts import ArtefactsConfig
    from ddpw.job import JobConfig, JobMode
    from ddpw.wrapper import Wrapper

    from src import MyDataset, MyModel, MyOptimiser, MyTrainer

    # configure the platform
    p_config = PlatformConfig(platform=Platform.GPU, n_gpus=4, cpus_per_task=2)

    # configure the artefacts (model, dataset, optimiser, etc.)
    a_config = ArtefactsConfig(
        train_set=MyDataset(train=True),
        test_set=MyDataset(train=False),
        batch_size=64,
        model=MyModel(),
        optimiser_config=MyOptimiser(lr=0.1)
    )

    # configure the job
    j_config = JobConfig(job_type=JobMode.TRAIN, start_at=0, epochs=50,
                        save_every=5)


    # call the job
    w = Wrapper(p_config, a_config)
    w.start(MyTrainer(j_config))

Refer to the :ref:`example with MNIST <MNIST example>` to see how the custom
artefacts are implemented.
