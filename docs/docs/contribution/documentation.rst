Documentation
#############

Set up
^^^^^^

.. code-block:: bash

  # with conda
  conda env create --file environment.yaml
  conda activate ddpw-docs
  pip install -r requirements.txt

Freeze environment
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  sh freeze.sh

  # alternatively
  conda env export --no-build --from-history > environment.yaml
  pip list --not-required --format=freeze > requirements.txt

Update environment
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  conda env update --file environment.yaml
