#!/bin/bash

conda env export \
  --no-build \
  --from-history | grep -v "^prefix: " > environment.yaml
pip list --not-required --format=freeze > requirements.txt