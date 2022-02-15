#!/bin/sh

conda env export --no-build --from-history > environment.yaml
pip list --not-required --format=freeze > requirements.txt
