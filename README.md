# DDPW

The Distributed Data Parallel wrapper (DDPW) is created as a utility package to
encapsulate the scaffolding for PyTorch's Distributed Data Parallel.

This code is written in Python 3.8. The [DDP
documentation](http://ddpw.projects-tvs.surge.sh) contains details on how to use
this package.

## Set up

```bash
# with conda
conda env create --file environment.yaml
conda activate ddpw

# with pip
pip install -r requirements.txt
```

**Freeze environment**

```bash
sh freeze.sh

# alternatively
conda env export --no-build --from-history > environment.yaml
pip list --format=freeze > requirements.txt
```

**Update environent**

```bash
conda env update --file environment.yaml
```
