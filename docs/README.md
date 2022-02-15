# DDPW Documentation

This is the source for the [DDPW
Documentation](http://ddpw.projects-tvs.surge.sh) written in Python 3.8 with
[Sphinx](https://www.sphinx-doc.org/en/master/).

## Set up

```bash
conda env create --file environment.yaml
conda activate ddpw-docs
pip install -r requirements.txt
```

**Freeze environment**

```bash
sh freeze.sh

# alternatively
conda env export --no-build --from-history > environment.yaml
pip list --format=freeze > requirements.txt
```

**Update environment**

```bash
conda env update --file environment.yaml
pip install -r requirements.txt
```

## Publish manually

```bash
sh publish.sh
```