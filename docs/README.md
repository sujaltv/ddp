# DDPW Documentation

This is the source for the [DDPW
Documentation](https://ddpw.projects.sujal.tv) written in Python 3.8 with
[Sphinx](https://www.sphinx-doc.org/en/master/).

## Set up

```bash
> conda env create --file environment.yaml # root folder
> conda activate ddpw
> pip install -r requirements.txt # required for building this documentation
```

**Making the documentation**

```bash
> cd docs
> make html
```
