from setuptools import setup, find_packages

from ddpw import __version__, __build__


with open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()

setup(
  name="ddpw",
  version=__version__,
  build=__build__,
  author="Sujal T.V.",
  url="https://ddpw.projects.sujal.tv",
  description=r"""A utility package to scaffold PyTorch's DDP""",
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux"
  ],
  python_requires='>=3.10',
  install_requires=['numpy>=1.22.2', 'submitit>=1.2.1', 'torch>=2.0.0']
)
