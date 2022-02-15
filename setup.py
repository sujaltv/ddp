import os
from setuptools import setup, find_packages

from ddpw import __version__, __build__


install_requires = []
req_path = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
with open(req_path) as f:
  install_requires = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()

setup(
  name="ddpw",
  version=__version__,
  build=__build__,
  author="Sujal T.V.",
  url="http://ddpw.projects-tvs.surge.sh",
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
  python_requires='>=3.8',
  install_requires=install_requires
)
