#!/usr/bin/env python
import re
import codecs
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='cogitare',
    version=find_version('cogitare', '__init__.py'),
    url='https://github.com/cogitare-ai/cogitare',
    description=' Cogitare - A Modern, Powerful, and Modular Deep Learning and Machine Learning framework in Python.',
    author='Aron Bordin',
    author_email='aron.bordin@gmail.com',
    keywords=['deep learning', 'framework', 'PyTorch'],
    packages=find_packages(exclude=('tests', 'tests.*')),
    install_requires=[
        'six>=1.11.0',
        'numpy>=1.17.2',
        'tqdm>=4.36.1',
        'matplotlib>=3.1.1',
        'dask>=2.4.0',
        'toolz>=0.10.0',
        'cloudpickle>=1.2.2',
        'futures>=3.1.1',
        'coloredlogs>=10.0',
        'humanize>=0.5.1',
        'torch>=1.2.0',
        'torchvision>=0.4.0',
    ]
)
