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
        'six>=1.10.0',
        'numpy>=1.12.1',
        'tqdm>=4.11.2',
        'matplotlib>=2.0.2',
        'dask>=0.15.2',
        'toolz>=0.8.2',
        'cloudpickle>=0.4.0',
        'futures>=3.1.1',
        'coloredlogs>=7.3',
        'humanize>=0.5.1'
    ]
)
