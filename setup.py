#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='cogitare',
    version='0.1.1',
    url='https://github.com/cogitare-ai/cogitare',
    description=' Cogitare - A Modern, Powerful, and Modular Deep Learning and Machine Learning framework in Python.',
    author='Aron Bordin',
    packages=find_packages(exclude=('tests', 'tests.*')),
)
