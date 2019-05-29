#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import os
import platform
import re

from setuptools import find_packages
from setuptools import setup

__location__ = os.path.join(os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))

__version__ = '0.0.4'


def gh(name, version):
    package = name.split('/')[1]
    if 'GHE_ACCESS_TOKEN' in os.environ:
        proto = 'git+https://{}@'.format(os.environ['GHE_ACCESS_TOKEN'])
    elif 'CDP_BUILD_VERSION' in os.environ:
        proto = 'git+https://'
    else:
        proto = 'git+ssh://git@'
    return '{proto}github.com/{name}.git@{version}' \
           '#egg={package}-{version}'.format(**locals())


py_major_version, py_minor_version, _ = (
    int(re.sub('[^\d]+.*$', '', v)) for v in platform.python_version_tuple())  # dealing with 2.7.2+ and 2.7.15rc1


def load_requirements_file(path):
    content = open(os.path.join(__location__, path)).read().splitlines()
    requires = [req for req in content if req != '' and not req.startswith("#")]
    return requires


setup(
    name='aspect based_sentiment_analysis',
    packages=find_packages(),
    version=__version__,
    description='Aspect Based Sentiment Analysis',
    long_description=open('README.md').read(),
    keywords='Aspect Based Sentiment Analysis',
    author='Amit Kushwaha',
    url='https://github.com/yardstick17/AspectBasedSentimentAnalysis',
    setup_requires=['flake8'],
    install_requires=load_requirements_file('requirements.txt'),
    dependency_links=[
        gh('yardstick17/PyAthena', 'v0.0.4'),
    ]

)
