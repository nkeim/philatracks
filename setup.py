#!/usr/bin/env python

from distutils.core import setup

setup(
    name='philatracks',
    version='0.2',
    author='Nathan C. Keim',
    author_email='nkeim@seas.upenn.edu',
    url='https://github.com/nkeim/philatracks',
    packages=['philatracks'],
    install_requires=['numpy', 'scipy', 'pandas'],
    )
