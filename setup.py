#!/usr/bin/env python

import os
from setuptools import setup

with open('keras_gtar/version.py') as version_file:
    exec(version_file.read())

setup(name='keras-gtar',
      author='Matthew Spellings',
      author_email='matthew.p.spellings@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Save and load keras models from GTAR trajectory files',
      extras_require={},
      install_requires=['gtar', 'tensorflow >= 2'],
      license='MIT',
      packages=[
          'keras_gtar',
      ],
      python_requires='>=3',
      version=__version__
      )
