#!/usr/bin/env python

from setuptools import setup, find_packages


name = 'fairface_classification'
version = '1'
description = 'Code Research'
author = 'Artem Trybushenko'

setup(
      name=name,
      version=version,
      description=description,
      author=author,
      packages=find_packages(where='src'),
      package_dir={'':'src'},
)