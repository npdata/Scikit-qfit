#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for skqfit.

    This file was generated with PyScaffold 2.4.4, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import sys, os
import versioneer
from setuptools import setup

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    install_requires = ['numpy >= 1.8.2', 'scipy >= 0.13.3']
    tests_require = ['pytest_cov', 'pytest']
else:
    install_requires = []
    tests_require = []

def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []

    setup(version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          setup_requires=['six', 'pyscaffold>=2.4rc1,<2.5a0'] + sphinx,
          install_requires = install_requires,
          tests_require=tests_require,
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
