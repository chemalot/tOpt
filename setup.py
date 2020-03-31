#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="/",
    author_email='gobbi.alberto@gene.com',
    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Pytorch Multiconformer tensor optimiser",
    entry_points={
        'console_scripts': [
            'sdfANIOptimizer.py=tOpt.SDFANIMOptimizer:main',
            #'sdfGeometric.py=tOpt.geometric.SDFGeomeTRIC:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='tOpt',
    name='tOpt',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='git@github.roche.com:albertgo/tOpt.git',
    version='0.1.0',
    zip_safe=False,
)
