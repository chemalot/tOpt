#!/usr/bin/env python
# encoding: utf-8

'''
Wrapper around SDFMOptimize.py to load pyNeuroChem first.
Loading pyNeuroChem after pytorch causes a mysterious crash

@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

import warnings

try:
    import pyNeuroChem as neuro
except ImportError:
    warnings.warn('pyNeuroChem module not found!')
    pass

import sys
from tOpt import sdf_multi_optimizer
from tOpt.NNP_computer_factory import ExampleNNPComputerFactory


def main():
    sdf_multi_optimizer.main(ExampleNNPComputerFactory)
    

if __name__ == "__main__":
    sys.exit(main())
