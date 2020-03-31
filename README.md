# Pytorch Tensor Optimizer

This is a parallelized version of the lbfgs optimization algorithm for molecular conformations. A whole set of conformations can be optimized in parallel significantly reducing the time required for conformational searches.

This code was developed based on the lbfgs optimization algorithm as implemented in pytorch [lbfgs.py](https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html).
The intended use is for conformational searches using Nerual Net Potentials. 
An example interface to the [ANI](https://github.com/isayev/ASE_ANI) potential and 
a dummy implementation of a potential in pytorch is included.

This code is released under the [MIT license](License.txt).


## Installation

To use this with the ANI Neural net Potential (NNP):
   - install [ASE ANI](https://github.com/isayev/ASE_ANI) and ensure that it runs 
     correctly including the pyton interface (nerurochem package)
     
   - install the [cddlib](??) package 

   - download(git clone) this source code

   - install it into your python environment:
     cd into thee root direcotry of this package
     ```bash
     pip install .
     ```

   - setup the environment as necessary. Note: ASE_ANI requires CUDA.
     ```bash
     export ASE_ANI_DIR=<your ASE_ANI_DIR>
     export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$ASE_ANI_DIR/lib
     export PYTHONPATH=${PYTHONPATH}:$ASE_ANI_DIR/lib
     ```

   - run on a test case:
     ```bash
      sdfANIOptimizer.py -in data/C2H6.sdf -out C2H6.ani.sdf -conf $ANICONF -computeForce
     ```


## Use with other pytorch based NNP's

This requires the implementation of three wrapper classes:

1. a pytorch [module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) that extends from [CoordinateModelInterface](tOpt/coordinates_batch.py)
2. a [NNPComputerFactoryInterface](tOpt/NNP_computer_factory.py) class
3. a tiny command line wrapper like [sdfANIMOptimizer](sdfANIMOptimizer.py)


----------------
##### 1. Pytorch module
This encapsulates an actual NNP implementation. The `forward()` method takes an argument of type  [SameSizeCoordsBatch](tOpt/coordinates_batch.py) that provides access to the information on the conformations to minimize:
- n_confs: the number of conformations
- n_atom_per_conf: the number of atoms per conformation (all conformations have the same number of atoms)
- coords: the coordinates
- atom_types: the atomic numbers of the atoms. note: the conformations may vary in the type of atoms.

A very simplified NNP that pulls all atoms to the coordinates (-0.703,-0.703, -0.703) is given below:


```
class DummyNet(CoordinateModelInterface):
    """
        A dummy pytoch module that computes a potential that pulls all atoms
        towards having coordinate = -0.703 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, same_size_coords_batch:SameSizeCoordsBatch):
        c = same_size_coords_batch.coords
        c = c*5
        e = c.pow(2) + c.exp()
        e = e.reshape(c.shape[0],-1).sum(-1)
        # min (y=(5x)^2 + e^(5x)) ~ y(-0.703) = 0.8272
        return e, e   # fake stdev with e, will not affect tests   
```


------------------------
##### 2. The NNP Computer Factory

This Interface takes a string parameter and provides a factory for the NeuralNet Potential. The String parameter can be a directory name as used to point to the configuration directory for the [ANI_computer](tOpt.ANI_computer.py), a filename 
or any other string that will be used to constuct your PyTorch Module.

An Example can be found in ExampleNNPComputerFactory ([NNPComputerFactoryInterface](tOpt/NNP_computer_factory.py))


------------------------
##### 3. Wrapper for command line interface
To create a command line program that you can call to minimize conformations with
your NNP you have to create a tiny wrapper that constructs the NNP Computer Factory and passes it to the [sdf_multi_optimizer](tOpt/sdf_multi_optimizer.py).

An Example can be found in [SDFANIMOptimizer](tOpt/SDFANIMOptimizer.py).


## Acknowledgments

I would like to thank Justin S. Smith and Adrian Reutberg for making the ASE_ANI available and for help to set it up.

I would like to thanks Man-Ling Lee, the Genentech Incubator project and the Genentech Computational Chemistry group for allowing me to work on this.

## References:

[ASE_ANI](https://github.com/isayev/ASE_ANI)

[Original implementation of the lbfgs optimization algorithm in pytorch](https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html)


## License
```
###############################################################################
## The MIT License
##
## SPDX short identifier: MIT
##
## Copyright 2020 Genentech Inc. South San Francisco
##
## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included
## in all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
## OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.
###############################################################################
```

