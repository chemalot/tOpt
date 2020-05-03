'''
Created on May 2, 2020

@author: ping lin
'''

import torch
from tOpt.coordinates_batch import SameSizeCoordsBatch, CoordinateModelInterface
from tOpt.unit_conversion import HARTREE_TO_KCALMOL

import warnings

try:
    import torchani
except ImportError:
    warnings.warn('pyNeuroChem module not found!')
    pass


class ANI1xNet(CoordinateModelInterface):
    """
        A ANI1x pytoch module that computes a ANI1x potential
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchani.models.ANI1x().to(self.device)
        self.code = [1, 6, 7, 8] # {1: 0, 6: 1, 7: 2, 8: 3}

    def forward(self, same_size_coords_batch: SameSizeCoordsBatch):
        c = same_size_coords_batch.coords.to(self.device)
        a = same_size_coords_batch.atom_types.to(self.device)
        b = torch.zeros_like(a)
        for i, s in enumerate(self.code):
            b[a == s] = i
        e = self.model((b, c)).energies
        e = e * HARTREE_TO_KCALMOL
        std = torch.zeros_like(e)
        return e, std  # fake stdev with e, will not affect tests


class ANI2xNet(CoordinateModelInterface):
    """
        A dummy pytoch module that computes a potential that pulls all atoms
        towards having coordinate = -0.703
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchani.models.ANI2x().to(self.device)
        self.code = [1, 6, 7, 8, 9, 16, 17] # {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

    def forward(self, same_size_coords_batch: SameSizeCoordsBatch):
        c = same_size_coords_batch.coords.to(self.device)
        a = same_size_coords_batch.atom_types.to(self.device)
        b = torch.zeros_like(a)
        for i, s in enumerate(self.code):
            b[a == s] = i
        e = self.model((b, c)).energies
        e = e * HARTREE_TO_KCALMOL
        std = torch.zeros_like(e)
        return e, std  # fake stdev with e, will not affect tests

