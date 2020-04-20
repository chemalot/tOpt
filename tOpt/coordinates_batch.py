from typing import Sequence
import numpy as np
import torch
from torch import nn
from cddlib.chem.mol import BaseMol


class SameSizeCoordsBatch(object):
    
    def __init__(self, allowed_atom_num:Sequence[int], dtype ):
        """
        This represents a batch of conformations that all have the same number of atoms.
        
        Arguments:
            allowed_atom_num: list of atomic numbers supported
            dtype: torch dtype of the coordinates
        """
        
        self.n_confs         = 0
        self.n_atom_per_conf = 0
        self.uniq_at_types   = None  # np.array of unique atomic numbers in batch
        self.at_type_count   = None  # list of counts for each unique atomic number
        self.coords          = None  # tensor nConf,nAtoms, 3
        self.atom_types      = None  # tensor nconf,nAtoms  with atomic numbers
        self._coords_list    = []
        self._at_type_list   = []
        self.allowed_atom_num = frozenset(allowed_atom_num)
        self.dtype           = dtype


    def addConformerMol(self, mol:BaseMol) -> None:
        xyz = mol.coordinates
        atomTypes = np.array(mol.atom_types, dtype=np.int64)
         
        self.addConformer(xyz, atomTypes)
        
        
    def addConformer(self, xyz:np.ndarray, atomic_nums:np.ndarray):
        """ Add a conformer to this batch
        
            xyz: array of xyz coordinates for each atom
            atomicNums: np.array.int64 of atomic numbers for each atom
        """
        
        assert(len(xyz) == len(atomic_nums))
        assert len(atomic_nums) > 0
        assert all( an in self.allowed_atom_num for an in atomic_nums), "Invalid atom type"
        
        if self.n_atom_per_conf == 0:
            self.n_atom_per_conf = len(atomic_nums)
        else:
            if len(atomic_nums) != self.n_atom_per_conf:
                raise ValueError("only molecules with one size are allowed {}".format(self.n_atom_per_conf))
            
        self._coords_list.append(xyz)
        self._at_type_list.append(atomic_nums)
        self.n_confs += 1
        
        
    def collectConformers(self, requires_xyz_grad, device):
        """
            This is called when all conformers of a given batch have been added to 
            finalize the internal data structure and to move the data onto the device.
        """
        self.coords     = torch.tensor(self._coords_list, dtype=self.dtype)
        self.atom_types = torch.tensor(self._at_type_list, dtype=torch.int64)
        if device is not None:
            self.coords     = self.coords.to(device=device, non_blocking=True)
            self.atom_types = self.atom_types.to(device=device, non_blocking=True)
        self.coords.requires_grad_(requires_xyz_grad)
        
        np_atom_types = np.array(self._at_type_list)
        self.uniq_at_types, self.at_type_count = np.unique(np_atom_types, return_counts=True)
        self.at_type_count = self.at_type_count.tolist()
        
        del self._coords_list, self._at_type_list
    
    
    def filter_(self, fltr:torch.tensor) -> None:
        """ Keep only coordinates conforming where fltr == 1 """
        
        assert '_coords_list' not in self.__dict__, "Filter works only after calling collectConformers()" 
        
        self.coords = self.coords[fltr].detach_().requires_grad_(self.coords.requires_grad)
        self.atom_types = self.atom_types[fltr]
        self.n_confs = self.coords.shape[0]
        
        uniq_at_types, at_type_count = torch.unique(self.atom_types.reshape(-1),return_counts=True,sorted=True)
        self.uniq_at_types = uniq_at_types.cpu().numpy()
        self.at_type_count = at_type_count.cpu().numpy().tolist()
        

    def filter(self, fltr:torch.tensor):
        """ Return copy with only coordinates conforming where fltr == 1 """
        
        assert '_coords_list' not in self.__dict__, "Filter works only after calling collectConformers()" 
        
        cln = SameSizeCoordsBatch(self.allowed_atom_num, self.coords.dtype)
        cln.coords = self.coords[fltr]
        cln.atom_types = self.atom_types[fltr]
        cln.n_confs = cln.coords.shape[0]
        
        uniq_at_types, at_type_count = torch.unique(self.atom_types.reshape(-1),return_counts=True,sorted=True)
        cln.uniq_at_types = uniq_at_types.cpu().numpy()
        cln.at_type_count = at_type_count.cpu().numpy().tolist()
        cln.n_atom_per_conf = self.n_atom_per_conf
        
        del cln._coords_list, cln._at_type_list
        
        return cln

    def clone(self):
        assert '_coords_list' not in self.__dict__, "Clone works only after calling collectConformers()" 

        cln = SameSizeCoordsBatch(self.allowed_atom_num, self.coords.dtype)
        cln.coords = self.coords.detach().clone()
        cln.coords.requires_grad_(self.coords.requires_grad)
        cln.atom_types = self.atom_types.clone()
        cln.n_confs = self.n_confs
        cln.uniq_at_types = self.uniq_at_types.copy()
        cln.at_type_count = self.at_type_count.copy()
        cln.n_atom_per_conf = self.n_atom_per_conf
        
        del cln._coords_list, cln._at_type_list    
        return cln    
                
    def zero_grad(self):
        if self.coords.grad is not None:
            self.coords.grad.detach_()
            self.coords.grad.zero_()
        
        
class CoordinateModelInterface(nn.Module):
    """
        To use the optimizer with any pytorch based NNP implement a wrapper that
        derives from this class and transforms the input form the SameSizeCoordsBatch
        to the input needed for the NNP.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, input:SameSizeCoordsBatch) -> (torch.tensor,torch.tensor,torch.tensor):
        r""" compute energies, std, and range from SameSizeCoordsBatch
        
        std and range may be None if not computed 

        Should be overridden by all subclasses."""
        raise NotImplementedError
