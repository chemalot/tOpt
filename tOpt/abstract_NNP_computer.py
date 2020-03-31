'''
Created on Oct, 2019

@author: albertgo
'''

import logging
import torch
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

from typing import List, ClassVar, Sequence
from tOpt.unit_conversion import Units
from tOpt.coordinates_batch import SameSizeCoordsBatch
from cddlib.chem.mol import BaseMol


log = logging.getLogger(__name__)


class AbstractNNPComputer(metaclass=ABCMeta):
    '''
        An NNP computer provides access to a compute engine for Energies and gradients 
        from batches for conformations in SameSizeCoordsBatch.
        
        The actual computation is wrapped in a class derived from EnergyAndGradHelperInterface.
        
        Interface that all wrappers around NNP engines need to implement for use
        in tOpt.
    '''


    def __init__(self, outputGrads:bool, compute_stdev:bool, coords_dtype, 
                 mol_mem_GB:int, bytes_per_mol:int, batch_by_atom_order:bool, 
                 energyOutUnits=Units.KCAL):
        '''
            Arguments:
            outputGrads: output gradients
            compute_stdev: compute standard deviation
            coords_dtype:   torch dtype of coordinates
            energyOutUnits: units of energies to output
            mol_mem_GB:     GB of memory available for computations
            bytes_per_mol:  approximate bytes required for each conformation
            batch_by_atom_order: if false input molecules are grouped by atom count 
                                 if true in addtion they must have the same atom 
                                    types in the same order
        '''
        
        self.outputGrads = outputGrads
        self.compute_stdev = compute_stdev
        self.coods_dtype = coords_dtype
        self.energyOutUnits = energyOutUnits
        self.mol_mem_GB = mol_mem_GB           # GB of memory available for computations
        self.BYTES_PER_MOL = bytes_per_mol     # approximate bytes required for each conformation
        self.batch_by_atom_order = batch_by_atom_order
        
                
             
    @abstractmethod
    def create_energy_helper(self, coordsBatch:SameSizeCoordsBatch,
                                  harm_constraint:Sequence[float] = None, fixed_atoms_list:List[bool] = None ):
        """ Create an energy helper class that can compute energies and gradients for batches
            of conformations depending on the presence of constraints
            
            arguments:
            harm_constraint: list of harmonic constraint. 
                                mols.shape[0] must be same size harm_constraint
        """

        raise NotImplementedError()
        
        
    def computeBatch(self, mols:List[BaseMol]):
        """ Return iterator returning (mol,e,std,grad) for all mols 
        
            Parameter:
            mols: array of Mol Objects, currently must be conformers of same molecule
        """
        
        class ComputeResultIterator(object): 
            _energyBatch:ClassVar[np.ndarray] = None
            _stdevBatch:ClassVar[np.ndarray] = None
            _gradBatch:ClassVar[np.ndarray] = None

    
            def __init__(self, nnpCmptr:AbstractNNPComputer, mols:List[BaseMol]):
                """
                Attributes:
                  _molBatch    array of Mol's
                  _energyBatch np.array for float
                  _coords np.                 
                """
                assert len(mols) > 0
                
                self._molBatch    = mols
                self._energyBatch = None
                self._gradBatch   = None
                self._coords      = None
                self._curentMol   = 0
                self.outputGrads  = nnpCmptr.outputGrads
                self.nnpCmptr     = nnpCmptr

                coordsBatch = SameSizeCoordsBatch(nnpCmptr.allowed_atom_num, self.nnpCmptr.coods_dtype)
        
                for mol in mols:            
                    xyz = mol.coordinates
                    atomTypes = np.array(mol.atom_types, dtype=np.int64)
                    coordsBatch.addConformer(xyz, atomTypes)
            
                coordsBatch.collectConformers(self.outputGrads, nnpCmptr.device)
                e_help    = nnpCmptr.create_energy_helper(coordsBatch)
                pred, std = e_help.compute_energy()
                
                if nnpCmptr.outputGrads:
                    self._gradBatch = e_help.compute_grad().cpu().numpy()
                
                self._energyBatch = pred.detach().cpu().numpy()
                if std is not None: 
                    self._stdevBatch  = std.detach_().cpu().numpy()

        
            def __iter__(self):
                return self
            
            def __next__(self):
                """ returns tuple molecule (unchanged, energy float, gradient(numpy))
                    gradient will be None unless the nnpCmpr was configured with outputGrads True
                """
                if self._curentMol >= len(self._molBatch):
                    raise StopIteration
                                    
                mol  = self._molBatch[self._curentMol]
                e    = self._energyBatch[self._curentMol]
                if self._stdevBatch is not None:
                    std  = self._stdevBatch[self._curentMol]
                else:
                    std = None
                      
                grad = None
                if self.outputGrads:
                    grad   = self._gradBatch[self._curentMol]
                
                log.debug("coodsA=%s e=%.2f g=%s [%s]"%(
                     mol.coordinates.reshape(-1),e,grad,self.nnpCmptr.energyOutUnits))

                self._curentMol += 1
                log.debug("e=%.2f [%s]"%(e,self.nnpCmptr.energyOutUnits))
                return mol,e,std,grad
            
            
        return ComputeResultIterator(self, mols)
    


class EnergyAndGradHelperInterface(metaclass=ABCMeta):
    """ The EnergyAndGradHelperInterface provides an implementation independend
        interface to computing energies and gradients for batches of conformations
        stored in a single SameSizeCoordsBatch.
    """
        
    def __init__(self, eFactor:float, fFactor:float):
        """
            arguments:
               eFactor -- multiply energy returned by engine by this factor
               fFactor -- multiply forces returned by engine by this factor
        """
        
        self.eFactor = eFactor
        self.fFactor = fFactor


    @abstractmethod
    def compute_energy(self) -> torch.tensor:
        """ return the energy of the conformations in self.coords_batch """
        
        raise NotImplementedError()


    @abstractmethod
    def compute_energy_with_filter(self, fltr:torch.tensor) -> torch.tensor:
        """
            Compute Energies for conformations for which fltr is 1
        """
        
        raise NotImplementedError()
    

    @abstractmethod
    def energy_no_constraint(self):
        """ 
            Return energies of the last computation without any imposed constraints
        """ 
        raise NotImplementedError()

    

    @abstractmethod
    def compute_grad(self):
        """ 
            Return gradients of the last computation without any imposed constraints
        """ 
        raise NotImplementedError()
        

    @abstractmethod
    def filter_(self, fltr: torch.tensor):
        """
            Reduce conformations to those with fltr == 1 
        """
        raise NotImplementedError()
        
        
class AbstractEnergyAndGradHelper(EnergyAndGradHelperInterface, metaclass=ABCMeta):
    """ Provides some common functionality needed by all sub classes of EnergyAndGradHelperInterface
    """
    def __init__(self, coords_batch:SameSizeCoordsBatch, 
                 eFactor:float, fFactor:float):
        
        super().__init__(eFactor, fFactor)
        
        self.coords_batch = coords_batch
        

    def filter_(self, fltr: torch.tensor):
        self.coords_batch.filter_(fltr)
        self.n_confs = self.coords_batch.n_confs
        
    
    
class EnergyAndGradHelperHarmConstraint(EnergyAndGradHelperInterface):
    """ energy Helper that adds a harmonic constraint to the energy and forces 
        pulling the conformation back towards its input coordinates """
        
    def __init__(self, e_helper:EnergyAndGradHelperInterface, harm_constraint:torch.tensor):
        
        super().__init__(e_helper.eFactor, e_helper.fFactor)
        
        self.e_helper = e_helper
        self.in_coords = e_helper.coords_batch.coords.detach().clone()
        self.harm_constraint = harm_constraint
        
        self.delta = None


    def compute_energy(self) -> torch.tensor:
        e, std = self.e_helper.compute_energy()

        self.delta = self.in_coords - self.e_helper.coords_batch.coords
        delta2 = self.delta.pow(2)

        harmE = (self.harm_constraint * delta2.reshape(self.e_helper.n_confs,-1).sum(-1))
        if self.eFactor != 1.: harmE * self.eFactor
        
        e = e +  harmE
        
        return e, std
    
    
    def compute_energy_with_filter(self, fltr:torch.tensor) -> torch.tensor:
        e, std = self.e_helper.compute_energy_with_filter(fltr)
    
        delta = self.in_coords[fltr] - self.e_helper.coords_batch.coords[fltr]
        delta2 = delta.pow(2)
        self.delta[fltr] = delta
        
        e = e + self.harm_constraint[fltr] * delta2.reshape(e.shape[0],-1).sum(-1)
        return e,std
    
    
    def energy_no_constraint(self):
        """ Return energies without any imposed constraints
            To be overwritten
        """ 
        return self.e_helper.energy_no_constraint()
    
    
    def compute_grad(self):
        """ compute gradients of last energy evaluation """
        # we could back propagate from the energy with constraint
        # in that case we would not need to keep  self.delta instead we would keep
        # energy_with_constraint
        
        grad = self.e_helper.compute_grad()
        grad = grad - (2. * self.harm_constraint.view(-1,1,1) * self.delta) * self.fFactor
        
        return grad


    def filter_(self, fltr: torch.tensor):
        self.e_helper.filter_(fltr)
        self.in_coords = self.in_coords[fltr]
        self.harm_constraint = self.harm_constraint[fltr]
        
        
class EnergyAndGradHelperFixedAtoms(EnergyAndGradHelperInterface):
    """ energy Helper that keeps some atoms fixed in position """
        
    def __init__(self, e_helper:EnergyAndGradHelperInterface, fixed_atoms:torch.tensor):
        """
            e_helper: EnergyAndGradHelper used to compute energies and gradients
            fixed_atoms: tensor of indexes or boolean tensor specifing fixed atoms
        """
        super().__init__(e_helper.eFactor, e_helper.fFactor)
        
        assert fixed_atoms.shape[0] == e_helper.coords_batch.n_confs
        assert fixed_atoms.shape[1] == e_helper.coords_batch.n_atom_per_conf
            
        self.e_helper = e_helper
        self.fixed_atoms = fixed_atoms
        

    def compute_energy(self) -> torch.tensor:
        """ return the energy of the conformations in self.coords_batch """
        
        return self.e_helper.compute_energy()


    def compute_energy_with_filter(self, fltr:torch.tensor) -> torch.tensor:
        """
            Compute Energies for conformations for which fltr is 1
        """
        
        return self.e_helper.compute_energy_with_filter(fltr)
    

    def energy_no_constraint(self):
        """ 
            Return energies of the last computation without any imposed constraints
        """ 
        return self.e_helper.energy_no_constraint()
    
    
    def compute_grad(self):
        """ compute gradients of last energy evaluation """

        grad = self.e_helper.compute_grad()
        grad[self.fixed_atoms] = 0.
        
        return grad


    def filter_(self, fltr: torch.tensor):
        self.e_helper.filter_(fltr)
        self.fixed_atoms = self.fixed_atoms[fltr]





