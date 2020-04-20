# encoding: utf-8
'''
tOpt.optimizer -- Reads sdf files and optimizes them using NNP.

@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

from typing import Sequence, Iterable, Type
import numpy as np
import re
import sys
import math
import copy
import time

import logging
from tOpt.opt_util import OPT_HARM_CONSTRAINT_TAG, ConvergenceOpts,\
    OPT_ENERGY_TAG, OPT_STATUS_TAG, OPT_STEPS, OPT_STD_TAG, OPT_FORCE_TAG,\
    Status, DEFAULT_CONVERGENCE_OPTS
from tOpt.coordinates_batch import SameSizeCoordsBatch
from tOpt.batch_lbfgs import BatchLBFGS
from tOpt.abstract_NNP_computer import AbstractNNPComputer
from cddlib.chem.mol import BaseMol
from cddlib.util import iterate
log = logging.getLogger(__name__)


MULTI_SPACE_RE  = re.compile("\\s+")
SPACE_TO_CSV_RE = re.compile("([0-9.\\]]) ([-+0-9\\[])")


class BatchOptimizer(object):
    """ Iterator that takes an iterator of Mol's and returns an iterator of Mol's
        The output Mol will have the updated geometry the following output fields:
            - NNP_Energy_kcal_mol
            - NNP_Energy_status
            - NNP_STEPS
        The conformations in the input can be unsorted, however by sorting by atom count
        the implementation might be significantly faster as the GPU implemnetation will
        only work on batches of consecutive molecules with the same atom count.        
    """
    def __init__(self, nnp_computer:AbstractNNPComputer, molInStream:Iterable[Type[BaseMol]], 
                 conv_opts:ConvergenceOpts = DEFAULT_CONVERGENCE_OPTS, 
                 learn_rate: float = 0.3,
                 constraint: str = None,  
                 harm_constr: Sequence[float] = None,
                 line_search:str  = None, lbfgs_hist_size:int = 100,
                 out_grad:bool = False,
                 prune_high_energy_freq:int = 2, prune_high_energy_fract:float = .3,
                 plot_name:str = None ):
        
        """ Arguments
            ---------
            nnp_computer: engine to compute energies and gradient
            molInStream: iterator over input molecules
            conv_opts: convergence parameters for terminating optimization
            learn_rate initial learning rate
            constraint: currently None|"heavyAtom"
            harm_contraint: harmonic constraint pulling back to initial coordinates in [kcal/mol/a^2]
            line_search: one of None|"Armijo"|"Wolfe" specifying the algorithm for a line search at each step
            lbfgs_hist_size: size of history used to approximate second derivative
            out_grad: if true the gradients will be added to the output conformations
            prune_high_energy_freq: cf. batch_lbgf
            prune_high_energy_fract: cf. batch_lbgf
            plot_name: if given a png file with cycle vs energy will be stored 
        """
        self.molIn = iterate.PushbackIterator(molInStream)
        self.learn_rate = learn_rate
        self.conv_opts = conv_opts
        self.constraint = constraint 
        self.harm_constr = harm_constr
        self.n_harm = 1 if not harm_constr else len(harm_constr)
        self.line_search = line_search
        self.out_grad = out_grad
        self.prune_high_energy_freq = prune_high_energy_freq
        self.prune_high_energy_fract = prune_high_energy_fract
        self.plot_name = plot_name
        self._currentMol = 0
        self.countIn = 0
        self.countMin = 0
        self._molBatch   = []
        self._batch_harm_constr = []
        self.nnp_computer = nnp_computer        
        self.device = nnp_computer.device
        self.countBatch = 0
        self.start = time.time()
        self.lbfgs_hist_size = lbfgs_hist_size



    def __enter__(self):
        return self

    def __iter__(self):
        return self
    
    def __exit__(self, *args):
        pass;
        

    def __next__(self):
        """ Iterate over optimized molecules """
        
        if self._currentMol >= len(self._molBatch):
            self._molBatch.clear()
            nMols = 0
            batch_natoms = 0
            batch_size = 1    # initial value will be adjusted in first record
            while nMols < batch_size and self.molIn.has_next():
                mol = self.molIn.__next__()
                
                for at in mol.atoms:
                    if at.atomic_num not in self.nnp_computer.allowed_atom_num:
                        log.warning(f"Unknown atom type {at.atomic_num} in {mol.title}")
                        break
                else:    
                    nAt = mol.num_atoms
                    if batch_natoms == 0: 
                        batch_natoms = nAt
                        
                        # assume memory requirement goes with nAtom^3 
                        batch_size = int(self.nnp_computer.mol_mem_GB * math.pow(1024/batch_natoms,3) 
                                         / self.nnp_computer.BYTES_PER_MOL)                          
                        log.debug(f"nAt: {batch_natoms}, batchSize: {batch_size}")

                    if nAt != batch_natoms:
                        self.molIn.pushback(mol)
                        break
                    if self.harm_constr:
                        self._batch_harm_constr.extend(self.harm_constr)
                        for cnstr in self.harm_constr:
                            mol[OPT_HARM_CONSTRAINT_TAG] = cnstr
                            self._molBatch.append(mol)
                            mol = copy.deepcopy(mol)
                    else:
                        self._molBatch.append(mol)
                    nMols += 1
                    self.countIn += 1
                    self.countMin += self.n_harm
                    if log.isEnabledFor(logging.WARN):
                        if self.countIn % 50 == 0:  print('.', file=sys.stderr, flush=True, end="")
                        if self.countIn % 2000 ==0: log.warning(f' MOptimizer: {self.countIn}')

            if nMols == 0:
                if log.isEnabledFor(logging.WARN):
                    end = time.time()
                    runtime = end - self.start
                    if self.countMin > self.countIn:
                        log.warning(f'\nFinished MOptimizer: {self.countIn} inputs, {self.countMin} minimization in {runtime:.1f}sec')
                    else:
                        log.warning(f'\nFinished MOptimizer: {self.countIn} mols in {runtime:.1f}sec')
                raise StopIteration
        
            self.optimizeMols(self._molBatch,self._batch_harm_constr)
            self._currentMol = 0
            self.countBatch += 1

        res= self._molBatch[self._currentMol]
        self._currentMol += 1
        return res


    def optimizeMols(self, mol_batch: Sequence[Type[BaseMol]], harm_constr: Sequence[float] ) \
                    -> None:
        """
            Optimize a batch of Mols
           
            Parameter
            ---------
               mol_batch (potentially large) batch of cddlib.chem.Mol objects 
               harm_constr harmonic constraint to add to potential, None allowed
        """

        coordsBatch = SameSizeCoordsBatch(self.nnp_computer.allowed_atom_num, self.nnp_computer.coods_dtype)
        fixed_atoms_list = []
        for mol in mol_batch:            
            xyz = mol.coordinates
            atomTypes = np.array(mol.atom_types, dtype=np.int64)
            coordsBatch.addConformer(xyz, atomTypes)
            if self.constraint == "heavyAtom":
                ha = atomTypes != 1
                fixed_atoms_list.append(ha)
    
        coordsBatch.collectConformers(True, self.device)
        
        p_name = self.plot_name
        if p_name: p_name += str(self.countBatch)
        optimizer = BatchLBFGS(lr=self.learn_rate, convergence_opts=self.conv_opts, line_search_fn=self.line_search,
                           history_size=self.lbfgs_hist_size,
                           prune_high_energy_freq=self.prune_high_energy_freq, prune_high_energy_fract=self.prune_high_energy_fract,
                           plot_name=p_name, device=self.device)

        energyHelper = self.nnp_computer.create_energy_helper(coordsBatch, harm_constr, fixed_atoms_list)
        
        coords, energies, grads, std, status, n_steps = optimizer.optimize(coordsBatch, energyHelper)
        
        energies = energies.detach().cpu().numpy()
        stds      = std.detach().cpu().numpy() if std is not None else [ None ] * len(energies)
        n_steps  = n_steps.cpu().numpy()
        status   = [ Status.to_string(stat) for stat in status.cpu().numpy()]

        for mol,conf,e,std,g,stat, n_step in zip(mol_batch,coords.detach().cpu().numpy(), energies, stds, grads.detach().cpu().numpy(), status, n_steps):
            mol.coordinates = conf 
            mol[OPT_ENERGY_TAG] = f'{e:.1f}'
            mol[OPT_STATUS_TAG] = stat
            mol[OPT_STEPS] = n_step
            if std: mol[OPT_STD_TAG] = f'{std:.1f}'
            if self.out_grad: 
                frc = np.array_str(-1 * g,-1, 3, True)
                frc = re.sub(MULTI_SPACE_RE, " ",frc)
                frc = re.sub(SPACE_TO_CSV_RE,"\\1,\\2",frc)
                mol[OPT_FORCE_TAG] = frc
