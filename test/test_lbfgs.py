import numpy as np
import numpy.testing as npt

import os
import torch 
import torch.nn as nn

from tOpt.batch_lbfgs import MultiInstanceHistory, BatchLBFGS
from cddlib.chem.oechem.io import MolInputStream

import logging.config as logConfig

from tOpt.coordinates_batch import SameSizeCoordsBatch
from tOpt.opt_util import ConvergenceOpts
from tOpt.pytorch_computer import EnergyAndGradHelper, DummyNet
from tOpt.abstract_NNP_computer import EnergyAndGradHelperHarmConstraint,\
    EnergyAndGradHelperFixedAtoms


logIni = os.path.dirname(__file__) + "/../tOpt/log.debug.ini"
logConfig.fileConfig(logIni, disable_existing_loggers=False)



def test_multi_inst__history():
    history_size = 5
    n_confs = 3
    
    mih = MultiInstanceHistory(history_size, n_confs, 1, torch.float)
    
    mih.push_if(torch.tensor([1,0,1],dtype=torch.uint8).bool(), torch.tensor([-1.0,0.1,0.2]).unsqueeze(-1))
    npt.assert_allclose(mih.container[0].squeeze(), [-1.,0.,0.2])
    npt.assert_equal(mih.count_hist, np.array([1,0,1]))
    
    mih.push_if(torch.tensor([1,0,1],dtype=torch.uint8).bool(), torch.tensor([-1.0,0.1,0.2]).unsqueeze(-1))
    npt.assert_allclose(mih.container[0:2].squeeze(), [[-1.,0.,0.2],[-1.,0.,0.2]])
    npt.assert_equal(mih.count_hist, np.array([2,0,2]))
    
    mih.push_if(torch.tensor([0,1,0],dtype=torch.uint8).bool(), torch.tensor([-1.0,0.1,0.2]).unsqueeze(-1))
    npt.assert_allclose(mih.container[0:2].squeeze(), [[-1.,0.1,0.2],[-1.,0.,0.2]])
    npt.assert_equal(mih.count_hist, np.array([2,1,2]))
    
    mih.push_if(torch.tensor([1,0,1],dtype=torch.uint8).bool(), torch.tensor([-2.0,0.1,0.2]).unsqueeze(-1))
    mih.push_if(torch.tensor([1,0,1],dtype=torch.uint8).bool(), torch.tensor([-3.0,0.1,0.2]).unsqueeze(-1))
    mih.push_if(torch.tensor([1,0,1],dtype=torch.uint8).bool(), torch.tensor([-4.0,0.1,0.2]).unsqueeze(-1))
    mih.push_if(torch.tensor([1,1,1],dtype=torch.uint8).bool(), torch.tensor([-5.0,2.1,0.2]).unsqueeze(-1))
    npt.assert_allclose(mih.container.squeeze(), [[-5.,2.1,0.2],[-4.,0.1,0.2],[-3.,0.,0.2],[-2.,0.,0.2],[-1.,0.0,0.2]])
    npt.assert_equal(mih.count_hist, np.array([5,2,5]))


class TestBatchLBFGS(object):
    def setup(self):
        self.coords_batch = SameSizeCoordsBatch([7], torch.float)
    
        sdName = "data/N2_1.0_0.9.sdf"
        with MolInputStream(sdName) as molS:
            for mol in molS:
                xyz = mol.coordinates
                atomicNumbers = mol.atom_types
                
                self.coords_batch.addConformer(xyz,atomicNumbers)
    
        self.coords_batch.collectConformers(True, None)
        self.dummy_model = DummyNet()
        
        self.conv_opts = ConvergenceOpts(max_iter = 50)
        
        self.atomization_energies = torch.zeros((50,),dtype=torch.float)

        
    def test_opt(self):
        coords_batch = self.coords_batch.clone()
        e_helper = EnergyAndGradHelper(self.atomization_energies, self.dummy_model, coords_batch,1,1)
        lbfgs = BatchLBFGS(0.6, convergence_opts = self.conv_opts)
        min_coor, min_e, min_e_grad, std, stat, conf_step = lbfgs.optimize(coords_batch, e_helper)
        
        npt.assert_allclose( min_e.detach().cpu().numpy(), 4.9631, atol=1e-3)
        npt.assert_allclose( min_coor.detach().cpu().numpy(), -0.0703, atol=1e-2)
        npt.assert_allclose( min_e_grad.detach().cpu().numpy(), 0.0, atol=5e-2)
        npt.assert_equal(stat.cpu().numpy(), [7,7])
        npt.assert_equal(conf_step.cpu().numpy(), [15,16])
        

    def test_harmConstr(self):
        coords_batch = self.coords_batch.clone()
        coords = coords_batch.coords
        harm_constr = torch.full((coords.shape[0],), .5 )
        e_helper = EnergyAndGradHelper(self.atomization_energies, self.dummy_model, coords_batch,1,1)
        e_helper = EnergyAndGradHelperHarmConstraint(e_helper, harm_constr)
        
        lbfgs = BatchLBFGS(1, convergence_opts = self.conv_opts)
        min_coor, min_e, min_e_grad, std, stat, conf_step = lbfgs.optimize(coords_batch, e_helper)
        
        npt.assert_allclose( min_e.detach().cpu().numpy(), [4.970, 4.971], atol=1e-3)
        npt.assert_allclose( min_coor.detach().cpu().numpy(), 
                             [[[-0.069, -0.069, -0.069],[-0.056, -0.069, -0.069]],
                              [[-0.069, -0.069, -0.069],[-0.055, -0.069, -0.069]]], atol=1e-2)
        npt.assert_allclose( min_e_grad.detach().cpu().numpy(), 0.0, atol=5e-2)
        npt.assert_equal(stat.cpu().numpy(), [7,7])
        npt.assert_equal(conf_step.cpu().numpy(), [10,10])
        

    def test_fixConstr(self):
        coords_batch = self.coords_batch.clone()
        coords = coords_batch.coords
        fixed = torch.tensor([[False,False],[False,True]])
        e_helper = EnergyAndGradHelper(self.atomization_energies, self.dummy_model, coords_batch,1,1)
        e_helper = EnergyAndGradHelperFixedAtoms(e_helper, fixed)
        
        lbfgs = BatchLBFGS(1, convergence_opts = self.conv_opts)
        min_coor, min_e, min_e_grad, std, stat, conf_step = lbfgs.optimize(coords_batch, e_helper)
        
        npt.assert_allclose( min_e.detach().cpu().numpy(), [4.96, 177.89], atol=1e-2)
        npt.assert_allclose( min_coor.detach().cpu().numpy(), 
                             [[[-0.07, -0.07, -0.07],[-0.07, -0.07, -0.07]],
                              [[-0.07, -0.07, -0.07],[ 1.00,  0.00,  0.00]]], atol=1e-2)
        npt.assert_allclose( min_e_grad.detach().cpu().numpy(), 0.0, atol=5e-2)
        npt.assert_equal(stat.cpu().numpy(), [7,7])
        npt.assert_equal(conf_step.cpu().numpy(), [10,4])
        

    def test_opt_wolfe(self):
        coords_batch = self.coords_batch.clone()
        e_helper = EnergyAndGradHelper(self.atomization_energies, self.dummy_model, coords_batch,1,1)
        lbfgs = BatchLBFGS(0.6, convergence_opts = self.conv_opts, line_search_fn="Wolfe")
        min_coor, min_e, min_e_grad, std, stat, conf_step = lbfgs.optimize(coords_batch, e_helper)

        # Armijo does not converge for this steep case
        npt.assert_allclose( min_e.detach().cpu().numpy(), 4.9632, atol=1e-2)
        npt.assert_allclose( min_coor.detach().cpu().numpy(), -0.0703, atol=1e-2)
        npt.assert_allclose( min_e_grad.detach().cpu().numpy(), 0.0, atol=5e-2)
        npt.assert_equal(stat.cpu().numpy(), [7,7])
        npt.assert_equal(conf_step.cpu().numpy(), [12,19])
        

    def test_opt_armijo(self):
        coords_batch = self.coords_batch.clone()
        e_helper = EnergyAndGradHelper(self.atomization_energies, self.dummy_model, coords_batch,1,1)
        lbfgs = BatchLBFGS(0.6, convergence_opts = self.conv_opts, line_search_fn="Armijo")
        min_coor, min_e, min_e_grad, std, stat, conf_step = lbfgs.optimize(coords_batch, e_helper)
        
        # Armijo does not converge for this steep case
        npt.assert_allclose( min_e.detach().cpu().numpy(), 4.9632, atol=1e-3)
        npt.assert_allclose( min_coor.detach().cpu().numpy(), -0.0704, atol=1e-2)
        npt.assert_allclose( min_e_grad.detach().cpu().numpy(), 0.0, atol=1e-2)
        npt.assert_equal(stat.cpu().numpy(), [7,7])
        npt.assert_equal(conf_step.cpu().numpy(), [11,12])
        
