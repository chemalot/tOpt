# encoding: utf-8
'''
Factory class to hide the actual implementation of the NNP computer.

@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

import logging
import os
import errno
from abc import ABCMeta
from abc import abstractmethod
import torch

from tOpt.unit_conversion import Units
from tOpt.pytorch_computer import PytorchComputer
from tOpt.abstract_NNP_computer import AbstractNNPComputer

log = logging.getLogger(__name__)


class NNPComputerFactoryInterface(metaclass=ABCMeta):
    """
        Interface for a Factory of NNP Computers.
        This class must be overwritten when adapting a new NNP computer to be used
        for the optimizer.
    """
    
    
    @abstractmethod
    def __init__(self, nnpName:str):
        """
            use the nnpName parameter to instantiate the correct NNPComputer.
        """
        pass
    
    @abstractmethod
    def createNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs) \
        -> AbstractNNPComputer :
        pass
    
    
    
class ExampleNNPComputerFactory(NNPComputerFactoryInterface):
    """ Example Factory class for NNP_Computer """
    
    def __init__(self, nnpName:str):
        """
            Factory class that creates an AbstractNNPComputer given a NNP name.
            If the nnpName is a directory name it will assume that the NNP configuration is 
            for the neurochem implementation of ANI.
            If the nnPName is a file name if it will raise an exception
            If it is neither a directory nor a file name it will create a 
            Dummy NNP just for demo purposes.
        """
        # try path as given, if this does not work try prepending $NNP_PATH
        if not os._exists(nnpName):
            if os._exists(os.environ.get("NNP_PATH","") + "/" + nnpName):
                nnpName = os.environ.get("NNP_PATH","") + "/" + nnpName
                
        self.nnp_name = nnpName
        
        if os.path.isdir(nnpName):
            self.createNNP = self._createANI

        elif os.path.isfile(nnpName):
            self.createNNP = self._otherNNP

        elif self.nnp_name == 'torchani-ani1x':
            self.createNNP = self._ANI1xNNP

        elif self.nnp_name == 'torchani-ani2x':
            self.createNNP = self._ANI2xNNP

        else:
            self.createNNP = self._dummyNNP 

    

    def createNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        """ to be replaced by specific implementation"""
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.nnp_name)
    
        
    def _createANI(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        # import here to avoid dependency collision pytorch vs ANI
        from tOpt import ANI_computer
        return ANI_computer.ANIComputer(self.nnp_name, 
                                        outputGrad, compute_stdev, energyOutUnits=energyOutUnits)
        

    def _ANI1xNNP(self, outputGrad: bool, compute_stdev: bool, energyOutUnits: Units = Units.KCAL, **kwArgs):
        """
           just a call to use ANI2x pytorch potential
        """
        from tOpt.torchani_computer import ANI1xNet

        log.warning('Using Torchani-ANI1x model!!!!!')

        net = ANI1xNet()
        atoms = [1, 6, 7, 8]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        atomization_e = torch.tensor([0] * (max(atoms) + 1), dtype=torch.float).to(device)

        return PytorchComputer(net, atoms, atomization_e, outputGrad, compute_stdev, torch.float, 10, 1, False)


    def _ANI2xNNP(self, outputGrad: bool, compute_stdev: bool, energyOutUnits: Units = Units.KCAL, **kwArgs):
        """
           just a call to use ANI2x pytorch potential
        """
        from tOpt.torchani_computer import ANI2xNet

        log.warning('Using Torchani-ANI2x model!!!!!')

        net = ANI2xNet()
        atoms = [1, 6, 7, 8, 9, 16, 17]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        atomization_e = torch.tensor([0] * (max(atoms) + 1), dtype=torch.float).to(device)

        return PytorchComputer(net, atoms, atomization_e, outputGrad, compute_stdev, torch.float, 10, 1, False)


    def _dummyNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        """
           just a dummy pytorch potential that pulls all atom coordiantes to -0.703
        """
        from tOpt.pytorch_computer import DummyNet
        
        log.warning('Using DummyNet just for testing!!!!!')
        
        net = DummyNet()
        atoms = [1,6,7,8,9,16,17]
        atomization_e = torch.tensor([0]*(max(atoms)+1), dtype=torch.float)
        
        return PytorchComputer(net, atoms, atomization_e, outputGrad, compute_stdev, torch.float, 10, 1, False)
            
    def _otherNNP(self, outputGrad:bool, compute_stdev:bool, energyOutUnits:Units = Units.KCAL, **kwArgs):
        """
           Here we could call the constructor for a NNP_Computer implemented differently than NeuroChem
        """
        
        raise NotImplemented("No other NNP was implemented")