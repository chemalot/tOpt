# encoding: utf-8
'''
This file contains the main method to run the tOpt optimizer over an sdf file.
The main() method implements the command line parsing.
By default this uses the ExampleNNPComputerFactory to create the NNPComputer
This supports the neurochem based NNPComputer and a DummyComputer using pytorch.


To adapt it to a different computer implement a subclass of AbstractNNPComputer
and NNPComputerFactoryInterface so that your NNP computer is used. 
Then implement a script similar to SDFANIMOptimizer.py to overwrite the default
factory method when calling main().


@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

import sys

import argparse
import torch
from argparse import RawDescriptionHelpFormatter

from cddlib.chem.io import get_mol_input_stream, get_mol_output_stream
from tOpt.batch_optimizer import BatchOptimizer
import logging
from tOpt.NNP_computer_factory import ExampleNNPComputerFactory, NNPComputerFactoryInterface
log = logging.getLogger(__name__)

TESTRUN = 0
PROFILE = 0


torch.set_num_threads(4)


def main(nnp_comput_factory:NNPComputerFactoryInterface = ExampleNNPComputerFactory, 
         argv=sys.argv): # IGNORE:C0111
    """
        nnp_comput_factory: factory for nnp computer to be used
        argv: list of command line arguments to overwrite sys.argv, 0th argument is not used
    """
    
    from cddlib.util import log_helper
    sys.argv = argv

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Optimize all molecules in sdf", formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-in",  dest="inFile", type=str, default=".sdf",
                        help="input file (def=.sdf)")
    parser.add_argument("-out", dest="outFile",  type=str, default=".sdf",
                        help="output file (def=.sdf)")
    parser.add_argument('-conf' ,  metavar='fileName', type=str, required=True,
                        help='nnp configuration file *.json or ANI directory name')
    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')
    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')
    parser.add_argument('-harm_constr' ,  metavar='k[kcal/mol/a]' ,  type=str, default=None,
                        help='Force Constant of harmonic constraint that pulls input molecules back to their input coordinates. ' \
                             'if this is a "," separated list of n values each input conf is minimized n times with different constraints.')
    parser.add_argument('-constraint' ,  metavar='type' ,  type=str, default=None, choices=['heavyAtom'],
                        help='Supported: heavyAtom ')
    parser.add_argument('-trust',   type=float, default=1, help='Starting trust radius.')
    parser.add_argument('-maxiter', type=int, default=300, help='Maximum number of optimization steps.')
    parser.add_argument('-lineSearch', type=str, default=None, choices=['Armijo', 'Wolfe'], help='Line search algorithm to be used.')
    parser.add_argument('-lbfgsHistSize', type=int, default=200, help='History size used to approximate second derivatives.')
    parser.add_argument('-prune_high_energy', nargs=2, default=None, 
                        help='2 arguments (freq drop_fract) every freq sycles the reamining drop_fract confs will be dropped. '
                            +'Note: this makes only sense if all conformer have the same structure.')
    parser.add_argument('-computeForce', default=False, action='store_true',
                        help='Output force at optimized coordinates')
    parser.add_argument('-plotName', type=str, help='if given a plot and tab separated file with the cycles is generated.')

    # Process arguments
    args = parser.parse_args()

    inFile          = args.inFile
    outFile         = args.outFile
    constraint      = args.constraint
    harm_constr     = args.harm_constr
    maxiter         = args.maxiter
    trust           = args.trust
    line_search     = args.lineSearch
    out_grad        = args.computeForce
    plot_name       = args.plotName
    lbfgs_hist_size = args.lbfgsHistSize
    prune_high_energy = args.prune_high_energy
    prune_high_energy_freq = None
    prune_high_energy_fract = None
    if prune_high_energy:
        prune_high_energy_freq = int(prune_high_energy[0])
        prune_high_energy_fract = float(prune_high_energy[1])
    
    log_helper.initialize_loggger(__name__, args.logFile)    

    nnp_factory = nnp_comput_factory(args.conf)
    nnp_computer = nnp_factory.createNNP(True, True, **vars(args))
    if harm_constr:
        harm_constr = [ float(v) for v in harm_constr.split(",")]
     
    with get_mol_output_stream(outFile) as out, \
         get_mol_input_stream(inFile) as molS,  \
         BatchOptimizer(nnp_computer, molS, maxiter, learn_rate=trust, lbfgs_hist_size=lbfgs_hist_size,
                        constraint=constraint, harm_constr=harm_constr,
                        prune_high_energy_freq=prune_high_energy_freq, prune_high_energy_fract=prune_high_energy_fract,
                        line_search=line_search, out_grad=out_grad, plot_name=plot_name) as sdfOptizer:
        for mol in sdfOptizer:
            out.write_mol(mol)
    
    return 0


if __name__ == "__main__":
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE == 1:
        import cProfile
        import pstats
        profile_filename = 'tOpt.optimizer_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = sys.stderr
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    elif PROFILE == 2:
        import line_profiler
        import atexit
        from tOpt.batch_lbfgs import BatchLBFGS
        
        prof = line_profiler.LineProfiler()
        atexit.register(prof.print_stats)

        prof.add_function(BatchLBFGS.step)
        
        prof_wrapper = prof(main)
        prof_wrapper()
        sys.exit(0)
        
    elif PROFILE == 3:
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()

        main()
                
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        sys.exit(0)
    
    elif PROFILE == 4:
        # this needs lots of memory
        import torch
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            main()
        # use chrome://tracing  to open file
        prof.export_chrome_trace("profile_Chrome.json")
        sys.exit(0)

    sys.exit(main())
