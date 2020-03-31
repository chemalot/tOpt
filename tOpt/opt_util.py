from enum import IntEnum, unique


OPT_HARM_CONSTRAINT_TAG = "NNP_HARM_CONSTR"
OPT_ENERGY_TAG   = "NNP_Energy_kcal_mol"
OPT_STD_TAG      = "NNP_Energy_stdev_kcal_mol"
OPT_RANGE_TAG    = "NNP_Energy_range_kcal_mol"
OPT_STATUS_TAG   = "NNP_Energy_status"
OPT_FORCE_TAG    = "NNP_Force_kcal_mol_A"
OPT_MAX_FORCE_TAG= "NNP_MAX_Force_kcal_mol_A"
OPT_STEPS        = "NNP_STEPS"


class ConvergenceOpts(object):
    def __init__(self, max_iter: int = 20, max_it_without_decrease: int = 5,
                       convergence_e:    float = 1e-2,
                       convergence_grms: float = 0.1,  convergence_gmax: float = 0.15,
                       convergence_drms: float = 0.0006, convergence_dmax: float = 0.005):
        """ Default values from geometric, converted to kcal and A """
        self.max_iter = max_iter
        self.max_it_without_decrease = max_it_without_decrease
        self.convergence_es    = convergence_e * convergence_e
        self.convergence_gms   = convergence_grms * convergence_grms 
        self.convergence_gmaxs = convergence_gmax * convergence_gmax 
        self.convergence_dms   = convergence_drms * convergence_drms
        self.convergence_dmaxs = convergence_dmax * convergence_dmax

DEFAULT_CONVERGENCE_OPTS = ConvergenceOpts()


@unique
class Status(IntEnum):
    IN_PROGRESS = 0
    GRADIENT_CONVERGED    = 1
    GEOMETRY_CONVERGED    = 2
    ENERGY_CONVERGED      = 4
    ENERGY_NOT_DECREASING = 8
    ITERATIONS_EXCEEDED   = 16
    HIGH_ENERGY           = 32
    
    ALL_CONVERGED = ENERGY_CONVERGED | GEOMETRY_CONVERGED | GRADIENT_CONVERGED
    
    @staticmethod
    def to_string(status_byte):
        if status_byte == Status.IN_PROGRESS:
            return str(Status.IN_PROGRESS)
        
        res = ''
        for stat in set(Status) - {Status.IN_PROGRESS}:
            if status_byte & stat == stat: res += str(stat) + ", "
            
        if len(res) > 0: res = res[0:-2]
        
        return res
    