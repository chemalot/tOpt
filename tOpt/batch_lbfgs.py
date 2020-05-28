import torch


##################################################################################
# The initial code was copyed from the pytorch implementation in 2019
# therefore part of this code is 
# Copyright (c) 2016-     Facebook, Inc and the developers of the lbfgs.py package
###################################################################################   


import logging
from tOpt.opt_util import *
from tOpt.coordinates_batch import SameSizeCoordsBatch
from tOpt.abstract_NNP_computer import EnergyAndGradHelperInterface
log = logging.getLogger(__name__)

torch.set_printoptions(linewidth=200)

INF = float("Inf")


class OptState(object):
    """ This holds all optimization variables that can be reduced when a conformation completes 
        during a batch optimization.
        All variables have n_conf as first dimension
    """
    
    def __init__(self, lbfgs, n_confs, n_atoms, dtype):
        
        self.n_confs = n_confs
        self.dtype= dtype
        
        self.in_confIdx   = torch.arange(n_confs, dtype=torch.int64, device=lbfgs.device)
        #self.status_actives = torch.ones(n_confs, dtype=torch.uint8, device=lbfgs.device)

        self.lr = torch.full((n_confs,), lbfgs.initial_lr, device=lbfgs.device)
        self.loss = None
        self.std  = None
        self.flat_grad = None
        self.prev_flat_grad = None
        #self.abs_grad_sum = None
        self.d = None
        self.t = None
        self.old_dirs = MultiInstanceHistory(lbfgs.history_size, n_confs, n_atoms*3, dtype, lbfgs.device)
        self.old_stps = MultiInstanceHistory(lbfgs.history_size, n_confs, n_atoms*3, dtype, lbfgs.device)
        self.H_diag = torch.full((n_confs,), 1, dtype=dtype, device=lbfgs.device)
        self.prev_loss = None
        self.last_decreased = torch.zeros(n_confs, dtype=torch.int16, device=lbfgs.device)
            
        self.iters_with_no_linsearch   = torch.zeros(n_confs, dtype=torch.int16, device=lbfgs.device)
        self.iters_with_high_linsearch = torch.zeros(n_confs, dtype=torch.int16, device=lbfgs.device)
        
        
    def filter_(self, fltr:torch.tensor):
        self.in_confIdx = self.in_confIdx[fltr]
        self.lr = self.lr[fltr]
        self.loss = self.loss[fltr]
        if self.std is not None: self.std  = self.std[fltr]
        self.flat_grad = self.flat_grad[fltr]
        self.prev_flat_grad = self.prev_flat_grad[fltr]
        self.d = self.d[fltr] 
        self.t = self.t[fltr]
        self.H_diag = self.H_diag[fltr] 
        self.prev_loss = self.prev_loss[fltr]
        self.last_decreased = self.last_decreased[fltr]

        self.old_dirs.filter_(fltr)
        self.old_stps.filter_(fltr)
            
        self.iters_with_no_linsearch   = self.iters_with_no_linsearch[fltr]
        self.iters_with_high_linsearch = self.iters_with_high_linsearch[fltr]

        self.n_confs = self.in_confIdx.shape[0]
        #self.status_actives = torch.ones(self.n_confs, dtype=torch.uint8, device=self.status_actives.device)


class BatchLBFGS():
    """Implements L-BFGS algorithm for a set of conformation.

    Arguments:
        param([tensor]): coordinates to optimize
        lr (float): learning rate (default: 1)
        convergence_opts: settings for convergence
        history_size (int): update history size (default: 100).
    """

    def __init__(self, lr: int = 1,  
                 convergence_opts:ConvergenceOpts = DEFAULT_CONVERGENCE_OPTS,
                 history_size:int = 100, line_search_fn:str=None,
                 prune_high_energy_freq:int = 9e9, prune_high_energy_fract:float = 0,
                 plot_name:str = None, device = None):
        """
            Arguments:
            convergence_opts -- Configuration ofr the convergence criteria
            history_size -- size of history used to approximate second derivatives
            line_search_fn -- currently 'None|Armijo|Wolfe' (Wolfe is not well tested'
            prune_high_energy_freq -- after how many cycles to try pruning high energy conformations
            prune_high_energy_fract -- each time when pruning high energy conformation 
                                       this fraction of a conformations is dropped 
            plot_name -- if given a lot displaying the cyvle vs energy trends is created
        """
        
        assert prune_high_energy_freq is None or prune_high_energy_freq > 0
        assert prune_high_energy_fract is None \
               or (prune_high_energy_fract >= 0 and prune_high_energy_fract <= 1)
        
        self.initial_lr = lr
        self.convergence_opts = convergence_opts
        self.history_size=history_size
        self.line_search_fn=line_search_fn
        self.prune_high_energy_freq  = prune_high_energy_freq  if prune_high_energy_freq else 9e9
        self.prune_high_energy_fract = prune_high_energy_fract if prune_high_energy_fract else 0
        self.plot_name=plot_name
        self.plot_data = [] if plot_name else None
        self.device = device
    
    def _add_grad(self, coords:torch.tensor, trust:torch.tensor, 
                  update:torch.tensor ) -> torch.tensor:
        """ add gradient to coordinates to take a step """
        # view as to avoid deprecated point wise semantics
        step = trust.reshape(-1,1,1)*update.view_as(coords)
                
        coords.data.add_(step)
        return trust #* cutoff_multiplier
    

    def trust_by_step_size(self, coords:torch.tensor, update:torch.tensor, 
                           max_displace:float ) -> torch.tensor:
        """ compute the trust necessary to reach a given step size
            given an update direction
        """
        step = update.view_as(coords)
        
        # Scale to move at most trust
        abs_step = torch.sum(step*step, dim=-1).sqrt_()
        abs_step = abs_step.max(dim=-1)[0]
        
        if log.isEnabledFor(logging.DEBUG):
            log.debug("    max Step %s", abs_step.max())
        
        multiplier = max_displace / abs_step
        return multiplier
    

    def _add_grad_wFilter(self, coords:torch.tensor, trust:torch.tensor,
                          update:torch.tensor, fltr:torch.tensor ) -> torch.tensor:
        """ Add gradient to coordinates that are still active.
        """
        # view as to avoid deprecated point wise semantics
        trust = trust[fltr]
        update = update[fltr]
        dat = coords[fltr]
        step = trust.reshape(-1,1,1)*update.view_as(dat)
                
        coords.data[fltr] += step
        return trust #* cutoff_multiplier


    def wolfe_lineSearch(self, n_iter:int, coords:torch.tensor, 
                              st:OptState, energy_helper:EnergyAndGradHelperInterface) -> int:
        """ Do a Wolfe Line search: https://en.wikipedia.org/wiki/Wolfe_conditions
            !!!!!!!!!!!!!!!!!! I think this is still buggy !!!!!!
        """
        
        # directional derivative
        gtd = torch.sum(st.flat_grad * st.d, dim=1)  # g * d

        ls_func_evals = 0
        c1 = 1e-4
        c2 = 0.8
        max_ls = 10
        org_coords = coords.detach().clone()
        st.t = self._add_grad(coords, st.t, st.d)
        
        F_new, std = energy_helper.compute_energy()
        st.flat_grad = energy_helper.compute_grad().reshape(st.n_confs,-1)
        gtd_new = torch.sum(st.flat_grad * st.d, dim=1)  # g * d
        
        ls_step = 0
        alpha = torch.full_like(st.t, 0.)
        beta  = torch.full_like(st.t, INF)
        while ls_step < max_ls:
            ls_func_evals += 1
            
            bad_armjio = F_new > st.loss + c1*st.t*gtd # Armijo condition
            #bad_armjio &= (st.status_actives & Status.ALL_CONVERGED) < Status.ALL_CONVERGED
            
            bad_wolfe = (gtd_new.abs() > c2 * gtd.abs()) & ~bad_armjio
            #bad_wolfe &= (st.status_actives & Status.ALL_CONVERGED) < Status.ALL_CONVERGED
            
            any_line_search = bad_armjio | bad_wolfe
            
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f'{n_iter}.{ls_step} loss {st.loss[0:5].detach().cpu().numpy()} F_new{F_new[0:5].detach().cpu().numpy()}')

            if any_line_search.sum() == 0:
                break
            
            lower_e = F_new<st.loss
            gtd[lower_e] = gtd_new[lower_e]
            st.loss[lower_e] = F_new[lower_e]
            if not st.std is None: st.std[lower_e]  = std[lower_e]

            beta[bad_armjio] = st.t[bad_armjio]
            alpha[bad_wolfe] = st.t[bad_wolfe]
            
            st.t[any_line_search] = 0.5 * (alpha + beta)[any_line_search]
            st.t[bad_wolfe & (beta == INF)] = alpha[bad_wolfe & (beta == INF)] * 2.
            
            if log.isEnabledFor(logging.DEBUG):
#                         log.debug(f'{n_iter}.{ls_step} loss {st.loss[0:5].detach().cpu().numpy()} F_new{F_new[0:5].detach().cpu().numpy()}')
#                         log.debug(f'   x12 {coords[0:5,0:2,0].detach().cpu().numpy()}')                    
                log.debug(f'    alpha {alpha} beta {beta} t {st.t}')
            
            coords.data.copy_(org_coords)
            self._add_grad_wFilter(coords, st.t, st.d, any_line_search)
            
            # filtering did not save any time
            F_new, std = energy_helper.compute_energy() 
            st.flat_grad = energy_helper.compute_grad().reshape(st.n_confs,-1)
            gtd_new = torch.sum(st.flat_grad * st.d, dim=1)  # g * d
            #F_new[bad_armjio] = energy_helper.compute_energy_with_filter(bad_armjio) 
            
            ls_func_evals += 1
            ls_step += 1
        
        st.lr = st.t
        st.lr[st.lr < 0.000001] = 0.000001

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f'lr: {st.lr} t:{st.t}')
        
        st.loss = F_new
        st.std  = std
        
        if n_iter != self.convergence_opts.max_iter:
            # re-evaluate function only if not in last iteration
            # the reason we do this: in a stochastic setting,
            # no use to re-evaluate that function here
            #st.abs_grad_sum = st.flat_grad.abs().sum(1)
            if log.isEnabledFor(logging.DEBUG):
                log.debug('{} loss: {}, coords[0:5,1,0] {}'.format(
                    n_iter, st.loss[0:5].detach().cpu().numpy(), coords[0:5,0,0].detach().cpu().numpy()))
    
        return ls_func_evals
    
    
    def armijo_line_search(self, n_iter:int, coords:torch.tensor, 
                               st:OptState, energy_helper:EnergyAndGradHelperInterface) -> int:
        """
            Perform an Armijo line search.
        """
        
        eta = 4   # the higher eta the closer to the original move will be the line search
        c1 = 1e-4
        max_ls = 5
        
        # directional derivative
        gtd = torch.sum(st.flat_grad * st.d, dim=1)  # g * d

        line_search_done = torch.zeros(st.n_confs, dtype=torch.int16, device=self.device)
        st.t = self._add_grad(coords, st.t, st.d)
        
        F_new, std = energy_helper.compute_energy()
        
        ls_step = 0
        ls_func_evals = 0
        while ls_step < max_ls:
            ls_func_evals += 1
            bad_steps = F_new > st.loss + c1*st.t*gtd # Armijo condition
            #bad_steps &= (st.status_actives & Status.ALL_CONVERGED) < Status.ALL_CONVERGED
            line_search_done += bad_steps.to(dtype=torch.int16)

            if log.isEnabledFor(logging.DEBUG):
                log.debug(f'{n_iter}.{ls_step} loss {st.loss[0:5].detach().cpu().numpy()} F_new{F_new[0:5].detach().cpu().numpy()}')
#                         log.debug(f'   x12 {coords[0:5,0:2,0].detach().cpu().numpy()}')                    
                         
            if bad_steps.sum() == 0:
                break
            
            t_new = st.t.clone()
            if ls_step == 0:
                st.t[bad_steps] = st.t[bad_steps]/eta

            if log.isEnabledFor(logging.DEBUG):
                log.debug(f'    bad {bad_steps} t-t_new {st.t-t_new}')
            
            self._add_grad_wFilter(coords, st.t-t_new, st.d, bad_steps)
            
            # filtering did not save any time
            F_new, std = energy_helper.compute_energy() 
            #F_new[bad_steps] = energy_helper.compute_energy_with_filter(bad_steps) 
            
            ls_func_evals += 1
            ls_step += 1
        
        ##############################################
        # adjust learning rate so that we do have occasional but not
        # frequent line searches
        
        st.iters_with_no_linsearch[line_search_done == 0]           += 1
        st.iters_with_no_linsearch[line_search_done > 0]            -= 1
        st.iters_with_no_linsearch[st.iters_with_no_linsearch < 0]   = 0
        st.lr[st.iters_with_no_linsearch >= 5]                      *= 1.2
        st.iters_with_no_linsearch[st.iters_with_no_linsearch >= 5] -= 2

        st.lr[line_search_done > 0] = st.t[line_search_done > 0]
        st.lr[st.lr < 0.000001] = 0.000001
        
#                 st.iters_with_high_linsearch                                   += line_search_done
#                 st.iters_with_high_linsearch[st.iters_with_high_linsearch > 6]  = 6
# #                 st.iters_with_high_linsearch[line_search_done == 0]          -= 1
# #                 st.iters_with_high_linsearch[st.iters_with_high_linsearch < 0]   = 0
#                 st.lr[st.iters_with_high_linsearch >= 4]                        /= 1.2
#                 st.iters_with_high_linsearch[st.iters_with_high_linsearch >= 4] -= 1
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f'lr: {st.lr}')
        
        st.loss = F_new
        st.std  = std
        
        if n_iter != self.convergence_opts.max_iter:
            # re-evaluate function only if not in last iteration
            # the reason we do this: in a stochastic setting,
            # no use to re-evaluate that function here
            st.flat_grad = energy_helper.compute_grad().reshape(st.n_confs,-1)
            #st.abs_grad_sum = st.flat_grad.abs().sum(1)
            if log.isEnabledFor(logging.DEBUG):
                log.debug('{} loss: {}, coords[0:5,1,0] {}'.format(
                    n_iter, st.loss[0:5].detach().cpu().numpy(), coords[0:5,0,0].detach().cpu().numpy()))

        return ls_func_evals
        
    
    
    def optimize(self, coords_batch: SameSizeCoordsBatch, energy_helper):
        """Performs a geometry optimization on a batch of conformations

        Arguments:
            energy_helper: wrapper around NNP engine
            coords_batch: batch of conformations

        Returns
        -------
        min_coords: tensor coordinates at which lowest energy was found
        Energies: tensor, 
        Status: tensor of ints corresponding to Status enum
        conf_step: tensor step number for each conformation
                
        Side Effects
        ------------
        The coordinates in coords_batch will have changed but will not necessarily
        be the coordinates of the minimum 
        """

        a_coords = coords_batch.coords
        n_total_confs = a_coords.shape[0]
        n_atoms = a_coords.shape[1]
        
        func_evals = 0

        # evaluate initial f(x) and df/dx
        loss, std = energy_helper.compute_energy()    # loss[nConf]
        st = OptState(self, n_total_confs, n_atoms, loss.dtype)
        st.loss = loss
        st.std = std
        min_loss = st.loss.detach().clone()
        
        minE_no_constraints = energy_helper.energy_no_constraint().detach().clone()
        min_std = torch.full_like(minE_no_constraints, -1)
        st.flat_grad = energy_helper.compute_grad().reshape(st.n_confs,-1)
        min_grad_square_max = torch.full((n_total_confs,), 9e20, dtype=a_coords.dtype, device=self.device)
        #st.abs_grad_sum = st.flat_grad.abs().sum(1)
        
        status      = torch.zeros((n_total_confs,),dtype=torch.uint8, device=self.device)
        is_active   = torch.ones((n_total_confs,), dtype=torch.uint8, device=self.device).bool()
        conf_steps  = torch.full((n_total_confs,), -1, dtype=torch.int16, device=self.device)
        minE_coords = a_coords.detach().clone()
        minE_grad   = torch.full((n_total_confs,n_atoms*3), -999, dtype=a_coords.dtype, device=self.device)
        
        current_evals = 1
        func_evals += 1
        n_iter = 0

        # optimize for a max of max_iter iterations
        while n_iter < self.convergence_opts.max_iter:
            # keep track of nb of iterations
            n_iter += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if n_iter == 1:
                st.d = st.flat_grad.neg()
            else:
                # d: direction of step
                # s: step = direction * trust of step
                # y: delta gradient in step (grad - prev_grad) = vector of gradient change
                # ys: sum(y * step) 
                # do lbfgs update (update memory)
                y = st.flat_grad.sub(st.prev_flat_grad)
                
                s = st.d*st.t.reshape(-1,1)
                
                ys = torch.sum(y * s, dim=1)
                
                is_valid_step = ys > 1e-10 # DIAL BACK TO 10E-6,4,5, look at RFO, rational function optimization
                                           # reach out to lee-ping or roland king (optking)
                                           # try occasionally setting h_diag to 1
                                           # look into getting code from psi4 to convert, (little bit of a mess) cartesian to internal pyoptking
                                           # pyberny: has nicer code for internal coordinates
                                           #    maybe can get initial hessian guess in internal coordinates and project back to xyz and use as first guess
                st.old_dirs.push_if(is_valid_step, y)
                st.old_stps.push_if(is_valid_step, s)
                y = y[is_valid_step]         
                st.H_diag[is_valid_step] = ys[is_valid_step] / torch.sum(y * y, dim=1)
                d_not_valid_steps = st.flat_grad[~is_valid_step].neg()   #d[~is_valid_step]
                                
                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                
                ro = 1. / torch.sum(st.old_dirs.container * st.old_stps.container, dim=2)
                ro[torch.isinf(ro)] = 1e-10

                al = torch.zeros((self.history_size,st.n_confs), dtype=loss.dtype, device=self.device)
                
                num_old = st.old_dirs.count_hist.max()
#                 log.debug("old_dirs {}\n{}".format(num_old, st.old_dirs.container[0:num_old]))
                
                q = st.flat_grad.neg()
                for i in range(num_old):
                    al[i] = torch.sum(st.old_stps.container[i]* q, dim=1) * ro[i]
                    q.add_(-al[i].reshape(-1,1) * st.old_dirs.container[i])
                st.d = r = q * st.H_diag.reshape(-1,1)                
#                 log.debug("al {}".format(al[0:num_old]))
#                 log.debug("q {}".format(q))
#                 log.debug("d {}".format(st.d))
#                 log.debug("H_diag {}".format(st.H_diag))
#                 
                for i in range(num_old - 1, -1, -1):
                    be_i = torch.sum(st.old_dirs.container[i] * r, dim=1) * ro[i]
#                     log.debug("{} od {}".format(i,st.old_dirs.container[i]))
#                     log.debug("{} r  {}".format(i,r))
#                     log.debug("{} ro {}".format(i,ro[i]))
#                     log.debug("{} bei {}".format(i,be_i))
                    r.add_((al[i] - be_i).reshape(-1,1) * st.old_stps.container[i])
#                     log.debug("{} r {}".format(i,r))
#                 st.d[~is_valid_step] = d_not_valid_steps

            if st.prev_flat_grad is None:
                st.prev_flat_grad = st.flat_grad.clone()
            else:
                st.prev_flat_grad.copy_(st.flat_grad)
            st.prev_loss = st.loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for trust
            if n_iter == 1:
                st.t = self.trust_by_step_size(a_coords, st.d, 0.1)                
            else:
#                 log.debug(lr)
                st.t = st.lr.clone()
                #if n_iter > 10: st.t = st.t + st.t * random.gauss(0, 0.1)

            if self.line_search_fn == "Armijo":
                ls_func_evals = self.armijo_line_search(n_iter, a_coords, st, energy_helper)
                
            elif self.line_search_fn == "Wolfe":
                ls_func_evals =self.wolfe_lineSearch(n_iter, a_coords, st, energy_helper)
            
            else:
                # directional derivative
                #gtd = torch.sum(st.flat_grad * st.d, dim=1)  # g * d

                # no line search, simply move with fixed-step
                st.t = self._add_grad(a_coords, st.t, st.d)
                
                if n_iter != self.convergence_opts.max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    st.loss, st.std = energy_helper.compute_energy()
                    st.flat_grad = energy_helper.compute_grad().reshape(st.n_confs,-1)
                    #st.abs_grad_sum = st.flat_grad.abs().sum(1)   # not needed
                    ls_func_evals = 1


            # update func eval
            current_evals += ls_func_evals
            func_evals  += ls_func_evals


            ############################################################
            # check conditions
            ############################################################
            #
            # active conformers are conformers that have not convereged
            # all variable in st. (OptState) are limited to active conformers
            #
            # local variables that have only elements for active conformers in the following evaluation
            # code have an "a_" prefix
            #
            
            status[is_active] = 0
            
            
            # a_ prefix flags tensors on active conformers only, just as st.
            a_flat_grad_sqare      = st.flat_grad.pow(2)
            a_flat_grad_MSE        = a_flat_grad_sqare.sum(1) / n_atoms 
            a_flat_grad_square_max = a_flat_grad_sqare.max(1)[0]

            if log.isEnabledFor(logging.DEBUG):
                log.debug('{} loss: {}, max_grad: {} a_coords[0:5,1,0] {}'.format(
                    n_iter, 
                    st.loss[0:5].detach().cpu().numpy(),
                    a_flat_grad_square_max.sqrt().detach().cpu().numpy(),
                    a_coords[0:5,0,0].detach().cpu().numpy()))

            a_deltaLoss = st.loss - min_loss[is_active]
            st.last_decreased[a_deltaLoss < 0] = n_iter

            # check if energy is converged
            a_e_decreased = a_deltaLoss * a_deltaLoss < self.convergence_opts.convergence_es
            e_decreased_idx = st.in_confIdx[a_e_decreased]
            status[e_decreased_idx] |= Status.ENERGY_CONVERGED
                            
            # flag geometry as "decreased" if it went down or stated ~ same but gradient decreased 
            # allow for 10x more tolerance because if forces are decreased
            a_deltaGrad = a_flat_grad_square_max - min_grad_square_max[is_active]
            a_e_decreased = (  (a_deltaLoss < 0)
                             | ((a_deltaLoss <= self.convergence_opts.convergence_es * 10) 
                               &(a_deltaGrad < 0))) 
            e_decreased_idx = st.in_confIdx[a_e_decreased]
            
            # store best geometry when geom is found
            if e_decreased_idx:
                minE_coords[e_decreased_idx] = a_coords[a_e_decreased].detach_()
                minE_no_constraints[e_decreased_idx] = \
                            energy_helper.energy_no_constraint()[a_e_decreased].detach_()
                minE_grad[e_decreased_idx] = st.flat_grad[a_e_decreased].detach_()
                min_loss [e_decreased_idx] = st.loss[a_e_decreased].detach().clone()
                if not st.std is None: min_std[e_decreased_idx] = st.std[a_e_decreased]
                min_grad_square_max[e_decreased_idx] = a_flat_grad_square_max[a_e_decreased]
            
            dummy = n_iter - st.last_decreased >= self.convergence_opts.max_it_without_decrease
            status[st.in_confIdx[dummy]] |=   Status.ENERGY_NOT_DECREASING
                
            dummy =   (a_flat_grad_MSE        < self.convergence_opts.convergence_gms) \
                    & (a_flat_grad_square_max < self.convergence_opts.convergence_gmaxs)
            status[st.in_confIdx[dummy]] |= Status.GRADIENT_CONVERGED 

            dt_square = st.d*st.t.reshape(-1,1)
            dt_square *= dt_square
            dummy =   (dt_square.sum(1)/ n_atoms <= self.convergence_opts.convergence_dms) \
                    & (dt_square.max(1)[0]       <= self.convergence_opts.convergence_dmaxs)
            status[st.in_confIdx[dummy]] |= Status.GEOMETRY_CONVERGED
            
            if self.plot_data is not None:
                rec = torch.full((n_total_confs,), float('nan'), dtype=st.loss.dtype, device=self.device)
                rec[is_active] = st.loss
                self.plot_data.append([n_iter, rec])
                
            actives_finished = (status[is_active] >= Status.ALL_CONVERGED) 

            actives_finished_in_idx = st.in_confIdx[actives_finished]
            # set conf_steps for not-finished conformations
            conf_steps[actives_finished_in_idx] = n_iter

            if (~actives_finished).sum() == 0:
                log.info(f"all finished (nIter={n_iter}): {minE_no_constraints}")                                         
                break

            if n_iter == self.convergence_opts.max_iter:
                log.info(f"MAX_ITER reached: {minE_no_constraints}")
                status[ st.in_confIdx[~actives_finished] ] |= Status.ITERATIONS_EXCEEDED
                break
            if current_evals >= self.convergence_opts.max_iter * 3:                                  
                status[ st.in_confIdx[~actives_finished] ] |= Status.ITERATIONS_EXCEEDED                                         
                log.info(f"MAX_EVAL reached: {minE_no_constraints}")                                         
                break
            
            # filter out completed conformations
            if actives_finished.sum() > 0:
                a_future_actives = ~actives_finished
                is_active[actives_finished_in_idx] = 0
                st.filter_(a_future_actives)
                energy_helper.filter_(a_future_actives) # also filters coords_batch
                a_coords = coords_batch.coords

            if n_iter % self.prune_high_energy_freq == 0 and n_iter > 5:
                # this is a global minimum search, to speed up: prune 
                # conformations with the highest energy
                drop_count = int(st.loss.shape[0] * self.prune_high_energy_fract)
                if drop_count < 1: continue
                a_to_drop = st.loss.argsort(descending=True)[0:int(st.loss.shape[0] * self.prune_high_energy_fract)]
                to_drop_idx = st.in_confIdx[a_to_drop]
                is_active[to_drop_idx] = 0
                status[to_drop_idx] = Status.HIGH_ENERGY
                conf_steps[to_drop_idx] = n_iter
                a_future_actives = torch.ones_like(st.loss, dtype=torch.uint8).bool()
                a_future_actives[a_to_drop] = 0
                st.filter_(a_future_actives)
                energy_helper.filter_(a_future_actives) # also filters coords_batch
                a_coords = coords_batch.coords
            
        if self.plot_data:
            self.plot(n_total_confs)

        # set conf_steps for not-finished conformations
        conf_steps[conf_steps == -1] = n_iter
            
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f'lbfgs completed e={minE_no_constraints}, maxgrad^2={min_grad_square_max}')
        status[(status > Status.ALL_CONVERGED) & (status < Status.HIGH_ENERGY)] = Status.ALL_CONVERGED

        if st.std is None: min_std = None
        return minE_coords, minE_no_constraints, minE_grad, min_std, status, conf_steps


    def plot(self, n_confs):
        """ Plot cycle vs energy curves """
        
        import pandas as pd
        import numpy as np
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import csv
                
        n_iter = len(self.plot_data)
        
        data = np.ndarray((n_iter, n_confs+1))
        data[:,0] = [i[0] for i in self.plot_data]
        data[:,1:] = [i[1].detach().cpu().numpy() for i in self.plot_data]

        df=pd.DataFrame(data)
        names = ['iter']
        for i in range(n_confs): names.append(f'c{i+1}')
        df.columns = names
        df.to_csv(f"{self.plot_name}.tab", sep="\t", quoting=csv.QUOTE_NONE)        

        d = data[:,1:].reshape(-1)
        d = d[~np.isnan(d)]
        mine = d.min() - 0.01
        for i in range(n_confs): 
            data[:,i+1] -= mine
        
        df=pd.DataFrame(data)
        names = ['iter']
        for i in range(n_confs): names.append(f'c{i+1}')
        df.columns = names
        
        colors = (0,0,0)
        area = 10
         
        # Plot
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1,1,1)
        for i in range(n_confs):
            ax.plot('iter', f'c{i+1}', data=df)
        ax.set_yscale('log')

        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.savefig(f'{self.plot_name}.png')


class MultiInstanceHistory(object):
    """ Container for the history of the optimization needed to approximate the 
        second derivatives for a batch of n_instance conformations.
    """
    def __init__(self, hist_size:int, n_instance:int, instance_size:int, dtype, device=None):
        """
        Arguments:
            hist_size: size of history, the larger the more accurate the 2nd derivatives will be over time
            n_instance: number of conformations
            instance_size: number of atoms
            dtype: data type of the energy
        """
        self.hist_size = hist_size
        self.hist_sizet = torch.tensor([hist_size], dtype=torch.int64, device=device)
        self.n_instance = n_instance
        self.container  = torch.zeros((hist_size,n_instance, instance_size), dtype=dtype, device=device)
        self.count_hist = torch.zeros((n_instance,), dtype=torch.int64, device=device)
        
        
    def push_if(self, condition:torch.tensor, values:torch.tensor):
        """ add values to container for instances where condition is 1
        
        Parameter:
        ---------
        condition tensor [n_instance,dtype=bool] flagging elements in "values" that
                  should be added to the container 
        values tensor[n_instance, instance_size] to be added to this history
        
        """

        condition_float = condition.to(dtype=values.dtype)
        
        shifted = self.container * condition_float.unsqueeze(-1)
        shifted = shifted[0:self.hist_size-1]
        
        not_shifted = self.container * (1. - condition_float).unsqueeze(-1)
        not_shifted = not_shifted[1:self.hist_size]
        
        self.container[1:self.hist_size] = shifted + not_shifted
        self.container[0,condition] = values[condition]
        
        self.count_hist = torch.min(self.count_hist + condition.to(dtype=torch.int64), self.hist_sizet)
        

    def filter_(self,fltr:torch.tensor):
        """ Filter instances from this history, such that only items with one in fltr are retained """
        self.container  = self.container[:,fltr]
        self.count_hist = self.count_hist[fltr]


