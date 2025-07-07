"""
Contains one single class that simulates controller-system (agent-environment) loops.
The system can be of three types:
    
- discrete-time deterministic
- continuous-time deterministic or stochastic
- discrete-time stochastic (to model Markov chains)

"""

import numpy as np
import scipy as sp

from utilities import rej_sampling_rvs

class Simulator:
    def __init__(self, sys_type,
                 closed_loop_rhs,
                 sys_out,
                 state_init,
                 disturb_init=[],
                 action_init=[],
                 t0=0,
                 t1=1,
                 dt=1e-2,
                 max_step=0.5e-2,
                 first_step=1e-6,
                 atol=1e-5,
                 rtol=1e-3,
                 is_disturb=0,
                 is_dyn_ctrl=0):
        
        self.sys_type = sys_type
        self.closed_loop_rhs = closed_loop_rhs
        self.sys_out = sys_out
        self.dt = dt
        
        # Build full state of the closed-loop
        if is_dyn_ctrl:
            if is_disturb:
                state_full_init = np.concatenate([state_init, disturb_init, action_init])
            else:
                state_full_init = np.concatenate([state_init, action_init])
        else:
            if is_disturb:
                state_full_init = np.concatenate([state_init, disturb_init])
            else:
                state_full_init = state_init
            
        self.state_full = state_full_init
            
        self.t = t0
        self.state = state_init
        self.dim_state = state_init.shape[0]
        self.observation = self.sys_out(state_init)
        
        if sys_type == "diff_eqn":
            
            # Store these for reset purposes
            self.state_full_init = state_full_init
            self.t0 = t0
            self.t1 = t1
            self.max_step = dt/2
            self.first_step = first_step
            self.atol = atol
            self.rtol = rtol
                        
            # self.ODE_solver = sp.integrate.RK45(closed_loop_rhs, t0, state_full_init, t1, max_step = dt/2, first_step=first_step, atol=atol, rtol=rtol)
            self.ODE_solver = sp.integrate.RK45(self.closed_loop_rhs,
                                                self.t0,
                                                self.state_full_init,
                                                self.t1,
                                                max_step=self.dt/2, 
                                                first_step=self.first_step,
                                                atol=self.atol,
                                                rtol=self.rtol
                                                )
               
    def sim_step(self):
        """
        Do one simulation step and update current simulation data (time, system state and output). 

        """
        if self.sys_type == "diff_eqn":
            self.ODE_solver.step()
            
            self.t = self.ODE_solver.t
            self.state_full = self.ODE_solver.y
            
            self.state = self.state_full[0:self.dim_state]
            self.observation = self.sys_out(self.state)
            
        elif self.sys_type == "discr_fnc":
            self.t = self.t + self.dt
            self.state_full = self.closed_loop_rhs(self.t, self.state_full)
            
            self.state = self.state_full[0:self.dim_state]
            self.observation = self.sys_out(self.state)
            
        elif self.sys_type == "discr_prob":
            self.state_full = rej_sampling_rvs(self.dim_state, self.closed_loop_rhs, 10)
            
            self.t = self.t + self.dt
            
            self.state = self.state_full[0:self.dim_state]
            self.observation = self.sys_out(self.state)           
        else:
            raise ValueError('Invalid system description')
            
    def get_sim_step_data(self):
        """
        Collect current simulation data: time, system state and output, and, for completeness, full closed-loop state.

        """
        
        t, state, observation, state_full = self.t, self.state, self.observation, self.state_full
        
        return t, state, observation, state_full
    
    def reset(self):
        if self.sys_type == "diff_eqn":
            # The only real way to reset the solver seems to be recreating it altogether
            self.ODE_solver = sp.integrate.RK45(self.closed_loop_rhs,
                                    self.t0,
                                    self.state_full_init,
                                    self.t1,
                                    max_step=self.dt/2, 
                                    first_step=self.first_step,
                                    atol=self.atol,
                                    rtol=self.rtol
                                    )
            # self.ODE_solver.status = 'running'
            # self.ODE_solver.t = self.t0
            # # self.ODE_solver.observation = self.state_full_init
            # # self.state_full = self.state_full_init
            # self.ODE_so        Collect current simulation data: time, system state and output, and, for completeness, full closed-loop state.
