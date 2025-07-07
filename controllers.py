"""
Contains controllers a.k.a. agents.

"""

from utilities import dss_sim
from utilities import rep_mat
from utilities import uptria2vec
from utilities import push_vec
import models
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from scipy.stats import multivariate_normal
from scipy.linalg import solve_discrete_are
from numpy.linalg import lstsq
from numpy import reshape
import warnings
import math
# For debugging purposes
from tabulate import tabulate
import os

def ctrl_selector(t, observation, action_manual, ctrl_nominal, ctrl_benchmarking, mode):
    """
    Main interface for various controllers.

    Parameters
    ----------
    mode : : string
        Controller mode as acronym of the respective control method.

    Returns
    -------
    action : : array of shape ``[dim_input, ]``.
        Control action.

    """
    
    if mode=='manual': 
        action = action_manual
    elif mode=='nominal': 
        action = ctrl_nominal.compute_action(t, observation)
    else: # Controller for benchmakring
        action = ctrl_benchmarking.compute_action(t, observation)
        
    return action


class ControllerOptimalPredictive:    
    def __init__(self,
                 dim_input,
                 dim_output,
                 mode='MPC',
                 ctrl_bnds=[],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 buffer_size=20,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 run_obj_struct='quadratic',
                 run_obj_pars=[],
                 observation_target=[],
                 state_init=[],
                 obstacle=[],
                 seed=1):
  
        np.random.seed(seed)
        print(seed)

        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        # Controller: common
        self.Nactor = Nactor 
        self.pred_step_size = pred_step_size
        
        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor) 
        self.action_sqn_init = []
        self.state_init = []

        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
            self.action_init = self.action_min/10
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)
        
        
        self.action_buffer = np.zeros( [buffer_size, dim_input] )
        self.observation_buffer = np.zeros( [buffer_size, dim_output] )        
        
        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys   
        
        # Learning-related things
        self.buffer_size = buffer_size
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.run_obj_struct = run_obj_struct
        self.run_obj_pars = run_obj_pars
        self.observation_target = observation_target
        
        self.accum_obj_val = 0
        print('---Critic structure---', self.critic_struct)

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 + (self.dim_output + self.dim_input) ) 
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 )
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_output + self.dim_input
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-mix':
            self.dim_critic = int( self.dim_output + self.dim_output * self.dim_input + self.dim_input )
            self.Wmin = -1e3*np.ones(self.dim_critic)  
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'poly3':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input ) )
            self.Wmin = -1e3*np.ones(self.dim_critic)  
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'poly4':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 * 3)
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = np.ones(self.dim_critic) 
        self.N_CTRL = N_CTRL()
        self.LQRcontroller = LQRcontroller(dt=self.sampling_time)

    def reset(self,t0):   

        # Controller: common

        if len(self.action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
            self.action_init = self.action_min/10
        else:
            self.action_curr = self.action_init
            self.action_sqn_init = rep_mat( self.action_init , 1, self.Nactor)
        
        self.action_buffer = np.zeros( [self.buffer_size, self.dim_input] )
       # self.observation_buffer = np.zeros( [self.buffer_size, self.dim_output] ) 
        self.observation_target = np.array([1.0, 1.0, 0.0])       

        self.critic_clock = t0
        self.ctrl_clock = t0
    
    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state
    
    def upd_accum_obj(self, observation, action):

        self.accum_obj_val += self.run_obj(observation, action)*self.sampling_time
                 
    def run_obj(self, observation, action):
        run_obj = 1
        #####################################################################################################
        ################################# write down here cost-function #####################################
        #####################################################################################################
        if self.run_obj_struct == 'quadratic':
            R1 = self.run_obj_pars[0]
            chi = np.concatenate([observation,action])
            run_obj = chi.T @ R1 @ chi
        else:
            raise ValueError(f"Running objective structure")
        return run_obj

    def _actor_cost(self, action_sqn, observation):
        
        my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        
        observation_sqn = np.zeros([self.Nactor, self.dim_output])
        
        # System observation prediction
        observation_sqn[0, :] = observation
        state = self.state_sys
        for k in range(1, self.Nactor):
            state = state + self.pred_step_size * self.sys_rhs([], state, my_action_sqn[k-1, :])  # Euler scheme
            
            observation_sqn[k, :] = self.sys_out(state)
        
        J = 0         
        if self.mode=='MPC':
            for k in range(self.Nactor):
                J += self.gamma**k * self.run_obj(observation_sqn[k, :], my_action_sqn[k, :])

        return J
    
    def _actor_optimizer(self, observation):

        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 40, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 40, 'maxfev': 60, 'disp': False, 'adaptive': True, 'xatol': 1e-3, 'fatol': 1e-3}
       
        isGlobOpt = 0
        
        my_action_sqn_init = np.reshape(self.action_sqn_init, [self.Nactor*self.dim_input,])
        
        bnds = sp.optimize.Bounds(self.action_sqn_min, self.action_sqn_max, keep_feasible=True)
        
        try:
            if isGlobOpt:
                minimizer_kwargs = {'method': actor_opt_method, 'bounds': bnds, 'tol': 1e-3, 'options': actor_opt_options}
                action_sqn = basinhopping(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                          my_action_sqn_init,
                                          minimizer_kwargs=minimizer_kwargs,
                                          niter = 10).x
            else:
                action_sqn = minimize(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                      my_action_sqn_init,
                                      method=actor_opt_method,
                                      tol=1e-3,
                                      bounds=bnds,
                                      options=actor_opt_options).x        

        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            action_sqn = self.action_curr
        
        return action_sqn[:self.dim_input]    # Return first action
                    
    def compute_action(self, t, observation):     
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            
            if self.mode == 'MPC':  
                
                action = self._actor_optimizer(observation)

            elif self.mode == "N_CTRL":
                
                action = self.N_CTRL.pure_loop(observation)

            elif self.mode == "LQR":
                if not self.observation_target or len(self.observation_target) < 3:
                    self.observation_target = [0.0, 0.0, 0.0]
                    print("Warning: observation_target was invalid or empty. Setting to default:", self.observation_target)
                action = self.LQRcontroller.compute_control(observation, self.observation_target)

            self.action_curr = action
            
            return action    
    
        else:
            return self.action_curr

class N_CTRL:
        def __init__(self):
             self.linear_speed = 2
             self.angular_speed = 3
             self.counter = 0 # used for time tracking or logging
    
        def pure_loop(self, observation):
            x, y, theta = observation
            
            x_goal, y_goal = 0.0, 0.0

            k_rho = 2.5
            k_beta = -1.5
            k_alpha =  3.2 
            
            #Error
            dx = x_goal - x
            dy = y_goal - y 
            
            # Normalize angles
            rho = math.hypot(dx, dy)
            alpha = math.atan2(dy, dx) - theta
            beta = -theta - alpha

            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
            beta = (beta + np.pi) % (2 * np.pi) - np.pi
            
            # Control Law
            v = k_rho * rho
            w = k_alpha * alpha + k_beta * beta

            # goal
            if -(np.pi) < alpha <=  -(np.pi)/2 or (np.pi)/2 < alpha <= (np.pi):
               v=-v

            return [v,w]

class LQRcontroller:
    def __init__(self, dt=0.1):
        self.dt = dt

        # Cost matrices
        self.Q = np.diag([8, 8, 6])
        self.R = np.diag([2, 4])
        #self.Q = np.diag([40, 40, 20])
        #self.R = np.diag([10, 15])

    def compute_control(self, current_state, goal_state):
        # Add debug to confirm inputs
        print(f"[LQR] current_state: {current_state}")
        print(f"[LQR] goal_state: {goal_state}")
        if len(goal_state) < 3:
            raise ValueError("goal state must have at least3 elment")
        theta_g = goal_state[2] 
        v_guess = 0.5  # small nominal speed for linearization

        # Linearized A matrix
        A = np.array([
        [0, 0, -v_guess * np.sin(theta_g)],
        [0, 0,  v_guess * np.cos(theta_g)],
        [0, 0, 0]
        ])

        # Linearized B matrix
        B = np.array([
        [np.cos(theta_g), 0],
        [np.sin(theta_g), 0],
        [0, 1]
        ])

        # Discretize with Euler
        A_d = np.eye(3) + A * self.dt
        B_d = B * self.dt

        # Solve DARE
        P = solve_discrete_are(A_d, B_d, self.Q, self.R)

        # Compute gain
        K = np.linalg.inv(self.R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)

        error = current_state - goal_state
        print(f"[LQR] error: {error}")

        control = -K @ error

        # Clip control to safe limits
        v_max, w_max = 1.0, 1.0
        control[0] = np.clip(control[0], -v_max, v_max)
        control[1] = np.clip(control[1], -w_max, w_max)

        print(f"[LQR] computed control (clipped): {control}")
        return control

class MPCcontroller:
    def __init__(self, dt = 1, N = 6):
        self.dt = dt
        self.N = 15

        # Cost weights
        self.Q = np.diag([4, 5, 3])
        self.R = np.diag([2, 3])
        self.QF = np.diag([80, 80, 20])

        self.n = 3
        self.m = 2

        self.A = np.eye(self.n)
        self.B = self.dt * np.array([
            [1, 0],
            [0, 0],
            [0, 1]
        ]) 

    def compute_control(self, current_state, goal_state):

        # Decision variables
        x = np.Variable((self.n, self.N + 1))
        u = np.Variable((self.m, self.N))

        cost = 0
        constraints = []

        # Initial condition
        constraints += [x[:, 0] == current_state]

        for k in range(self.N):
            # Stage cost
            cost += np.quad_form(x[:, k] - goal_state, self.Q)
            cost += np.quad_form(u[:, k], self.R)

            # Dynamics constraint
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]

            # Input constraints
            constraints += [np.abs(u[0, k]) <= 2.0]  # linear velocity limits
            constraints += [np.abs(u[1, k]) <= 2.0]  # angular velocity limits


            # Terminal cost
            cost += np.quad_form(x[:, self.N] - goal_state, self.QF)


            # Solve the problem
            prob = np.Problem(np.Minimize(cost), constraints)
            prob.solve(solver=np.OSQP)

        # Return the first control input
        if u.value is not None:
            return u.value[:, 0]
        else:
            print("MPC failed to solve.")
            return np.array([0.0, 0.0])
