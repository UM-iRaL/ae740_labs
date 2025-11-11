from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
import casadi as cs
from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt,  cos, sin, norm_2, tanh, GenMX_zeros
from scipy.linalg import block_diag
from .quadrotor_simplified_model import QuadrotorSimplified
from threading import Thread
from time import sleep, time
from pathlib import Path
import importlib
import sys

class SelfAdaptiveMPC:
    def __init__(self, name: str, quadrotor: QuadrotorSimplified, horizon: float, num_steps: int, rf_dict=None, code_export_directory : Path=Path('acados_generated_files')):
        self.model_name = name
        self.quad = quadrotor
        self.horizon = horizon
        self.num_steps = num_steps
        self.ocp_solver = None
        self.solver_locked = False
        self.hover_control = np.array([0., 0., 0., self.quad.gravity*self.quad.mass])
        # self.acados_generated_files_path = Path(__file__).parent.resolve() / 'acados_generated_files'
        self.acados_generated_files_path = code_export_directory
        
        #### NEW VARIABLES FOR SELF-ADAPTIVE MPC
        rate = 50 # Hz
        self.mpc_dt = self.horizon/self.num_steps
        # number of update calls within one mpc time step 
        # for horizon of 1sec, num_steps=10, rate=50Hz -> 5 updates per mpc time step
        self.step_size = (self.horizon/self.num_steps)/(1/rate) # number of updates per mpc time step

        #### Initialize symbolic variables
        # Declare model variables
        self.p = cs.MX.sym('p', 3)  # position
        self.v = cs.MX.sym('v', 3)  # velocity
        self.e = cs.MX.sym('e', 3)  # euler angles

        # Full state vector (9-dimensional)
        self.x = cs.vertcat(self.p, self.v, self.e)
        self.state_dim = 9

        # Control input vector
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')
        u3 = cs.MX.sym('u3')
        u4 = cs.MX.sym('u4')
        self.u = cs.vertcat(u1, u2, u3, u4)
        self.u_dim = 4

        self.dt = cs.MX.sym('dt')
        self.rf_dict = rf_dict # dictionary containing parameters for RFF

        self.target_state_dim = 3 # only position of target is considered for learning residuals

        # symbolic variables
        self.learning_rate = self.rf_dict['lr']
        self.n_rf = self.rf_dict['n_rf'] # number of random features
        self.omega = self.rf_dict['omega'] # 'w' in 'wz+b'
        self.b = self.rf_dict['b'] # 'b' in 'wz+b'
        self.target_mask = self.rf_dict['target'] # mask that maps from features to output states of predictors
        # Bh is the state space of target, not pursuer
        self.Bh = np.eye(self.target_state_dim)[self.target_mask].T # Map learned residuals to state space 
        
        #### NEW VARIABLES FOR SELF-ADAPTIVE MPC
        self.memory_horizon = self.rf_dict['mh']
        self.predictor_type = self.rf_dict['p_type']
        if self.predictor_type=='single_learner':
            self.target_memory_dim = 3*self.memory_horizon
            self.input_mask = self.rf_dict['input'] # mask that only passes the chosen features among available 
            self.Bz = np.eye(self.state_dim+self.u_dim+self.target_memory_dim)[self.input_mask] # Map states/control to feature space
            # assuming a fixed rate for calling expert, we need to preallocate memory for target position history 
            self.target_dense_pose_log = np.zeros((3, int(2*(self.memory_horizon+self.num_steps)*self.step_size))) # memory horizon for target position history
        
        # FILL THIS FOR BONUS POINTS
        if self.predictor_type=='multiple_learners':
            raise NotImplementedError('multiple_learners predictor type not implemented yet')
        

        self.x_last = None
        
        # We need nominal model for our robot now
        # setup the symbolic dynamic model
        x_dot = self.nominal_model()
        f = cs.Function('x_dot', [self.x, self.u], [x_dot], ['x', 'u'], ['x_dot'])
        self.f=f # this can be used by substituting numerical values eg. self.f(self.x, self.u)

    def nominal_model(self):
        # [TODO] SSI PART: Model based on symbolic Casadi variables and additional parametric terms based on learned model
        #
        # Instructions: 
        # - Note that it is crucial to use the symbolic state/control variables initialized inside the __init__() function only. 
        #   These include 'self.x', 'self.u', 'self.p', 'self.v', 'self.e', 'u1', 'u2', 'u3', 'u4' only. 
        #   You must write the model (mathematically same as Lab 3) only by modifying these variables. Writing something
        #   like cos_pitch = cos(self.x[7]) or cos(self.e[1]) are both okay. 
        #
        # - You must use the Casadi functions imported in line 4. 
        #
        #   Keep the matrix sizes, and terms in mind to understand how the final prediction model works.
        #


        return cs.vertcat(p_dot, v_dot, e_dot) 
    
    def generate_mpc(self):

        print('Acados cython code not generated. Generating cython code now...')

        model = AcadosModel()
        model.x = self.x
        model.u = self.u
        model.f_expl_expr = self.f(x=self.x, u=self.u)['x_dot']
        model.name = self.model_name

        # Define the optimal control problem
        ocp = AcadosOcp()
        ocp.model = model

        ocp.code_export_directory = self.acados_generated_files_path / ('c_generated_code')
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of controls
        ny = nx + nu  # size of intermediate cost reference vector in least squares objective
        ny_e = nx # size of terminal reference vector

        N = self.num_steps
        Tf = self.horizon
        ocp.dims.N = N
        ocp.solver_options.tf = Tf

        # [TODO] Contraints and cost function values
        # 
        # Instructions:
        # - Write the appropriately-sized matrices for the quadratic costs 
        # - After finishing all other parts, tune the values for smooth performance
        # - Write the appropriate limits for the state and control variables
        # - Make sure to use the given variable names
        # 
        # Q = ... (quadratic cost for state -> x^T*Qx)
        # R = ... (quadratic cost for control -> u^T*R*u)
        # W = block_diag(Q,R) (as it is -> this is the combined cost matrix for all except terminal step) 
        # 
        # max_angle = ... [rad]
        # max_thrust = ... [N]
        # max_height = ... [m]
        # max_velocity = ... [m/s]
        # max_X = ... [m]
        # max_Y = ... [m]


        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.Vx = np.vstack([np.identity(nx), np.zeros((nu,nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx,nu)), np.identity(nu)])
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(ny)

        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W_e = Q
        ocp.cost.Vx_e = np.vstack([np.identity(nx)])
        ocp.cost.yref_e = np.zeros(ny_e)
  
        # bounds on control
        max_angle = np.radians(15) # [rad]
        max_thrust = 0.477627618 # [N]
        ocp.constraints.lbu = np.array([-max_angle, -max_angle, -np.radians(10), 0.])
        ocp.constraints.ubu = np.array([max_angle, max_angle, np.radians(10), max_thrust])
        ocp.constraints.idxbu = np.array([0,1,2,3])

        ocp.constraints.lbx = np.array([-max_X, -max_Y, 0., -max_velocity, -max_velocity, -max_velocity, -max_angle, -max_angle, -np.radians(180)])
        ocp.constraints.ubx = np.array([max_X, max_Y, max_height, max_velocity, max_velocity, max_velocity, max_angle, max_angle, np.radians(180)])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        

        # initial state
        ocp.constraints.x0 = np.zeros(9)

        json_file = str(self.acados_generated_files_path / ('acados_ocp.json'))
        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.tol = 1e-3
        ocp.solver_options.qp_tol = 1e-3
        ocp.solver_options.nlp_solver_max_iter = 20
        ocp.solver_options.qp_solver_iter_max = 30
        ocp.solver_options.print_level = 0
        
        AcadosOcpSolver.generate(ocp, json_file=json_file)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)

        if self.acados_generated_files_path.is_dir():
            sys.path.append(str(self.acados_generated_files_path))
        acados_ocp_solver_pyx = importlib.import_module('c_generated_code.acados_ocp_solver_pyx')
        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)

    def update_step(self, dt, x_now, target_now):
        
        # first call to this function, no past memory
        if self.x_last is None: 
            self.x_last = x_now # last state same as current
            self.u_last = np.zeros(self.u_dim) # zero control 
            #### NEW ADDITION FOR SELF-ADAPTIVE MPC
            if self.predictor_type=='single_learner':
                self.alpha_last = np.zeros((len(self.target_mask), self.n_rf)) # zero alpha
                # initialize target dense pose log with current target position for all times in memory horizon (assuming 50Hz update rate)
                self.target_dense_pose_log[:, :] = np.tile(target_now[:3].reshape(3,1), (1, self.target_dense_pose_log.shape[1]))
                self.target_last = target_now
            if self.predictor_type=='multiple_learners':
                raise NotImplementedError('multiple_learners predictor type not implemented yet')

        if dt == 0: # to avoid any errors
            dt = self.horizon/self.num_steps

        # input for update based on previous predictions
        alpha_in = self.alpha_last
        x_in = self.x_last
        u_in = self.u_last
        


        # [TODO] SELF-ADAPTIVE PART: The recursive least-squares update based on Algorithm 1 in the paper. 
        # 
        # Note:
        # - You may use the already defined self.f() to make predictions based on previous data.
        # - The crux on the algorithm lies in predicting the current target position based on previous data
        #   and comparing it with the actual target position to get the prediction error.
        # - Based on the prediction error, you will update the alpha parameters using the RLS update rule.
        # - You might need to construct the random features based on previous data, where you can take
        #   inspiration from the definitions of the symbolic variables in the __init__() function.  
        # - In the end, store the updated value of alpha as alpha_out.
        # - The time elapsed from the previous update 'dt' is important to make sure 

        # if self.predictor_type=='single_learner':
        #     # update target pose memory with a new entry
        #     self.target_dense_pose_log = np.roll(self.target_dense_pose_log, shift=1, axis=1)
        #     self.target_dense_pose_log[:,0] = np.copy(target_now[:3]) # only x,y,z position
        #
        #     # construct target memory vector starting from 2nd index to compare with most recent target position
        #     start = 1
        #     step = int(self.step_size)
        #     num_indices = int(self.memory_horizon)
        #     target_memory_vector = self.target_dense_pose_log[:, start:start+num_indices*step:step].T.flatten()  # shape (memory_horizon*3, )
        #   
        #     # # ... intermediate steps ...
        #    
        #     alpha_out = ... # updated alpha after one step of RLS update
        #
        # if self.predictor_type=='multiple_learners':
        #     raise NotImplementedError('multiple_learners predictor type not implemented yet')
        


        # log latest values
        self.alpha_last = np.copy(alpha_out)
        self.x_last = np.copy(x_now) # for the next iteration
        self.target_last = np.copy(target_now) # for the next iteration

        return
    
    
    def get_prediction_error(self):
        # [TODO] SELF-ADAPTIVE PART: Function to compute prediction error over last MPC horizon
        # Instructions:
        # - You will need to predict the target trajectory over the last MPC horizon
        #   using the previous values of target positions stored in self.target_dense_pose_log
        # - The prediction must be done in an iterative manner, where you predict one step ahead
        #   and use that to predict the next step using appropriately sized target_memory 
        # - This is basically a moving horizon (fixed length) feature space where at each step you 
        #   add the latest predicted target position and remove the oldest one.
        # - After predicting the full trajectory over the MPC horizon, compute the RMS error
        #   between the velocity of predicted and actual target positions over the horizon. 
        # - Look at the 'Algorithm parameters' in Experiments section and 'AS' module in the paper

        # if self.predictor_type=='single_learner':
        #     # # past target position -> one mpc horizon back
        #     target_past = self.target_dense_pose_log[:, int(self.num_steps*self.step_size)]

        #     # reconstruct target trajectory starting from past position
        #     predicted_traj = np.zeros((3, self.num_steps+1)) 
        #     actual_traj = np.zeros((3, self.num_steps+1))
        #     predicted_traj[:,0] = target_past[:3]
        #     actual_traj[:,0] = target_past[:3]

        #     start = int(2*self.num_steps*self.step_size) # starting index for 'past' memory vector
        #     # # IMP: this is start vector for memory at the time of 'target_past', so 2*mpc_horizon back
        #     step = int(self.step_size)
        #     num_indices = int(self.memory_horizon)
        #     past_memory_vector = self.target_dense_pose_log[:, start:start+num_indices*step:step].T.flatten() # flatten combines row first

        #     # iterate over mpc horizon
        #     for i in range(self.num_steps):
        #         # # ... intermediate steps ...          
        #         target_next = ...

        #         predicted_traj[:,i+1] = ...
        #         actual_traj[:,i+1] = ...
        #         past_memory_vector = ... # shift by 3 for x,y,z and then add the latest predicted position
            
        #     # compute error as difference in increments
        #     pred_increments = ...
        #     actual_increments = ...
        #     vel_error_matrix = (pred_increments - actual_increments)/self.mpc_dt  
        #     overall_rms_error = ...  # scalar that depicts how much error in velocity prediction over the mpc horizon

        # if self.predictor_type=='multiple_learners':
        #     raise NotImplementedError('multiple_learners predictor type not implemented yet')


        return overall_rms_error # scalar


    def solve_mpc(self, x0, target_now, dt):
        if self.solver_locked:
            print('mpc solver locked, skipping...')
            return
        self.solver_locked = True

        
        N = self.num_steps
        nx = self.state_dim
        nu = self.u_dim

        #### NEW ADDITION FOR SELF-ADAPTIVE MPC
        if self.predictor_type=='single_learner':
            yref = np.zeros((nx, N))
            alpha_in = self.alpha_last
            x_in = x0
            u_in = self.u_last
            target_in = target_now

            # construct target memory vector starting from 1st index to compare with most recent target position
            start = 0
            step = int(self.step_size)
            num_indices = int(self.memory_horizon)
            target_memory_vector = self.target_dense_pose_log[:, start:start+num_indices*step:step].T.flatten() # this is at the current time, will change during prediction
            
            # [TODO] SELF-ADAPTIVE PART: Predict target trajectory over the MPC horizon using the learned model
            # Instructions:
            # - Similar to the previous function, you will need to iteratively predict the target
            #   trajectory over the MPC horizon using the learned model.
            # - At each step, you will need to update the target memory vector by adding the
            #   latest predicted target position and removing the oldest one.
            # - Store the full predicted target trajectory in the yref variable to be used
            #   as reference in the MPC solver.
            # - In the end, store the final predicted target state in yref_e variable.
            # - Note that yref has shape (nx, N) and yref_e has shape (nx, ), where nx=9, hence fill appropriately.

            # for i in range(N):
            #     # # ... intermediate steps ...
            #     target_next = ...
            #     # # ... intermediate steps ...
            #     yref[...] = ...
            #     target_in = np.copy(target_next)
            #     # roll memory vector and update with latest predicted position
            #     target_memory_vector = ... # shift by 3 for x,y,z and then add the latest predicted position
                
            # yref_e = ... # final predicted target state 

         
        if self.predictor_type=='multiple_learners':
            raise NotImplementedError('multiple_learners predictor type not implemented yet')

        
        # set reference commands as recieved from the main node
        for i in range(N):
            self.ocp_solver.set(i, 'yref', np.array([*yref[:,i], *self.hover_control]))
        self.ocp_solver.set(N, 'yref', yref_e)

        # set the initial state in the solver 
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)

        # solve the optimization problem
        status = self.ocp_solver.solve()

        x_mpc = np.zeros((N+1, nx))
        u_mpc = np.zeros((N, nu))

        # extract state and control solution from solver
        for i in range(N):
            x_mpc[i,:] = self.ocp_solver.get(i, "x")
            u_mpc[i,:] = self.ocp_solver.get(i, "u")
        x_mpc[N,:] = self.ocp_solver.get(N, "x")

        self.u_last = np.copy(u_mpc[0,:])

        self.solver_locked = False

        #### NEW ADDITION FOR SELF-ADAPTIVE MPC
        return status, x_mpc, u_mpc, yref, yref_e