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

class SSIMpc:
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

        # symbolic variables
        self.learning_rate = self.rf_dict['lr']
        self.n_rf = self.rf_dict['n_rf'] # number of random features
        self.omega = self.rf_dict['omega'] # 'w' in 'wz+b'
        self.b = self.rf_dict['b'] # 'b' in 'wz+b'
        self.target_mask = self.rf_dict['target'] # mask that maps from features to output states of predictors
        self.input_mask = self.rf_dict['input'] # mask that only passes the chosen features among available 
        self.Bh = np.eye(self.state_dim)[self.target_mask].T # Map learned residuals to state space
        self.Bz = np.eye(self.state_dim+self.u_dim)[self.input_mask] # Map states/control to feature space

        Z = cs.vertcat(self.x, self.u) # available feature variables (all states and control inputs)
        self.rf = 1/cs.sqrt(self.n_rf) * cs.cos(self.omega @ self.Bz @ Z + self.b) # random features
        self.alpha = cs.MX.sym('alpha', len(self.target_mask), self.n_rf) # alpha parameter to be updated online
        
        # flattened symbolic variable to be passed as real-time paramter for disturbance function
        self.alpha_p = cs.reshape(self.alpha, (self.n_rf*len(self.target_mask),1)) 

        self.x_last = None
        self.alpha_last = np.zeros((len(self.target_mask), self.n_rf))


        # setup the symbolic dynamic model
        x_dot = self.ssi_augmented_model()
        f = cs.Function('x_dot', [self.x, self.u, self.alpha], [x_dot], ['x', 'u', 'alpha'], ['x_dot'])
        self.f=f # this can be used by substituting numerical values eg. self.f(self.x, self.u, self.alpha)


    def ssi_augmented_model(self):
        # [TODO] SSI PART: Model based on symbolic Casadi variables and additional parametric terms based on learned model
        #
        # Instructions: 
        # - Note that it is crucial to use the symbolic state/control variables inititialized inside the __init()__ function only. 
        #   These include 'self.x', 'self.u', 'self.p', 'self.v', 'self.e', 'u1', 'u2', 'u3', 'u4' only. 
        #   You must write the model (mathematically same as Lab 3) only by modifying these variables. Writing something
        #   like cos_pitch = cos(self.x[7]) or cos(self.e[1]) are both okay. 
        #
        # - You must use the Casadi functions imported in line 4. 
        #
        # - Augmented term using the learned dynamics is already provided in the 'return' function. 
        #   Keep the matrix sizes, and terms in mind to understand how the final prediction model works.
        #
        # - You can note that the augemented term also has direct dependence to current state/control variables 
        #   self.x and self.u because of the definition of self.rf in the __init__().  
        

        

        return cs.vertcat(p_dot, v_dot, e_dot) + cs.vertcat(cs.mtimes(self.alpha, self.rf))

    
    def generate_mpc(self):

        print('Acados cython code not generated. Generating cython code now...')

        #### SSI changes
        model = AcadosModel()
        model.x = self.x
        model.u = self.u
        model.p = self.alpha_p
        model.f_expl_expr = self.f(x=self.x, u=self.u, alpha=self.alpha)['x_dot']
        model.name = self.model_name

        # Define the optimal control problem
        ocp = AcadosOcp()
        ocp.model = model

        ocp.code_export_directory = self.acados_generated_files_path / ('c_generated_code')
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of controls
        ny = nx + nu  # size of intermediate cost reference vector in least squares objective
        ny_e = nx # size of terminal reference vector

        # Initialize real-time parameters
        n_p = model.p.size()[0] # number of parameters
        ocp.dims.np = n_p
        ocp.parameter_values = np.zeros(n_p) # initialize with zeros (here parameters are alpha)

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

    def update_step(self, dt, x_now):
        
        # first call to this function, no past memory
        if self.x_last is None: 
            self.x_last = x_now # last state same as current
            self.u_last = np.zeros(self.u_dim) # zero control 

        if dt == 0: # to avoid any errors
            dt = self.horizon/self.num_steps

        # input for update based on previous predictions
        alpha_in = self.alpha_last
        x_in = self.x_last
        u_in = self.u_last


        # [TODO] SSI Part: The recursive least-squares update based on Algorithm 1 in the paper. 
        # 
        # Note:
        # - You may use the already defined self.f() to make predictions based on previous data.
        # - The crux on the algorith lies in predicting based on past error and using it to update alpha
        # - You might need to construct the random features based on previous data, where you can take
        #   inspiration from the definitions of the symbolic variables in the __init__() function. 
        # - In the end, store the updated value of alpha as alpha_out.
        # - The time elapsed from the previous update 'dt' is important to make sure 
        # 
        # [...intermediate steps...]
        # alpha_out = ... 
        


        # log latest values
        self.alpha_last = np.copy(alpha_out)
        self.x_last = np.copy(x_now) # for the next iteration

        return


    def solve_mpc(self, x0, yref, yref_e, dt):
        if self.solver_locked:
            print('mpc solver locked, skipping...')
            return
        self.solver_locked = True

        # Update the alpha value based on the current observed state
        self.update_step(dt, x_now=x0) # 'dt' is the time elapsed from the previous update

        N = self.num_steps
        nx = self.state_dim
        nu = self.u_dim

        if yref.shape[1] != self.num_steps:
            raise Exception('incorrect size of yref')
        
        # set reference commands as recieved from the main node
        for i in range(N):
            self.ocp_solver.set(i, 'yref', np.array([*yref[:,i], *self.hover_control]))
        self.ocp_solver.set(N, 'yref', yref_e)

        # set the last update alpha value from the update step
        for i in range(N):
            # we use transpose before flatten because Casadi and Numpy have opposite conventions for row/column
            alpha_array = np.reshape(self.alpha_last.T, (self.n_rf*len(self.target_mask),1)).flatten()
            self.ocp_solver.set(i, 'p', np.copy(alpha_array))

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

        return status, x_mpc, u_mpc