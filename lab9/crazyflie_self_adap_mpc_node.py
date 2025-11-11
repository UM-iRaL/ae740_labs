import numpy as np

from crazyflie_py import *
import rclpy
import rclpy.node

from .quadrotor_simplified_model import QuadrotorSimplified
from .self_adaptive_mpc import SelfAdaptiveMPC

from crazyflie_interfaces.msg import AttitudeSetpoint

import pathlib

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Empty

from ament_index_python.packages import get_package_share_directory

import tf_transformations

from enum import Enum
from collections import deque

class Motors(Enum):
    MOTOR_CLASSIC = 1 # https://store.bitcraze.io/products/4-x-7-mm-dc-motor-pack-for-crazyflie-2 w/ standard props
    MOTOR_UPGRADE = 2 # https://store.bitcraze.io/collections/bundles/products/thrust-upgrade-bundle-for-crazyflie-2-x

class CrazyflieMPC(rclpy.node.Node):
    def __init__(self, node_name: str, quadrotor_dynamics: QuadrotorSimplified, mpc_N: int, mpc_tf: float, rate: int):
        super().__init__(node_name)

        name = self.get_name()
        prefix = '/' + name

        target_name = 'cf_1'
        target_prefix = '/' + target_name
        
        self.is_connected = True

        self.rate = rate

        self.odometry = Odometry()

        self.mpc_N = mpc_N
        self.mpc_tf = mpc_tf

        self.position = []
        self.velocity = []
        self.attitude = []

        self.trajectory_changed = True

        self.flight_mode = 'idle'
        self.trajectory_t0 = self.get_clock().now()
        self.trajectory_type = 'lemniscate'
        self.plot_trajectory = True
        
        self.motors = Motors.MOTOR_CLASSIC # MOTOR_CLASSIC, MOTOR_UPGRADE

        self.takeoff_duration = 5.0
        self.land_duration = 5.0

        self.takeoff_height = 1.0

        #### Initialization of new variables for SSI ####
        self.n_mpc_nodes = self.mpc_N
        self.t_horizon = self.mpc_tf
        self.control_freq_factor = 5
        self.opt_dt = self.t_horizon / (self.n_mpc_nodes * self.control_freq_factor)
        self.rate = 1/(self.opt_dt)
        self.n_rf = 50



        ############################################################################################################
        # [TODO] SELF-ADAPTIVE PART: Initialize the self-adaptive MPC experts here
        # Note:
        #   - You need to initialize all the variables mentioned here, some have been defined
        #   - Uncomment all these variables after filling the values in place of ...
        #
        # # PREDEFINED VARIABLES
        # self.mpc_expert_list = []
        # self.last_selection_time = 0.0
        # self.selected_expert_idx = 0
        
        # # Tuning parameters for the 'AS' module
        # self.selection_interval = ... # seconds
        # self.gamma = ... # discount factor for the expert loss update
        # self.a = ... # scaling factor for the reward function (see paper)
        # self.eta = ... # learning rate for the expert weight update (see paper)

        # # All the experts go here sequantially by calling self.init_expert() function

        # name = 'expert_1'
        # kernel = 'Gaussian' # kernel type, we only use 'Gaussian' for now
        # kernel_std = ... # standard deviation
        # lr = ... # learning rate
        # mh = ... # memory horizon for past target position history
        # p_type = 'single_learner' # predictor type BONUS PART: 'multiple_learners'
        # self.init_expert(name, kernel, kernel_std, lr, mh, p_type, quadrotor_dynamics, mpc_N, mpc_tf)

        # name = 'expert_2'
        # kernel = 'Gaussian' # kernel type, we only use 'Gaussian' for now
        # kernel_std = ... # standard deviation
        # lr = ... # learning rate
        # mh = ... # memory horizon for past target position history
        # p_type = 'single_learner' # predictor type
        # self.init_expert(name, kernel, kernel_std, lr, mh, p_type, quadrotor_dynamics, mpc_N, mpc_tf)

        # # numpy array to store the expert error at each update step (at rate 50Hz)
        # self.expert_error_array = np.zeros(len(self.mpc_expert_list))
        # self.list_of_expert_error_arrays = [] # a list to store all the expert error arrays over time

        # # initialize probability distribution over experts
        # self.P_t = ... # uniform distribution initially, 1D array with size = number of experts

        # # discounted sum of losses for each expert (array)
        # self.S_t = ... # zeros array initially, 1D array with size = number of experts








        self.control_queue = None
        self.is_flying = False

        self.update_rff_dict = None
        self.last_time_stamp = None
        self.current_time_stamp = None
        ####
    
        self.get_logger().info('Initialization completed...')


        ############################################################################################################
        # [TODO] PART 1: Add ROS2 subscriber for the Crazyflie state data, and publishers for the control command and MPC trajectory solution

        # (a) Position subscriber
        # topic type -> PoseStamped
        # topic name -> {prefix}/pose (e.g., '/cf_1/pose')
        # callback -> self._pose_msg_callback

        # (b) Velocity subscriber
        # topic type -> TwistStamped
        # topic name -> {prefix}/twist
        # callback -> self._twist_msg_callback

        # (c) MPC solution path publisher
        # topic type -> Path
        # topic name -> {prefix}/mpc_solution_path
        # publisher variable -> self.mpc_solution_path_pub

        # (d) Attitude setpoint command publisher
        # topic type -> AttitudeSetpoint
        # topic name -> {prefix}/cmd_attitude_setpoint
        # publisher variable -> self.attitude_setpoint_pub

        
        
        ############################################################################################################
        # [TODO] SELF-ADAPTIVE PART: Add ROS2 subscriber for the target state data, and publishers for the target prediction path

        # (a) Target Position subscriber
        # topic type -> PoseStamped
        # topic name -> {target_prefix}/pose 
        # callback -> self._target_pose_msg_callback

        # (b) Target prediction path publisher
        # topic type -> Path
        # topic name -> 'cf_3/mpc_solution_path' # we will use dummy cf_name cf_3 for the target prediction visualization
        # publisher variable -> self.target_prediction_path_pub

        


        
        # disabling takeoff service because we want to start tracking the target right away
        # self.takeoffService = self.create_subscription(Empty, f'/all/mpc_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/mpc_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/mpc_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/mpc_hover', self.hover, 10)
        self.teleopService = self.create_subscription(Empty, f'/all/mpc_teleop', self.teleop, 10)



        # [TODO] PART 2: Add ROS2 timers for the main control loop (callback -> self._main_loop) and 
        #                the MPC solver loop (self._mpc_solver_loop). 
        #  Hint: Keep in mind that the variable self.rate is the control update rate specified in Hz



    
    #### NEW ADDITIONAL FUNCTION FOR SELF-ADAPTIVE PART 
    def init_expert(self, name, kernel, kernel_std, lr, mh, p_type, quadrotor_dynamics, mpc_N, mpc_tf): 

        target_mask = [0,1,2] # we want to learn only the position dynamics of the target
        input_mask = list(range(13,13+3*mh)) # we use the past target position (3*mh) as features
        # indices 0-12 are used for pursuer robot state and control inputs 
        n_features = len(input_mask)

        omega, b = self.set_random_features(kernel, kernel_std, n_features)
        self.rf_dict = {'n_rf':self.n_rf, 'omega':omega, 'b':b, 'input':input_mask, 'target':target_mask, 'lr':lr, 'mh':mh, 'p_type':p_type}        
        acados_c_generated_code_path = pathlib.Path(get_package_share_directory('controller_pkg')).resolve() / 'acados_generated_files'
        self.mpc_expert_list.append(SelfAdaptiveMPC(name, quadrotor_dynamics, mpc_tf, mpc_N, self.rf_dict, code_export_directory=acados_c_generated_code_path))
        self.mpc_expert_list[-1].generate_mpc()

        return

    def set_random_features(self, kernel, kernel_std, n_features):

        # [TODO] SELF-ADAPTIVE PART: Draw random features (w,b in the paper) based on Gaussian kernel. 
        # 
        # This function is similar to SSI part, but has some argument changes because we want to define
        # the std, number of features for each expert differently.
        #
        # From the __init__() function, the Gaussian distribution has standard deviation self.kernel_std.
        # Use the variable self.omega and self.b to store the samples drawn
        # 
        # Hints:
        #   1. Based on Sec. V(A) of the paper, you know that 'w' is i.i.d, from a Gaussian 
        #      distribution of given std. dev and 'b' is drawn from uniform distribution [0,2pi].
        #   2. The size of the variables omega = (self.n_rf, n_features).
        #      Length of input mask is the actual number of features used for learning. 
        # 
        # omega = ...
        # b = ... 


        

        return omega, b


    # [TODO] PART 3: Parse the ROS2 position messages. Make sure to use given variable names.
    # 
    # NOTE: 
    # - Position is a Python list (not numpy array) containing the (x,y,z coordinates).
    # - Attitude is a Python list of the Euler angles 
    #
    # Hints: 
    #   1. Look the PoseStamped (and similarly others) message structure at https://docs.ros2.org/foxy/api/geometry_msgs/msg/PoseStamped.html.
    #   2. You can use tf_transformations for the conversion into different orientation representations. 
    #   3. Be sure to wrap the attitude angles between -pi to +pi. 

    def _pose_msg_callback(self, msg: PoseStamped):
        
        # self.position = ...  
        # self.attitude = ...

        # return # remove this statement after finishing this part 

        

        #### SSI Part: New additions
        self.current_time_stamp = msg.header.stamp

    def _twist_msg_callback(self, msg: TwistStamped):
        # self.velocity = ...

        return # remove this statement after finishing this part 


    #### NEW
    def _target_pose_msg_callback(self, msg: PoseStamped):
        # [TODO] SELF-ADAPTIVE PART: Parse the ROS2 target position messages.
        #
        # self.target_position = ...  
        # self.target_attitude = ...
        
        return # remove this statement after finishing this part

    def start_trajectory(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'trajectory'

    def teleop(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'teleop'

    def takeoff(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'takeoff'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        self.takeoff_height])
        
    def hover(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'hover'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        self.position[2]])

    def land(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'land'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        0.1])
        
    

    # [TODO] PART 5: Implement the cmd_attitude_setpoint function to publish attitude setpoint commands.
    # Instructions:
    # - Create an AttitudeSetpoint message
    # - Set the roll, pitch, yaw_rate, and thrust fields from the input parameters
    # - Publish the message using self.attitude_setpoint_pub
    # - See the structure of the message in 
    #       ae740_crazyflie_sim/ros2_ws/src/crazyswarm2/crazyflie_interfaces/msg/AttitudeSetpoint.msg
    #
    # def cmd_attitude_setpoint(...


    def thrust_to_pwm(self, collective_thrust: float) -> int:
        # omega_per_rotor = 7460.8*np.sqrt((collective_thrust / 4.0))
        # pwm_per_rotor = 24.5307*(omega_per_rotor - 380.8359)
        collective_thrust = max(collective_thrust, 0.) #  make sure it's not negative
        if self.motors == Motors.MOTOR_CLASSIC:
            return int(max(min(24.5307*(7460.8*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))
        elif self.motors == Motors.MOTOR_UPGRADE:
            return int(max(min(24.5307*(6462.1*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))
        
    def adaptive_expert_selection(self):
        # [TODO]: SELF-ADAPTIVE PART: Implement the adaptive expert selection strategy here.
        # 
        # Hints:
        #   1. You can use numpy.random.choice() function to sample from a discrete probability distribution
        #   2. Make sure to use self.P_t as the probability distribution over experts
        #   3. Important variables:
        #       self.P_t: probability distribution over experts
        #       self.list_of_expert_error_arrays: list of expert error arrays during an interval
        #       self.a, self.gamma, self.eta: tuning parameters
        #   4. Read the 'Algorithm Parameters' subsection in the Experiments section of the paper
        #   5. Calculate the loss for each expert based on normalized loss, calculated from 
        #      the mean error to reduce the noise at each step (due to update rate of 50Hz)
        #
        # selected_expert_idx = ...  # index of the selected expert to return
        return selected_expert_idx
        
        

        

    def _mpc_solver_loop(self):
        if not self.is_flying:
            return
        
        if self.trajectory_changed:
            self.trajectory_start_position = self.position
            self.trajectory_t0 = self.get_clock().now()
            self.trajectory_changed = False

        t = (self.get_clock().now() - self.trajectory_t0).nanoseconds / 10.0**9

        #### SSI Part: New additions 
        if self.last_time_stamp == None:
            self.last_time_stamp = self.current_time_stamp
            dt = 0.0
        else:
            dt = (self.current_time_stamp.nanosec - self.last_time_stamp.nanosec)/1e9 + (self.current_time_stamp.sec - self.last_time_stamp.sec)
            self.last_time_stamp = self.current_time_stamp # for the next iteration

        # [TODO] SELF-ADAPTIVE PART: Main solver loop for update (learning) step, expert error (AS), and solving the MPC problem.
        #                   
        # Hints: 
        #   1. Remember that self.position etc. are all python lists (not numpy arrays)
        #   2. Use the solve_mpc() method from the mpc_solver object, see the function in the self_adaptive_mpc.py file
        #   3. We will generate the y_ref inside the solve_mpc() function now, so you don't need to generate it here anymore
        #   4. See that solve_mpc() must now return yref and yref_e which are target predictions 
        #   5. Make sure to update the expert only at every self.selection_interval seconds (not at every iteration of this loop)
        #
        # IMPORTANT: make sure to check arguments to solve_mpc() as it now includes 'dt'
        # 
        # x0 = ... (numpy array (size=9) of the crazyflie state -> position, velocity, attitude)
        # target_pose = ... (numpy array (size=3) of the target position)
        #
        # COMPLETED CODE
        # for expert_idx, expert in enumerate(self.mpc_expert_list):
        #     # # Update the alpha value based on the current observed state
        #     expert.update_step(dt, x_now=x0, target_now=target_pose) # 'dt' is the time elapsed from the previous update
        #
        #     # # get prediction error metric for the expert
        #     self.expert_error_array[expert_idx] = expert.get_prediction_error() 
        #
        # # # list of expert errors
        # self.list_of_expert_error_arrays.append(self.expert_error_array.copy())
        #
        #
        # #### CREATE CONDITION WHEN TO UPDATE THE EXPERT (this is based on self.selection_interval)
        # #### We want that this does not happen at every iteration of this loop, but rather at a lower frequency
        # #### This interval is defined by self.selection_interval variable (0.25 seconds for example)
        # #### You will use self.adaptive_expert_selection() function here
        # if ...
        #
        # SOLVER CALL AT EVERY ITERATION
        # status, x_mpc, u_mpc, yref, yref_e = ... 
    


        self.control_queue = deque(u_mpc)

        if self.plot_trajectory:
            mpc_solution_path = Path()
            mpc_solution_path.header.frame_id = 'world'
            mpc_solution_path.header.stamp = self.get_clock().now().to_msg()

            for i in range(self.mpc_N):
                mpc_pose = PoseStamped()
                mpc_pose.pose.position.x = x_mpc[i,0]
                mpc_pose.pose.position.y = x_mpc[i,1]
                mpc_pose.pose.position.z = x_mpc[i,2]
                mpc_solution_path.poses.append(mpc_pose)

            self.mpc_solution_path_pub.publish(mpc_solution_path)

            target_prediction_path = Path()
            target_prediction_path.header.frame_id = 'world'
            target_prediction_path.header.stamp = self.get_clock().now().to_msg()

            for i in range(self.mpc_N):
                target_pose = PoseStamped()
                target_pose.pose.position.x = yref[0,i]
                target_pose.pose.position.y = yref[1,i]
                target_pose.pose.position.z = yref[2,i]
                target_prediction_path.poses.append(target_pose)

            self.target_prediction_path_pub.publish(target_prediction_path)

    def _main_loop(self):
        if self.flight_mode == 'idle':
            return

        if not self.position or not self.velocity or not self.attitude:
            self.get_logger().warning("Empty state message.")
            return
        
        if not self.is_flying:
            self.is_flying = True
            self.cmd_attitude_setpoint(0.,0.,0.,0)

        if self.control_queue is not None:
            control = self.control_queue.popleft()
            thrust_pwm = self.thrust_to_pwm(control[3])
            yawrate = 3.*(np.degrees(self.attitude[2]))
            self.cmd_attitude_setpoint(np.degrees(control[0]), 
                                    np.degrees(control[1]), 
                                    yawrate, 
                                    thrust_pwm)

def main():

    rclpy.init()

    # Quadrotor Parameters (same as MPC template for consistency)
    mass = 0.028
    arm_length = 0.044
    Ixx = 2.3951e-5
    Iyy = 2.3951e-5
    Izz = 3.2347e-5
    tau = 0.08  

    # MPC problem parameters
    mpc_N = 10 # number of steps in the MPC problem
    mpc_tf = 1 # MPC time horizon (in sec)
    rate = 50 # control update rate (in Hz)
    quad_name = 'cf_2'

    quadrotor_dynamics = QuadrotorSimplified(mass, arm_length, Ixx, Iyy, Izz, tau)
    node = CrazyflieMPC(quad_name, quadrotor_dynamics, mpc_N, mpc_tf, rate)
    
    # Standard node commands
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
   main()
