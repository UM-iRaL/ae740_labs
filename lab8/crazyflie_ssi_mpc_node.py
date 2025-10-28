import numpy as np

from crazyflie_py import *
import rclpy
import rclpy.node

from .quadrotor_simplified_model import QuadrotorSimplified
from .ssi_mpc import SSIMpc

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
        self.model_name = "Nominal"
        self.n_rf = 50
        self.lr = 0.1
        self.kernel = 'Gaussian'
        self.kernel_std = 1.0

        # Input mask decides what features to use amongst the available ones
        self.input_mask = [0,1,2,3,4,5,6,7,8] # takes the features at the given indices 
        # the give input mask indicates that all 9 states (no control inputs) are used as features

        # Output mask decides dynamics of which states is predicted
        self.target_mask = [0,1,2,3,4,5,6,7,8] # usually we predict the velocity dot (acceleration)
        # the given target mask indicates that we learn the error prediction function for all 9 states

        self.set_random_features()

        # dictionary for random features set-up (just one time)
        self.rf_dict = {'n_rf':self.n_rf, 'omega':self.omega, 'b':self.b, 'input':self.input_mask, 'target':self.target_mask, 'lr':self.lr}
        
        acados_c_generated_code_path = pathlib.Path(get_package_share_directory('controller_pkg')).resolve() / 'acados_generated_files'
        self.mpc_solver = SSIMpc('crazyflie', quadrotor_dynamics, mpc_tf, mpc_N, self.rf_dict, code_export_directory=acados_c_generated_code_path)
        self.mpc_solver.generate_mpc()

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

    
        
        
        self.takeoffService = self.create_subscription(Empty, f'/all/mpc_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/mpc_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/mpc_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/mpc_hover', self.hover, 10)
        self.teleopService = self.create_subscription(Empty, f'/all/mpc_teleop', self.teleop, 10)



        # [TODO] PART 2: Add ROS2 timers for the main control loop (callback -> self._main_loop) and 
        #                the MPC solver loop (self._mpc_solver_loop). 
        #  Hint: Keep in mind that the variable self.rate is the control update rate specified in Hz


    def set_random_features(self):

        # [TODO] SSI PART: Draw random features (w,b in the paper) based on Gaussian kernel. 
        #
        # From the __init__() function, the Gaussian distribution has standard deviation self.kernel_std.
        # Use the variable self.omega and self.b to store the samples drawn
        # 
        # Hints:
        #   1. Based on Sec. V(A) of the paper, you know that 'w' is i.i.d, from a Gaussian 
        #      distribution of given std. dev and 'b' is drawn from uniform distribution [0,2pi].
        #   2. The size of the variables self.omega = (self.n_rf, len(self.input_mask)).
        #      Length of input mask is the actual number of features used for learning. 
        # 
        # self.omega = ...
        # self.b = ... 

        return


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

        return # remove this statement after finishing this part 

    def _twist_msg_callback(self, msg: TwistStamped):
        # self.velocity = ...

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
        

    # [TODO] PART 4: Implement the trajectory type 'horizontal_circle' starting at self.trajectory_start_position.
    # Instructions:
    # - In the trajectory_function, add a case for 'horizontal_circle'.
    # - Use self.trajectory_start_position as the starting position (not the center).
    # - Set the radius (e.g., a) and angular velocity (e.g., omega).
    # - Compute the reference position (pxr, pyr, pzr) and velocity (vxr, vyr, vzr).
    # - Return these values in the output array as [pxr, pyr, pzr, vxr, vyr, vzr, 0., 0., 0.]
    # - The last three values (zeros) are the euler angles (attitude references)

    # def trajectory_function(self, t):
    #     if self.trajectory_type == 'horizontal_circle':  
    #       pxr = 
    #       pyr = 
    #       ...
    #       vzr = 

    #     return np.array([pxr,pyr,pzr,vxr,vyr,vzr,0.,0.,0.])

        

    def navigator(self, t):
        if self.flight_mode == 'takeoff':
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N+1)
            yref = np.array([np.array([*((self.go_to_position - self.trajectory_start_position)*(1./(1. + np.exp(-(12.0 * (t_mpc - self.takeoff_duration) / self.takeoff_duration + 6.0)))) + self.trajectory_start_position),0.,0.,0.,0.,0.,0.]) for t_mpc in t_mpc_array]).T
            # yref = np.repeat(np.array([[*self.go_to_position,0,0,0]]).T, self.mpc_N, axis=1)
        elif self.flight_mode == 'land':
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N+1)
            yref = np.array([np.array([*((self.go_to_position - self.trajectory_start_position)*(1./(1. + np.exp(-(12.0 * (t_mpc - self.land_duration) / self.land_duration + 6.0)))) + self.trajectory_start_position),0.,0.,0.,0.,0.,0.]) for t_mpc in t_mpc_array]).T
            # yref = np.repeat(np.array([[*self.go_to_position,0,0,0]]).T, self.mpc_N, axis=1)
        elif self.flight_mode == 'trajectory':
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N+1)
            yref = np.array([self.trajectory_function(t_mpc) for t_mpc in t_mpc_array]).T
        elif self.flight_mode == 'hover':
            yref = np.repeat(np.array([[*self.go_to_position,0.,0.,0.,0.,0.,0.]]).T, self.mpc_N, axis=1)
        return yref
    

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

    def _mpc_solver_loop(self):
        if not self.is_flying:
            return
        
        if self.trajectory_changed:
            self.trajectory_start_position = self.position
            self.trajectory_t0 = self.get_clock().now()
            self.trajectory_changed = False

        t = (self.get_clock().now() - self.trajectory_t0).nanoseconds / 10.0**9
        trajectory = self.navigator(t)

        #### SSI Part: New additions 
        if self.last_time_stamp == None:
            self.last_time_stamp = self.current_time_stamp
            dt = 0.0
        else:
            dt = (self.current_time_stamp.nanosec - self.last_time_stamp.nanosec)/1e9 + (self.current_time_stamp.sec - self.last_time_stamp.sec)
            self.last_time_stamp = self.current_time_stamp # for the next iteration

        # [TODO] PART 6: Load the initial state variable and reference variable for the MPC problem
        #                   and solve the MPC problem at the current time step
        # 
        # x0 = ... (numpy array (size=9) of the crazyflie state -> position, velocity, attitude)
        # yref = ... (2D numpy array of the reference trajectory) (shape = NUM_STATE_VAR, NUM_MPC_STEPS)
        # yref_e = ... (1D numpy array for the terminal state variable (size=NUM_STATE_VAR))
        #
        # status, x_mpc, u_mpc = ... 
        #
        # Hints: 
        #   1. Study the structure of trajectory from the self.navigator(t) function 
        #   2. Remember that self.position etc. are all python lists (not numpy arrays)
        #   3. Use the solve_mpc() method from the mpc_solver object, see the function in the tracking_mpc.py file
        #
        #
        # IMPORTANT: make sure to check arguements to solve_mpc() as it now includes 'dt'



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
    quad_name = 'cf_1'

    quadrotor_dynamics = QuadrotorSimplified(mass, arm_length, Ixx, Iyy, Izz, tau)
    node = CrazyflieMPC(quad_name, quadrotor_dynamics, mpc_N, mpc_tf, rate)
    
    # Standard node commands
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
   main()
