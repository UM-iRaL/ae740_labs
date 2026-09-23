import numpy as np

from crazyflie_py import *
import rclpy
import rclpy.node

from crazyflie_interfaces.msg import ForceTorqueCmd

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Empty

import tf_transformations

# TODO overview:
#   PART 1: Choose the gains of the geometric controller (self.Kp, self.Kv, self.KR, self.Kw).
#   PART 2: Add the ROS2 subscribers for the state data and the publishers for the command and reference path.
#   PART 3: Parse the pose and twist messages into the state variables used by the controller.
#   PART 4: Implement the 'horizontal_circle' reference trajectory (position, velocity, acceleration, yaw).
#   PART 5: Implement the geometric controller that computes the collective thrust and body torques.
#   PART 6: Implement cmd_force_torque to publish the force-torque command to the Crazyflie.
#   PART 7: Complete the control loop to compute and send the command, and stop the motors after landing.


class CrazyflieGeometricController(rclpy.node.Node):
    def __init__(self, node_name: str, mass: float, inertia: np.ndarray, rate: int):
        super().__init__(node_name)

        name = self.get_name()
        prefix = '/' + name

        self.rate = rate

        # Quadrotor parameters
        self.m = mass
        self.g = 9.81
        self.J = inertia
        self.e3 = np.array([0., 0., 1.])

        ############################################################################################################
        # [TODO] PART 1: Choose the gains of the geometric controller. Make sure to use given variable names.
        #
        # Each gain is a numpy array of size 3 (one gain per axis):
        # - self.Kp -> position gain         (N/m),      axes (x, y, z) of the world frame
        # - self.Kv -> velocity gain         (N s/m),    axes (x, y, z) of the world frame
        # - self.KR -> attitude gain         (Nm/rad),   axes (x, y, z) of the body frame
        # - self.Kw -> angular velocity gain (Nm s/rad), axes (x, y, z) of the body frame
        #
        # Hints:
        #   1. Start by tuning the gains for takeoff and hover, then the circular trajectory.
        #   2. The attitude loop runs off-board with state feedback at 50 Hz from the crazyflie_server, so the
        #      attitude gains cannot be arbitrarily large. In the simulator, roll/pitch KR above ~1e-2 oscillates,
        #      and below ~4e-3 the drone cannot hold hover.


        # Command limits
        self.max_thrust = 0.55      # N  (hover thrust is about 0.27 N)
        self.max_torque = 2.0e-3    # Nm

        self.position = []
        self.velocity = []
        self.attitude = []
        self.R_WB = np.eye(3)
        self.omega_B = np.zeros(3)

        self.trajectory_changed = True

        self.flight_mode = 'idle'
        self.trajectory_t0 = self.get_clock().now()
        self.trajectory_type = 'horizontal_circle'
        self.plot_trajectory = True

        self.takeoff_duration = 5.0
        self.land_duration = 5.0

        self.takeoff_height = 1.0

        self.is_flying = False

        self.get_logger().info('Initialization completed...')


        ############################################################################################################
        # [TODO] PART 2: Add ROS2 subscribers for the Crazyflie state data, and publishers for the control command
        #                and the reference trajectory
        #
        # (a) Position subscriber
        # topic type -> PoseStamped
        # topic name -> {prefix}/pose (e.g., '/cf_1/pose')
        # callback -> self._pose_msg_callback

        # (b) Velocity subscriber
        # topic type -> TwistStamped
        # topic name -> {prefix}/twist
        # callback -> self._twist_msg_callback

        # (c) Reference trajectory path publisher
        # topic type -> Path
        # topic name -> {prefix}/geo_reference_path
        # publisher variable -> self.reference_path_pub

        # (d) Force-torque command publisher
        # topic type -> ForceTorqueCmd
        # topic name -> {prefix}/force_torque_cmd
        # publisher variable -> self.force_torque_pub


        self.takeoffService = self.create_subscription(Empty, f'/all/geo_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/geo_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/geo_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/geo_hover', self.hover, 10)

        # Single control loop running at self.rate (Hz)
        self.create_timer(1./self.rate, self._control_loop)


    # [TODO] PART 3: Parse the ROS2 state messages. Make sure to use given variable names.
    #
    # NOTE:
    # - self.position -> numpy array (x, y, z) of p^W
    # - self.attitude -> numpy array of the Euler angles (roll, pitch, yaw), wrapped between -pi and +pi
    # - self.R_WB     -> 3x3 numpy array, rotation matrix R^W_B of the current attitude
    # - self.velocity -> numpy array of v^W (velocity in the WORLD frame)
    # - self.omega_B  -> numpy array of omega^B (angular velocity in the BODY frame, rad/s)
    #
    # Hints:
    #   1. Look at the PoseStamped and TwistStamped message structures at
    #      https://docs.ros2.org/foxy/api/geometry_msgs/msg/PoseStamped.html.
    #   2. tf_transformations has functions for Euler angles and rotation matrices from a quaternion
    #      (quaternion_matrix returns a 4x4 homogeneous matrix).
    #   3. The twist message is filled by the crazyflie_server from the firmware log variables
    #      kalman.statePX/PY/PZ (velocity in the BODY frame) and gyro.x/y/z (angular rates in deg/s).

    def _pose_msg_callback(self, msg: PoseStamped):

        # self.position = ...
        # self.attitude = ...
        # self.R_WB = ...

        return # remove this statement after finishing this part


    def _twist_msg_callback(self, msg: TwistStamped):
        # self.velocity = ...
        # self.omega_B = ...

        return # remove this statement after finishing this part


    def start_trajectory(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'trajectory'

    def takeoff(self, msg):
        if len(self.position) == 0:
            self.get_logger().warning("No pose received yet, ignoring takeoff.")
            return
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
                                        0.05])


    # [TODO] PART 4: Implement the trajectory type 'horizontal_circle' starting at self.trajectory_start_position.
    # Instructions:
    # - Use self.trajectory_start_position as the starting position (not the center).
    # - Use a radius a = 1.0 m, and let the angular velocity (the rate of change of the angle along the circle)
    #   ramp up smoothly as omega(t) = 0.75 * tanh(0.1 t).
    # - Compute the reference position (pxr, pyr, pzr), velocity (vxr, vyr, vzr) and acceleration (axr, ayr, azr).
    # - Keep the reference yaw angle yawr = 0.
    # - Return [pxr, pyr, pzr, vxr, vyr, vzr, axr, ayr, azr, yawr].

    # def trajectory_function(self, t):
    #     if self.trajectory_type == 'horizontal_circle':
    #       pxr =
    #       pyr =
    #       ...
    #       yawr =

    #     return np.array([pxr,pyr,pzr,vxr,vyr,vzr,axr,ayr,azr,yawr])


    def navigator(self, t):
        # Returns the desired p^W_d, v^W_d, p_ddot^W_d and yaw psi at time t
        if self.flight_mode == 'takeoff' or self.flight_mode == 'land':
            T = self.takeoff_duration if self.flight_mode == 'takeoff' else self.land_duration
            k = 12.0 / T
            s = 1./(1. + np.exp(-(k*(t - T) + 6.0)))       # smooth step from 0 to 1 over 2T seconds
            s_dot = k*s*(1. - s)
            delta = self.go_to_position - self.trajectory_start_position
            ref = np.array([*(self.trajectory_start_position + s*delta), *(s_dot*delta), 0., 0., 0., 0.])
        elif self.flight_mode == 'trajectory':
            ref = self.trajectory_function(t)
        else:   # hover
            ref = np.array([*self.go_to_position, 0., 0., 0., 0., 0., 0., 0.])
        return ref[0:3], ref[3:6], ref[6:9], ref[9]


    # [TODO] PART 5: Implement the geometric controller discussed in class.
    # Inputs:  desired position p_d, velocity v_d, acceleration a_d (all in the world frame) and desired yaw yaw_d.
    # Outputs: collective thrust f_z (scalar, N) and body torques tau (numpy array of size 3, Nm).
    #
    # Instructions:
    # - Use the current state from PART 3 (self.position, self.velocity, self.R_WB, self.omega_B).
    # - Use the gains self.Kp, self.Kv, self.KR, self.Kw and the parameters self.m, self.g, self.J.
    # - You can assume that the desired angular velocity is zero.
    # - Saturate f_z to [0, self.max_thrust] and each torque to [-self.max_torque, self.max_torque].

    # def geometric_controller(self, p_d, v_d, a_d, yaw_d):
    #     ...
    #     return f_z, tau


    # [TODO] PART 6: Implement the cmd_force_torque function to publish the force-torque commands.
    # Instructions:
    # - Create a ForceTorqueCmd message
    # - Set the thrust_si, torque_x, torque_y and torque_z fields from the input parameters
    #   (the message fields are float64, convert numpy values with float())
    # - Publish the message using self.force_torque_pub
    # - See the structure of the message in
    #       ae740_crazyflie_sim/ros2_ws/src/crazyswarm2/crazyflie_interfaces/msg/ForceTorqueCmd.msg
    #
    # def cmd_force_torque(self, thrust, torque):


    def publish_reference_path(self, t):
        reference_path = Path()
        reference_path.header.frame_id = 'world'
        reference_path.header.stamp = self.get_clock().now().to_msg()

        for t_ref in np.linspace(t, t + 1.0, 20):
            p_d, _, _, _ = self.navigator(t_ref)
            ref_pose = PoseStamped()
            ref_pose.pose.position.x = p_d[0]
            ref_pose.pose.position.y = p_d[1]
            ref_pose.pose.position.z = p_d[2]
            reference_path.poses.append(ref_pose)

        self.reference_path_pub.publish(reference_path)

    def _control_loop(self):
        if self.flight_mode == 'idle':
            return

        if len(self.position) == 0 or len(self.velocity) == 0 or len(self.attitude) == 0:
            self.get_logger().warning("Empty state message.")
            return

        if not self.is_flying:
            self.is_flying = True

        if self.trajectory_changed:
            self.trajectory_start_position = self.position.copy()
            self.trajectory_t0 = self.get_clock().now()
            self.trajectory_changed = False

        t = (self.get_clock().now() - self.trajectory_t0).nanoseconds / 10.0**9

        # [TODO] PART 7: Compute and send the geometric control command at the current time step
        #
        # p_d, v_d, a_d, yaw_d = ... (reference at time t, see self.navigator(t))
        # f_z, tau = ... (geometric control law, PART 5)
        # ... (publish the command, PART 6)
        #
        # Once the landing is finished (flight_mode == 'land' and t > 2*self.land_duration), send a zero
        # command to stop the motors, and set self.flight_mode = 'idle' and self.is_flying = False.


        if self.plot_trajectory:
            self.publish_reference_path(t)

def main():
    rclpy.init()

    # Quadrotor Parameters
    mass = 0.028
    Ixx = 2.3951e-5
    Iyy = 2.3951e-5
    Izz = 3.2347e-5
    inertia = np.diag([Ixx, Iyy, Izz])

    rate = 100 # control update rate (in Hz)
    quad_name = 'cf_1'

    node = CrazyflieGeometricController(quad_name, mass, inertia, rate)

    # Standard node commands
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
   main()
