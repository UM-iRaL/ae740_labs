from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Crazyflie MPC Controller Node
        Node(
            package='controller_pkg',
            executable='crazyflie_mpc_controller',
            output='screen',
            parameters=[
                # Add any parameters if needed
            ]
        ),
        
        # Target Tracking Controller Node
        Node(
            package='controller_pkg',
            executable='target_tracking_controller', 
            output='screen',
            parameters=[
                # Add any parameters if needed
            ]
        ),
    ])