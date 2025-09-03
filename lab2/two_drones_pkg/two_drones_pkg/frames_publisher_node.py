import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class FramesPublisherNode(Node):
    def __init__(self):
        super().__init__('frames_publisher_node')

        # TODO: Instantiate the Transform Broadcaster
    
        # self.tf_broadcaster = ...
        

    def on_publish(self):
        # NOTE: This method is called at 50Hz.

        # 1. TODO: Compute time elapsed in seconds since the node started
        # elapsed_duration = ...
        
        
        
        
        # Declare the two TransformStamped messages
        world_T_av1 = TransformStamped()
        world_T_av2 = TransformStamped()
        
        
        # 2. TODO: Populate the two transforms for the AVs, using the variable "time"
        #       computed above. Specifically:
        #     - world_T_av1 should have origin in [cos(time), sin(time), 0.0] and
        #       rotation such that:
        #        i) its y axis stays tangent to the trajectory and
        #       ii) the z vector stays parallel to that of the world frame
        #
        #     - world_T_av2 should have origin in [sin(time), 0.0, cos(2*time)], the
        #       rotation is irrelevant to our purpose.
        #    NOTE: world_T_av1's orientation is crucial for the rest of the
        #    assignment, make sure you get it right
        #
        #    HINTS:
        #    - check out the ROS tf2 Tutorials:
        #    https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html
        #    - consider using tf_transformations.quaternion_from_euler() for world_T_av1
        #      or tf2_ros.transformations module depending on your ROS2 version
        #    - The header needs a timestamp and frame_id.
        #   	world_T_av1.header.stamp = self.get_clock().now().to_msg()
        #   	world_T_av1.header.frame_id = 'world'
        #   	world_T_av1.child_frame_id = 'av1'
        #    - Set the translation vector (x, y, z).
        #      world_T_av1.transform.translation.x = ...
        #    - Set the orientation as a quaternion (x, y, z, w). For world_T_av1,
        #      you'll need to calculate the correct yaw angle and convert it.

        

        # 3. TODO: Publish the transforms, namely:
        #     - world_T_av1 with frame_id "world", child_frame_id "av1"
        #     - world_T_av2 with frame_id "world", child_frame_id "av2"
        # HINTS:
        #     1. you need to define a tf2_ros.TransformBroadcaster as a member of the
        #        node class and use its sendTransform method below
        #     2. the frame names are crucial for the rest of the assignment,
        #        make sure they are as specified, "av1", "av2" and "world"
	


def main(args=None):
    rclpy.init(args=args)
    node = FramesPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
