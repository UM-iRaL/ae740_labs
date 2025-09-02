import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque
import math


class TrajTrail:
    """Class to manage trajectory trails for visualization"""
    
    id_counter = 0
    
    def __init__(self, parent_node, ref_frame, dest_frame, buffer_size=160):
        self.parent = parent_node
        self.ref_frame = ref_frame
        self.dest_frame = dest_frame
        self.buffer_size = buffer_size
        self.poses = deque(maxlen=buffer_size)
        
        # Create marker
        self.marker = Marker()
        self.marker.header.frame_id = ref_frame
        self.marker.ns = "trails"
        self.marker.id = TrajTrail.id_counter
        TrajTrail.id_counter += 1
        self.marker.type = Marker.LINE_STRIP
        self.marker.action = Marker.ADD
        self.marker.color.a = 0.8
        self.marker.scale.x = 0.02
        self.marker.lifetime = Duration(seconds=1.0).to_msg()
        
    def set_color(self, r, g, b):
        """Set the color of the trail"""
        self.marker.color.r = r
        self.marker.color.g = g
        self.marker.color.b = b
        
    def set_namespace(self, ns):
        """Set the namespace of the marker"""
        self.marker.ns = ns
        
    def set_dashed(self):
        """Make the trail dashed by using LINE_LIST instead of LINE_STRIP"""
        self.marker.type = Marker.LINE_LIST
        
    def update(self):
        """Update the trail with the latest transform"""
        try:
            # ~~~~~~~~~~~~~~~~~~~~~~  BEGIN OF EDIT SECTION ~~~~~~~~~~~~~~~~~~~~~~~~~

            # The transform object needs to be populated with the most recent
            # transform from ref_frame to dest_frame as provided by tf.
            #
            # Relevant variables in this scope:
            #   - ref_frame: the frame of reference relative to which the trajectory
            #                needs to be plotted (given)
            #   - dest_frame: the frame of reference of the object whose trajectory
            #                 needs to be plotted (given)
            #   - parent.tf_listener: a tf2_ros.TransformListener object (given)
            #   - transform: the transform object that needs to be populated
            #
            # HINT: use "lookup_transform", see
            # https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Python.html
            #
            # NOTE: only using "lookup_transform", you might see a lot of errors in
            # your terminal!
            #       Unlike other topic types - which are typically received on a
            # callback basis -
            #       tf can be polled at any time and therefore "lookup_transform"
            # might fail
            #       due to timing issues. To fix it, consider the method
            # "can_transform" with
            #       a small wait time.
            
            # TODO: fill in here

            pass # to avoid error
                
            # ~~~~~~~~~~~~~~~~~~~~~~~~  END OF EDIT SECTION ~~~~~~~~~~~~~~~~~~~~~~~~~
                
        except TransformException as ex:
            self.parent.get_logger().error(f"Transform lookup failed: {ex}")
            
    def get_marker(self):
        """Get the updated marker for publishing"""
        self.update()
        
        # Update marker header
        self.marker.header.stamp = self.parent.get_clock().now().to_msg()
        
        # Clear and update points
        self.marker.points.clear()
        for pose in self.poses:
            self.marker.points.append(pose)
            
        # For dashed lines (LINE_LIST), we need even number of points
        if (self.marker.type == Marker.LINE_LIST and 
            len(self.marker.points) % 2 == 1):
            self.marker.points = self.marker.points[:-1]
            
        return self.marker


class PlotsPublisherNode(Node):
    """Node that publishes visualization markers for drone trajectories"""
    
    def __init__(self):
        super().__init__('plots_publisher_node')
        
        self.get_logger().info("Started plot node")
        
        # Initialize tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publisher for visualization markers
        self.markers_pub = self.create_publisher(MarkerArray, 'visuals', 10)
        
        # Create timer for publishing at 50Hz (0.02 seconds)
        self.timer = self.create_timer(0.02, self.on_publish)
        
        self.get_logger().info("Started heartbeat timer")
        
        # Create trajectory trails
        self.av1_trail = TrajTrail(self, "world", "av1", 300)
        self.av1_trail.set_color(0.25, 0.52, 1.0)
        self.av1_trail.set_namespace("Trail av1-world")
        
        self.av2_trail = TrajTrail(self, "world", "av2", 300)
        self.av2_trail.set_color(0.8, 0.4, 0.26)
        self.av2_trail.set_namespace("Trail av2-world")
        
        self.av2_trail_rel = TrajTrail(self, "av1", "av2", 160)
        self.av2_trail_rel.set_dashed()
        self.av2_trail_rel.set_color(0.8, 0.4, 0.26)
        self.av2_trail_rel.set_namespace("Trail av2-av1")
        
    def create_drone_marker(self, frame_id, marker_id, color_r, color_g, color_b):
        """Create a drone visualization marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "AVs"
        marker.id = marker_id
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = "package://two_drones_pkg/mesh/quadrotor.dae"
        marker.action = Marker.ADD
        
        # Set pose (identity quaternion)
        marker.pose.orientation.w = 1.0
        
        # Set scale
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Set color
        marker.color.r = color_r
        marker.color.g = color_g
        marker.color.b = color_b
        marker.color.a = 1.0
        
        # Set lifetime
        marker.lifetime = Duration(seconds=1.0).to_msg()
        
        return marker
        
    def on_publish(self):
        """Publish visualization markers"""
        visuals = MarkerArray()
        
        # Create drone markers
        av1_marker = self.create_drone_marker("av1", 0, 0.25, 0.52, 1.0)
        av2_marker = self.create_drone_marker("av2", 1, 0.8, 0.4, 0.26)
        
        visuals.markers.append(av1_marker)
        visuals.markers.append(av2_marker)
        
        # Add trajectory trails
        visuals.markers.append(self.av1_trail.get_marker())
        visuals.markers.append(self.av2_trail.get_marker())
        visuals.markers.append(self.av2_trail_rel.get_marker())
        
        # Publish all markers
        self.markers_pub.publish(visuals)


def main(args=None):
    rclpy.init(args=args)
    
    plots_node = PlotsPublisherNode()
    
    try:
        rclpy.spin(plots_node)
    except KeyboardInterrupt:
        pass
    finally:
        plots_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
