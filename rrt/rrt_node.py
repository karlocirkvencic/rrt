#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
import random 
import math
import tf2_geometry_msgs
from tf2_ros import TransformException, Buffer, TransformListener
from geometry_msgs.msg import Point, PoseStamped 
import tf_transformations
from visualization_msgs.msg import Marker


class rrtNode(Node): 
    def __init__(self):
        super().__init__("rrt_node") 

        #Parameters
        self.declare_parameter('topic_pose', "/ego_racecar/odom")
        self.declare_parameter('topic_path', "driverPath")

        #Parameters for sample free
        self.declare_parameter('sampling_radius', 4.0)
        self.declare_parameter('min_sampling_dist', 0.5)
        self.declare_parameter('forward_angle_limit_deg', 45.0) # Declare in degrees for easier human input

        self.sampling_radius = self.get_parameter('sampling_radius').value
        self.min_sampling_dist = self.get_parameter('min_sampling_dist').value
        self.forward_angle_limit = math.radians(self.get_parameter('forward_angle_limit_deg').value) 


        #Get parameters to variables
        self.pose_topic = self.get_parameter('topic_pose').get_parameter_value().string_value
        self.path_topic = self.get_parameter('topic_path').get_parameter_value().string_value

        #Subscriptions
        self.path_sub_ = self.create_subscription(Path, self.path_topic, self.path_callback, 10)
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 10)
        self.get_logger().info(f'Subscribing to topic: {self.pose_topic}')

        # Publishers
        self.sampled_point_viz_publisher = self.create_publisher(Marker, '/rrt_sampled_points_viz', 10)


        #Other initialized variables
        self.path = []
        self.path_array = None
        self.robot_pose = None
        self.enable_sampling_viz = True
        self.debug_marker_id_counter = 0

        # Timers
        if self.enable_sampling_viz:
            self.debug_sampling_timer = self.create_timer(0.5, self.debug_publish_sampled_point)


        

    def pose_callback(self, msg: PoseStamped):
        """Callback for receiving the robot's pose."""
        self.robot_pose = msg
        self.get_logger().info(f"Robot pose: {self.robot_pose.pose.pose.position.x}, {self.robot_pose.pose.pose.position.y}", once=True)
        test = self._random_point()
        self.get_logger().info(f"Random point: {test}", once=True)

   
    def path_callback(self,path_msg):  #callback is not tested !!!!!!!!!!!!!!!!!!
        """!
        @brief Retrieving information about Path waypoints

        Subscribe to topic and save the data from the messages that has been published to the topic

        """
        if not self.path: 
            #Accesing the coordinates of waypoints inside the Path message
            self.path = [[pose_stamped.pose.position.x, pose_stamped.pose.position.y] for pose_stamped in path_msg.poses]
                
            self.path_array = np.array(self.path) #
            #self.get_logger().info(f"List of waypoints: {self.path_array}", once=True)
            
        else:
            pass #we want to retrieve path information only once 

    def _get_local_target_point():
        pass

    def _random_point(self):
        """
        Generates a random point (x, y) within the defined sampling_radius around the robot,
        ensuring the point is in front of the robot (within forward_angle_limit).
        Returns a geometry_msgs.msg.Point object or None if robot_pose is not available.
        """
        if self.robot_pose is None:
            self.get_logger().warn("Robot pose not available for sampling in sample_free.")
            return None

        #Get current robot position
        robot_x = self.robot_pose.pose.pose.position.x
        robot_y = self.robot_pose.pose.pose.position.y

        #Get robot orientation in euler
        _, _, robot_yaw = tf_transformations.euler_from_quaternion([
            self.robot_pose.pose.pose.orientation.x,
            self.robot_pose.pose.pose.orientation.y,
            self.robot_pose.pose.pose.orientation.z,
            self.robot_pose.pose.pose.orientation.w
        ])

        #sample radius and angle in polar coordinates
        r = random.uniform(self.min_sampling_dist, self.sampling_radius)
        angle_min = robot_yaw - self.forward_angle_limit
        angle_max = robot_yaw + self.forward_angle_limit
        theta = random.uniform(angle_min, angle_max)
        
        #converting polar coordinates to global cartesian coordinates
        sampled_x_global = robot_x + r * math.cos(theta)
        sampled_y_global = robot_y + r * math.sin(theta)

        sampled_point = Point()
        sampled_point.x = sampled_x_global
        sampled_point.y = sampled_y_global
        sampled_point.z = 0.0 # Assuming 2D planning

        return sampled_point

    def debug_publish_sampled_point(self):
        if not self.enable_sampling_viz:
            return

        sampled_point = self._random_point()

        if sampled_point:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rrt_sampled_points"
            marker.id = self.debug_marker_id_counter
            self.debug_marker_id_counter += 1

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = sampled_point
            marker.pose.orientation.w = 1.0 # Identity for sphere
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0

            self.sampled_point_viz_publisher.publish(marker)

    def _find_nearest_point():
        pass
    
    def _add_node_to_tree():
        pass

    def _check_goal():
        pass

    def _publish_rrt_markers():
        #Ili to u add node to tree?
        pass

    def _reconstruct_path():
        pass

    def rrt_planning_loop():
        pass

    

    


def main(args=None):
    rclpy.init(args=args)
    node = rrtNode() 
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()


'''
funkcija za dohvaćanje poze robota --------------------

funkcija za generiranje random točke -------------------------
- gledamo polarne koordinate

funkcija za izračun udaljenosti točke od svih ostalih točaka

funkcija za crtanje vizualizacijskih markera nove točke i parent točke (spaja 2 najbliže)

Napredno:

funkcija za provjeru prepreke
- occupancy grid mogu dobiti preko statične mape

funkcija za provjeru blizine cilja (threshold)
- tak dugo dodaje točke dok nije u blizini thresholda
- radijus


imat ću listu točaka lokalnog RRT-a koju će pure pursuit dohvaćat



globalni cilj će on dohvatat puno veći nego što će generirati lokalni cilj
t
'''