#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid 
import random 
import math
import tf_transformations
from geometry_msgs.msg import Point, PoseStamped 
from visualization_msgs.msg import Marker
import numpy as np

# --- 1. Data Structure for a single Point (internal RRT representation) ---
class RRTPoint:
    """A simple class to represent a 2D point with x and y coordinates for RRT internal use."""
    def __init__(self, x_val, y_val):
        self.x = x_val
        self.y = y_val

    def __repr__(self):
        return f"RRTPoint(x={self.x:.2f}, y={self.y:.2f})"

    # Helper to convert to ROS Point message
    def to_ros_point(self):
        ros_point = Point()
        ros_point.x = self.x
        ros_point.y = self.y
        ros_point.z = 0.0 # Assuming 2D planning
        return ros_point

# --- 2. Data Structure for a single Node in the RRT tree ---
class RRTNodeData:
    """
    Represents a node in the RRT tree.
    Each node stores its RRTPoint and a reference to its parent RRTNodeData.
    """
    def __init__(self, point, parent = None):
        self.point = point  # The actual RRTPoint (x, y coordinates) of this node
        self.parent = parent # Reference to the parent RRTNodeData object in the tree (None for the root/start node)

    def __repr__(self):
        if self.parent:
            return f"RRTNodeData(point={self.point}, parent_coords=({self.parent.point.x:.2f},{self.parent.point.y:.2f}))"
        else:
            return f"RRTNodeData(point={self.point}, parent=None)"


# --- Your MAIN RRT Planner Class (named rrtNode in your code) ---
class rrtNode(Node): 
    def __init__(self):
        super().__init__("rrt_node") 

        # Parameters
        self.declare_parameter('topic_pose', "/ego_racecar/odom")
        self.declare_parameter('topic_path', "driverPath")
        self.declare_parameter('sampling_radius', 20.0)
        self.declare_parameter('min_sampling_dist', 0.5)
        self.declare_parameter('forward_angle_limit_deg', 45.0) 
        self.declare_parameter('rrt_step_size', 0.5) 
        self.declare_parameter('goal_threshold', 0.1) 
        self.declare_parameter('max_rrt_iterations', 10000) 
        self.declare_parameter('rrt_planning_frequency', 1.0) 
        self.declare_parameter('map_topic', "/map") 
        self.declare_parameter('obstacle_threshold', 50) 
        self.declare_parameter('debug_publish_rrt_tree', True) 
        self.declare_parameter('look_ahead_distance', 5.0) 
        self.declare_parameter('path_update_threshold', 7.0) # Adjusted to 0.2 as you tested

        self.sampling_radius = self.get_parameter('sampling_radius').value
        self.min_sampling_dist = self.get_parameter('min_sampling_dist').value
        self.forward_angle_limit = math.radians(self.get_parameter('forward_angle_limit_deg').value) 
        self.rrt_step_size = self.get_parameter('rrt_step_size').value
        self.goal_threshold = self.get_parameter('goal_threshold').value
        self.max_rrt_iterations = self.get_parameter('max_rrt_iterations').value
        self.rrt_planning_frequency = self.get_parameter('rrt_planning_frequency').value
        self.map_topic = self.get_parameter('map_topic').value 
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value 
        self.debug_publish_rrt_tree = self.get_parameter('debug_publish_rrt_tree').value
        self.look_ahead_distance = self.get_parameter('look_ahead_distance').value
        self.path_update_threshold = self.get_parameter('path_update_threshold').value

        # Get parameters to variables
        self.pose_topic = self.get_parameter('topic_pose').get_parameter_value().string_value
        self.path_topic = self.get_parameter('topic_path').get_parameter_value().string_value

        # Subscriptions
        self.path_sub_ = self.create_subscription(Path, self.path_topic, self.path_callback, 10)
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 10)
        self.get_logger().info(f'Subscribing to topic: {self.pose_topic}, {self.path_topic} and {self.map_topic}')

        # Publishers
        self.sampled_point_viz_publisher = self.create_publisher(Marker, '/rrt_sampled_points_viz', 10)
        self.rrt_path_publisher = self.create_publisher(Path, '/rrt_generated_path', 10) 
        self.rrt_tree_viz_publisher = self.create_publisher(Marker, '/rrt_tree_viz', 10) 
        self.rrt_goal_viz_publisher = self.create_publisher(Marker, '/rrt_goal_viz', 10)

        # Other initialized variables
        self.global_path_waypoints = [] 
        self.robot_current_point = None 
        self.robot_current_yaw = 0.0 
        self.goal_point = None 
        self.current_path_waypoint_idx = 0 

        self.rrt_tree = [] 
        self.closest_node_to_goal = None 

        self.enable_sampling_viz = True
        self.debug_marker_id_counter = 0

        # Map data storage
        self.map_data = None 
        self.map_resolution = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_width_px = 0
        self.map_height_px = 0


        # Timers
        if self.enable_sampling_viz:
            self.debug_sampling_timer = self.create_timer(0.5, self.debug_publish_sampled_point)
        
        self.rrt_planning_timer = self.create_timer(1.0 / self.rrt_planning_frequency, self.rrt_planning_loop)
        self.get_logger().info(f"RRT planning loop set to run at {self.rrt_planning_frequency} Hz.")

        self.get_logger().info("RRTNode (Planner) initialized.")

    def map_callback(self, msg):
        """
        Callback for receiving the OccupancyGrid map data.
        """
        if self.map_data is None or \
           (self.map_resolution != msg.info.resolution or \
            self.map_width_px != msg.info.width or \
            self.map_height_px != msg.info.height): 

            self.map_resolution = msg.info.resolution
            self.map_origin_x = msg.info.origin.position.x
            self.map_origin_y = msg.info.origin.position.y
            self.map_width_px = msg.info.width
            self.map_height_px = msg.info.height
            
            self.map_data = np.array(msg.data).reshape((self.map_height_px, self.map_width_px))
            
            self.get_logger().info(f"MAP_CB: Map received/updated: Resolution={self.map_resolution:.2f}m, "
                                   f"Origin=({self.map_origin_x:.2f},{self.map_origin_y:.2f}), "
                                   f"Dimensions={self.map_width_px}x{self.map_height_px} pixels.")

    def _world_to_map_coords(self, world_point): 
        """
        Converts world coordinates (RRTPoint) to map grid coordinates (col, row).
        """
        if self.map_data is None:
            return -1, -1 # Indicate map not loaded

        col = int((world_point.x - self.map_origin_x) / self.map_resolution)
        row = int((world_point.y - self.map_origin_y) / self.map_resolution)
        
        return col, row

    def _is_within_map_bounds(self, col, row):
        """
        Checks if the given map grid coordinates (col, row) are within the map's boundaries.
        """
        if self.map_data is None:
            return False 
        
        return 0 <= col < self.map_width_px and 0 <= row < self.map_height_px

    def _get_occupancy_value(self, col, row):
        """
        Gets the occupancy value of a specific map grid cell.
        Returns -1 for out of bounds or if map not loaded.
        """
        if self.map_data is None or not self._is_within_map_bounds(col, row):
            return -1 

        return self.map_data[row, col] 

    def _is_point_collision(self, point):
        """
        Checks if a single RRTPoint is in collision with an obstacle or out of map bounds.
        """
        if self.map_data is None:
            self.get_logger().warn("COLLISION_CHECK: Map data not loaded for point collision check. Assuming no collision for now.", once=True)
            return False 

        col, row = self._world_to_map_coords(point)
        
        if not self._is_within_map_bounds(col, row):
            return True # Out of bounds is considered a collision

        occupancy = self._get_occupancy_value(col, row)
        
        # Treat unknown (-1) or occupied (>= threshold) as collision
        if occupancy >= self.obstacle_threshold or occupancy == -1:
            return True 
            
        return False 

    def pose_callback(self, msg): 
        """Callback for receiving the robot's Odometry message."""
        new_robot_point = RRTPoint(msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # Only update and potentially reset tree if robot has moved significantly
        if self.robot_current_point is None or \
           self._calculate_distance(self.robot_current_point, new_robot_point) > 0.1: 
            self.robot_current_point = new_robot_point
            self.robot_current_yaw = tf_transformations.euler_from_quaternion([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])[2] # Only yaw is needed for 2D planning

            # Reset the tree here so it always plans from current robot position when it moves
            # This is important for replanning in dynamic environments / with dynamic goals
            if self.robot_current_point and (not self.rrt_tree or self._calculate_distance(self.rrt_tree[0].point, self.robot_current_point) > 0.1):
                self.rrt_tree = [RRTNodeData(point=self.robot_current_point, parent=None)]
                self.closest_node_to_goal = self.rrt_tree[0]
                self.get_logger().debug(f"POSE_CB: RRT tree reset with start node: {self.robot_current_point}")


        # Initial tree setup if it's empty (e.g. at startup)
        if not self.rrt_tree and self.robot_current_point is not None:
            start_node_data = RRTNodeData(point=self.robot_current_point, parent=None)
            self.rrt_tree.append(start_node_data)
            self.closest_node_to_goal = start_node_data 
            self.get_logger().info(f"POSE_CB: RRT tree initialized with start node: {self.robot_current_point}")
            
    def path_callback(self, path_msg):
        """!
        @brief Retrieving information about Path waypoints

        Subscribe to topic and save the data from the messages that has been published to the topic
        """
        # Only process if path hasn't been received or if it has significantly changed (for simplicity, only once here)
        if not self.global_path_waypoints: 
            self.global_path_waypoints = [
                RRTPoint(pose_stamped.pose.position.x, pose_stamped.pose.position.y)
                for pose_stamped in path_msg.poses
            ]
            self.get_logger().info(f"PATH_CB: Global path with {len(self.global_path_waypoints)} waypoints received. First waypoint: {self.global_path_waypoints[0]}")
            
        else:
            self.get_logger().debug("PATH_CB: Global path already received, skipping update.") 

    def _update_rrt_goal(self):
        """
        Dynamically updates the RRT's goal point by finding a look-ahead point
        on the global path, handling circular paths.
        """
        self.get_logger().debug("UPDATE_GOAL: _update_rrt_goal started.") # Debug start of function

        if self.robot_current_point is None:
            self.get_logger().info("UPDATE_GOAL: Robot current point is None. Cannot update goal.")
            return
        
        if not self.global_path_waypoints:
            self.get_logger().info("UPDATE_GOAL: Global path waypoints list is empty. Cannot update goal.")
            return

        # 1. Find the closest waypoint index to the robot's current position
        min_dist_to_robot = float('inf')
        temp_closest_idx = -1
        for i, wp in enumerate(self.global_path_waypoints):
            dist = self._calculate_distance(self.robot_current_point, wp)
            if dist < min_dist_to_robot:
                min_dist_to_robot = dist
                temp_closest_idx = i
        
        if temp_closest_idx == -1: 
            self.get_logger().warn("UPDATE_GOAL: Could not find closest waypoint to robot on global path. This should not happen if path exists.")
            return
        
        self.current_path_waypoint_idx = temp_closest_idx
        self.get_logger().debug(f"UPDATE_GOAL: Closest waypoint index to robot: {self.current_path_waypoint_idx} (dist: {min_dist_to_robot:.2f})")

        # 2. Check if the current RRT goal needs updating based on path_update_threshold
        # Only update if the current goal is None (first time) OR if robot is close enough to it
        if self.goal_point is not None:
            dist_to_current_goal = self._calculate_distance(self.robot_current_point, self.goal_point)
            if dist_to_current_goal > self.path_update_threshold:
                self.get_logger().debug(f"UPDATE_GOAL: Goal update skipped. Robot dist to current goal ({self.goal_point}): {dist_to_current_goal:.2f}m > Threshold: {self.path_update_threshold:.2f}m")
                return # If the robot is still far from the current RRT goal, keep it stable.
            else:
                self.get_logger().debug(f"UPDATE_GOAL: Robot is close to current goal ({self.goal_point}, dist: {dist_to_current_goal:.2f}m). Proceeding to find new look-ahead goal.")
        else:
            self.get_logger().debug("UPDATE_GOAL: goal_point is None (first time setting goal). Proceeding to find new look-ahead goal.")

        # 3. Calculate the look-ahead point by iterating along the path
        current_look_ahead_dist = 0.0
        look_ahead_idx = self.current_path_waypoint_idx
        
        path_len = len(self.global_path_waypoints)
        if path_len == 0: 
            self.get_logger().warn("UPDATE_GOAL: Global path became empty during look-ahead calculation.")
            return 

        start_idx_for_loop = self.current_path_waypoint_idx 
        
        # Iterate through waypoints, accumulating distance until look_ahead_distance is met
        # Loop twice the path length to ensure it can wrap around multiple times for very long look_ahead_distance
        found_look_ahead = False
        for _ in range(path_len * 2): 
            next_idx = (look_ahead_idx + 1) % path_len
            
            # This condition handles cases where look_ahead_distance is very large,
            # or if it's a non-circular path and we've reached the end.
            if next_idx == start_idx_for_loop and current_look_ahead_dist < self.look_ahead_distance:
                # If we've looped back to the start and haven't covered enough distance,
                # use the current look_ahead_idx as the furthest valid point.
                self.get_logger().debug(f"UPDATE_GOAL: Look-ahead distance ({self.look_ahead_distance:.2f}m) exceeds available path segment length ({current_look_ahead_dist:.2f}m). Using furthest point found at index {look_ahead_idx}.")
                found_look_ahead = True
                break 

            segment_dist = self._calculate_distance(self.global_path_waypoints[look_ahead_idx], 
                                                    self.global_path_waypoints[next_idx])
            
            if (current_look_ahead_dist + segment_dist) >= self.look_ahead_distance:
                # The next segment would exceed the look_ahead_distance, so this is our target waypoint
                look_ahead_idx = next_idx # Take the point that crosses the threshold
                found_look_ahead = True
                break
            
            current_look_ahead_dist += segment_dist
            look_ahead_idx = next_idx
        
        if not found_look_ahead:
             # This can happen if the path is too short or logic error; default to current_path_waypoint_idx
            look_ahead_idx = self.current_path_waypoint_idx
            self.get_logger().warn(f"UPDATE_GOAL: Could not find sufficient look-ahead point. Defaulting to current path index {look_ahead_idx}.")

        new_goal_point = self.global_path_waypoints[look_ahead_idx]
        self.get_logger().debug(f"UPDATE_GOAL: Candidate new goal point: {new_goal_point} (from global path index {look_ahead_idx}).")

        # 4. Update self.goal_point if it's different from the previously set goal
        # Use a small epsilon to avoid floating point comparison issues
        if self.goal_point is None or \
           (self._calculate_distance(self.goal_point, new_goal_point) > 0.01): 
            old_goal_point = self.goal_point
            self.goal_point = new_goal_point
            self.get_logger().info(f"UPDATE_GOAL: --- Goal point ACTUALLY UPDATED. Old: {old_goal_point}, New: {self.goal_point}. Path index: {look_ahead_idx}. Dist to robot: {self._calculate_distance(self.robot_current_point, new_goal_point):.2f}m.")
            
            # Reset RRT tree only if a *new* look-ahead goal is set
            if self.robot_current_point:
                self.rrt_tree = [RRTNodeData(point=self.robot_current_point, parent=None)]
                self.closest_node_to_goal = self.rrt_tree[0]
                self.get_logger().debug("UPDATE_GOAL: RRT tree reset due to new look-ahead goal point.")
        else:
            self.get_logger().debug(f"UPDATE_GOAL: New goal point ({new_goal_point}) is effectively same as old ({self.goal_point}). No update needed.")


    def _sample_free(self):
        """
        Generates a random point (x, y) within the defined sampling_radius around the robot,
        ensuring the point is in front of the robot (within forward_angle_limit),
        and that the point is not in collision with an obstacle.
        Returns an RRTPoint object or None if no free point is found after some tries.
        """
        if self.robot_current_point is None:
            return None

        max_sample_tries = 100 
        for i in range(max_sample_tries):
            robot_x = self.robot_current_point.x
            robot_y = self.robot_current_point.y
            robot_yaw = self.robot_current_yaw

            r = random.uniform(self.min_sampling_dist, self.sampling_radius)
            angle_min = robot_yaw - self.forward_angle_limit
            angle_max = robot_yaw + self.forward_angle_limit
            theta = random.uniform(angle_min, angle_max)
            
            sampled_x_global = robot_x + r * math.cos(theta)
            sampled_y_global = robot_y + r * math.sin(theta)

            sampled_point = RRTPoint(sampled_x_global, sampled_y_global)

            if not self._is_point_collision(sampled_point):
                return sampled_point
        
        self.get_logger().warn(f"SAMPLE_FREE: Failed to find a free sample point after {max_sample_tries} tries. Consider adjusting sampling parameters or map.", once=True)
        return None 


    def debug_publish_sampled_point(self):
        """
        Periodically publishes a sampled point for visualization in RViz.
        Used for debugging the sampling area.
        """
        if not self.enable_sampling_viz:
            return

        sampled_point_rrt = self._sample_free()

        if sampled_point_rrt:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rrt_sampled_points"
            marker.id = self.debug_marker_id_counter
            self.debug_marker_id_counter += 1

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = sampled_point_rrt.to_ros_point() 
            marker.pose.orientation.w = 1.0 
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0

            self.sampled_point_viz_publisher.publish(marker)

    def _find_nearest(self, x_rand):
        """
        Finds the nearest node in the RRT tree (self.rrt_tree) to the given random point (x_rand).
        """
        if not self.rrt_tree:
            self.get_logger().warn("FIND_NEAREST: RRT tree is empty, cannot find nearest node.")
            return None

        tree_points_np = np.array([[node.point.x, node.point.y] for node in self.rrt_tree])
        x_rand_np = np.array([x_rand.x, x_rand.y])
        distances = np.linalg.norm(tree_points_np - x_rand_np, axis=1)
        min_distance_index = np.argmin(distances)
        nearest_node_data = self.rrt_tree[min_distance_index]

        return nearest_node_data

    def _steer(self, x_nearest_data, x_rand): 
        """
        Extends from x_nearest_data's point towards x_rand by a fixed step_size.
        """
        p_nearest = x_nearest_data.point 
        p_rand = x_rand
        
        distance = self._calculate_distance(p_nearest, p_rand) 

        if distance <= self.rrt_step_size or distance == 0:
            return p_rand 
        
        direction_vector_x = p_rand.x - p_nearest.x
        direction_vector_y = p_rand.y - p_nearest.y

        unit_vector_x = direction_vector_x / distance
        unit_vector_y = direction_vector_y / distance

        x_new_x = p_nearest.x + unit_vector_x * self.rrt_step_size
        x_new_y = p_nearest.y + unit_vector_y * self.rrt_step_size

        return RRTPoint(x_new_x, x_new_y)
        

    def _calculate_distance(self, point1, point2): 
        '''
        Calculates the Euclidean distance between two RRTPoint objects.
        '''
        np_point1 = np.array([point1.x, point1.y])
        np_point2 = np.array([point2.x, point2.y])

        distance = np.linalg.norm(np_point1 - np_point2)
    
        return float(distance) 

    def _is_segment_collision_free(self, start_point, end_point):
        """
        Checks if the straight-line segment between start_point and end_point is collision-free.
        It steps along the segment and checks each point's occupancy.
        """
        if self.map_data is None:
            self.get_logger().warn("SEGMENT_COLLISION: Map data not loaded for segment collision check. Assuming free.", once=True)
            return True 

        segment_length = self._calculate_distance(start_point, end_point)
        if segment_length == 0: 
            return not self._is_point_collision(start_point)

        # Number of steps to check along the segment, based on map resolution
        num_steps = int(segment_length / (self.map_resolution / 2.0)) 
        if num_steps < 2: num_steps = 2 

        for i in range(num_steps + 1):
            alpha = float(i) / num_steps # Interpolation factor (0.0 to 1.0)
            interp_x = start_point.x + alpha * (end_point.x - start_point.x)
            interp_y = start_point.y + alpha * (end_point.y - start_point.y)
            
            current_point_rrt = RRTPoint(interp_x, interp_y)

            if self._is_point_collision(current_point_rrt):
                # self.get_logger().debug(f"SEGMENT_COLLISION: Collision detected at {current_point_rrt} between {start_point} and {end_point}.")
                return False # Collision detected along the segment
                
        return True # No collision detected

    def _generate_rrt_path(self, goal_node_data):
        """
        Reconstructs the path from the goal_node_data back to the start node.
        """
        path = []
        current_node = goal_node_data
        while current_node is not None:
            path.append(current_node.point)
            current_node = current_node.parent
        return path[::-1] # Reverse to get path from start to goal

    def _publish_path(self, path_points):
        """
        Publishes the generated RRT path as a nav_msgs/Path message for visualization.
        """
        if not path_points:
            return

        path_msg = Path()
        path_msg.header.frame_id = "map" 
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for rrt_point in path_points:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = path_msg.header.frame_id
            pose_stamped.header.stamp = path_msg.header.stamp
            pose_stamped.pose.position = rrt_point.to_ros_point() 
            pose_stamped.pose.orientation.w = 1.0 # Assuming 2D, no rotation for path points
            path_msg.poses.append(pose_stamped)
        
        self.rrt_path_publisher.publish(path_msg)

    def _publish_rrt_tree_viz(self):
        """
        Publishes the entire RRT tree for visualization in RViz using Marker messages.
        Nodes as spheres, edges as lines.
        """
        if not self.debug_publish_rrt_tree or not self.rrt_tree:
            return

        marker_array = Marker()
        marker_array.header.frame_id = "map"
        marker_array.header.stamp = self.get_clock().now().to_msg()
        marker_array.ns = "rrt_tree_edges"
        marker_array.id = 0 
        marker_array.type = Marker.LINE_LIST 
        marker_array.action = Marker.ADD
        marker_array.pose.orientation.w = 1.0 
        marker_array.scale.x = 0.02 # Line thickness
        marker_array.color.a = 0.8 # Alpha (opacity)
        marker_array.color.r = 1.0 # Red
        marker_array.color.g = 0.0
        marker_array.color.b = 0.0

        node_markers = Marker()
        node_markers.header.frame_id = "map"
        node_markers.header.stamp = self.get_clock().now().to_msg()
        node_markers.ns = "rrt_tree_nodes"
        node_markers.id = 1
        node_markers.type = Marker.SPHERE_LIST 
        node_markers.action = Marker.ADD
        node_markers.pose.orientation.w = 1.0
        node_markers.scale.x = 0.08 # Sphere size
        node_markers.scale.y = 0.08
        node_markers.scale.z = 0.08
        node_markers.color.a = 0.8 
        node_markers.color.r = 0.0
        node_markers.color.g = 0.0
        node_markers.color.b = 1.0 # Blue

        for node_data in self.rrt_tree:
            node_markers.points.append(node_data.point.to_ros_point())
            
            if node_data.parent:
                marker_array.points.append(node_data.parent.point.to_ros_point())
                marker_array.points.append(node_data.point.to_ros_point())
        
        self.rrt_tree_viz_publisher.publish(marker_array)
        self.rrt_tree_viz_publisher.publish(node_markers)

    def _publish_goal_point_viz(self):
        """
        Publishes the current RRT goal point for visualization in RViz.
        """
        if self.goal_point is None:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_goal_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = self.goal_point.to_ros_point()
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3 # Size of the goal sphere
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0 # Alpha (opacity)
        marker.color.r = 0.0 
        marker.color.g = 1.0 # Green
        marker.color.b = 0.0 

        self.rrt_goal_viz_publisher.publish(marker)

    def rrt_planning_loop(self):
        """
        Executes the main RRT planning loop periodically.
        """
        self.get_logger().debug("RRT_LOOP: rrt_planning_loop started.") # Debug start of main loop

        # Always try to update the RRT goal point dynamically
        self._update_rrt_goal() 
        
        # Always publish the goal point for visualization
        self._publish_goal_point_viz()
        
        # Ensure robot pose, goal, and map are available before planning
        if self.robot_current_point is None:
            self.get_logger().info("RRT_LOOP: Waiting for robot pose (robot_current_point is None).")
            return
        if self.goal_point is None:
            self.get_logger().info("RRT_LOOP: Waiting for dynamic goal point to be set (goal_point is None).")
            return
        if self.map_data is None:
            self.get_logger().info("RRT_LOOP: Waiting for map data (map_data is None).")
            return

        # If RRT has found a path to the current dynamic goal (within its goal_threshold)
        # then we might not need to plan new nodes in this tick, just maintain.
        if self.closest_node_to_goal and self._calculate_distance(self.closest_node_to_goal.point, self.goal_point) <= self.goal_threshold:
            self.get_logger().debug("RRT_LOOP: Current look-ahead goal reached or very close. Publishing current best path.")
            # Publish the path that leads to the current look-ahead goal
            final_path_points = self._generate_rrt_path(self.closest_node_to_goal)
            self._publish_path(final_path_points)
            # IMPORTANT: The 'return' here stops further tree expansion for THIS loop cycle
            # if the current look-ahead goal is already met. This is to stabilize planning.
            # If your global path has a definite end, this will cause planning to effectively stop
            # once the last global path point (which becomes the goal_point) is reached.
            return 

        # Ensure the tree is initialized with the robot's current position
        if not self.rrt_tree:
            self.get_logger().warn("RRT_LOOP: RRT tree is empty. This should have been initialized in pose_callback.")
            return 

        # Number of iterations to run RRT in this planning tick
        iterations_per_tick = 50 
        
        self.get_logger().debug(f"RRT_LOOP: Running {iterations_per_tick} iterations. Tree size: {len(self.rrt_tree)}. Closest node dist to current goal: {self._calculate_distance(self.closest_node_to_goal.point, self.goal_point):.2f}")

        for i in range(iterations_per_tick):
            # Check if goal is met BEFORE trying to add more nodes (within this loop)
            if self.closest_node_to_goal and self._calculate_distance(self.closest_node_to_goal.point, self.goal_point) <= self.goal_threshold:
                self.get_logger().debug(f"RRT_LOOP: Goal reached within planning tick. Path to {self.goal_point} found. Breaking iteration loop.")
                break 
            
            # 1. Sample a random point in free space
            x_rand = self._sample_free()
            if x_rand is None:
                self.get_logger().debug(f"RRT_LOOP (Iter {i+1}): _sample_free returned None.")
                continue 

            # 2. Find the nearest node in the tree to x_rand
            x_nearest_data = self._find_nearest(x_rand)
            if x_nearest_data is None: 
                self.get_logger().warn(f"RRT_LOOP (Iter {i+1}): _find_nearest returned None, tree might be empty or corrupted.")
                continue 

            # 3. Steer from x_nearest_data's point towards x_rand to get the new point (x_new_point)
            x_new_point = self._steer(x_nearest_data, x_rand)

            # 4. Check if the segment from x_nearest_data to x_new_point is collision-free
            if self._is_segment_collision_free(x_nearest_data.point, x_new_point):
                # 5. Create a new RRTNodeData for x_new_point and add it to the tree
                x_new_node_data = RRTNodeData(point=x_new_point, parent=x_nearest_data)
                self.rrt_tree.append(x_new_node_data)
                
                # Update the closest node to the goal
                if self._calculate_distance(x_new_node_data.point, self.goal_point) < \
                   self._calculate_distance(self.closest_node_to_goal.point, self.goal_point):
                    self.closest_node_to_goal = x_new_node_data
                    self.get_logger().debug(f"RRT_LOOP (Iter {i+1}): New closest node to goal found at {x_new_node_data.point}. Distance: {self._calculate_distance(x_new_node_data.point, self.goal_point):.2f}")
            else:
                self.get_logger().debug(f"RRT_LOOP (Iter {i+1}): Segment from {x_nearest_data.point} to {x_new_point} is NOT collision free.")

        # --- Debugging Visualizations ---
        # Always publish the current best path at the end of each planning tick,
        # if the tree has grown beyond the start node.
        if len(self.rrt_tree) > 1:
            current_best_path = self._generate_rrt_path(self.closest_node_to_goal)
            self._publish_path(current_best_path)
        
        # Publish RRT tree visualization periodically (less often than path, as it's more data)
        # Publishes every 200 new nodes added to the tree
        if self.debug_publish_rrt_tree and len(self.rrt_tree) > 1 and len(self.rrt_tree) % 200 == 0: 
            self.get_logger().info(f"RRT_LOOP: Periodically publishing RRT tree visualization. Tree size: {len(self.rrt_tree)}.")
            self._publish_rrt_tree_viz()


def main(args=None):
    rclpy.init(args=args)
    node = rrtNode() 
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()