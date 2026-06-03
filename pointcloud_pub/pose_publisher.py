import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header


class PosePublisher(Node):
    def __init__(self, topic_name='poses'):
        super().__init__('pose_publisher')
        self.publisher = self.create_publisher(PoseArray, topic_name, 10)

    def create_pose_array(self, points, frame):


        points = np.asarray(points, dtype=np.float32)
        assert points.ndim == 2, f"points must be 2D, got {points.ndim}D"

        msg = PoseArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame

        for row in points:
            x, y, z = row[:3]  # ignore the class column

            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(z)

            # neutral orientation
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0

            msg.poses.append(pose)

        return msg