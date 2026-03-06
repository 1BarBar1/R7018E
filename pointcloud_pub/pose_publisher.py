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
        """
        points: numpy array of shape (n,3) or (n,4)
        columns: [x,y,z,(optional class)]
        """

        assert points.ndim == 1


        points = np.asarray(points, dtype=np.float32)

        msg = PoseArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame

        pose = Pose()

        pose.position.x = float(points[0])
        pose.position.y = float(points[1])
        pose.position.z = float(points[2])

        # neutral orientation
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        msg.poses.append(pose)


        return msg