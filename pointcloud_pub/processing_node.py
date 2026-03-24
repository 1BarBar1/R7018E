from pointcloud_pub.vlm import Clipseg
import numpy as np
import math
import time
from pointcloud_pub.pointcloud_publisher import PointCloudPublisher
from pointcloud_pub.pose_publisher import PosePublisher
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



class ProcessingNode(Node):
    def __init__(self,seg_human, seg_obstacle):
        super().__init__('processing_node')
        self.seg_human = seg_human
        self.seg_obstacle = seg_obstacle
        self.bridge = CvBridge()
        self.pc_human_pub = PointCloudPublisher(topic_name='human')
        self.pc_obstacle_pub = PointCloudPublisher(topic_name='obstacle')
        self.po_pub = PosePublisher()
        self.K = None

        self.frame ="camera1_depth_optical_frame"

        self.color_frame = None
        self.depth_frame = None

        # Match the Image topics (Reliable + Transient Local)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Match the Camera Info (Reliable + Volatile)
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Image subscribers (using the Transient Local profile)
        self.color_sub = Subscriber(self, Image, '/camera1/color/image_raw', qos_profile=image_qos)
        self.depth_sub = Subscriber(self, Image, '/camera1/depth/image_rect_raw', qos_profile=image_qos)

        # Camera Info subscriber (using the Volatile profile)
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera1/depth/camera_info',
            self.info_cb,
            info_qos
        )


        self.time_sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=5,
            slop = 0.05
        )
        self.time_sync.registerCallback(self.synced_cb)


    def info_cb(self, msg):
            if self.K is None:
                self.K = np.array(msg.k).reshape(3,3)
                self.frame = msg.header.frame_id

                self.get_logger().info("Camera info received")

                self.destroy_subscription(self.info_sub)


    def transformation(self, points):

        xyz = points[:, :3]      # spatial coordinates
        classes = points[:, 3]   # class labels

        # --- Rotation Z (90°)
        a = math.radians(-90)
        Rz = np.array([
            [math.cos(a), -math.sin(a), 0],
            [math.sin(a),  math.cos(a), 0],
            [0, 0, 1]
        ])

        # --- Rotation Y (90°)
        a = math.radians(90)
        Ry1 = np.array([
            [math.cos(a), 0, math.sin(a)],
            [0, 1, 0],
            [-math.sin(a), 0, math.cos(a)]
        ])

        # --- Rotation Y (42°)
        a = math.radians(50)
        Ry2 = np.array([
            [math.cos(a), 0, math.sin(a)],
            [0, 1, 0],
            [-math.sin(a), 0, math.cos(a)]
        ])


        rotated = xyz @ Rz.T
        rotated = rotated @ Ry1.T
        rotated = rotated @ Ry2.T


        rotated[:, 2] += 1.2

        return np.column_stack((rotated, classes))
    def synced_cb(self,color_msg,depth_msg):
        if self.K is None:
            print("no K")
            return

        color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        depth_image = depth_frame.astype(np.uint16)
        fx = self.K[0,0]
        fy = self.K[1,1]
        cx = self.K[0,2]
        cy = self.K[1,2]

        if self.seg_human:
                mask_points = []
                mask_human, logits = self.seg_human.get_segmentation(color_frame)
                conf_human = mask_human.max(axis=0)
                ys, xs = np.where(conf_human > 0.08)


                for v,u in zip(ys, xs):
                    Z = depth_image[v,u] / 1000.0
                    if Z <= 0:
                        continue
                    if Z > 6.0:
                        continue


                    X = (u-cx)*Z/fx
                    Y = (v-cy)*Z/fy

                    mask_points.append([X,Y,Z,0])
                try:
                    mask_points = np.asarray(mask_points)
                    human = mask_points[mask_points[:,3]==0]

                    X = human[:, :3]        # xyz
                    classes = human[:, 3]   # class labels

                    k = 1
                    max_iters = 2

                    centroids = np.zeros((k, 3))

                    for _ in range(max_iters):

                        if X.shape[0] == 0:
                            break

                        current_mean = np.mean(X, axis=0)

                        if np.allclose(centroids[0], current_mean, atol=1e-4):
                            break

                        centroids[0] = current_mean

                        distances = np.linalg.norm(X - centroids[0], axis=1)
                        distance_mask = distances <= 0.8

                        X = X[distance_mask]
                        classes = classes[distance_mask]


                    if classes.size > 0:
                        centroid_class = np.bincount(classes.astype(int)).argmax()
                    else:
                        centroid_class = -1


                    centroid_output = np.hstack((centroids, [[centroid_class]]))



                    centroid_transformed = self.transformation(centroid_output)
                    human = self.transformation(human)


                    if human.size > 0:

                        msg_human = self.pc_human_pub.create_pointcloud2(human,self.frame)
                        self.pc_human_pub.publisher.publish(msg_human)

                    msg_pose = self.po_pub.create_pose_array(centroid_transformed, self.frame)
                    self.po_pub.publisher.publish(msg_pose)
                except (IndexError,ValueError):
                    print("mask error")

        #start = time.time()
        if self.seg_obstacle:
            mask_points = []

            mask_obstacle, logits = self.seg_obstacle.get_segmentation(color_frame)
            conf_obstacle = mask_obstacle.max(axis=0)

            ys, xs = np.where(conf_obstacle > 0.08)
            #start1 = time.time()
            for v,u in zip(ys, xs):
                Z = depth_image[v,u] / 1000.0
                if Z <= 0:
                    continue
                if Z > 3.0:
                    continue


                X = (u-cx)*Z/fx
                Y = (v-cy)*Z/fy

                mask_points.append([X,Y,Z,1])
            try:

                mask_points = np.asarray(mask_points)
                obstacle = mask_points[mask_points[:,3]==1]
                obstacle = transformation(obstacle)
                if obstacle.size > 0:
                    msg_obs = self.pc_obstacle_pub.create_pointcloud2(obstacle,self.frame)
                    self.pc_obstacle_pub.publisher.publish(msg_obs)
            except (IndexError,ValueError):
                print("mask error")


        self.color_frame = None
        self.depth_frame = None


def main(args=None):
    track_obs = False
    if track_obs:
        #Clipseg(prompts=["human"])
        seg_human = None
        seg_obstacle = Clipseg(["obstacle"])
    else:
        seg_human = Clipseg(prompts=["human"])
        seg_obstacle = None

    rclpy.init(args=args)
    node = ProcessingNode(seg_human, seg_obstacle)


    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pc_human_pub.destroy_node()
        node.pc_obstacle_pub.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

