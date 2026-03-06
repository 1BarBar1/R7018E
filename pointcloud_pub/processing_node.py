from pointcloud_pub.pointcloud_publisher import PointCloudPublisher
from pointcloud_pub.pose_publisher import PosePublisher
from message_filters import Subscriber, ApproximateTimeSynchronizer
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
#import open3d as o3d
from pointcloud_pub.depth import Depth
from pointcloud_pub.vlm import Clipseg
import time


class ProcessingNode(Node):
    def __init__(self,seg):
        super().__init__('processing_node')
        self.seg = seg
        self.bridge = CvBridge()
        self.pc_pub = PointCloudPublisher()
        self.po_pub = PosePublisher()

        self.K = np.array([
            [389.0664978027344,   0.0,                319.8500061035156],
            [0.0,                 389.0664978027344,  237.91696166992188],
            [0.0,                 0.0,                1.0]
            ], dtype=np.float32)
        self.frame ="camera_depth_optical_frame"

        self.color_frame = None
        self.depth_frame = None

        qos = QoSProfile(depth=10)

        self.color_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_rect_raw')
        #self.create_subscription(Image, '/camera/depth/camera_info', self.info_cb, 10)

        self.time_sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=5,
            slop = 0.05
        )
        self.time_sync.registerCallback(self.synced_cb)



    '''
    def color_cb(self, msg):

        self.try_process_frame()

    def depth_cb(self, msg):

        self.try_process_frame()
    '''
    '''
    def info_cb(self, msg):
        self.K = np.array(msg.k).reshape(3,3)
        self.frame = msg.frame_id
        self.try_process_frame()
    '''

    def synced_cb(self,color_msg,depth_msg):

        color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        mask,logits = self.seg.get_segmentation(color_frame)

        #depth = Depth(depth_frame, self.K)
        # Convert to numpy points
        #points = depth.o3dPoints_to_np()
        depth_image = depth_frame.astype(np.uint16)

        mask_points = []
        points_all = []
        class_map = mask.argmax(axis=0)
        conf = mask.max(axis=0)

        ys, xs = np.where(conf > 0.08)
        fx = self.K[0,0]
        fy = self.K[1,1]
        cx = self.K[0,2]
        cy = self.K[1,2]
        '''
        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                Z = depth_image[v,u] / 1000.0
                if Z <= 0:
                    continue
                if Z > 4.0:
                    continue

                X = (u-cx)*Z/fx
                Y = (v-cy)*Z/fy

                points_all.append([X,Y,Z])

        '''
        for v,u in zip(ys, xs):
            Z = depth_image[v,u] / 1000.0
            if Z <= 0:
                 continue
            if class_map[v,u] == 1:
                if Z > 2.0:
                    continue
            if class_map[v,u] == 1:
                if Z > 6.0:
                    continue
            X = (u-cx)*Z/fx
            Y = (v-cy)*Z/fy

            mask_points.append([X,Y,Z,class_map[v,u]])



        #seg.visulize(mask)
        try:
            point_all = np.asarray(points_all)
            mask_points = np.asarray(mask_points)
            human = mask_points[mask_points[:,3]==0]
            obstacle = mask_points[mask_points[:,3]==1]
            # data: shape (n_samples, 4)
            print("Human shape",human.shape)
            X = human[:, :3]   # keep x,y,z only

            k = 1
            max_iters = 2


            # Initialize centroids with a starting value
            centroids = np.array([0.0, 0.0, 0.0])
            original_X = X.copy() # Keep the original data if you need it
            print(original_X.shape)
            for _ in range(max_iters):
                if X.shape[0] == 0:
                    break

                # 1. Compute current mean
                current_mean = np.mean(X, axis=0)

                # 2. Check for convergence (did the mean move?)
                if np.allclose(centroids, current_mean, atol=1e-4):
                    break

                centroids = current_mean

                # 3. Filter points for the NEXT iteration
                distances = np.linalg.norm(X - centroids, axis=1)
                # Strategy: Use a relative threshold or a fixed one if units are known
                distance_mask = distances <= 1.2
                X = X[distance_mask, :]
            print(X.shape)
            print(centroids.shape)
            # Publish using your publisher
            if human.size > 0:
                msg_human = self.pc_pub.create_pointcloud2(human,self.frame)
                self.pc_pub.publisher.publish(msg_human)

            msg_pose = self.po_pub.create_pose_array(centroids, self.frame)
            self.po_pub.publisher.publish(msg_pose)
            '''
            if obstacle.size > 0:
                msg_obs = self.pc_pub.create_pointcloud2(obstacle,self.frame)
                self.pc_pub.publisher.publish(msg_obs)
            '''
            '''
            msg_points = self.pc_pub.create_pointcloud2(points_points,self.frame)
            self.pc_pub.publisher.publish(msg_points)
            '''
        except (ValueError):
            print("mask error")
        # Clear frames after processing

        self.color_frame = None
        self.depth_frame = None


def main(args=None):
    seg = Clipseg()
    rclpy.init(args=args)
    node = ProcessingNode(seg)


    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pc_pub.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

