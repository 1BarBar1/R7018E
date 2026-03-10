from pointcloud_pub.pointcloud_publisher import PointCloudPublisher
from pointcloud_pub.pose_publisher import PosePublisher
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo
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
        self.pc_human_pub = PointCloudPublisher(topic_name='human')
        self.pc_obstacle_pub = PointCloudPublisher(topic_name='obstacle')
        self.po_pub = PosePublisher()
        self.K = None
        '''
        np.array([
            [908.531982421875,   0.0,                639.4365844726562],
            [0.0,                 907.80322265625,  365.8372497558594],
            [0.0,                 0.0,                1.0]
            ], dtype=np.float32)
        '''
        self.frame ="camera_depth_optical_frame"

        self.color_frame = None
        self.depth_frame = None

        qos = QoSProfile(depth=10)

        self.color_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_rect_raw')

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.info_cb,
            10
        )
        #self.create_subscription(Image, '/camera/depth/camera_info', self.info_cb, 10)

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


    def synced_cb(self,color_msg,depth_msg):
        if self.K is None:
            print("no K")
            return

        color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        #start = time.time()
        mask, logits = self.seg.get_segmentation(color_frame)
        #print("seg time:", time.time() - start)


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

        #start1 = time.time()
        for v,u in zip(ys, xs):
            Z = depth_image[v,u] / 1000.0
            if Z <= 0:
                 continue
            if Z > 2.0:
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
        #print("proj time:", time.time() - start1)


        #seg.visulize(mask)
        try:
            point_all = np.asarray(points_all)
            mask_points = np.asarray(mask_points)
            human = mask_points[mask_points[:,3]==0]
            obstacle = mask_points[mask_points[:,3]==1]
            # data: shape (n_samples, 4)
            X = human[:, :3]   # keep x,y,z only

            k = 1
            max_iters = 2


            # Initialize centroids with a starting value
            centroids = np.array([0.0, 0.0, 0.0])
            original_X = X.copy() # Keep the original data if you need it
            #start2 = time.time()
            print('proj',X.shape[0] )
            for _ in range(max_iters):
                if X.shape[0] == 0:
                    break

                # 1. Compute current mean
                current_mean = np.mean(X, axis=0)

                # 2. Check for convergence (did the mean move?)
                if np.allclose(centroids, current_mean, atol=1e-4):
                    pass

                centroids = current_mean
                print(centroids)
                # 3. Filter points for the NEXT iteration
                distances = np.linalg.norm(X - centroids, axis=1)
                # Strategy: Use a relative threshold or a fixed one if units are known
                distance_mask = distances <= 1.2
                X = X[distance_mask, :]
            #print("centroid time:", time.time() - start2)
            # Publish using your publisher

            if human.size > 0:
                msg_human = self.pc_human_pub.create_pointcloud2(human,self.frame)
                self.pc_human_pub.publisher.publish(msg_human)

            msg_pose = self.po_pub.create_pose_array(centroids, self.frame)
            self.po_pub.publisher.publish(msg_pose)

            if obstacle.size > 0:
                msg_obs = self.pc_obstacle_pub.create_pointcloud2(obstacle,self.frame)
                self.pc_obstacle_pub.publisher.publish(msg_obs)



        except (IndexError,ValueError):
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
        node.pc_human_pub.destroy_node()
        node.pc_obstacle_pub.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

