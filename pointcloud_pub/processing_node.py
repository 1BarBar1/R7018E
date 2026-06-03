import numpy as np
import math
import time
from pointcloud_pub.vlm import Clipseg
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
from pointcloud_pub.KD_tree import KD_tree_node, KD_tree


def euclidean_clustering(points, tree, distance_threshold, min_cluster_size=10):
    """Groups points into distinct clusters based on spatial proximity."""
    points = points[:, :3]      # spatial coordinates
    clusters = []
    processed = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if processed[i]:
            continue

        current_cluster = []
        queue = [i]
        processed[i] = True

        while len(queue) > 0:
            current_idx = queue.pop(0)
            current_cluster.append(current_idx)

            query_point = points[current_idx]
            neighbor_indices = tree.search_radius(query_point, distance_threshold)

            for neighbor_idx in neighbor_indices:
                if not processed[neighbor_idx]:
                    processed[neighbor_idx] = True
                    queue.append(neighbor_idx)

        if len(current_cluster) >= min_cluster_size:
            clusters.append(current_cluster)

    return clusters


def extract_centroids(points, clusters):
    """
    Calculates the centroid of each cluster and appends the cluster index.
    Returns an (M, 4) array where M is the number of valid clusters.
    """
    num_clusters = len(clusters)

    # 1. Pre-allocate an (M, 4) array. It defaults to all zeros.
    centroids_array = np.zeros((num_clusters, 4))

    # 2. Iterate through the clusters provided by the KD-Tree
    for i, cluster_indices in enumerate(clusters):

        # Extract just the (x, y, z) data for the points in this specific cluster
        object_points = points[cluster_indices, :3]

        # Calculate the geometric center (mean along the column axis)
        centroid_xyz = np.mean(object_points, axis=0)

        # Assign the (x, y, z) mean to the first 3 columns
        centroids_array[i, :3] = centroid_xyz

        # Assign the cluster index 'i' to the 4th column
        centroids_array[i, 3] = i

    return centroids_array


class ProcessingNode(Node):
    def __init__(self,seg_human, seg_obstacle):
        super().__init__('processing_node')
        print("hallå")
        self.seg_human = seg_human
        self.seg_obstacle = seg_obstacle
        self.bridge = CvBridge()
        self.pc_human_pub = PointCloudPublisher(topic_name='human_point')
        self.pc_obstacle_pub = PointCloudPublisher(topic_name='obstacle_point')
        self.po_pub_human = PosePublisher(topic_name='human')
        self.po_pub_obstacle = PosePublisher(topic_name='obstacle')
        self.K = None
        '''
        np.array([
            [908.531982421875,   0.0,                639.4365844726562],
            [0.0,                 907.80322265625,  365.8372497558594],
            [0.0,                 0.0,                1.0]
            ], dtype=np.float32)
        '''
        self.frame ="camera1_depth_optical_frame"

        self.color_frame = None
        self.depth_frame = None

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )


        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )


        self.color_sub = Subscriber(self, Image, '/camera1/color/image_raw', qos_profile=image_qos)
        self.depth_sub = Subscriber(self, Image, '/camera1/depth/image_rect_raw', qos_profile=image_qos)


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
                self.fx = self.K[0,0]
                self.fy = self.K[1,1]
                self.cx = self.K[0,2]
                self.cy = self.K[1,2]

    def segment_pointcloud(self):
        mask_points = []
        mask, logits = self.seg_human.get_segmentation(self.color_frame)
        #conf= mask.max(axis=0)
        print(mask.shape)
        ys, xs = np.where(mask[0] > 0.2)



        for v,u in zip(ys, xs):
            Z = self.depth_image[v,u] / 1000.0
            if Z <= 0:
                continue
            if Z > 6.0:
                continue
            X = (u-self.cx)*Z/self.fx
            Y = (v-self.cy)*Z/self.fy

            mask_points.append([X,Y,Z,0])

        return mask_points

    def transformation(self, points):
        print(points.shape)
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
        a = math.radians(30)
        Ry2 = np.array([
            [math.cos(a), 0, math.sin(a)],
            [0, 1, 0],
            [-math.sin(a), 0, math.cos(a)]
        ])


        rotated = xyz @ Rz.T
        rotated = rotated @ Ry1.T
        #rotated = rotated @ Ry2.T
        rotated[:, 2] += 1.2

        return np.column_stack((rotated, classes))

    def voxel_downsmaple(self,points_classed, voxel_size):
        points = points_classed[:, :3]      # spatial coordinates
        classes = points_classed[:, 3]
        # 1. Compute voxel indices
        min_bound = np.min(points, axis=0)
        voxel_indices = np.floor((points - min_bound) / voxel_size).astype(np.int32)

        # 2. Find unique voxels and the mapping for each point
        unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

        # 3. Calculate the centroid of each voxel
        num_unique_voxels = len(unique_voxels)

        # Accumulate coordinate sums
        voxel_sums = np.zeros((num_unique_voxels, 3))
        np.add.at(voxel_sums, inverse_indices, points)

        # Count points per voxel
        voxel_counts = np.bincount(inverse_indices)

        # Compute mean
        centroids = voxel_sums / voxel_counts[:, np.newaxis]
        N = centroids.shape[0]
        return np.column_stack((centroids, np.zeros(N)))

    def search_nearest(self, query_point):
        """
        The public method your ROS2 node will actually call.
        """
        # It calls an internal recursive search function behind the scenes
        best_node, best_dist = self._closest_point(self.root, query_point)
    def search_radius(self, query_point, radius):
        """Public method to find all points within a specific distance."""
        results = []
        self._search_radius(self.root, query_point, radius, results)
        return results

    def synced_cb(self,color_msg,depth_msg):
        if self.K is None:
            print("no K")
            return

        self.color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        #color_frame = cv2.rotate(color_frame, cv2.ROTATE_180)
        #depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_180)
        self.depth_image = depth_frame.astype(np.uint16)
        fx = self.K[0,0]
        fy = self.K[1,1]
        cx = self.K[0,2]
        cy = self.K[1,2]

        if self.seg_human:
                mask_points_human = self.segment_pointcloud()

                try:

                    mask_points_human = np.asarray(mask_points_human)

                    human = mask_points_human[mask_points_human[:,3]==0]

                    voxel_size = 0.08
                    print("human shape",human[:, :3].shape)

                    human = self.voxel_downsmaple(human,voxel_size)

                    print("shape voxels, ",human.shape)
                    if len(human) == 0:
                        return

                    # 3. Build the KD-Tree using ONLY the downsampled points
                    spatial_tree = KD_tree(human)

                    # 4. Extract Clusters
                    # The threshold must bridge the gap between your voxels!
                    cluster_tolerance = voxel_size * 0.50
                    min_points_per_cluster = 5

                    clusters = euclidean_clustering(
                        points=human,
                        tree=spatial_tree,
                        distance_threshold=cluster_tolerance,
                        min_cluster_size=min_points_per_cluster
                    )
                    '''
                    X = human[:, :3]
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
                    '''
                    # Get the (M, 4) array of centroids [x, y, z, cluster_idx]
                    centroid_output = extract_centroids(human, clusters)

                    centroid_transformed_human = self.transformation(centroid_output)
                    human = self.transformation(human)
                    if human.size > 0:
                        msg_human = self.pc_human_pub.create_pointcloud2(human,self.frame)
                        self.pc_human_pub.publisher.publish(msg_human)
                    msg_pose_human = self.po_pub_human.create_pose_array(centroid_transformed_human, self.frame)
                    self.po_pub_human.publisher.publish(msg_pose_human)
                except (ValueError, IndexError):
                    print("mask error")


        if self.seg_obstacle:
            mask_points_obstacle = []

            mask_obstacle, logits = self.seg_obstacle.get_segmentation(color_frame)
            print("det är här du ksa kolla: ",mask_obstacle.shape)
            conf_obstacle = mask_obstacle.max(axis=0)

            ys, xs = np.where(conf_obstacle > 0.4)
            #start1 = time.time()
            for v,u in zip(ys, xs):
                Z = depth_image[v,u] / 1000.0
                if Z <= 0:
                    continue
                if Z > 3.0:
                    continue


                X = (u-cx)*Z/fx
                Y = (v-cy)*Z/fy

                mask_points_obstacle.append([X,Y,Z,1])
            try:

                mask_points_obstacle = np.asarray(mask_points_obstacle)
                obstacle = mask_points_obstacle[mask_points_obstacle[:,3]==1]
                X = obstacle[:, :3]        # xyz
                classes = obstacle[:, 3]   # class labels

                k = 1
                max_iters = 10

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



                centroid_transformed_obstacle = self.transformation(centroid_output)
                obstacle = self.transformation(obstacle)
                if obstacle.size > 0:
                    print("obstacale",obstacle.shape)
                    msg_obs = self.pc_obstacle_pub.create_pointcloud2(obstacle,self.frame)
                    self.pc_obstacle_pub.publisher.publish(msg_obs)

                msg_pose_obstacle = self.po_pub_obstacle.create_pose_array(centroid_transformed_obstacle, self.frame)
                self.po_pub_obstacle.publisher.publish(msg_pose_obstacle)
            except (ValueError, IndexError):
                print("mask error")


        self.color_frame = None
        self.depth_frame = None



def main(args=None):
    track_obs = False
    print("hallåå")
    if track_obs:
        #
        seg_human = Clipseg(prompts=["human"])
        seg_obstacle = Clipseg(["Blocks","Legos", "obstacles"])
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
        node.po_pub_human.destroy_node()
        node.po_pub_obstacle.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

