import numpy as np

class KD_tree_node:
    def __init__(self, point, index, axis, left=None, right=None):
        self.point = point[:3]  # The coordinate vector (e.g., [x, y, z])
        # print(self.point) # You might want to remove this so it doesn't spam your ROS terminal!
        self.index = index      # Fixed typo here
        self.axis = axis
        self.left = left
        self.right = right
        self.category = point[3]

class KD_tree:
    def __init__(self, points):
        original_indices = np.arange(len(points))
        self.root = self._build_tree(points, original_indices)

    def _build_tree(self, points, indices, depth=0):
        if len(points) == 0:
            return None

        k = points.shape[1] - 1  # Excellent adaptation for N,4 arrays!
        axis = depth % k

        sort_order = points[:, axis].argsort()

        sorted_points = points[sort_order]
        sorted_indices = indices[sort_order]

        median_idx = len(sorted_points) // 2

        return KD_tree_node(
            point=sorted_points[median_idx],
            index=sorted_indices[median_idx],
            axis=axis,
            left=self._build_tree(
                sorted_points[:median_idx],
                sorted_indices[:median_idx],
                depth + 1
            ),
            right=self._build_tree(
                sorted_points[median_idx + 1:],
                sorted_indices[median_idx + 1:],
                depth + 1
            )
        )

    def search_radius(self, query_point, radius):
        results = []
        self._search_radius(self.root, query_point, radius, results)
        return results

    def _search_radius(self, node, query_point, radius, results, depth=0):
        if node is None:
            return

        k = len(query_point)
        axis = depth % k

        dist = np.linalg.norm(node.point - query_point)
        if dist <= radius:
            results.append(node.index)

        if query_point[axis] < node.point[axis]:
            next_branch = node.left
            opposite_branch = node.right
        else:
            next_branch = node.right
            opposite_branch = node.left

        self._search_radius(next_branch, query_point, radius, results, depth + 1)

        if abs(query_point[axis] - node.point[axis]) <= radius:
            self._search_radius(opposite_branch, query_point, radius, results, depth + 1)

        # Removed the erroneous return statement here