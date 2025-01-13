import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import distance
import open3d as o3d

class BoundingBoxCalculator:
    def __init__(self):
        self.body_parts = {
            'head': None,
            'torso': None,
            'left_arm': None,
            'right_arm': None,
            'left_hand': None,
            'right_hand': None,
            'left_leg': None,
            'right_leg': None,
            'left_foot': None,
            'right_foot': None
        }
        
        self.colors = {
            'head': [1, 0, 0],      # Red
            'torso': [0, 1, 0],     # Green
            'left_arm': [0, 0, 1],  # Blue
            'right_arm': [1, 1, 0],  # Yellow
            'left_hand': [1, 0, 1],  # Magenta
            'right_hand': [0, 1, 1], # Cyan
            'left_leg': [0.5, 0, 0.5],  # Purple
            'right_leg': [0.5, 0.5, 0], # Olive
            'left_foot': [1, 0.5, 0],   # Orange
            'right_foot': [0, 0.5, 1]   # Light blue
        }

    def estimate_skeleton(self, points):
        """Estimate key joint positions from point cloud."""
        # First, find the torso center using DBSCAN
        clustering = DBSCAN(eps=0.1, min_samples=5).fit(points)
        largest_cluster = np.where(clustering.labels_ == 0)[0]
        torso_center = np.mean(points[largest_cluster], axis=0)

        # Estimate key joints using local point density and geometric constraints
        joints = {}
        
        # Find head by looking for top cluster
        top_points = points[points[:, 1] > np.percentile(points[:, 1], 85)]
        if len(top_points) > 0:
            joints['head'] = np.mean(top_points, axis=0)

        # Find shoulders using width and height
        shoulder_height = np.percentile(points[:, 1], 80)
        left_points = points[(points[:, 1] > shoulder_height * 0.9) & 
                           (points[:, 1] < shoulder_height * 1.1) &
                           (points[:, 0] < torso_center[0])]
        right_points = points[(points[:, 1] > shoulder_height * 0.9) & 
                            (points[:, 1] < shoulder_height * 1.1) &
                            (points[:, 0] > torso_center[0])]
        
        if len(left_points) > 0:
            joints['left_shoulder'] = np.mean(left_points, axis=0)
        if len(right_points) > 0:
            joints['right_shoulder'] = np.mean(right_points, axis=0)

        # Find hands using extremity detection
        extremities = self._find_extremities(points, 5)
        if len(extremities) >= 2:
            # Classify extremities as hands based on height and distance from shoulders
            potential_hands = extremities[extremities[:, 1] > np.median(points[:, 1])]
            if len(potential_hands) >= 2:
                left_hand = potential_hands[np.argmin(potential_hands[:, 0])]
                right_hand = potential_hands[np.argmax(potential_hands[:, 0])]
                joints['left_hand'] = left_hand
                joints['right_hand'] = right_hand

        # Find feet using lowest points
        bottom_points = points[points[:, 1] < np.percentile(points[:, 1], 15)]
        if len(bottom_points) > 0:
            # Split into left and right
            left_foot = bottom_points[bottom_points[:, 0] < np.median(bottom_points[:, 0])]
            right_foot = bottom_points[bottom_points[:, 0] >= np.median(bottom_points[:, 0])]
            
            if len(left_foot) > 0:
                joints['left_foot'] = np.mean(left_foot, axis=0)
            if len(right_foot) > 0:
                joints['right_foot'] = np.mean(right_foot, axis=0)

        return joints

    def _find_extremities(self, points, n_extremities):
        """Find extremity points using distance from centroid."""
        centroid = np.mean(points, axis=0)
        distances = distance.cdist([centroid], points)[0]
        return points[np.argsort(distances)[-n_extremities:]]

    def segment_using_joints(self, points, joints):
        """Segment points using estimated joint positions."""
        segments = {}
        
        for i, point in enumerate(points):
            # Calculate distances to each joint
            distances = {part: np.linalg.norm(point - pos) 
                       for part, pos in joints.items()}
            
            # Assign point to nearest body part with some geometric constraints
            if 'head' in distances and distances['head'] < 0.2:
                segments[i] = 'head'
            elif point[1] > np.percentile(points[:, 1], 60):  # Upper body
                if point[0] < joints.get('torso', [0])[0]:
                    if 'left_hand' in joints and np.linalg.norm(point - joints['left_hand']) < 0.15:
                        segments[i] = 'left_hand'
                    else:
                        segments[i] = 'left_arm'
                else:
                    if 'right_hand' in joints and np.linalg.norm(point - joints['right_hand']) < 0.15:
                        segments[i] = 'right_hand'
                    else:
                        segments[i] = 'right_arm'
            elif point[1] < np.percentile(points[:, 1], 20):  # Lower body
                if point[0] < joints.get('torso', [0])[0]:
                    if 'left_foot' in joints and np.linalg.norm(point - joints['left_foot']) < 0.1:
                        segments[i] = 'left_foot'
                    else:
                        segments[i] = 'left_leg'
                else:
                    if 'right_foot' in joints and np.linalg.norm(point - joints['right_foot']) < 0.1:
                        segments[i] = 'right_foot'
                    else:
                        segments[i] = 'right_leg'
            else:  # Middle body
                segments[i] = 'torso'
                
        return segments

    def calculate_bboxes(self, points):
        """Calculate bounding boxes using improved joint-based segmentation."""
        # Normalize points
        normalized_points = self._normalize_points(points)
        
        # Estimate skeleton joints
        joints = self.estimate_skeleton(normalized_points)
        
        # Segment points using joints
        segments = self.segment_using_joints(normalized_points, joints)
        
        # Calculate bounding boxes for each segment
        for part_name in self.body_parts.keys():
            self.body_parts[part_name] = self._compute_bbox(normalized_points, segments, part_name)
        
        return self.body_parts
    
    def _normalize_points(self, points):
        """Normalize point cloud to standard position and scale."""
        centroid = np.mean(points, axis=0)
        normalized = points - centroid
        
        # Scale to unit height
        height = np.max(normalized[:, 1]) - np.min(normalized[:, 1])
        if height != 0:
            normalized = normalized / height
            
        return normalized
    
    def _segment_points(self, points):
        """Segment points into body parts using height-based thresholds."""
        segments = {}
        heights = np.percentile(points[:, 1], [0, 5, 25, 55, 85, 95, 100])
        median_x = np.median(points[:, 0])
        
        for i, point in enumerate(points):
            x, y, z = point
            
            # Segment based on height and left/right position
            if y >= heights[5]:  # Head region
                segments[i] = 'head'
            elif y >= heights[3]:  # Torso and arms region
                if x < median_x - 0.15:
                    segments[i] = 'left_arm' if y < heights[4] else 'left_hand'
                elif x > median_x + 0.15:
                    segments[i] = 'right_arm' if y < heights[4] else 'right_hand'
                else:
                    segments[i] = 'torso'
            elif y >= heights[1]:  # Legs region
                if x < median_x:
                    segments[i] = 'left_leg'
                else:
                    segments[i] = 'right_leg'
            else:  # Feet region
                if x < median_x:
                    segments[i] = 'left_foot'
                else:
                    segments[i] = 'right_foot'
        
        return segments
    
    def _compute_bbox(self, points, segments, part_name):
        """Compute bounding box for a specific body part."""
        # Get points for this part
        part_indices = [i for i, part in segments.items() if part == part_name]
        if not part_indices:
            return None
            
        part_points = points[part_indices]
        
        # Compute tight bounding box
        min_coords = np.min(part_points, axis=0)
        max_coords = np.max(part_points, axis=0)
        
        # Add small padding
        padding = 0.01
        min_coords -= padding
        max_coords += padding
        
        return {
            'min': min_coords.tolist(),
            'max': max_coords.tolist(),
            'center': ((min_coords + max_coords) / 2).tolist(),
            'dimensions': (max_coords - min_coords).tolist()
        }
    
    def color_points(self, points, bounding_boxes):
        """Color points based on their body part segment."""
        colors = np.ones((len(points), 3)) * 0.7  # Default gray
        
        for i, point in enumerate(points):
            for part_name, bbox in bounding_boxes.items():
                if bbox is not None and self._point_in_bbox(point, bbox):
                    colors[i] = self.colors[part_name]
                    break
        
        return colors
    
    def _point_in_bbox(self, point, bbox):
        """Check if point is inside bounding box."""
        min_coords = np.array(bbox['min'])
        max_coords = np.array(bbox['max'])
        return np.all(point >= min_coords) and np.all(point <= max_coords)
    
    def save_colored_ply(self, points, output_path):
        """Save point cloud with colors to PLY file."""
        colors = self.color_points(points, self.body_parts)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)