# creates a pointcloud file where points are in red if they overlap (within a certain tiny chamfer distance)
# all other points are white
# takes in two pointclouds, outputs one

import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def compute_chamfer_distance(pcd1, pcd2):
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
    
    distances = []
    for point in pcd1.points:
        [_, idx, _] = pcd2_tree.search_knn_vector_3d(point, 1)
        distances.append(np.linalg.norm(np.asarray(point) - np.asarray(pcd2.points)[idx[0]]))
    
    return np.array(distances)

def mark_overlapping_points(pcd1, pcd2, threshold):
    distances1 = compute_chamfer_distance(pcd1, pcd2)
    distances2 = compute_chamfer_distance(pcd2, pcd1)
    print(distances1)
    print(distances2)
    
    colors1 = np.array([[1, 0, 0] if d < threshold else [1, 1, 1] for d in distances1])
    colors2 = np.array([[1, 0, 0] if d < threshold else [0, 0, 1] for d in distances2])
    
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)

def combine_point_clouds(pcd1, pcd2):
    combined_pcd = pcd1 + pcd2
    return combined_pcd

def save_point_cloud(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd)

def scale_point_cloud(pcd, scale_factor):
    pcd.scale(scale_factor, center=pcd.get_center())
    return pcd

def main(ply_file1, ply_file2, output_file, threshold=5):
    pcd1 = load_point_cloud(ply_file1)
    pcd2 = load_point_cloud(ply_file2)
    
    # Compute bounding boxes
    bbox1 = pcd1.get_axis_aligned_bounding_box()
    bbox2 = pcd2.get_axis_aligned_bounding_box()
    
    # Determine scaling factor
    scale_factor = min(bbox1.get_extent()) / max(bbox2.get_extent())
    
    # Scale the larger point cloud
    if bbox1.volume() < bbox2.volume():
        pcd2 = scale_point_cloud(pcd2, scale_factor)
    else:
        pcd1 = scale_point_cloud(pcd1, scale_factor)
    
    mark_overlapping_points(pcd1, pcd2, threshold)
    
    combined_pcd = combine_point_clouds(pcd1, pcd2)
    
    save_point_cloud(combined_pcd, output_file)

if __name__ == "__main__":
    ply_file1 = "/afs/inf.ed.ac.uk/user/s21/s2150959/pifuhd/pointcloud/results/dataset_pc_compact.ply"
    ply_file2 = "/afs/inf.ed.ac.uk/user/s21/s2150959/pifuhd/pointcloud/results/pifuhd_pc_compact.ply"
    output_file = "/afs/inf.ed.ac.uk/user/s21/s2150959/pifuhd/pointcloud/results/combined_pointcloud.ply"
    
    main(ply_file1, ply_file2, output_file)