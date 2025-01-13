import numpy as np
import open3d as o3d

def load_point_cloud(file_path):
    """
    Load point cloud from various file formats.
    Supports: .ply, .obj, .pcd
    
    Args:
        file_path (str): Path to the point cloud file
        
    Returns:
        numpy.ndarray: Array of shape (N, 3) containing point coordinates
    """
    # Get file extension
    ext = file_path.split('.')[-1].lower()
    
    if ext == 'ply':
        pcd = o3d.io.read_point_cloud(file_path)
    elif ext == 'obj':
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
    elif ext == 'pcd':
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Convert to numpy array
    points = np.asarray(pcd.points)
    
    return points

def preprocess_point_cloud(points, normalize=True, remove_outliers=True):
    """
    Preprocess point cloud data.
    
    Args:
        points (numpy.ndarray): Input point cloud
        normalize (bool): Whether to normalize the point cloud
        remove_outliers (bool): Whether to remove statistical outliers
        
    Returns:
        numpy.ndarray: Preprocessed point cloud
    """
    if remove_outliers:
        # Convert to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Remove statistical outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        points = np.asarray(cl.points)
    
    if normalize:
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit height
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        points = points / height
    
    return points

def save_point_cloud(points, file_path):
    """
    Save point cloud to file.
    
    Args:
        points (numpy.ndarray): Point cloud to save
        file_path (str): Output file path
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    o3d.io.write_point_cloud(file_path, pcd)