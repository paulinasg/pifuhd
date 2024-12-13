import numpy as np
import pymeshlab

def get_bounding_box_dimensions(meshset):
    bbox = meshset.current_mesh().bounding_box()
    min_corner = bbox.min()
    max_corner = bbox.max()
    dimensions = max_corner - min_corner
    return dimensions

def scale_mesh_to_fit_bounding_box(meshset, target_bbox):
    # Compute the current bounding box of the mesh
    current_bbox = meshset.current_mesh().bounding_box()
    current_height = current_bbox.dim_y()

    # Compute target height
    target_height = target_bbox[4] - target_bbox[1]  # height

    # Compute uniform scaling factor based on height
    scale_factor = target_height / current_height

    # Translate the mesh to the origin
    center = current_bbox.center()
    meshset.apply_filter('compute_matrix_from_translation', axisx=-center[0],
                         axisy=-center[1], axisz=-center[2])

    # Apply the uniform scaling transformation
    meshset.apply_filter('compute_matrix_from_scaling_or_normalization',
                         axisx=scale_factor, axisy=scale_factor, axisz=scale_factor,
                         uniformflag=True, scalecenter='origin', freeze=True)

    # Translate the mesh back to its original position
    meshset.apply_filter('compute_matrix_from_translation', axisx=center[0],
                         axisy=center[1], axisz=center[2])

    # Return the updated meshset
    return meshset

# Load the OBJ file
ms = pymeshlab.MeshSet()
#ms.load_new_mesh("/afs/inf.ed.ac.uk/user/s21/s2150959/pifuhd/pointcloud/dataset_samples/rp_posed_00178_29.obj")
ms.load_new_mesh("/afs/inf.ed.ac.uk/user/s21/s2150959/pifuhd/results/pifuhd_final/recon/result_test_512.obj")

# Define the target bounding box dimensions
target_bbox = [0, 0, 0, 0.4650, 1.5390, 0.5020]  # [x_min, y_min, z_min, x_max, y_max, z_max]

# Scale the mesh to fit within the target bounding box
ms = scale_mesh_to_fit_bounding_box(ms, target_bbox)


# Compute a point cloud from the mesh
ms.generate_sampling_poisson_disk(samplenum=10000)

# Get and print bounding box dimensions
dimensions = get_bounding_box_dimensions(ms)
print(f"Bounding box dimensions: {dimensions}")

# Save the point cloud
ms.save_current_mesh("results/pifuhd_pc_compact.ply", binary=False)
#ms.save_current_mesh("results/dataset_pc_compact.ply", binary=False)