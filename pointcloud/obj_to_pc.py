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
    current_dims = [
        current_bbox.dim_x(),
        current_bbox.dim_y(),
        current_bbox.dim_z()
    ]

    # Compute target dimensions
    target_dims = [
        target_bbox[3] - target_bbox[0],  # width
        target_bbox[4] - target_bbox[1],  # height
        target_bbox[5] - target_bbox[2]   # depth
    ]

    # Compute scaling factors for each axis
    scaling_factors = [
        target_dims[0] / current_dims[0],  # Scale factor for X axis
        target_dims[1] / current_dims[1],  # Scale factor for Y axis
        target_dims[2] / current_dims[2]   # Scale factor for Z axis
    ]

    # Apply the scaling transformation
    meshset.apply_filter('compute_matrix_from_scaling_or_normalization',
                         axisx=scaling_factors[0],
                         axisy=scaling_factors[1],
                         axisz=scaling_factors[2],
                         uniformflag=False,  # Allow non-uniform scaling
                         scalecenter='origin',  # Scale with respect to origin
                         freeze=True)  # Apply the transformation immediately

    # Return the updated meshset
    return meshset

# Load the OBJ file
ms = pymeshlab.MeshSet()
ms.load_new_mesh("/afs/inf.ed.ac.uk/user/s21/s2150959/pifuhd/pointcloud/dataset_samples/rp_posed_00178_29.obj")
#ms.load_new_mesh("/afs/inf.ed.ac.uk/user/s21/s2150959/pifuhd/results/pifuhd_final/recon/result_test_512.obj")

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
#ms.save_current_mesh("results/pifuhd_pc_compact.ply", binary=False)
ms.save_current_mesh("results/dataset_pc_compact.ply", binary=False)